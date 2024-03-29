from collections import deque
import numpy as np
import torch
from torch.utils.data.sampler import WeightedRandomSampler


class Memory:
    def __init__(self, capacity, state_shape, feature_shape, action_shape, device):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.feature_shape = feature_shape
        self.action_shape = action_shape
        self.device = device
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(
        self, state, feature, action, reward, next_state, done, episode_done=None
    ):
        self._append(state, feature, action, reward, next_state, done)

    def _append(self, state, feature, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        feature = np.array(feature, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)

        if self.is_image:
            state = (state * 255).astype(np.uint8)
            next_state = (next_state * 255).astype(np.uint8)

        self.states[self._p] = state
        self.features[self._p] = feature
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.next_states[self._p] = next_state
        self.dones[self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        if self.is_image:
            states = self.states[indices].astype(np.float32) / 255.0
            next_states = self.next_states[indices].astype(np.float32) / 255.0
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]

        states = torch.FloatTensor(states).to(self.device)
        features = torch.FloatTensor(self.features[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)

        return states, features, actions, rewards, next_states, dones

    def __len__(self):
        return self._n

    def reset(self):
        self._n = 0
        self._p = 0

        self.states = np.zeros(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.features = np.zeros((self.capacity, *self.feature_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.features[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
        )

    def load_memory(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(slice(self._p, self._p + num_data), batch, slice(0, num_data))
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            self._insert(slice(self._p, self.capacity), batch, slice(0, mid_index))
            self._insert(slice(0, end_index), batch, slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        states, features, actions, rewards, next_states, dones = batch
        self.states[mem_indices] = states[batch_indices]
        self.features[mem_indices] = features[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]


class MyMultiStepBuff:
    keys = ["state", "feature", "action", "reward"]

    def __init__(self, maxlen=3):
        super().__init__()
        self.maxlen = int(maxlen)
        self.memory = {key: deque(maxlen=self.maxlen) for key in self.keys}

    def append(self, state, feature, action, reward):
        self.memory["state"].append(state)
        self.memory["feature"].append(feature)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        feature = self._multi_step_feature(gamma)
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        _ = self.memory["feature"].popleft()
        return state, feature, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([r * (gamma**i) for i, r in enumerate(self.memory["reward"])])

    def _multi_step_feature(self, gamma):
        return np.sum(
            [f * (gamma**i) for i, f in enumerate(self.memory["feature"])], 0
        )

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f"There is no key {key} in MultiStepBuff.")
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory["state"])


class MyMultiStepMemory(Memory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        device,
        gamma=0.99,
        multi_step=3,
    ):
        super().__init__(capacity, state_shape, feature_shape, action_shape, device)

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MyMultiStepBuff(maxlen=self.multi_step)

    def append(
        self, state, feature, action, reward, next_state, done, episode_done=False
    ):
        if self.multi_step != 1:
            self.buff.append(state, feature, action, reward)

            if len(self.buff) == self.multi_step:
                state, feature, action, reward = self.buff.get(self.gamma)
                self._append(state, feature, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self._append(state, feature, action, reward, next_state, done)


class MyPrioritizedMemory(MyMultiStepMemory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        device,
        gamma=0.99,
        multi_step=3,
        alpha=0.6,
        beta=0.4,
        beta_annealing=0.001,
        epsilon=1e-4,
    ):

        super().__init__(
            capacity,
            state_shape,
            feature_shape,
            action_shape,
            device,
            gamma,
            multi_step,
        )
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def append(
        self,
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        error,
        episode_done=False,
    ):
        if self.multi_step != 1:
            self.buff.append(state, feature, action, reward)

            if len(self.buff) == self.multi_step:
                state, feature, action, reward = self.buff.get(self.gamma)
                self.priorities[self._p] = self.calc_priority(error)
                self._append(state, feature, action, reward, next_state, done)

            if episode_done or done:
                self.buff.reset()
        else:
            self.priorities[self._p] = self.calc_priority(error)
            self._append(state, feature, action, reward, next_state, done)

    def update_priority(self, indices, errors):
        self.priorities[indices] = np.reshape(self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        self.beta = min(1.0 - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(self.priorities[: self._n, 0], batch_size)
        indices = list(sampler)
        batch = self._sample(indices)

        p = self.priorities[indices] / np.sum(self.priorities[: self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        return batch, indices, weights

    def reset(self):
        super().reset()
        self.priorities = np.empty((self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.features[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
            self.priorities[valid],
        )

    def _insert(self, mem_indices, batch, batch_indices):
        states, features, actions, rewards, next_states, dones, priorities = batch
        self.states[mem_indices] = states[batch_indices]
        self.features[mem_indices] = features[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.priorities[mem_indices] = priorities[batch_indices]


# ======================== RNN replay buffer =======================#


class RNNMemory:
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        hidden_shape0,
        hidden_shape1,
        device,
    ):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.feature_shape = feature_shape
        self.action_shape = action_shape
        self.hidden_shape0 = hidden_shape0
        self.hidden_shape1 = hidden_shape1
        self.device = device
        self.is_image = len(state_shape) == 3
        self.state_type = np.uint8 if self.is_image else np.float32

        self.reset()

    def append(
        self,
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        hin0,
        hout0,
        hin1,
        hout1,
        episode_done=None,
    ):
        self._append(
            state, feature, action, reward, next_state, done, hin0, hout0, hin1, hout1
        )

    def _append(
        self, state, feature, action, reward, next_state, done, hin0, hout0, hin1, hout1
    ):
        state = np.array(state, dtype=np.float32)
        feature = np.array(feature, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        hin0 = hin0.cpu()
        hout0 = hout0.cpu()
        hin1 = hin1.cpu()
        hout1 = hout1.cpu()

        if self.is_image:
            state = (state * 255).astype(np.uint8)
            next_state = (next_state * 255).astype(np.uint8)

        self.states[self._p] = state
        self.features[self._p] = feature
        self.actions[self._p] = action
        self.rewards[self._p] = reward
        self.next_states[self._p] = next_state
        self.dones[self._p] = done
        self.hin0[self._p] = hin0
        self.hout0[self._p] = hout0
        self.hin1[self._p] = hin1
        self.hout1[self._p] = hout1

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.randint(low=0, high=self._n, size=batch_size)
        return self._sample(indices)

    def _sample(self, indices):
        if self.is_image:
            states = self.states[indices].astype(np.float32) / 255.0
            next_states = self.next_states[indices].astype(np.float32) / 255.0
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]

        states = torch.FloatTensor(states).to(self.device)
        features = torch.FloatTensor(self.features[indices]).to(self.device)
        actions = torch.FloatTensor(self.actions[indices]).to(self.device)
        rewards = torch.FloatTensor(self.rewards[indices]).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(self.dones[indices]).to(self.device)
        hin0 = torch.FloatTensor(self.hin0[indices]).to(self.device).unsqueeze(0)
        hout0 = torch.FloatTensor(self.hout0[indices]).to(self.device).unsqueeze(0)
        hin1 = torch.FloatTensor(self.hin1[indices]).to(self.device).unsqueeze(0)
        hout1 = torch.FloatTensor(self.hout1[indices]).to(self.device).unsqueeze(0)

        return (
            states,
            features,
            actions,
            rewards,
            next_states,
            dones,
            hin0,
            hout0,
            hin1,
            hout1,
        )

    def __len__(self):
        return self._n

    def reset(self):
        self._n = 0
        self._p = 0

        self.states = np.zeros(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.features = np.zeros((self.capacity, *self.feature_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_states = np.zeros(
            (self.capacity, *self.state_shape), dtype=self.state_type
        )
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.hin0 = np.zeros((self.capacity, *self.hidden_shape0), dtype=np.float32)
        self.hout0 = np.zeros((self.capacity, *self.hidden_shape0), dtype=np.float32)
        self.hin1 = np.zeros((self.capacity, *self.hidden_shape1), dtype=np.float32)
        self.hout1 = np.zeros((self.capacity, *self.hidden_shape1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.features[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
            self.hin0[valid],
            self.hout0[valid],
            self.hin1[valid],
            self.hout1[valid],
        )

    def load_memory(self, batch):
        num_data = len(batch[0])

        if self._p + num_data <= self.capacity:
            self._insert(slice(self._p, self._p + num_data), batch, slice(0, num_data))
        else:
            mid_index = self.capacity - self._p
            end_index = num_data - mid_index
            self._insert(slice(self._p, self.capacity), batch, slice(0, mid_index))
            self._insert(slice(0, end_index), batch, slice(mid_index, num_data))

        self._n = min(self._n + num_data, self.capacity)
        self._p = (self._p + num_data) % self.capacity

    def _insert(self, mem_indices, batch, batch_indices):
        (
            states,
            features,
            actions,
            rewards,
            next_states,
            dones,
            hin0,
            hout0,
            hin1,
            hout1,
        ) = batch
        self.states[mem_indices] = states[batch_indices]
        self.features[mem_indices] = features[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.hin0[mem_indices] = hin0[batch_indices]
        self.hout0[mem_indices] = hout0[batch_indices]
        self.hin1[mem_indices] = hin1[batch_indices]
        self.hout1[mem_indices] = hout1[batch_indices]


class MyMultiStepRNNBuff:
    keys = ["state", "feature", "action", "reward", "hin0", "hout0", "hin1", "hout1"]

    def __init__(self, maxlen=3):
        super().__init__()
        self.maxlen = int(maxlen)
        self.memory = {key: deque(maxlen=self.maxlen) for key in self.keys}

    def append(self, state, feature, action, reward, hin0, hout0, hin1, hout1):
        self.memory["state"].append(state)
        self.memory["feature"].append(feature)
        self.memory["action"].append(action)
        self.memory["reward"].append(reward)
        self.memory["hin0"].append(hin0)
        self.memory["hout0"].append(hout0)
        self.memory["hin1"].append(hin1)
        self.memory["hout1"].append(hout1)

    def get(self, gamma=0.99):
        assert len(self) == self.maxlen
        reward = self._multi_step_reward(gamma)
        feature = self._multi_step_feature(gamma)
        state = self.memory["state"].popleft()
        action = self.memory["action"].popleft()
        hin0 = self.memory["hin0"].popleft()
        hout0 = self.memory["hout0"].popleft()
        hin1 = self.memory["hin1"].popleft()
        hout1 = self.memory["hout1"].popleft()

        _ = self.memory["reward"].popleft()
        _ = self.memory["feature"].popleft()
        return state, feature, action, reward, hin0, hout0, hin1, hout1

    def _multi_step_reward(self, gamma):
        return np.sum([r * (gamma**i) for i, r in enumerate(self.memory["reward"])])

    def _multi_step_feature(self, gamma):
        return np.sum(
            [f * (gamma**i) for i, f in enumerate(self.memory["feature"])], 0
        )

    def __getitem__(self, key):
        if key not in self.keys:
            raise Exception(f"There is no key {key} in MultiStepBuff.")
        return self.memory[key]

    def reset(self):
        for key in self.keys:
            self.memory[key].clear()

    def __len__(self):
        return len(self.memory["state"])


class MyMultiStepRNNMemory(RNNMemory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        hidden_shape0,
        hidden_shape1,
        device,
        gamma=0.99,
        multi_step=3,
    ):
        super().__init__(
            capacity,
            state_shape,
            feature_shape,
            action_shape,
            hidden_shape0,
            hidden_shape1,
            device,
        )

        self.gamma = gamma
        self.multi_step = int(multi_step)
        if self.multi_step != 1:
            self.buff = MyMultiStepRNNBuff(maxlen=self.multi_step)

    def append(
        self,
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        hin0,
        hout0,
        hin1,
        hout1,
        episode_done=False,
    ):
        if self.multi_step != 1:
            self.buff.append(state, feature, action, reward, hin0, hout0, hin1, hout1)

            if len(self.buff) == self.multi_step:
                (
                    state,
                    feature,
                    action,
                    reward,
                    hin0,
                    hout0,
                    hin1,
                    hout1,
                ) = self.buff.get(self.gamma)
                self._append(
                    state,
                    feature,
                    action,
                    reward,
                    next_state,
                    done,
                    hin0,
                    hout0,
                    hin1,
                    hout1,
                )

            if episode_done or done:
                self.buff.reset()
        else:
            self._append(
                state,
                feature,
                action,
                reward,
                next_state,
                done,
                hin0,
                hout0,
                hin1,
                hout1,
            )


class MyPrioritizedRNNMemory(MyMultiStepRNNMemory):
    def __init__(
        self,
        capacity,
        state_shape,
        feature_shape,
        action_shape,
        hidden_shape0,
        hidden_shape1,
        device,
        gamma=0.99,
        multi_step=3,
        alpha=0.6,
        beta=0.4,
        beta_annealing=0.001,
        epsilon=1e-4,
    ):

        super().__init__(
            capacity,
            state_shape,
            feature_shape,
            action_shape,
            hidden_shape0,
            hidden_shape1,
            device,
            gamma,
            multi_step,
        )
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon

    def append(
        self,
        state,
        feature,
        action,
        reward,
        next_state,
        done,
        hin0,
        hout0,
        hin1,
        hout1,
        error,
        episode_done=False,
    ):
        if self.multi_step != 1:
            self.buff.append(state, feature, action, reward, hin0, hout0, hin1, hout1)

            if len(self.buff) == self.multi_step:
                (
                    state,
                    feature,
                    action,
                    reward,
                    hin0,
                    hout0,
                    hin1,
                    hout1,
                ) = self.buff.get(self.gamma)
                self.priorities[self._p] = self.calc_priority(error)
                self._append(
                    state,
                    feature,
                    action,
                    reward,
                    next_state,
                    done,
                    hin0,
                    hout0,
                    hin1,
                    hout1,
                )

            if episode_done or done:
                self.buff.reset()
        else:
            self.priorities[self._p] = self.calc_priority(error)
            self._append(
                state,
                feature,
                action,
                reward,
                next_state,
                done,
                hin0,
                hout0,
                hin1,
                hout1,
            )

    def update_priority(self, indices, errors):
        self.priorities[indices] = np.reshape(self.calc_priority(errors), (-1, 1))

    def calc_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def sample(self, batch_size):
        self.beta = min(1.0 - self.epsilon, self.beta + self.beta_annealing)
        sampler = WeightedRandomSampler(self.priorities[: self._n, 0], batch_size)
        indices = list(sampler)
        batch = self._sample(indices)

        p = self.priorities[indices] / np.sum(self.priorities[: self._n])
        weights = (self._n * p) ** -self.beta
        weights /= np.max(weights)
        weights = torch.FloatTensor(weights).to(self.device)

        return batch, indices, weights

    def reset(self):
        super().reset()
        self.priorities = np.empty((self.capacity, 1), dtype=np.float32)

    def get(self):
        valid = slice(0, self._n)
        return (
            self.states[valid],
            self.features[valid],
            self.actions[valid],
            self.rewards[valid],
            self.next_states[valid],
            self.dones[valid],
            self.hin0[valid],
            self.hout0[valid],
            self.hin1[valid],
            self.hout1[valid],
            self.priorities[valid],
        )

    def _insert(self, mem_indices, batch, batch_indices):
        (
            states,
            features,
            actions,
            rewards,
            next_states,
            dones,
            hin0,
            hout0,
            hin1,
            hout1,
            priorities,
        ) = batch
        self.states[mem_indices] = states[batch_indices]
        self.features[mem_indices] = features[batch_indices]
        self.actions[mem_indices] = actions[batch_indices]
        self.rewards[mem_indices] = rewards[batch_indices]
        self.next_states[mem_indices] = next_states[batch_indices]
        self.dones[mem_indices] = dones[batch_indices]
        self.hin0[mem_indices] = hin0[batch_indices]
        self.hout0[mem_indices] = hout0[batch_indices]
        self.hin1[mem_indices] = hin1[batch_indices]
        self.hout1[mem_indices] = hout1[batch_indices]

        self.priorities[mem_indices] = priorities[batch_indices]
