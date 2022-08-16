from collections import deque
import numpy as np
import torch


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


class MultiStepBuff:
    keys = ["state", "feature", "action", "reward"]

    def __init__(self, maxlen=3):
        super(MultiStepBuff, self).__init__()
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
        state = self.memory["state"].popleft()
        feature = self.memory["feature"].popleft()
        action = self.memory["action"].popleft()
        _ = self.memory["reward"].popleft()
        return state, feature, action, reward

    def _multi_step_reward(self, gamma):
        return np.sum([r * (gamma**i) for i, r in enumerate(self.memory["reward"])])

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
            self.buff = MultiStepBuff(maxlen=self.multi_step)

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
