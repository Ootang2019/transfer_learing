import pprint
import random
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return np.array([self.buffer[i] for i in range(rand, rand + batch_size)])
        else:
            indexes = np.random.choice(
                np.arange(len(self.buffer)), size=batch_size, replace=False
            )
            return np.array([self.buffer[i] for i in indexes])

    def clear(self):
        self.buffer.clear()


class SoftQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, alpha=10, sizes=[64, 64]) -> None:
        super().__init__()
        self.alpha = alpha
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.observation_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], self.action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def getV(self, q_value):
        """v(s) = alpha*log E[ exp(1/alpha*Q(s,a')) / q(a') ]
        where Q(s,a') is the q value and q(a') is importance weight

        Args:
            q_value (_type_): _description_

        Returns:
            _type_: _description_
        """

        v = self.alpha * torch.log(
            torch.sum(torch.exp(q_value / self.alpha), dim=1, keepdim=True)
        )
        return v

    def choose_action(self, observation):
        """the entropy policy: pi(a|s) = exp( 1/alpha * ( Q(s,a) - V(s) )  )

        Args:
            observation (_type_): _description_

        Returns:
            _type_: _description_
        """
        observation = torch.FloatTensor(observation).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.forward(observation)
            v = self.getV(q).squeeze()
            dist = torch.exp((q - v) / self.alpha)
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.item()


class StochasticPolicy(nn.Module):
    """Stochastic NN policy

    Args:
        nn (_type_): _description_
    """

    def __init__(self, observation_dim, action_dim, sizes=[64, 64], squash=True):
        super.__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash

        self.fc1 = nn.Linear(self.observation_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], self.action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = nn.tanh(x) if self.squash else x
        return x

    def get_action(self, observation):
        return self.get_actions(observation)[0], None

    def get_actions(self, observations):
        action = self.acitons_for(observations)
        return action

    def acitons_for(self, observations, n_action_samples=1):
        n_state_samples = observations[0]
        if n_action_samples > 1:
            observations = observations[:, None, :]
            latent_shape = (n_state_samples, n_action_samples, self.action_dim)
        else:
            latent_shape = (n_state_samples, self.action_dim)

        latents = torch.normal(0, 1, size=latent_shape)

        raw_actions = self.forward((observations, latents))

        return raw_actions


class DiscreteSQLAgent:
    """soft q learning with discrete action space"""

    def __init__(
        self,
        env,
        total_timesteps: int = 1e6,
        n_timesteps: int = 200,
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 8,
        td_target_update_interval: int = 1,
        reward_scale: float = 1,
        replay_buffer_size: int = 1e5,
        batch_size: int = 128,
        min_n_experience: int = 1024,  # minimum number of training experience
    ) -> None:
        self.env = env

        self.total_timesteps = int(total_timesteps)
        self.n_timesteps = int(n_timesteps)
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha

        self.td_target_update_interval = int(td_target_update_interval)

        self.reward_scale = reward_scale
        self.replay_buffer_size = int(replay_buffer_size)
        self.batch_size = int(batch_size)
        self.min_n_experience = int(min_n_experience)

        self.learn_steps = 0
        self.begin_learn_td = False

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.behavior_net = SoftQNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            alpha=self.alpha,
        ).to(device)

        self.target_net = SoftQNetwork(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
            alpha=self.alpha,
        ).to(device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())

        self.optimizer = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def train(self):
        episode_reward = 0
        n_epochs = int(self.total_timesteps / self.n_timesteps)

        for epoch in range(n_epochs):
            state = self.env.reset()
            episode_reward = 0

            for ts in range(self.n_timesteps):
                action = self.behavior_net.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                reward = self.reward_scale * reward
                episode_reward += reward
                self.replay_buffer.add((state, next_state, action, reward, done))

                if self.replay_buffer.size() > self.min_n_experience:
                    self.td_update()

                if done:
                    break

                state = next_state

            wandb.log({"episode_reward": episode_reward})
            wandb.watch(self.behavior_net)

    def td_update(self):
        if self.begin_learn_td is False:
            print("begin learning q function")
            self.begin_learn_td = True

        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

        batch = self.replay_buffer.sample(self.batch_size, False)
        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_reward,
            batch_done,
        ) = zip(*batch)

        batch_state = torch.FloatTensor(batch_state).to(device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(device)
        batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

        with torch.no_grad():
            next_q = self.target_net(batch_next_state)
            next_v = self.target_net.getV(next_q)
            target = batch_reward + (1 - batch_done) * self.gamma * next_v

        self.optimizer.zero_grad()
        behav_q = self.behavior_net(batch_state).gather(1, batch_action.long())
        loss = F.mse_loss(behav_q, target)
        loss.backward()
        self.optimizer.step()

        metrics = {
            "loss": loss,
            "q": next_q.mean(),
            "v": next_v.mean(),
            "target": target.mean(),
        }
        wandb.log(metrics)


class ContinuousSQLAgent:
    """soft q learning with continuous action space"""

    def __init__(
        self,
        env,
        total_timesteps: int = 1e6,
        n_timesteps: int = 200,
        lr: float = 1e-3,
        gamma: float = 0.99,
        alpha: float = 8,
        value_n_particles: int = 16,
        td_target_update_interval: int = 1,
        kernel_n_particles: int = 16,
        kernel_update_ratio: float = 0.5,
        reward_scale: float = 1,
        replay_buffer_size: int = 1e5,
        batch_size: int = 128,
        min_n_experience: int = 1024,  # minimum number of training experience
    ) -> None:
        self.env = env

        self.total_timesteps = int(total_timesteps)
        self.n_timesteps = int(n_timesteps)
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha

        self.value_n_particles = int(value_n_particles)
        self.td_target_update_interval = int(td_target_update_interval)

        self.kernel_n_particles = int(kernel_n_particles)
        self.kernel_update_ratio = kernel_update_ratio

        self.reward_scale = reward_scale
        self.replay_buffer_size = int(replay_buffer_size)
        self.batch_size = int(batch_size)
        self.min_n_experience = int(min_n_experience)

        self.learn_steps = 0
        self.begin_learn_td = False

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        self.behavior_net = SoftQNetwork(
            input_dim=self.observation_dim + self.action_dim,
            output_dim=self.action_dim,
            alpha=self.alpha,
        ).to(device)

        self.target_net = SoftQNetwork(
            input_dim=self.observation_dim + self.action_dim,
            output_dim=self.action_dim,
            alpha=self.alpha,
        ).to(device)
        self.target_net.load_state_dict(self.behavior_net.state_dict())

        self.policy = StochasticPolicy(
            observation_dim=self.observation_dim,
            action_dim=self.action_dim,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def train(self):
        episode_reward = 0
        n_epochs = int(self.total_timesteps / self.n_timesteps)

        for epoch in range(n_epochs):
            state = self.env.reset()
            episode_reward = 0

            for ts in range(self.n_timesteps):
                action = self.behavior_net.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                reward = self.reward_scale * reward
                episode_reward += reward
                self.replay_buffer.add((state, next_state, action, reward, done))

                if self.replay_buffer.size() > self.min_n_experience:
                    self.td_update()

                if done:
                    break

                state = next_state

            wandb.log({"episode_reward": episode_reward})
            wandb.watch(self.behavior_net)

    def td_update(self):
        if self.begin_learn_td is False:
            print("begin learning q function")
            self.begin_learn_td = True

        self.learn_steps += 1

        if self.learn_steps % self.td_target_update_interval == 0:
            self.target_net.load_state_dict(self.behavior_net.state_dict())

        batch = self.replay_buffer.sample(self.batch_size, False)
        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_reward,
            batch_done,
        ) = zip(*batch)

        batch_state = torch.FloatTensor(batch_state).to(device)
        batch_next_state = torch.FloatTensor(batch_next_state).to(device)
        batch_action = torch.FloatTensor(batch_action).unsqueeze(1).to(device)
        batch_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
        batch_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)

        with torch.no_grad():
            target_actions = (
                2
                * torch.rand(
                    batch_state.shape[0], self.value_n_particles, self.action_dim
                )
                - 1
            )
            q_value_targets = self.target_net(
                torch.cat((batch_next_state, target_actions), 1)
            )
            v_value_target = self.target_net.getV(q_value_targets)
            target = batch_reward + (1 - batch_done) * self.gamma * v_value_target

            # next_q = self.target_net(batch_next_state)
            # next_v = self.target_net.getV(next_q)
            target = batch_reward + (1 - batch_done) * self.gamma * next_v

        self.optimizer.zero_grad()
        behav_q = self.behavior_net(batch_state).gather(1, batch_action.long())
        loss = F.mse_loss(behav_q, target)
        loss.backward()
        self.optimizer.step()

        metrics = {
            "loss": loss,
            "q": next_q.mean(),
            "v": next_v.mean(),
            "target": target.mean(),
        }
        wandb.log(metrics)


if __name__ == "__main__":
    default_dict = dict(lr=1e-3, alpha=4, total_timesteps=1e5, batch_size=128)

    wandb.init(config=default_dict)
    config = wandb.config

    env = gym.make("CartPole-v0")
    agent = DiscreteSQLAgent(env=env, **config)
    agent.train()
