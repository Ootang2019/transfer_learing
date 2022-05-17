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
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(
                np.arange(len(self.buffer)), size=batch_size, replace=False
            )
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


class SoftQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, alpha=10, sizes=[64, 64]) -> None:
        super().__init__()
        self.alpha = alpha
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], self.output_dim)
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

    def choose_action(self, state):
        """the entropy policy: pi(a|s) = exp( 1/alpha * ( Q(s,a) - V(s) )  )

        Args:
            state (_type_): _description_

        Returns:
            _type_: _description_
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q = self.forward(state)
            v = self.getV(q).squeeze()
            dist = torch.exp((q - v) / self.alpha)
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            a = c.sample()
        return a.item()


class SQLAgent:
    """soft q learning with discrete action space"""

    def __init__(
        self,
        env,
        total_timesteps=1e6,
        n_timesteps=200,
        lr=1e-3,
        gamma=0.99,
        reward_scale=1,
        replay_buffer_size=1e5,
        min_n_experience=1024,  # minimum number of training experience
        batch_size=128,
        td_target_update_interval=2,
        alpha=8,
    ) -> None:
        self.env = env

        self.lr = lr
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.replay_buffer_size = int(replay_buffer_size)
        self.batch_size = int(batch_size)
        self.td_target_update_interval = int(td_target_update_interval)
        self.total_timesteps = int(total_timesteps)
        self.n_timesteps = int(n_timesteps)

        self.learn_steps = 0
        self.begin_learn = False
        self.min_n_experience = min_n_experience

        self.behavior_net = SoftQNetwork(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n,
            alpha=alpha,
        ).to(device)
        self.target_net = SoftQNetwork(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n,
            alpha=alpha,
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
                    self.train_net()

                if done:
                    break

                state = next_state

            wandb.log({"episode_reward": episode_reward})
            wandb.watch(self.behavior_net)

    def train_net(self):
        if self.begin_learn is False:
            print("begin learning")
            self.begin_learn = True
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

        batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
        batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
        batch_action = torch.FloatTensor(np.array(batch_action)).unsqueeze(1).to(device)
        batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device)
        batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device)

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


if __name__ == "__main__":
    default_dict = dict(lr=1e-3, alpha=4, total_timesteps=1e5, batch_size=128)

    wandb.init(config=default_dict)
    config = wandb.config

    env = gym.make("CartPole-v0")
    agent = SQLAgent(env=env, **config)
    agent.train()
