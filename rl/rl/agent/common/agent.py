import gym
from pathlib import Path
import pickle
import torch
import numpy as np
from .replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AbstractAgent:
    def __init__(
        self,
        env: str,
        env_kwargs: dict = {},
        total_timesteps=1e6,
        n_timesteps=200,
        reward_scale: float = 1,
        replay_buffer_size: int = 1e6,
        mini_batch_size: int = 128,
        min_n_experience: int = 1024,  # minimum number of training experience
        save_path: str = "./",
        render: bool = False,
        replay_buffer=ReplayBuffer,
    ):
        self.save_path = save_path

        if isinstance(env, str):
            self.env = gym.make(env, **env_kwargs)
        else:
            self.env = env

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.render = render

        self.total_timesteps = int(total_timesteps)
        self.n_timesteps = int(n_timesteps)
        self.reward_scale = reward_scale
        self.replay_buffer_size = int(replay_buffer_size)
        self.mini_batch_size = int(mini_batch_size)
        self.min_n_experience = int(min_n_experience)

        self.replay_buffer = replay_buffer(self.replay_buffer_size)

    def train(self):
        raise NotImplementedError

    def sample_minibatch(self):
        batch = self.replay_buffer.sample(self.mini_batch_size, False)
        (
            batch_state,
            batch_next_state,
            batch_action,
            batch_reward,
            batch_done,
        ) = zip(*batch)

        return (
            torch.FloatTensor(np.array(batch_state)).to(device),
            torch.FloatTensor(np.array(batch_next_state)).to(device),
            torch.FloatTensor(np.array(batch_action)).to(device),
            torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device),
        )

    def save_config(self, config):
        config = dict(config)
        Path(self.save_path).mkdir(parents=True, exist_ok=True)
        with open(self.save_path + "/config.pkl", "wb") as f:
            pickle.dump(config, f)

    def load_config(self, path):
        with open(path + "/config.pkl", "rb") as f:
            config = pickle.load(f)
        return config
