import torch
import torch.nn as nn
from .util import get_sa_pairs
import numpy as np
from torch.distributions import Normal
from rltorch.network import create_linear_network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class StochasticPolicy(BaseNetwork):
    """Stochastic NN policy"""

    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        squash=True,
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash

        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], self.action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x) if self.squash else x
        return x

    def get_action(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.tensor(observation, dtype=torch.float32).to(device)
        return self.get_actions(observation).squeeze(0)[0].cpu().detach().numpy()

    def get_actions(self, observations):
        return self.acitons_for(observations)

    def acitons_for(self, observations, n_action_samples=1):
        if observations.ndim > 1:
            n_state_samples = observations.shape[0]
        else:
            observations = observations[None, :]
            n_state_samples = 1

        latent_shape = (n_action_samples, self.action_dim)
        latents = torch.normal(0, 1, size=latent_shape).to(device)

        s_a = get_sa_pairs(observations, latents)
        raw_actions = self.forward(s_a).view(
            n_state_samples, n_action_samples, self.action_dim
        )

        return raw_actions


class GaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, observation_dim, action_dim, sizes=[256, 256], squash=True):
        super(GaussianPolicy, self).__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim * 2
        self.sizes = sizes
        self.squash = squash

        self.fc1 = nn.Linear(self.observation_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], self.action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, states):
        x = self.relu(self.fc1(states))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x) if self.squash else x

        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(states)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)
