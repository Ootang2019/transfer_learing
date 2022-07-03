import torch
import torch.nn as nn
import numpy
import line_profiler
from util import get_sa_pairs


class StochasticPolicy(nn.Module):
    """Stochastic NN policy"""

    def __init__(
        self, observation_dim, action_dim, sizes=[64, 64], squash=True, device="cpu"
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.squash = squash
        self.device = device

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
        observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
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
        latents = torch.normal(0, 1, size=latent_shape).to(self.device)

        s_a = get_sa_pairs(observations, latents)
        raw_actions = self.forward(s_a).view(
            n_state_samples, n_action_samples, self.action_dim
        )

        return raw_actions
