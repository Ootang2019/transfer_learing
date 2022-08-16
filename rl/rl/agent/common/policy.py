import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import get_sa_pairs, check_dim
from .distribution import GaussianMixture
import numpy as np
from torch.distributions import Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPS = 1e-2


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

        s, a = get_sa_pairs(observations, latents)
        raw_actions = self.forward(torch.cat([s, a], -1)).view(
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

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x) if self.squash else x

        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, obs):
        # calculate Gaussian distribusion of (mean, std)
        means, log_stds = self.forward(obs)
        stds = log_stds.exp()
        normals = Normal(means, stds)
        # sample actions
        xs = normals.rsample()
        actions = torch.tanh(xs)
        # calculate entropies
        log_probs = normals.log_prob(xs) - torch.log(1 - actions.pow(2) + self.eps)
        entropies = -log_probs.sum(dim=1, keepdim=True)

        return actions, entropies, torch.tanh(means)


class GMMPolicy(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        sizes=[64, 64],
        n_gauss=10,
        reg=0.001,
        reparameterize=True,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes
        self.n_gauss = n_gauss
        self.reg = reg
        self.reparameterize = reparameterize

        self.model = GaussianMixture(
            input_dim=self.observation_dim,
            output_dim=self.action_dim,
            hidden_layers_sizes=sizes,
            K=n_gauss,
            reg=reg,
            reparameterize=reparameterize,
        )

    def forward(self, obs):
        obs, n_sample = check_dim(obs)

        act, logp = self.model(obs)
        act = torch.tanh(act)
        logp -= self.squash_correction(act)
        return act, logp

    def squash_correction(self, inp):
        return torch.sum(torch.log(1 - torch.tanh(inp) ** 2 + EPS), 1)

    def reg_loss(self):
        return self.model.reg_loss_t


class GMMChi(GMMPolicy):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        sizes=[64, 64],
        n_gauss=10,
        reg=0.001,
        reparameterize=True,
        action_strategy="merge",
    ) -> None:
        super.__init__(
            observation_dim=observation_dim,
            action_dim=feature_dim,
            sizes=sizes,
            n_gauss=n_gauss,
            reg=reg,
            reparameterize=reparameterize,
        )

        self.feature_dim = feature_dim
        self.action_strategy = action_strategy

    def act(self, obs, w):
        obs, n_sample = check_dim(obs)

        chi, logp = self.get_chi(obs)
        p = torch.exp(logp)

        if self.action_strategy == "sample":
            act = self.sample_action(chi, logp, w)
        elif self.action_strategy == "merge":
            act = self.merge_action(chi, p, w)
        else:
            raise NotImplementedError

        logpi = torch.sum(p * w * logp, 1) / torch.sum(p * w, 1)

        return act, logpi

    def get_chi(self, obs):
        chi, _ = self.model(obs)
        logp = self.model.log_p_x_mono_t
        chi = torch.tanh(chi)
        logp -= self.squash_correction_mono(chi)
        return chi, logp

    def merge_action(self, chi, p, w):
        act = torch.sum(p * w * chi, 1) / torch.sum(p * w, 1)
        return act

    def sample_action(self, chi, logp, w):
        logp = w * (torch.tanh(logp) + 1) / 2
        z = torch.multinomial(logp, num_samples=1)
        mask = F.one_hot(z[:, 0], self.feature_dim).bool()
        act = chi[mask]
        act = torch.clip(act, -1, 1)
        return act

    def squash_correction_mono(self, inp):
        return torch.log(1 - torch.tanh(inp) ** 2 + EPS)
