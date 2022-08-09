import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import get_sa_pairs
import numpy as np
from torch.distributions import Normal

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


LOG_SIG_CAP_MAX = 2
LOG_SIG_CAP_MIN = -20
EPS = 1e-6


class GaussianMixture(torch.nn.Module):
    def __init__(
        self,
        K,
        input_dim,
        output_dim,
        hidden_layers_sizes=[64, 64],
        reg=0.001,
        reparameterize=True,
    ):
        super().__init__()
        self._K = K
        self._input_dim = input_dim
        self._Dx = output_dim

        self._reg = reg
        self._layer_sizes = list(hidden_layers_sizes) + [self._K * (2 * self._Dx + 1)]
        self._reparameterize = reparameterize

        self.fc1 = nn.Linear(self._input_dim, self._layer_sizes[0])
        self.fc2 = nn.Linear(self._layer_sizes[0], self._layer_sizes[1])
        self.fc3 = nn.Linear(self._layer_sizes[1], self._layer_sizes[2])
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self._log_p_x_t = 0
        self._log_p_x_mono_t = 0
        self._reg_loss_t = 0
        self._x_t = 0
        self._mus_t = 0
        self._log_sigs_t = 0
        self._log_ws_t = 0
        self._N_pl = 0

    def get_distribution(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, self._K, 2 * self._Dx + 1)

        log_w_t = x[..., 0]
        mu_t = x[..., 1 : 1 + self._Dx]
        log_sig_t = x[..., 1 + self._Dx :]
        log_sig_t = torch.clip(log_sig_t, LOG_SIG_CAP_MIN, LOG_SIG_CAP_MAX)
        return log_w_t, mu_t, log_sig_t

    @staticmethod
    def _create_log_gaussian(mu_t, log_sig_t, t):
        normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)  # ... x D
        quadratic = -0.5 * torch.sum(normalized_dist_t**2, -1)
        # ... x (None)

        log_z = torch.sum(log_sig_t, axis=-1)  # ... x (None)
        D_t = torch.tensor(mu_t.shape[-1], dtype=torch.float32)
        log_z += 0.5 * D_t * np.log(2 * np.pi)

        log_p = quadratic - log_z

        return log_p  # ... x (None)

    @staticmethod
    def _create_log_mono_gaussian(mu_t, log_sig_t, t):
        normalized_dist_t = (t - mu_t) * torch.exp(-log_sig_t)  # ... x D
        quadratic = -0.5 * normalized_dist_t**2
        # ... x (D)

        log_z = log_sig_t  # ... x (D)
        D_t = torch.tensor(mu_t.shape[-1], dtype=torch.float32)
        log_z += 0.5 * D_t * np.log(2 * np.pi)

        log_p = quadratic - log_z

        return log_p  # ... x (D)

    def forward(self, obs):
        if obs.ndim > 1:
            N = obs.shape[0]
        else:
            obs = obs[None, :]
            N = 1

        Dx, K = self._Dx, torch.tensor(self._K).to(device)

        # create K gaussians
        log_ws_t, xz_mus_t, xz_log_sigs_t = self.get_distribution(obs)
        # (N x K), (N x K x Dx), (N x K x Dx)
        xz_sigs_t = torch.exp(xz_log_sigs_t)

        # Sample the latent code.
        log_ws_t = self.tanh(log_ws_t) + 1 + EPS  # me add this to make it logits
        z_t = torch.multinomial(log_ws_t, num_samples=1)  # N x 1

        # Choose mixture component corresponding to the latent.
        mask_t = F.one_hot(z_t[:, 0], K)
        mask_t = mask_t.bool()
        xz_mu_t = xz_mus_t[mask_t]  # N x Dx
        xz_sig_t = xz_sigs_t[mask_t]  # N x Dx

        # Sample x.
        x_t = xz_mu_t + xz_sig_t * torch.normal(0, 1, (N, Dx)).to(device)  # N x Dx

        if not self._reparameterize:
            x_t = x_t.detach()

        # log p(x|z_k) = log N(x | mu_k, sig_k)
        log_p_xz_t = self._create_log_gaussian(
            xz_mus_t, xz_log_sigs_t, x_t[:, None, :]
        )  # N x K

        # log p(x) = log sum_k p(z_k)p(x|z_k)
        log_p_x_t = torch.logsumexp(log_p_xz_t + log_ws_t, axis=1)
        log_p_x_t = log_p_x_t - torch.logsumexp(log_ws_t, axis=1)  # N

        # log p(x|z_k)
        log_p_xz_mono_t = self._create_log_mono_gaussian(
            xz_mus_t, xz_log_sigs_t, x_t[:, None, :]
        )  # N x K

        # log test p(x)
        log_ws_mono_t = log_ws_t[..., None]
        log_p_x_mono_t = torch.logsumexp(log_p_xz_mono_t + log_ws_mono_t, axis=1)
        log_p_x_mono_t = log_p_x_mono_t - torch.logsumexp(log_ws_mono_t, axis=1)  # N

        reg_loss_t = 0
        reg_loss_t += self._reg * 0.5 * torch.mean(xz_log_sigs_t**2)
        reg_loss_t += self._reg * 0.5 * torch.mean(xz_mus_t**2)

        self._log_p_x_t = log_p_x_t
        self._log_p_x_mono_t = log_p_x_mono_t
        self._reg_loss_t = reg_loss_t
        self._x_t = x_t

        self._log_ws_t = log_ws_t
        self._mus_t = xz_mus_t
        self._log_sigs_t = xz_log_sigs_t

        return x_t, log_p_x_t

    @property
    def log_p_x_mono_t(self):
        return self._log_p_x_mono_t

    @property
    def log_p_t(self):
        return self._log_p_x_t

    @property
    def reg_loss_t(self):
        return self._reg_loss_t

    @property
    def x_t(self):
        return self._x_t

    @property
    def mus_t(self):
        return self._mus_t

    @property
    def log_sigs_t(self):
        return self._log_sigs_t

    @property
    def log_ws_t(self):
        return self._log_ws_t

    @property
    def N_t(self):
        return self._N
