from numpy import argmax
import torch
import torch.nn as nn


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class QNetwork(BaseNetwork):
    def __init__(
        self, observation_dim, action_dim, sizes=[64, 64], activation=nn.ReLU
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes

        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, self.sizes[0])
        self.fc2 = nn.Linear(self.sizes[0], self.sizes[1])
        self.fc3 = nn.Linear(self.sizes[1], 1)
        self.activation = activation()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, observation_dim, action_dim, sizes=[64, 64], activation=nn.ReLU):
        super(TwinnedQNetwork, self).__init__()

        self.Q1 = QNetwork(observation_dim, action_dim, sizes, activation)
        self.Q2 = QNetwork(observation_dim, action_dim, sizes, activation)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class SFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        sizes=[64, 64],
        activation=nn.ReLU,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.sizes = sizes

        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, self.sizes[0])
        self.fc2 = nn.Linear(self.sizes[0], self.sizes[1])
        self.fc3 = nn.Linear(self.sizes[1], self.feature_dim)
        self.activation = activation()

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_chi(self, states, chi):
        target = torch.cat(
            [
                self.forward(states, chi[:, i].unsqueeze(1))
                for i in range(self.feature_dim)
            ],
            1,
        )
        mask = [i * (self.feature_dim + 1) for i in range(self.feature_dim)]
        return target[:, mask]


class TwinnedSFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        sizes=[64, 64],
        activation=nn.ReLU,
    ):
        super().__init__()

        self.SF1 = SFNetwork(
            observation_dim, feature_dim, action_dim, sizes, activation
        )
        self.SF2 = SFNetwork(
            observation_dim, feature_dim, action_dim, sizes, activation
        )

    def forward(self, states, actions):
        sf1 = self.SF1(states, actions)
        sf2 = self.SF2(states, actions)
        return sf1, sf2

    def forward_chi(self, states, chi):
        target1 = self.SF1.forward_chi(states, chi)
        target2 = self.SF2.forward_chi(states, chi)
        return target1, target2
