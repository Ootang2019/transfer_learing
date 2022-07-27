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


class SFMLP(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        action_dim,
        sf_dim,
        sizes=[64, 64],
        activation=nn.ReLU,
    ) -> None:
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sf_dim = sf_dim
        self.sizes = sizes

        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, self.sizes[0])
        self.fc2 = nn.Linear(self.sizes[0], self.sizes[1])
        self.fc3 = nn.Linear(self.sizes[1], self.sf_dim)
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


if __name__ == "__main__":
    observation_dim = 4
    action_dim = 5
    sizes = [64, 64, 64]

    m = SFMLP(observation_dim=observation_dim, action_dim=action_dim, sizes=sizes)
    print(m)
    print("\n")

    inp = torch.rand(5, observation_dim + action_dim)
    w = torch.arange(observation_dim).float()
    oup = m(inp)
    print("inp:", inp)
    print("inp shape:", inp.shape)
    print("\n")
    print("oup:", oup)
    print("oup shape:", oup.shape)
    print("oup dim: (n_data, n_features)")
    print("\n")
