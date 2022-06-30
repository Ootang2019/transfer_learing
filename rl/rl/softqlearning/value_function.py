import torch.nn as nn


class SoftQNetwork(nn.Module):
    def __init__(self, observation_dim, action_dim, alpha=10, sizes=[64, 64]) -> None:
        super().__init__()
        self.alpha = alpha
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, sizes[0])
        self.fc2 = nn.Linear(sizes[0], sizes[1])
        self.fc3 = nn.Linear(sizes[1], 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
