from numpy import argmax
import torch
import torch.nn as nn
from .util import check_dim, np2ts


class BaseNetwork(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def reg_loss(self):
        return 0


class QNetwork(BaseNetwork):
    def __init__(
        self, observation_dim, action_dim, sizes=[64, 64], activation=nn.SiLU
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.sizes = sizes

        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, self.sizes[0])
        self.ln1 = nn.LayerNorm(self.sizes[0])
        self.fc2 = nn.Linear(self.sizes[0], self.sizes[1])
        self.ln2 = nn.LayerNorm(self.sizes[1])
        self.fc3 = nn.Linear(self.sizes[1], 1)
        self.activation = activation()

        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.fc3.weight, 0.01)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, 1)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


class TwinnedQNetwork(BaseNetwork):
    def __init__(self, observation_dim, action_dim, sizes=[64, 64], activation=nn.SiLU):
        super().__init__()

        self.Q1 = QNetwork(observation_dim, action_dim, sizes, activation)
        self.Q2 = QNetwork(observation_dim, action_dim, sizes, activation)

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], dim=1)
        q1 = self.Q1(x)
        q2 = self.Q2(x)
        return q1, q2


class VNetwork(BaseNetwork):
    def __init__(self, observation_dim, sizes=[64, 64], activation=nn.SiLU) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.sizes = sizes

        self.fc1 = nn.Linear(self.observation_dim, self.sizes[0])
        self.ln1 = nn.LayerNorm(self.sizes[0])
        self.fc2 = nn.Linear(self.sizes[0], self.sizes[1])
        self.ln2 = nn.LayerNorm(self.sizes[1])
        self.fc3 = nn.Linear(self.sizes[1], 1)
        self.activation = activation()

        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.fc3.weight, 0.01)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, 1)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


class TwinnedVNetwork(BaseNetwork):
    def __init__(self, observation_dim, sizes=[64, 64], activation=nn.SiLU):
        super().__init__()

        self.observation_dim = observation_dim

        self.V1 = VNetwork(observation_dim, sizes, activation)
        self.V2 = VNetwork(observation_dim, sizes, activation)

    def forward(self, observations):
        observations = check_dim(observations, self.observation_dim)
        v1 = self.V1(observations)
        v2 = self.V2(observations)
        return v1, v2


class SFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        sizes=[64, 64],
        activation=nn.SiLU,
    ) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.sizes = sizes

        self.fc1 = nn.Linear(self.observation_dim + self.action_dim, self.sizes[0])
        self.ln1 = nn.LayerNorm(self.sizes[0])
        self.fc2 = nn.Linear(self.sizes[0], self.sizes[1])
        self.ln2 = nn.LayerNorm(self.sizes[1])
        self.fc3 = nn.Linear(self.sizes[1], self.feature_dim)
        self.activation = activation()

        self.apply(self._init_weights)
        nn.init.xavier_uniform_(self.fc3.weight, 0.01)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, 1)
            if module.bias is not None:
                module.bias.data.zero_()

        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, observations, actions):
        observations = check_dim(observations, self.observation_dim)
        actions = check_dim(actions, self.action_dim)
        x = torch.cat([observations, actions], dim=1)
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        return x


class TwinnedSFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        sizes=[64, 64],
        activation=nn.SiLU,
    ):
        super().__init__()

        self.SF0 = SFNetwork(
            observation_dim, feature_dim, action_dim, sizes, activation
        )
        self.SF1 = SFNetwork(
            observation_dim, feature_dim, action_dim, sizes, activation
        )

    def forward(self, observations, actions):
        sf0 = self.SF0(observations, actions)
        sf1 = self.SF1(observations, actions)
        return sf0, sf1


class ChiNetwork(SFNetwork):
    def forward_chi(self, observations, chi):
        target = torch.cat(
            [
                self.forward(observations, chi[:, i].unsqueeze(1))
                for i in range(self.feature_dim)
            ],
            1,
        )
        mask = [i * (self.feature_dim + 1) for i in range(self.feature_dim)]
        return target[:, mask]


class TwinnedChiNetwork(TwinnedSFNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        sizes=[64, 64],
        activation=nn.SiLU,
    ):
        super().__init__(observation_dim, feature_dim, action_dim, sizes, activation)

        self.CHI0 = ChiNetwork(
            observation_dim, feature_dim, action_dim, sizes, activation
        )
        self.CHI1 = ChiNetwork(
            observation_dim, feature_dim, action_dim, sizes, activation
        )

    def forward_chi(self, observations, chi):
        target1 = self.CHI0.forward_chi(observations, chi)
        target2 = self.CHI1.forward_chi(observations, chi)
        return target1, target2


class MultiheadSFNetwork(SFNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        n_heads,
        sizes=[64, 64],
        activation=nn.SiLU,
    ) -> None:
        super().__init__(observation_dim, feature_dim, action_dim, sizes, activation)

        self.n_heads = n_heads
        self.head_ls = []
        for _ in range(n_heads):
            self._add_head(self.sizes[1], self.feature_dim)
        self._update_heads()

    def forward(self, observations, actions, head_idx):
        x = self._forward_hidden(observations, actions)
        x = self.heads[head_idx](x)
        return x

    def forward_heads(self, observations, actions):
        sfs = []
        x = self._forward_hidden(observations, actions)

        for i in range(self.n_heads):
            sf = self.heads[i](x)
            sfs.append(sf)
        return torch.stack(sfs)

    def _forward_hidden(self, observations, actions):
        observations = check_dim(observations, self.observation_dim)
        actions = check_dim(actions, self.action_dim)

        x = torch.cat([observations, actions], dim=1)
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        return x

    def add_heads(self, output_size):
        self._add_head(self.sizes[1], output_size)
        self._update_heads()
        head_idx = len(self.head_ls) - 1
        return head_idx

    def _add_head(self, input_size, output_size):
        head = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(head.weight, 0.01)
        self.head_ls.append(head)

    def _update_heads(self):
        self.heads = nn.ModuleList(self.head_ls)


class MultiheadSFRNNNetwork(SFNetwork):
    def __init__(
        self,
        observation_dim,
        feature_dim,
        action_dim,
        n_heads,
        sizes=[256, 256],
        activation=nn.SiLU,
    ) -> None:
        super().__init__(observation_dim, feature_dim, action_dim, sizes, activation)

        self.rnn = torch.nn.GRU(self.sizes[0], self.sizes[0], 1, batch_first=True)

        self.n_heads = n_heads
        self.head_ls = []
        for _ in range(n_heads):
            self._add_head(self.sizes[1], self.feature_dim)
        self._update_heads()

    def forward(self, observations, actions, hidden, head_idx):
        x, hidden_rnn = self._forward_hidden(observations, actions, hidden)
        x = self.heads[head_idx](x)
        return x, hidden_rnn

    def forward_heads(self, observations, actions, hidden):
        sfs = []
        x, hidden_rnn = self._forward_hidden(observations, actions, hidden)

        for i in range(self.n_heads):
            sf = self.heads[i](x)
            sfs.append(sf)
        return torch.stack(sfs), hidden_rnn

    def _forward_hidden(self, observations, actions, hidden):
        observations = check_dim(observations, self.observation_dim)
        actions = check_dim(actions, self.action_dim)

        x = torch.cat([observations, actions], dim=1)
        x = self.activation(self.ln1(self.fc1(x)))

        x = x.view(-1, 1, self.sizes[0])
        x, hidden_rnn = self.rnn(x, hidden)
        x = x.squeeze(1)

        x = self.activation(self.ln2(self.fc2(x)))
        return x, hidden_rnn

    def add_heads(self, output_size):
        self._add_head(self.sizes[1], output_size)
        self._update_heads()
        head_idx = len(self.head_ls) - 1
        return head_idx

    def _add_head(self, input_size, output_size):
        head = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(head.weight, 0.01)
        self.head_ls.append(head)

    def _update_heads(self):
        self.heads = nn.ModuleList(self.head_ls)
