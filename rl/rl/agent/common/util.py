from re import A
from typing import Tuple
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_sa_pairs(s: torch.tensor, a: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
    """s is state, a is action particles
    pair every state with each action particle

    for example, 2 samples of state and 3 action particls
    s = [s0, s1]
    a = [a0, a1, a2]

    s_tile = [s0, s1, s0, s1, s0, s1]
    a_tile = [a0, a1, a2, a0, a1, a2]

    Args:
        s (torch.tensor): (number of samples, state dimension)
        a (torch.tensor): (number of particles, action dimension)

    Returns:
        Tuple[torch.tensor, torch.tensor]:
            s_tile (n_sample*n_particles, state_dim)
            a_tile (n_sample*n_particles, act_dim)
    """
    n_particles = a.shape[0]
    n_samples = s.shape[0]
    state_dim = s.shape[1]

    s_tile = torch.tile(s, (1, n_particles))
    s_tile = s_tile.reshape(-1, state_dim)

    a_tile = torch.tile(a, (n_samples, 1))
    return s_tile, a_tile


def get_sa_pairs_(
    s: torch.tensor, a: torch.tensor
) -> Tuple[torch.tensor, torch.tensor]:
    """s is state, a is action particles
    pair every state with each action particle

    Args:
        s (tensor): (number of samples, state dimension)
        a (tensor): (number of samples, number of particles, action dimension)

    Returns:
        Tuple[torch.tensor, torch.tensor]:
            s_tile (n_sample*n_particles, state_dim)
            a_tile (n_sample*n_particles, act_dim)
    """
    n_particles = a.shape[1]
    act_dim = a.shape[2]
    state_dim = s.shape[1]

    s_tile = torch.tile(s, (1, n_particles))
    s_tile = s_tile.reshape(-1, state_dim)

    a_tile = a.reshape(-1, act_dim)
    return s_tile, a_tile


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape
    assert len(tensor_shape) == len(
        expected_shape
    ), f"expect len a {len(tensor_shape)}, b {len(expected_shape)}"
    assert all(
        [a == b for a, b in zip(tensor_shape, expected_shape)][1:]
    ), f"expect shape a {tensor_shape}, b {expected_shape}"


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


def check_samples(obj):
    if obj.ndim > 1:
        n_samples = obj.shape[0]
    else:
        n_samples = 1
    return n_samples


def check_dim(obj, obj_dim):
    if isinstance(obj, np.ndarray):
        obj = np2ts(obj)

    n_samples = check_samples(obj)
    obj = obj.reshape(n_samples, obj_dim)

    assert_shape(obj, [None, obj_dim])
    return obj


def check_output_action(action, action_dim):
    if isinstance(action, torch.Tensor):
        action = ts2np(action)

    action = action.reshape(action_dim)

    assert isinstance(action, np.ndarray)
    assert action.shape == (action_dim,), f"shape a {action.shape}, b {(action_dim,)}"
    return action


def np2ts(obj: np.ndarray) -> torch.Tensor:
    if isinstance(obj, np.ndarray) or isinstance(obj, float):
        obj = torch.tensor(obj, dtype=torch.float32).to(device)
    return obj


def ts2np(obj: torch.Tensor) -> np.ndarray:
    return obj.cpu().detach().numpy()


def to_batch(
    batch_state, batch_action, batch_reward, batch_next_state, batch_done, device
):
    batch_state = torch.FloatTensor(np.array(batch_state)).to(device)
    batch_action = torch.FloatTensor(np.array(batch_action)).to(device)
    batch_reward = torch.FloatTensor(np.array(batch_reward)).unsqueeze(1).to(device)
    batch_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
    batch_done = torch.FloatTensor(np.array(batch_done)).unsqueeze(1).to(device)
    return batch_state, batch_action, batch_reward, batch_next_state, batch_done


def update_params(optim, network, loss, grad_clip=None, retain_graph=False):
    optim.zero_grad()
    loss.backward(retain_graph=retain_graph)
    if grad_clip is not None:
        for p in network.modules():
            torch.nn.utils.clip_grad_norm_(p.parameters(), grad_clip)
    optim.step()


def generate_grid_schedule(gap, feature_dim, feature_scale):
    l = []
    for _ in range(feature_dim):
        x = np.arange(0.0, 1.0 + gap, gap)
        l.append(x)
    g = np.meshgrid(*l)
    rgrid = []
    for i in range(len(g)):
        rgrid.append(g[i].reshape(-1, 1))
    grid = np.concatenate(rgrid, 1)
    grid = np.delete(grid, 0, 0)
    grid *= feature_scale
    return grid
