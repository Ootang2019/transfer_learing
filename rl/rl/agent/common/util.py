import torch
import numpy as np


def get_sa_pairs(s, a):
    """s is state, a is action particles
    Args:
        s (_type_): (number of states, state dimension)
        a (_type_): (number of particles, action dimension)

    Returns:
        _type_: _description_
    """
    s_tmp = torch.tile(s, (1, a.shape[0])).reshape(-1, s.shape[1])
    a_tmp = torch.tile(a, (s.shape[0], 1))
    return torch.cat([s_tmp, a_tmp], -1)


def get_sa_pairs_(s, a):
    """_summary_
    Args:
        s (_type_): (number of states, state dimension)
        a (_type_): (number of states, number of particles, action dimension)

    Returns:
        _type_: _description_
    """
    s_tmp = torch.tile(s, (1, a.shape[1])).reshape(-1, s.shape[1])
    a_tmp = a.reshape(-1, a.shape[2])
    return torch.cat([s_tmp, a_tmp], -1)


def assert_shape(tensor, expected_shape):
    tensor_shape = tensor.shape
    assert len(tensor_shape) == len(expected_shape)
    assert all([a == b for a, b in zip(tensor_shape, expected_shape)][1:])


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1.0 - tau) + s.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


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
