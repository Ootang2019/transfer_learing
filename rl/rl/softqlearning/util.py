import torch


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
