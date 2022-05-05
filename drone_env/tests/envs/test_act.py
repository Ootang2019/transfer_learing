from drone_env.drone_env.envs.common.abstract import AbstractEnv
import pytest
from drone_env.envs.common.gazebo_connection import GazeboConnection
from drone_env.envs.common.action import action_factory
import copy
import numpy as np


# ============== test env ==============#


def test_action():
    GazeboConnection.reset_sim()

    action_type = action_factory(env=AbstractEnv, config={"type": "ContinuousAction"})

    action = action_type.sample()
    action = np.zeros_like(action)
    action_type.act(action)
