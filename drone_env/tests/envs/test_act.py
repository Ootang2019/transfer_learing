from drone_env.envs.common.abstract import AbstractEnv
import pytest
from drone_env.envs.common.gazebo_connection import GazeboConnection
from drone_env.envs.common.action import action_factory
import copy
import numpy as np
import rospy

# ============== test env ==============#


def test_continuous_action():
    rospy.init_node(
        "test_act",
        anonymous=True,
        disable_signals=True,
    )

    GazeboConnection().reset_sim()
    GazeboConnection().unpause_sim()

    config = {"type": "ContinuousAction", "dbg_act": True, "act_noise_stdv": 0}
    action_type = action_factory(env=AbstractEnv, config=config)
    action = 0.457 * np.ones(action_type.act_dim)
    action_type.act(action)

    for i in range(100):
        rospy.sleep(0.1)


def test_continuous_differential_action():
    rospy.init_node(
        "test_act",
        anonymous=True,
        disable_signals=True,
    )

    GazeboConnection().reset_sim()
    GazeboConnection().unpause_sim()
    rospy.sleep(1)

    config = {
        "type": "ContinuousDifferentialAction",
        "dbg_act": True,
        "act_noise_stdv": 0,
    }
    action_type = action_factory(
        env=AbstractEnv,
        config=config,
    )
    action = 0.1 * np.ones(action_type.act_dim)
    for i in range(100):
        action_type.act(action)
        rospy.sleep(0.05)


test_continuous_action()
