from drone_env.envs.common.abstract import AbstractEnv
import pytest
from drone_env.envs.common.gazebo_connection import GazeboConnection
from drone_env.envs.common.observation import observation_factory
import copy
import numpy as np
import rospy

# ============== test env ==============#


class mock_action_type:
    def get_cur_act(self):
        return np.array([0, 0, 0, 0])


class mock_env:
    def __init__(self) -> None:
        self.action_type = mock_action_type()
        self.goal = {
            "task": 0,
            "orientation": np.zeros(4),
            "angular_velocity": np.zeros(3),
            "position": np.zeros(3),
            "velocity": np.zeros(3),
        }


def test_kinematics_observation():
    rospy.init_node(
        "test_obs",
        anonymous=True,
        disable_signals=True,
    )

    GazeboConnection().reset_sim()
    GazeboConnection().unpause_sim()

    config = {"type": "Kinematics", "DBG_OBS": True, "DBG_ROS": False}
    observation_type = observation_factory(env=mock_env(), config=config)

    for i in range(100):
        i += 1
        observation_type.observe()
        rospy.sleep(0.1)


def test_planar_kinematics_observation():
    rospy.init_node(
        "test_obs",
        anonymous=True,
        disable_signals=True,
    )

    GazeboConnection().reset_sim()
    GazeboConnection().unpause_sim()

    config = {"type": "PlanarKinematics", "DBG_OBS": True, "DBG_ROS": False}
    observation_type = observation_factory(env=mock_env(), config=config)

    for i in range(100):
        i += 1
        observation_type.observe()
        rospy.sleep(0.1)


test_planar_kinematics_observation()
