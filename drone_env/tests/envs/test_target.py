from drone_env.envs.common.abstract import AbstractEnv
import pytest
from drone_env.envs.common.gazebo_connection import GazeboConnection
from drone_env.envs.common.target import target_factory
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


def test_random_goal():
    rospy.init_node(
        "test_target",
        anonymous=True,
        disable_signals=True,
    )

    GazeboConnection().reset_sim()
    GazeboConnection().unpause_sim()

    config = {"type": "RandomGoal"}
    target_type = target_factory(env=mock_env(), config=config)
    target_type.sample_new_goal(
        min_dist_to_origin=5,
        range_dict=dict(
            x_range=(-10, 10),
            y_range=(-10, 10),
        ),
    )

    for _ in range(100):
        target = target_type.sample()
        print(target)
        rospy.sleep(0.1)


test_random_goal()
