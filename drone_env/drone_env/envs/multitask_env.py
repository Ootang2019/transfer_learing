from __future__ import absolute_import, division, print_function

from lib2to3.pytree import Base
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from drone_env.envs.common.action import Action
from drone_env.envs.base_env import BaseEnv
import copy

Observation = Union[np.ndarray, float]


class MultiTaskEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["simulation"].update({})
        config["observation"].update(
            {
                "type": "Kinematics",
                "noise_stdv": 0.02,
                "scale_obs": True,
            }
        )
        config["action"].update(
            {
                "type": "ContinuousVirtualAction",
                "act_noise_stdv": 0.05,
                "max_thrust": 1.0,
            }
        )
        config["target"].update(
            {
                "type": "FixedGoal",
                "target_name_space": "goal_",
            }
        )
        config.update(
            {
                "duration": 6000,
                "simulation_frequency": 200,  # [hz]
                "policy_frequency": 50,  # [hz]
                "success_threshhold": 5,  # [meters]
                "tasks": {
                    "tracking": {
                        "ori_diff": np.array([0.0, 0.0, 0.0, 0.0]),
                        "ang_diff": np.array([0.0, 0.0, 0.0]),
                        "angvel_diff": np.array([0.0, 0.0, 0.0]),
                        "pos_diff": np.array([0.5, 0.0, 0.0]),
                        "vel_diff": np.array([0.5, 0.0, 0.0]),
                        "vel_norm_diff": np.array([0.0]),
                    },
                    "success": np.array([0.0]),
                    "action": np.array([0.0]),
                },
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(config)
        self.tasks = self.config["tasks"]
        self.tracking_feature_len = np.concatenate(
            [v for k, v in self.tasks["tracking"].items()]
        ).shape[0]
        self.feature_len = (
            self.tracking_feature_len
            + self.tasks["success"].shape[0]
            + self.tasks["action"].shape[0]
        )
        self.obs, self.obs_info = None, None

    def reset(self) -> Observation:
        self.steps = 0
        self.done = False
        self._reset()

        self._sample_new_goal()

        obs, obs_info = self.observation_type.observe()
        self.obs, self.obs_info = obs, obs_info
        return obs

    def _sample_new_goal(self):
        if self.config["target"]["type"] == "MultiGoal" and self.config["target"].get(
            "enable_random_goal", True
        ):
            n_waypoints = np.random.randint(4, 8)
            self.target_type.sample_new_wplist(n_waypoints=n_waypoints)
        elif self.config["target"]["type"] == "RandomGoal":
            pos = self.config["simulation"]["position"]
            self.target_type.sample_new_goal(origin=np.array([pos[1], pos[0], -pos[2]]))
        elif self.config["target"]["type"] == "FixedGoal":
            self.target_type.sample_new_goal()

    def one_step(
        self,
        action: Action,
    ) -> Tuple[Observation, float, bool, dict]:
        """[perform a step action and observe result]

        Args:
            action (Action): action from the agent [-1,1] with size (4,)

        Returns:
            Tuple[Observation, float, bool, dict]:
                obs: np.array [-1,1] with size (9,),
                reward: scalar,
                terminal: bool,
                info: dictionary of all the step info,
        """
        self._simulate(action)
        obs, obs_info = self.observation_type.observe()
        self.obs, self.obs_info = obs, obs_info
        (reward, reward_info) = self._reward(
            obs.copy(), action, copy.deepcopy(obs_info)
        )
        terminal = self._is_terminal(copy.deepcopy(obs_info))
        info = {
            "step": self.steps,
            "obs": obs,
            "obs_info": obs_info,
            "act": action,
            "tasks": self.tasks,
            "reward": reward,
            "features": reward_info["features"],
            "reward_info": reward_info,
            "terminal": terminal,
        }

        self._update_goal_and_env(obs_info)
        self._step_info(info)

        return obs, reward, terminal, info

    def _reward(
        self,
        obs: np.array,
        act: np.array,
        obs_info: dict,
    ) -> Tuple[float, dict]:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: -ori_diff - angvel_diff - pos_diff - vel_diff
        action_reward: penalty for motor use
        tasks: defined by reward weights

        Args:
            obs (np.array): ("ori_diff", "angvel_diff", "pos_diff", "vel_diff", and so on)
            act (np.array): agent action [-1,1] with size (4,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        weights = self.get_tasks_weights()
        features = self.compute_features(obs, obs_info)
        reward = np.dot(weights, features)
        reward = np.clip(reward, -1, 1)
        reward_info = {
            "reward": reward,
            "features": features,
            "tasks": self.tasks,
        }

        return float(reward), reward_info

    def get_tasks_weights(self):
        tasks = self.tasks.copy()
        feature_weights = np.concatenate([v for k, v in tasks["tracking"].items()])
        return np.concatenate([feature_weights, tasks["success"], tasks["action"]])

    def get_features(self):
        return self.compute_features(self.obs, self.obs_info)

    def compute_features(self, obs, obs_info):
        tracking_features = -np.abs(obs[0 : self.tracking_feature_len])
        success_features = self.compute_success_feature(
            obs_info["obs_dict"]["position"], obs_info["goal_dict"]["position"]
        )
        action_features = np.array([self.action_type.action_rew()])
        features = np.concatenate(
            [tracking_features, success_features, action_features]
        )
        return features

    def compute_success_feature(self, pos: np.array, goal_pos: np.array) -> float:
        pos_success = self.position_task_successs(pos, goal_pos)
        return np.array([pos_success])

    def position_task_successs(self, pos: np.array, goal_pos: np.array) -> float:
        """task success if distance to goal is less than sucess_threshhold

        Args:
            pos ([np.array]): [position of machine]
            goal_pos ([np.array]): [position of planar goal]
            k (float): scaler for success

        Returns:
            [float]: [1 if success, otherwise 0]
        """
        return (
            1.0
            if np.linalg.norm(pos[0:3] - goal_pos[0:3])
            <= self.config["success_threshhold"]
            else 0.0
        )

    def _is_terminal(self, obs_info: dict) -> bool:
        """if episode terminate
        - time: episode duration finished

        Returns:
            bool: [episode terminal or not]
        """
        time = False
        if self.config["duration"] is not None:
            time = self.steps >= int(self.config["duration"]) - 1

        success = False

        fail = False
        if obs_info["obs_dict"]["position"][2] >= -5.0:
            fail = True

        return time or success or fail


if __name__ == "__main__":
    import copy
    from drone_env.envs.common.gazebo_connection import GazeboConnection
    from drone_env.envs.script import close_simulation

    # ============== profile ==============#
    # pip install snakeviz
    # python -m cProfile -o out.profile drone_env/drone_env/envs/multitask_env.py -s time
    # snakeviz multitask_env.profile

    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    ENV = MultiTaskEnv
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": auto_start_simulation,
            "position": (0, 0, 30),  # initial spawned position
        },
        "observation": {
            "DBG_ROS": False,
            "DBG_OBS": False,
            "noise_stdv": 0.02,
        },
        "action": {
            "DBG_ACT": False,
            "act_noise_stdv": 0.05,
        },
        "target": {
            "DBG_ROS": False,
        },
    }

    def env_step():
        env = ENV(copy.deepcopy(env_kwargs))
        env.reset()
        for _ in range(100000):
            action = env.action_space.sample()
            action = np.zeros_like(action)
            obs, reward, terminal, info = env.step(action)

        GazeboConnection().unpause_sim()

    env_step()
