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
                "type": "ContinuousAngularAction",
                "act_noise_stdv": 0.05,
                "max_thrust": 1.0,
            }
        )
        config["target"].update(
            {
                "type": "RandomGoal",
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
                    "orientation": np.array([0.0, 0.0, 0.0, 0.0]),
                    "angvel_diff": np.array([0.0, 0.0, 0.0]),
                    "position": np.array([0.5, 0.0, 0.0]),
                    "vel_diff": np.array([0.5, 0.0, 0.0]),
                    "vel_norm_diff": np.array([0.0]),
                    "action": np.array([0.0]),
                    "success": np.array([0.0]),
                },
            }
        )
        return config

    def one_step(
        self,
        action: Action,
    ) -> Tuple[Observation, float, bool, dict]:
        """[perform a step action and observe result]

        Args:
            action (Action): action from the agent [-1,1] with size (4,)
            tasks (dict): define current task

        Returns:
            Tuple[Observation, float, bool, dict]:
                obs: np.array [-1,1] with size (9,),
                reward: scalar,
                terminal: bool,
                info: dictionary of all the step info,
        """
        self._simulate(action)
        obs, obs_info = self.observation_type.observe()
        reward, reward_info = self._reward(obs.copy(), action, copy.deepcopy(obs_info))
        terminal = self._is_terminal(copy.deepcopy(obs_info))
        info = {
            "step": self.steps,
            "obs": obs,
            "obs_info": obs_info,
            "act": action,
            "reward": reward,
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
        tasks = self.config["tasks"]
        track_weights = np.concatenate(
            [
                tasks["orientation"],
                tasks["angvel_diff"],
                tasks["position"],
                tasks["vel_diff"],
                tasks["vel_norm_diff"],
            ]
        )

        success_reward = tasks["success"] * self.compute_success_rew(
            obs_info["position"], obs_info["goal_dict"]["position"]
        )
        tracking_reward = np.dot(track_weights, -np.abs(obs[0:14]))
        action_reward = tasks["action"] * self.action_type.action_rew()

        reward = success_reward + tracking_reward + action_reward
        reward = np.clip(reward, -1, 1)
        reward_info = {
            "rew_info": (reward, success_reward, tracking_reward, action_reward)
        }

        return float(reward), reward_info
