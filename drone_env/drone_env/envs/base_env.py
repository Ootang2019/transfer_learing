""" navigate env with position target """
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from drone_env.envs.common.abstract import ROSAbstractEnv
from drone_env.envs.common.action import Action
from geometry_msgs.msg import Point, Quaternion
from std_msgs.msg import Float32MultiArray
from drone_env.envs.script import close_simulation
import line_profiler
import copy

profile = line_profiler.LineProfiler()

Observation = Union[np.ndarray, float]


class BaseEnv(ROSAbstractEnv):
    """Navigate drone by path following decomposed to altitude and planar control"""

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
                "type": "ContinuousAction",
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
                "duration": 3000,
                "simulation_frequency": 100,  # [hz]
                "policy_frequency": 50,  # [hz]
                "reward_weights": np.array(
                    [100, 0.8, 0.2]
                ),  # success, tracking, action
                "tracking_reward_weights": np.array(
                    [0, 0, 0, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0, 0, 0.25]
                ),  # ("ori_diff", "angvel_diff", "pos_diff", "vel_diff", "vel_norm_diff")
                "success_threshhold": 5,  # [meters]
            }
        )
        return config

    def _create_pub_and_sub(self):
        super()._create_pub_and_sub()
        self.rew_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_reward", Float32MultiArray, queue_size=1
        )
        self.state_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_state", Float32MultiArray, queue_size=1
        )
        self.vel_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel", Point, queue_size=1
        )
        self.vel_diff_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_vel_diff", Point, queue_size=1
        )
        self.ang_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_ang", Point, queue_size=1
        )
        self.ang_diff_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_ang_diff", Point, queue_size=1
        )
        self.act_rviz_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_act", Quaternion, queue_size=1
        )
        self.pos_cmd_pub = rospy.Publisher(
            self.config["name_space"] + "/rviz_pos_cmd",
            Point,
            queue_size=1,
        )

    @profile
    def one_step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
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

    def _step_info(self, info: dict):
        """publish all the step information to rviz

        Args:
            info ([dict]): [dict contain all step information]
        """
        obs_info = info["obs_info"]
        proc_info = obs_info["proc_dict"]

        self.rew_rviz_pub.publish(
            Float32MultiArray(data=np.array(info["reward_info"]["rew_info"]))
        )
        self.state_rviz_pub.publish(
            Float32MultiArray(
                data=np.concatenate(
                    [
                        proc_info["ori_diff"],
                        proc_info["angvel_diff"],
                        proc_info["pos_diff"],
                        proc_info["vel_diff"],
                    ]
                )
            )
        )
        self.vel_rviz_pub.publish(Point(*obs_info["velocity"]))
        self.vel_diff_rviz_pub.publish(
            Point(
                obs_info["velocity_norm"],
                self.goal["velocity"],
                proc_info["vel_diff"],
            )
        )
        self.ang_rviz_pub.publish(Point(*obs_info["angle"]))
        self.ang_diff_rviz_pub.publish(Point(*(obs_info["angle"] - self.goal["angle"])))
        self.act_rviz_pub.publish(Quaternion(*info["act"]))

        self.pos_cmd_pub.publish(Point(*self.goal["position"]))

        if self.dbg:
            print(
                f"================= [ PlanarNavigateEnv ] step {self.steps} ================="
            )
            print("STEP INFO:", info)
            print("\r")

    def reset(self) -> Observation:
        self.steps = 0
        self.done = False
        self._reset()

        self._sample_new_goal()

        obs, _ = self.observation_type.observe()
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

    def _update_goal_and_env(self, obs_info={}):
        """update goal and env state"""
        self.goal = self.target_type.sample(**obs_info)

    def _reward(
        self, obs: np.array, act: np.array, obs_info: dict
    ) -> Tuple[float, dict]:
        """calculate reward
        total_reward = success_reward + tracking_reward + action_reward
        success_reward: +1 if agent stay in the vicinity of goal
        tracking_reward: -ori_diff - angvel_diff - pos_diff - vel_diff
        action_reward: penalty for motor use

        Args:
            obs (np.array): ("ori_diff", "angvel_diff", "pos_diff", "vel_diff", and so on)
            act (np.array): agent action [-1,1] with size (4,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        track_weights = self.config["tracking_reward_weights"].copy()
        reward_weights = self.config["reward_weights"].copy()

        success_reward = self.compute_success_rew(
            obs_info["position"], obs_info["goal_dict"]["position"]
        )
        tracking_reward = np.dot(track_weights, -np.abs(obs[0:14]))
        action_reward = self.action_type.action_rew()

        reward = np.dot(
            reward_weights,
            (
                success_reward,
                tracking_reward,
                action_reward,
            ),
        )
        reward = np.clip(reward, -1, 1)
        rew_info = (reward, success_reward, tracking_reward, action_reward)
        reward_info = {"rew_info": rew_info}

        return float(reward), reward_info

    def compute_success_rew(self, pos: np.array, goal_pos: np.array) -> float:
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
        if self.config["target"]["type"] == "MultiGoal":
            success = self.target_type.wp_index == self.target_type.wp_max_index
        else:
            success_reward = self.compute_success_rew(
                obs_info["position"], obs_info["goal_dict"]["position"]
            )
            success = success_reward >= 0.99

        return time or success

    def close(self) -> None:
        return super().close()


if __name__ == "__main__":
    import copy
    from drone_env.envs.common.gazebo_connection import GazeboConnection
    from drone_env.envs.script import close_simulation

    # ============== profile ==============#
    # 1. pip install line-profiler
    # 2. in terminal:
    # kernprof -l -v drone_env/envs/planar_navigate_env.py

    auto_start_simulation = False
    if auto_start_simulation:
        close_simulation()

    ENV = BaseEnv
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

    @profile
    def env_step():
        env = ENV(copy.deepcopy(env_kwargs))
        env.reset()
        for _ in range(100000):
            action = env.action_space.sample()
            action = np.zeros_like(action)
            obs, reward, terminal, info = env.step(action)

        GazeboConnection().unpause_sim()

    env_step()
