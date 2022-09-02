from __future__ import absolute_import, division, print_function

import copy
from lib2to3.pytree import Base
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from drone_env.envs.base_env import BaseEnv
from drone_env.envs.common.action import Action
from drone_env.envs.common.utils import extract_nparray_from_dict
from gym.spaces import Box
import copy

Observation = Union[np.ndarray, float]


class MultiTaskEnv(BaseEnv):
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config["simulation"].update(
            {
                "simulation_time_step": 0.005,  # 0.001 or 0.005 for 5x speed
            }
        )
        config["observation"].update(
            {
                "type": "Kinematics",
                "noise_stdv": 0.02,
                "scale_obs": True,
                "include_raw_state": True,
                "bnd_constraint": True,
            }
        )
        config["action"].update(
            {
                "type": "ContinuousVirtualAction",
                "act_noise_stdv": 0.05,
                "thrust_range": [-1, 1],
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
                "duration": 1500,  # [steps]
                "simulation_frequency": 50,  # [hz]
                "policy_frequency": 25,  # [hz]
                "position_success_threshhold": 1,  # [meters]
                "attitude_success_threshhold": 0.05,  # [rad]
                "tasks": {
                    "tracking": {
                        "ori_diff": np.array([1.0, 1.0, 1.0, 1.0]),
                        "ang_diff": np.array([1.0, 1.0, 1.0]),
                        "angvel_diff": np.array([1.0, 1.0, 1.0]),
                        "pos_diff": np.array([1.0, 1.0, 1.0]),
                        "vel_diff": np.array([1.0, 1.0, 1.0]),
                        "vel_norm_diff": np.array([1.0]),
                    },
                    "constraint": {
                        "survive": np.array([1.0]),
                        "action_cost": np.array([0.1, 0.1]),
                        "pos_ubnd_cost": np.array([0.1, 0.1, 0.1]),  # x,y,z
                        "pos_lbnd_cost": np.array([0.1, 0.1, 0.1]),  # x,y,z
                    },
                    "success": {
                        "att": np.array([10.0, 10.0, 10.0]),
                        "pos": np.array([10.0, 10.0, 10.0]),
                        "fail": np.array([-10.0]),
                    },  # x,y,z
                },
                "angle_reset": True,  # reset env if roll and pitch too large
            }
        )
        return config

    def __init__(self, config: Optional[Dict[Any, Any]] = None) -> None:
        super().__init__(config)

        self.angle_reset = self.config.get("angle_reset", True)
        self.max_episode_steps = self.config["duration"]
        self.tasks = self.config["tasks"]

        self.w = self.get_tasks_weights()
        self.feature_len = self.w.shape[0]
        self.tracking_feature_len = np.concatenate(
            extract_nparray_from_dict(self.tasks["tracking"])
        ).shape[0]

        self.feature_space = Box(
            low=-np.inf, high=np.inf, shape=(len(self.w),), dtype=np.float32
        )

        self.obs, self.obs_info = None, None
        if self.observation_type.constraint is not None:
            self.env_constraint = self.observation_type.constraint
            self.pos_ubnd = self.env_constraint["pos_ubnd"]
            self.pos_lbnd = self.env_constraint["pos_lbnd"]

        self.success_timer = 0

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
        reward, reward_info = self._reward(obs.copy(), action, obs_info)
        terminal, terminal_info = self._is_terminal(obs_info)

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
            "terminal_info": terminal_info,
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

        Args:
            obs (np.array):
            act (np.array): agent action [-1,1] with size (4,)
            obs_info (dict): contain all information of a step

        Returns:
            Tuple[float, dict]: [reward scalar and a detailed reward info]
        """
        features = self.calc_features(obs, obs_info)
        reward = np.dot(self.w, features)
        reward_info = {
            "reward": reward,
            "features": features,
            "tasks": self.tasks,
        }

        return float(reward), reward_info

    def reset(self) -> Observation:
        self.reset_params()
        self._reset()
        self._sample_new_goal()

        obs, obs_info = self.observation_type.observe()
        self.obs, self.obs_info = obs, obs_info
        obs_info["features"] = self.calc_features(obs, obs_info)
        return obs, obs_info

    def reset_params(self):
        self.steps = 0
        self.done = False
        self.succecss_timer = 0

    def update_tasks(self, task: dict):
        self.tasks = task
        self.w = self.get_tasks_weights()

    def get_tasks_weights(self):
        tasks = self.tasks
        feature_weights = np.concatenate(extract_nparray_from_dict(tasks["tracking"]))
        constr_weights = np.concatenate(extract_nparray_from_dict(tasks["constraint"]))
        success_weights = np.concatenate(extract_nparray_from_dict(tasks["success"]))
        return np.concatenate([feature_weights, constr_weights, success_weights])

    def calc_features(self, obs, obs_info):
        tracking_features = self.calc_track_features(obs)
        constraint_features = self.calc_constr_features(obs_info)
        success_features = self.calc_success_features(obs_info)
        return np.concatenate(
            [tracking_features, constraint_features, success_features]
        )

    def calc_track_features(self, obs: np.array) -> np.array:
        return -np.abs(obs[0 : self.tracking_feature_len])

    def calc_constr_features(self, obs_info: dict) -> np.array:
        pos = obs_info["obs_dict"]["position"]
        constr_dict = obs_info["constr_dict"]
        done, _ = self._is_terminal(obs_info)

        survive = np.array([done]).astype(np.float32)
        act = np.array(
            [self.action_type.action_rew(), self.action_type.action_cont_rew()]
        )
        pos_bnd = self.calc_dist_to_pos_bnd(pos, constr_dict)
        return np.concatenate([survive, act, pos_bnd])

    def calc_dist_to_pos_bnd(self, pos: np.array, constr_dict: dict) -> np.array:
        """closer to the bnd the higher the cost"""
        ubnd_pos = constr_dict["upper_boundary_position"]
        lbnd_pos = constr_dict["lower_boundary_position"]

        ubnd_cost = -np.exp(pos - ubnd_pos)
        lbnd_cost = -np.exp(-(pos - lbnd_pos))
        return np.clip(np.concatenate([ubnd_cost, lbnd_cost]), -10, 1)

    def calc_success_features(self, obs_info: dict) -> np.array:
        pos, goal_pos = (
            obs_info["obs_dict"]["position"],
            obs_info["goal_dict"]["position"],
        )
        pos_success = self.position_task_successs(pos, goal_pos)

        ang, goal_ang = (
            obs_info["obs_dict"]["angle"],
            obs_info["goal_dict"]["angle"],
        )
        att_success = self.attitude_task_successs(ang, goal_ang)

        _, ter_info = self._is_terminal(obs_info)
        fail = -np.array([ter_info["fail"]]).astype(float)
        return np.concatenate([att_success, pos_success, fail])

    def attitude_task_successs(self, ang: np.array, goal_ang: np.array) -> np.array:
        """task success if distance to goal is less than sucess_threshhold

        Args:
            pos ([np.array]): [position of machine]
            goal_pos ([np.array]): [position of goal]

        Returns:
            [np.array]: [1.0 if success, otherwise 0.0]
        """
        ref_ang = np.abs(ang[0:3] - goal_ang[0:3])
        return np.array(ref_ang <= self.config["attitude_success_threshhold"]).astype(
            float
        )

    def position_task_successs(self, pos: np.array, goal_pos: np.array) -> np.array:
        """task success if distance to goal is less than sucess_threshhold

        Args:
            pos ([np.array]): [position of machine]
            goal_pos ([np.array]): [position of goal]

        Returns:
            [np.array]: [1.0 if success, otherwise 0.0]
        """
        ref_pos = np.abs(pos[0:3] - goal_pos[0:3])
        return np.array(ref_pos <= self.config["position_success_threshhold"]).astype(
            float
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
        success = self.check_success(obs_info)

        fail = False
        if self.env_constraint is not None:
            if (obs_info["obs_dict"]["position"] <= self.pos_lbnd).any():
                fail = True
            if (obs_info["obs_dict"]["position"] >= self.pos_ubnd).any():
                fail = True

        if self.angle_reset:
            roll = obs_info["obs_dict"]["angle"][0]
            if (roll > np.pi - 1) or (roll < -np.pi + 1):
                fail = True
            pitch = obs_info["obs_dict"]["angle"][1]
            if (pitch > np.pi - 1) or (pitch < -np.pi + 1):
                fail = True

        return time or success or fail, {"time": time, "success": success, "fail": fail}

    def check_success(self, obs_info, success_region=0.1):
        w_pos_diff = np.array(self.tasks["tracking"]["pos_diff"])
        w_vel_diff = np.array(self.tasks["tracking"]["vel_diff"])
        w_ang_diff = np.array(self.tasks["tracking"]["ang_diff"])

        success_bnd = np.array([0.0, 0.0, 0.0])

        if (w_pos_diff > 0).any():
            value = obs_info["proc_dict"]["pos_diff"]
            diff = w_pos_diff
        elif (w_vel_diff > 0).any():
            value = obs_info["proc_dict"]["vel_diff"]
            diff = w_vel_diff
        elif (w_ang_diff > 0).any():
            value = obs_info["proc_dict"]["ang_diff"]
            diff = w_ang_diff
        else:
            return False

        success_bnd = success_region * np.array([diff[0] > 0, diff[1] > 0, diff[2] > 0])
        success_bnd[success_bnd == 0.0] += 1
        success = self.time_in_success_region(value, -success_bnd, success_bnd)
        return success

    def time_in_success_region(self, val, val_lbnd, val_ubnd, time=5):
        if (val > val_lbnd).all() and (val < val_ubnd).all():
            self.success_timer += 1
        else:
            self.succecss_timer = 0

        if self.succecss_timer > time * self.config["policy_frequency"]:
            return True

        return False

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

    ENV = MultiTaskEnv  # MultiTaskEnv, MultiTaskPIDEnv
    env_kwargs = {
        "DBG": True,
        "simulation": {
            "gui": True,
            "enable_meshes": True,
            "auto_start_simulation": auto_start_simulation,
            "position": (0, 0, 0.05),  # initial spawned position
        },
        "observation": {
            "DBG_ROS": False,
            "DBG_OBS": False,
            "noise_stdv": 0.0,
        },
        "action": {
            "DBG_ACT": True,
            "act_noise_stdv": 0.0,
            "thrust_range": [-1, 1],
        },
        "target": {
            "DBG_ROS": False,
        },
    }

    def env_step():
        env = ENV(env_kwargs)
        env.reset()
        action = env.action_space.sample()
        action = np.zeros_like(action)
        for _ in range(100000):
            action += 0.02 * np.ones_like(action)
            obs, reward, terminal, info = env.step(action)

        GazeboConnection().unpause_sim()

    env_step()
