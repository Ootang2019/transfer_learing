""" observation type """
#!/usr/bin/env python

from math import e
import subprocess
from typing import TYPE_CHECKING, Any, Dict, Tuple
import time
import numpy as np
import pandas as pd
import rospy
from drone_env.envs.common import utils
from drone_env.envs.script.drone_script import respawn_model, resume_simulation
from gym import spaces
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from transforms3d.euler import quat2euler


if TYPE_CHECKING:
    from drone_env.envs.common.abstract import AbstractEnv

GRAVITY = 9.81


class ObservationType:
    """abstract observation type"""

    def __init__(
        self, env: "AbstractEnv", **kwargs  # pylint: disable=unused-argument
    ) -> None:
        self.env = env

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()


class ROSObservation(ObservationType):
    """kinematirc obervation from sensors"""

    def __init__(
        self,
        env: "AbstractEnv",
        name_space="machine_0",
        DBG_ROS=False,
        DBG_OBS=False,
        real_experiment=False,
        pose_deltaT=1 / 150,  # [sec]
        **kwargs,  # pylint: disable=unused-argument
    ):
        super().__init__(env)
        self.name_space = name_space
        self.dbg_ros = DBG_ROS
        self.dbg_obs = DBG_OBS
        self.real_exp = real_experiment
        self.imu_namespace = (
            self.name_space + "/imu"
            if self.real_exp
            else self.name_space + "/ground_truth/imu"
        )
        self.pose_namespace = (
            self.name_space + "/pose"
            if self.real_exp
            else self.name_space + "/ground_truth/pose"
        )
        self.odo_namespace = (
            self.name_space + "/odometry"
            if self.real_exp
            else self.name_space + "/ground_truth/odometry"
        )

        self.pos_data = np.array([0, 0, 0])
        self.prev_pos_data = np.array([0, 0, 0])
        self.vel_data = np.array([0, 0, 0])
        self.acc_data = np.array([0, 0, 0])
        self.ori_data = np.array([0, 0, 0, 0])
        self.ang_data = np.array([0, 0, 0])
        self.ang_vel_data = np.array([0, 0, 0])

        self.pose_deltaT = pose_deltaT

        self.obs_dim = (
            len(self.pos_data)
            + len(self.vel_data)
            + len(self.acc_data)
            + len(self.ori_data)
            + len(self.ang_vel_data)
        )

        self.ros_cnt = 0

        self._create_pub_and_sub()

    def space(self) -> spaces.Space:
        return spaces.Box(
            low=-1,
            high=1,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    def _create_pub_and_sub(self):
        rospy.Subscriber(self.imu_namespace, Imu, self._imu_callback)
        rospy.Subscriber(self.pose_namespace, Pose, self._pose_callback)
        time.sleep(0.1)

    def _imu_callback(self, msg):
        """imu msg callback

        Args:
            msg ([Imu]): imu sensor raw data
        """
        self.ang_vel_data = utils.obj2array(msg.angular_velocity)

        acc = utils.obj2array(msg.linear_acceleration)
        if self.real_exp:
            acc[2] += GRAVITY
        else:
            acc[2] -= GRAVITY

        self.acc_data = self.to_NED(acc)

        if self.dbg_ros:
            self.ros_cnt += 1
            if self.ros_cnt % 1 == 0:
                print(
                    "[ KinematicObservation ] imu_callback: linear_acceleration",
                    self.acc_data,
                )
                print(
                    "[ KinematicObservation ] imu_callback: angular_velocity",
                    self.ang_vel_data,
                )

    def _pose_callback(self, msg):
        """pose msg callback

        Args:
            msg ([pose]): pose sensor data
        """
        self.prev_pos_data = self.to_NED(self.pos_data.copy())
        self.pos_data = self.to_NED(utils.obj2array(msg.position))
        self.ori_data = utils.obj2array(msg.orientation)
        self.ang_data = np.array(quat2euler(self.ori_data))
        self.vel_data = (self.pos_data - self.prev_pos_data) / self.pose_deltaT
        self.vel_data = self.to_NED(self.vel_data)

        if self.dbg_ros:
            print(
                "[ KinematicObservation ] pose_callback: position",
                self.pos_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: velocity",
                self.vel_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: orientation",
                self.ori_data,
            )
            print(
                "[ KinematicObservation ] pose_callback: angle",
                self.ang_data,
            )

    def to_NED(self, arr):
        return np.array([arr[1], arr[0], -arr[2]])

    def check_connection(self):
        """check ros connection"""
        while (self.pos_data == np.zeros(3)).all():
            rospy.loginfo("[ observation ] waiting for pose subscriber...")
            try:
                pose_data = rospy.wait_for_message(
                    self.pose_namespace,
                    Pose,
                    timeout=25,
                )
            except:
                rospy.loginfo("[ observation ] cannot find pose subscriber")
                self.obs_err_handle()
                pose_data = rospy.wait_for_message(
                    self.pose_namespace,
                    Pose,
                    timeout=25,
                )

            self.pos_data = utils.obj2array(pose_data.position)

        rospy.loginfo("[ observation ] pose ready")

        while (self.acc_data == np.zeros(3)).all():
            rospy.loginfo("[ observation ] waiting for imu subscriber...")
            try:
                acc_data = rospy.wait_for_message(
                    self.imu_namespace,
                    Imu,
                    timeout=25,
                )
            except:
                rospy.loginfo("[ observation ] cannot find imu subscriber")
                self.obs_err_handle()
                acc_data = rospy.wait_for_message(
                    self.imu_namespace,
                    Imu,
                    timeout=25,
                )
            self.acc_data = utils.obj2array(acc_data.linear_acceleration)

        rospy.loginfo("[ observation ] imu ready")

    def observe(self) -> np.ndarray:
        raise NotImplementedError

    def obs_err_handle(self):
        try:
            rospy.loginfo("[ observation ] respawn model...")
            reply = respawn_model(**self.env.config["simulation"])
            rospy.loginfo("[ observation ] respawn model status ", reply)
        except:
            rospy.loginfo("[ observation ] resume simulation...")
            reply = resume_simulation(**self.env.config["simulation"])
            rospy.loginfo("[ observation ] resume simulation status ", reply)
        return reply


class KinematicsObservation(ROSObservation):
    """kinematic observation with actuator feedback"""

    OBS = [
        "ori_diff",  # 4
        "ang_diff",  # 3
        "angvel_diff",  # 3
        "pos_diff",  # 3
        "vel_diff",  # 3
        "vel_norm_diff",  # 1
        "ori",  # 4
        "ang",  # 3
        "angvel",  # 3
        "pos",  # 3
        "vel",  # 3
        "acc",  # 3
        "goal_ori",  # 4
        "goal_ang",  # 3
        "goal_angvel",  # 3
        "goal_pos",  # 3
        "goal_vel",  # 3
    ]
    OBS_DIM = 52
    OBS_RANGE = {
        "ori_diff": [-1, 1],
        "ang_diff": [-np.pi, np.pi],
        "angvel_diff": [-50, 50],
        "pos_diff": [-50, 50],
        "vel_diff": [-50, 50],
        "vel_norm_diff": [-50, 50],
        "ori": [-1, 1],
        "ang": [-np.pi, np.pi],
        "angvel": [-50, 50],
        "pos": [-50, 50],
        "vel": [-50, 50],
        "acc": [-50, 50],
        "goal_ori": [-1, 1],
        "goal_ang": [-np.pi, np.pi],
        "goal_angvel": [-50, 50],
        "goal_pos": [-50, 50],
        "goal_vel": [-50, 50],
    }

    def __init__(
        self, env: "AbstractEnv", noise_stdv=0.02, scale_obs=True, **kwargs: dict
    ) -> None:
        super().__init__(env, **kwargs)
        self.noise_stdv = noise_stdv
        self.scale_obs = scale_obs

        self.obs_name = self.OBS.copy()
        self.obs_dim = self.OBS_DIM
        self.range_dict = self.OBS_RANGE.copy()

        self.actuator_list = [0, 1, 2, 3]
        self.obs_dim += len(self.actuator_list)
        self.obs_name.append("actuator")

    def observe(self) -> np.ndarray:
        obs, obs_dict = self._observe()
        while np.isnan(obs).any():
            rospy.loginfo("[ observation ] obs corrupted by NA")
            self.obs_err_handle()
            obs, obs_dict = self._observe()
        return obs, obs_dict

    def _observe(self) -> np.ndarray:
        goal_dict = self.env.goal
        obs_dict = {
            "position": self.pos_data,
            "velocity": self.vel_data,
            "velocity_norm": np.linalg.norm(self.vel_data),
            "linear_acceleration": self.acc_data,
            "acceleration_norm": np.linalg.norm(self.acc_data),
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angle_sin": np.sin(self.ang_data),
            "angle_cos": np.cos(self.ang_data),
            "angular_velocity": self.ang_vel_data,
        }

        proc_dict = self.process_obs(obs_dict, goal_dict, self.scale_obs)

        actuator = self.env.action_type.get_cur_act()[self.actuator_list]
        proc_dict.update({"actuator": actuator})

        proc_df = pd.DataFrame.from_records([proc_dict])
        processed = np.hstack(proc_df[self.obs_name].values[0])

        obs_info = dict(obs_dict=obs_dict, goal_dict=goal_dict, proc_dict=proc_dict)

        if self.dbg_obs:
            print("[ observation ] state", processed)
            print("[ observation ] obs obs_info", obs_info)

        return processed, obs_info

    def process_obs(
        self, obs_dict: dict, goal_dict: dict, scale_obs: bool = True
    ) -> dict:
        (
            obs_ori,
            obs_ang,
            obs_angvel,
            obs_pos,
            obs_vel,
            obs_acc,
            goal_ori,
            goal_ang,
            goal_angvel,
            goal_pos,
            goal_vel,
        ) = (
            obs_dict["orientation"],
            obs_dict["angle"],
            obs_dict["angular_velocity"],
            obs_dict["position"],
            obs_dict["velocity"],
            obs_dict["linear_acceleration"],
            goal_dict["orientation"],
            goal_dict["angle"],
            goal_dict["angular_velocity"],
            goal_dict["position"],
            goal_dict["velocity"],
        )

        state_dict = {
            "ori_diff": obs_ori - goal_ori,
            "ang_diff": self.compute_ang_diff(obs_ang - goal_ang),
            "angvel_diff": obs_angvel - goal_angvel,
            "pos_diff": obs_pos - goal_pos,
            "vel_diff": obs_vel - goal_vel,
            "vel_norm_diff": np.linalg.norm(obs_vel - goal_vel),
            "ori": obs_ori,
            "ang": obs_ang,
            "angvel": obs_angvel,
            "pos": obs_pos,
            "vel": obs_vel,
            "acc": obs_acc,
            "goal_ori": goal_ori,
            "goal_ang": goal_ang,
            "goal_angvel": goal_angvel,
            "goal_pos": goal_pos,
            "goal_vel": goal_vel,
        }

        if scale_obs:
            state_dict = self.scale_obs_dict(state_dict, self.noise_stdv)

        return state_dict

    def scale_obs_dict(self, state_dict: dict, noise_level: float = 0.0) -> dict:
        for key, val in state_dict.items():
            proc = utils.lmap(val, self.range_dict[key], [-1, 1])
            proc += np.random.normal(0, noise_level, proc.shape)
            proc = np.clip(proc, -1, 1)
            state_dict[key] = proc
        return state_dict

    def compute_ang_diff(self, ang_diff):
        ang_diff[ang_diff > np.pi] -= 2 * np.pi
        ang_diff[ang_diff < -np.pi] += 2 * np.pi
        return np.array(ang_diff)


class PlanarKinematicsObservation(ROSObservation):
    """Planar kinematics observation with actuator feedback"""

    OBS = ["z_diff", "planar_dist", "yaw_diff", "vel_diff", "vel", "yaw_vel"]
    OBS_range = {
        "z_diff": [-30, 30],
        "planar_dist": [0, 60 * np.sqrt(2)],
        "yaw_diff": [-np.pi, np.pi],
        "vel_diff": [-30, 30],
        "vel": [0, 70],
        "yaw_vel": [-30, 30],
    }

    def __init__(
        self, env: "AbstractEnv", noise_stdv=0.02, scale_obs=True, **kwargs: dict
    ) -> None:
        super().__init__(env, **kwargs)
        self.noise_stdv = noise_stdv
        self.scale_obs = scale_obs

        self.obs_name = self.OBS.copy()
        self.obs_dim = len(self.OBS)
        self.range_dict = self.OBS_range

        self.actuator_list = [0, 1, 2, 3]
        self.obs_dim += len(self.actuator_list)
        self.obs_name.append("actuator")

    def observe(self) -> np.ndarray:
        obs, obs_dict = self._observe()
        while np.isnan(obs).any():
            rospy.loginfo("[ observation ] obs corrupted by NA")
            self.obs_err_handle()
            obs, obs_dict = self._observe()
        return obs, obs_dict

    def _observe(self) -> np.ndarray:
        obs_dict = {
            "position": self.pos_data,
            "velocity": self.vel_data,
            "linear_acceleration": self.acc_data,
            "acceleration_norm": np.linalg.norm(self.acc_data),
            "orientation": self.ori_data,
            "angle": self.ang_data,
            "angular_velocity": self.ang_vel_data,
        }

        goal_dict = self.env.goal
        proc_dict = self.process_obs(obs_dict, goal_dict, self.scale_obs)

        actuator = self.env.action_type.get_cur_act()[self.actuator_list]
        proc_dict.update({"actuator": actuator})

        proc_df = pd.DataFrame.from_records([proc_dict])
        processed = np.hstack(proc_df[self.obs_name].values[0])

        obs_dict.update({"proc_dict": proc_dict})
        obs_dict.update({"goal_dict": goal_dict})

        if self.dbg_obs:
            print("[ observation ] state", processed)
            print("[ observation ] obs dict", obs_dict)

        return processed, obs_dict

    def process_obs(
        self, obs_dict: dict, goal_dict: dict, scale_obs: bool = True
    ) -> dict:
        obs_pos, obs_ang, goal_pos, next_goal_pos, goal_vel, goal_ang = (
            obs_dict["position"],
            obs_dict["angle"],
            goal_dict["position"],
            goal_dict["next_position"],
            goal_dict["velocity"],
            goal_dict["angle"],
        )
        vel = np.linalg.norm(obs_dict["velocity"])
        planar_dist = np.linalg.norm(obs_pos[0:2] - goal_pos[0:2])

        state_dict = {
            "z_diff": obs_pos[2] - goal_pos[2],
            "planar_dist": planar_dist,
            "yaw_diff": obs_ang[2] - goal_ang[2],
            "vel_diff": vel - goal_vel,
            "vel": vel,
            "yaw_vel": obs_dict["angular_velocity"][2],
        }

        if scale_obs:
            state_dict = self.scale_obs_dict(state_dict, self.noise_stdv)

        return state_dict

    def scale_obs_dict(self, state_dict: dict, noise_level: float = 0.0) -> dict:
        for key, val in state_dict.items():
            proc = utils.lmap(val, self.range_dict[key], [-1, 1])
            proc += np.random.normal(0, noise_level, proc.shape)
            proc = np.clip(proc, -1, 1)
            state_dict[key] = proc
        return state_dict


def observation_factory(env: "AbstractEnv", config: dict) -> ObservationType:
    """observation factory for different observation type"""
    if config["type"] == "Kinematics":
        return KinematicsObservation(env, **config)
    elif config["type"] == "PlanarKinematics":
        return PlanarKinematicsObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
