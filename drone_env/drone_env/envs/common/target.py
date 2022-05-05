from math import pi
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rospy
from drone_env.envs.common import utils
from gym import spaces
from librepilot.msg import AutopilotInfo
from transforms3d.euler import euler2quat, quat2euler
from visualization_msgs.msg import InteractiveMarkerInit, Marker, MarkerArray
from geometry_msgs.msg import Point

import time

if TYPE_CHECKING:
    from drone_env.envs.common.abstract import AbstractEnv


class WayPoint:
    def __init__(self, position=np.zeros(3), velocity=np.zeros(1), angle=np.zeros(3)):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.angle = np.array(angle)
        self.orientation = euler2quat(angle)

    def to_ENU(self):
        return np.array([self.position[1], self.position[0], -self.position[2]])


class TargetType:
    """abstract target type"""

    def __init__(
        self,
        env: "AbstractEnv",
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.env = env

    def space(self) -> spaces.Space:
        """get target space"""
        raise NotImplementedError

    def sample(self):
        """sample a goal"""
        raise NotImplementedError()

    def check_connection(self):
        pass


class RandomGoal(TargetType):
    """a random generated goal during training"""

    def __init__(
        self,
        env: "AbstractEnv",
        target_name_space="goal_0",
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env)

        self.target_name_space = target_name_space

        self.target_dim = 16
        self.pos_cmd_data = np.zeros(3)
        self.vel_cmd_data = np.zeros(3)
        self.ori_cmd_data = np.zeros(4)
        self.ang_cmd_data = np.zeros(3)
        self.angvel_cmd_data = np.zeros(3)

        self.range_dict = dict(
            x_range=(-50, 50),
            y_range=(-50, 50),
            z_range=(-5, -50),
            v_range=(-50, 50),
            phi_range=(-np.pi, np.pi),
            the_range=(-np.pi, np.pi),
            psi_range=(-np.pi, np.pi),
            phivel_range=(-10, 10),
            thevel_range=(-10, 10),
            psivel_range=(-10, 10),
        )

        self._pub_and_sub = False
        self._create_pub_and_sub()

    def space(self) -> spaces.Space:
        """gym space, only for testing purpose"""
        return spaces.Box(
            low=np.full((self.target_dim), -1),
            high=np.full((self.target_dim), 1),
            dtype=np.float32,
        )

    def _create_pub_and_sub(self) -> None:
        """create publicator and subscriber"""
        self.wp_viz_publisher = rospy.Publisher(
            self.target_name_space + "/rviz_pos_cmd", Marker, queue_size=1
        )
        self._pub_and_sub = True

    def publish_waypoint_toRviz(self, waypoint):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.action = marker.ADD
        marker.type = marker.SPHERE
        marker.id = 0
        marker.scale.x, marker.scale.y, marker.scale.z = 2, 2, 2
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = 1, 1, 1, 0
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            waypoint[1],
            waypoint[0],
            -waypoint[2],
        )  ## NED --> rviz(ENU)
        marker.pose.orientation.w = 1
        self.wp_viz_publisher.publish(marker)

    def generate_goal(
        self,
        x_range=(-50, 50),
        y_range=(-50, 50),
        z_range=(-5, -50),
        v_range=(-50, 50),
        phi_range=(-np.pi, np.pi),
        the_range=(-np.pi, np.pi),
        psi_range=(-np.pi, np.pi),
        phivel_range=(-10, 10),
        thevel_range=(-10, 10),
        psivel_range=(-10, 10),
    ):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)
        pos_cmd = np.array([x, y, z])

        vx = np.random.uniform(*v_range)
        vy = np.random.uniform(*v_range)
        vz = np.random.uniform(*v_range)
        v_cmd = np.array([vx, vy, vz])

        phi = np.random.uniform(*phi_range)
        the = np.random.uniform(*the_range)
        psi = np.random.uniform(*psi_range)
        ang_cmd = np.array([phi, the, psi])
        q_cmd = euler2quat(*ang_cmd)

        phivel = np.random.uniform(*phivel_range)
        thevel = np.random.uniform(*thevel_range)
        psivel = np.random.uniform(*psivel_range)
        angvel_cmd = np.array([phivel, thevel, psivel])
        return pos_cmd, v_cmd, ang_cmd, q_cmd, angvel_cmd

    def check_planar_distance(self, waypoint0, waypoint1, min_dist=0):
        """check if planar distance between 2 waypoints are greater than min_dist"""
        dist = np.linalg.norm(waypoint0[0:2] - waypoint1[0:2])
        return dist >= min_dist

    def sample_new_goal(
        self,
        origin=np.array([0, 0, -25]),
        min_dist_to_origin=0,
        range_dict={},
    ):
        self.range_dict.update(range_dict)
        far_enough = False
        while far_enough == False:
            pos_cmd, v_cmd, ang_cmd, q_cmd, angvel_cmd = self.generate_goal(
                **self.range_dict
            )
            far_enough = self.check_planar_distance(pos_cmd, origin, min_dist_to_origin)

        self.pos_cmd_data = pos_cmd
        self.vel_cmd_data = v_cmd
        self.ang_cmd_data = ang_cmd
        self.ori_cmd_data = q_cmd
        self.angvel_cmd_data = angvel_cmd

    def sample(self, **kwargs) -> Dict[str, np.ndarray]:
        """sample target state depend on machine_state

        machine_state (dict): machine position, veloctiy, orientation, etc.

        Returns:
            dict: target info dictionary
        """
        self.publish_waypoint_toRviz(self.pos_cmd_data)
        return {
            "orientation": self.ori_cmd_data,
            "angle": self.ang_cmd_data,
            "angular_velocity": self.angvel_cmd_data,
            "position": self.pos_cmd_data,
            "velocity": self.vel_cmd_data,
        }


class MultiGoal(TargetType):
    """a specified goal sequences."""

    def __init__(
        self,
        env: "AbstractEnv",
        target_name_space="goal_0",
        trigger_dist=5,  # [m] dist to trigger next waypoint
        wp_list=[
            (40, 40, -15, 3),
            (40, -40, -15, 3),
            (-40, -40, -15, 3),
            (-40, 40, -15, 3),
        ],  # [m] (x, y, z, v) in NED
        enable_dependent_wp=False,  # waypoint generated depend on previous waypoint
        dist_range=[10, 40],  # [m] new wp range of prev wp
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env)

        self.enable_dependent_wp = enable_dependent_wp

        self.target_name_space = target_name_space
        self.target_dim = 9
        self.min_dist = dist_range[0]
        self.max_dist = dist_range[1]

        self.wp_list = []
        for wp in wp_list:
            self.wp_list.append(WayPoint(wp[0:3], wp[3]))

        self.wp_max_index = len(self.wp_list)
        self.wp_index = 0
        self.next_wp_index = self.wp_index + 1
        self.wp = self.wp_list[self.wp_index]
        self.next_wp = self.wp_list[self.next_wp_index]

        self.trigger_dist = trigger_dist

        self._pub_and_sub = False
        self._create_pub_and_sub()

    def space(self) -> spaces.Space:
        """gym space, only for testing purpose"""
        return spaces.Box(
            low=np.full((self.target_dim), -1),
            high=np.full((self.target_dim), 1),
            dtype=np.float32,
        )

    def _create_pub_and_sub(self) -> None:
        """create publicator and subscriber"""
        self.wp_viz_publisher = rospy.Publisher(
            self.target_name_space + "/rviz_pos_cmd", Marker, queue_size=1
        )
        self.wplist_viz_publisher = rospy.Publisher(
            self.target_name_space + "/rviz_waypoint_list", MarkerArray, queue_size=10
        )
        self.path_viz_publisher = rospy.Publisher(
            self.target_name_space + "/rviz_path", Marker, queue_size=5
        )
        self._pub_and_sub = True

    def create_rviz_marker(
        self, waypoint: np.array, scale=(2, 2, 2), color=(1, 1, 1, 0)
    ):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.action = marker.ADD
        marker.type = marker.SPHERE
        marker.id = 0
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.a, marker.color.r, marker.color.g, marker.color.b = color
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = (
            waypoint[1],
            waypoint[0],
            -waypoint[2],
        )  ## NED --> rviz(ENU)
        marker.pose.orientation.w = 1
        return marker

    def publish_waypoint_toRviz(
        self, waypoint: WayPoint, scale: Tuple = (4, 4, 4), color: Tuple = (1, 0, 0, 1)
    ):
        marker = self.create_rviz_marker(waypoint.position, scale=scale, color=color)
        self.wp_viz_publisher.publish(marker)

    def publish_wplist_toRviz(self, wp_list: Tuple[WayPoint]):
        markerArray = MarkerArray()

        for wp in wp_list:
            marker = self.create_rviz_marker(wp.position)
            markerArray.markers.append(marker)

            id = 0
            for m in markerArray.markers:
                m.id = id
                id += 1

            self.wplist_viz_publisher.publish(markerArray)

    def check_planar_distance(self, waypoint0, waypoint1, min_dist=10):
        """check if planar distance between 2 waypoints are greater than min_dist"""
        return np.linalg.norm(waypoint0[0:2] - waypoint1[0:2]) > min_dist

    def close_enough(self, waypoint0, waypoint1, trigger_dist=5):
        """check if planar distance between 2 waypoints are less than trigger_dist"""
        return np.linalg.norm(waypoint0[0:2] - waypoint1[0:2]) < trigger_dist

    def _wp_index_plus_one(self):
        self.wp_index += 1
        if self.wp_index >= self.wp_max_index:
            self.wp_index = 0

        self.next_wp_index = self.wp_index + 1
        if self.next_wp_index >= self.wp_max_index:
            self.next_wp_index = 0

    def sample(
        self, machine_position: np.array = np.zeros(3), **kwargs
    ) -> Dict[str, np.ndarray]:
        """sample target state depend on machine_state

        machine_state (dict): machine position, veloctiy, orientation, etc.

        Returns:
            dict: target info dictionary
        """

        if self.env._pub_and_sub:
            if self.close_enough(
                self.wp_list[self.wp_index].position,
                machine_position,
                self.trigger_dist,
            ):
                self._wp_index_plus_one()
                self.wp, self.next_wp = (
                    self.wp_list[self.wp_index],
                    self.wp_list[self.next_wp_index],
                )

            self.publish_waypoint_toRviz(self.wp)
            self.publish_wplist_toRviz(self.wp_list)

        return {
            "position": self.wp.position,
            "velocity": self.wp.velocity,
            "angle": self.wp.angle,
            "next_position": self.next_wp.position,
        }

    def _generate_waypoint(
        self,
        x_range=np.array([-105, 105]),
        y_range=np.array([-105, 105]),
        z_range=np.array([-5, -210]),
        v_range=np.array([1.5, 7]),
    ):
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = np.random.uniform(*z_range)
        v = np.random.uniform(*v_range)
        return np.array([x, y, z]), np.array([v])

    def _generate_valid_waypoint(self, prev_pos_cmd=np.array([0, 0, -100])):
        far_enough = False
        if self.enable_dependent_wp:
            x_range = prev_pos_cmd[0] + np.array([-self.max_dist, self.max_dist])
            y_range = prev_pos_cmd[1] + np.array([-self.max_dist, self.max_dist])

        while far_enough == False:
            pos_cmd, v_cmd = self._generate_waypoint(x_range=x_range, y_range=y_range)
            far_enough = self.check_planar_distance(
                pos_cmd, prev_pos_cmd, min_dist=self.min_dist
            )
        return WayPoint(pos_cmd, v_cmd)

    def _generate_random_wplist(self, n_waypoints, origin=np.array([0, 0, -100])):
        wp_list = []
        wp = WayPoint(origin, 0)
        for _ in range(n_waypoints):
            wp = self._generate_valid_waypoint(prev_pos_cmd=wp.position)
            wp_list.append(wp)
        return wp_list

    def sample_new_wplist(self, n_waypoints=4):
        self.wp_list = self._generate_random_wplist(n_waypoints)
        self.wp_max_index = len(self.wp_list)
        self.wp_index = 0
        self.next_wp_index = self.wp_index + 1
        self.wp = self.wp_list[self.wp_index]
        self.next_wp = self.wp_list[self.next_wp_index]


def target_factory(env: "AbstractEnv", config: dict) -> TargetType:
    """generate different types of target

    Args:
        config (dict): [config should specify target type]

    Returns:
        TargetType: [a target will generate goal for RL agent]
    """
    if config["type"] == "RandomGoal":
        return RandomGoal(env, **config)
    elif config["type"] == "MultiGoal":
        return MultiGoal(env, **config)
    else:
        raise ValueError("Unknown target type")
