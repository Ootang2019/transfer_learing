""" action module of the environment """
import time
from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import rospy

from drone_env.envs.common import utils
from drone_env.envs.script import respawn_model, resume_simulation

from gym import spaces
from mav_msgs.msg import Actuators

from rospy.client import ERROR


if TYPE_CHECKING:
    from drone_env.envs.common.abstract import AbstractEnv

Action = Union[int, np.ndarray]


class ActionType:
    """abstract action type"""

    def __init__(
        self,
        env: "AbstractEnv",
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        self.env = env

    def space(self) -> spaces.Space:
        """action space"""
        raise NotImplementedError

    def act(self, action: Action) -> None:
        """perform action

        Args:
            action (Action): action
        """
        raise NotImplementedError

    def get_cur_act(self) -> np.ndarray:  # pylint: disable=no-self-use
        """return current action with a standardized format"""
        raise NotImplementedError

    def action_rew(self, scale: float) -> float:
        """calculate action reward from current action state

        Args:
            scale (float): [scale]

        Returns:
            float: [action reward indicates quality of current action state]
        """
        raise NotImplementedError


class ROSActionType(ActionType):
    """ROS abstract action type"""

    def __init__(
        self,
        env: "AbstractEnv",
        robot_id: str = "0",
        name_space: str = "machine_0",
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(env=env, **kwargs)

        self.robot_id = robot_id
        self.name_space = name_space
        self.actuator_name_space = self.name_space + "/command/motor_speed"

        self.act_dim = 4
        self.cur_act = self.init_act = np.zeros(self.act_dim)

        self._create_pub_and_sub()

    def _create_pub_and_sub(self):
        self.action_publisher = rospy.Publisher(
            self.actuator_name_space, Actuators, queue_size=1
        )
        time.sleep(0.1)

    def check_publishers_connection(self) -> None:
        """check actuator publisher connections"""
        waiting_time = 0

        while self.action_publisher.get_num_connections() == 0:
            rospy.loginfo("[ Action ] No subscriber to action_publisher yet, wait...")
            waiting_time += 1
            time.sleep(1)

            if waiting_time >= 10:
                waiting_time = 0
                self.act_err_handle()

        rospy.loginfo("[ Action ] action ready")
        return self.action_publisher.get_num_connections() > 0

    def act_err_handle(self):
        try:
            rospy.loginfo("[ Action ] respawn model...")
            reply = respawn_model(**self.env.config["simulation"])
            rospy.loginfo("[ Action ] respawn result:", reply)

        except TimeoutError:
            rospy.loginfo("[ Action ] resume simulation...")
            reply = resume_simulation(**self.env.config["simulation"])
            rospy.loginfo("[ Action ] resume simulation result:", reply)
        return reply

    def set_init_pose(self):
        """set initial actions"""
        self.check_publishers_connection()
        self.cur_act = self.init_act.copy()
        self.act(self.init_act.copy())

    def act(self, action: Action):
        raise NotImplementedError

    def action_rew(self, scale: float):
        raise NotImplementedError

    def space(self):
        raise NotImplementedError


class ContinuousAction(ROSActionType):
    """continuous action space
    action channel
    0: m0
    1: m1
    2: m2
    3: m3
    """

    ACTUATOR_RANGE = (0, 800)

    def __init__(
        self,
        env: "AbstractEnv",
        robot_id: str = "0",
        DBG_ACT: bool = False,
        name_space: str = "machine_0",
        max_thrust: float = 0.5,  # [%]
        act_noise_stdv: float = 0.05,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        super().__init__(
            env=env,
            robot_id=robot_id,
            name_space=name_space,
            **kwargs,
        )
        self.dbg_act = DBG_ACT

        self.act_dim = 4
        self.act_range = self.ACTUATOR_RANGE
        self.act_noise_stdv = act_noise_stdv
        self.max_thrust = max_thrust

        self.cur_act = self.init_act = np.zeros(self.act_dim)

    def space(self) -> spaces.Box:
        return spaces.Box(
            low=np.full((self.act_dim), -1),
            high=np.full((self.act_dim), 1),
            dtype=np.float32,
        )

    def process_action(self, action: np.array, noise_stdv: float = 0):
        """map agent action to actuator specification

        Args:
            action ([np.array]): agent action [-1, 1]

        Returns:
            [np.array]: formated action with 4 channels [0, 1000]
        """
        action += np.random.normal(0, noise_stdv, action.shape)
        proc = np.clip(action, 0, self.max_thrust)
        proc = utils.lmap(proc, [-1, 1], self.act_range)
        proc = proc.reshape(self.act_dim, 1)
        return proc

    def act(self, action: np.ndarray) -> None:
        """publish action

        Args:
            action (np.ndarray): agent action
        """
        self.cur_act = action
        processed_action = self.process_action(action, self.act_noise_stdv)

        act_msg = Actuators()
        act_msg.header.stamp = rospy.Time.now()
        act_msg.angular_velocities = processed_action

        self.action_publisher.publish(act_msg)

        if self.dbg_act:
            print("[ Action ] act: action:", action)
            print("[ Action ] act: processed_action:", processed_action)

    def action_rew(self, scale=0.5):
        """calculate action reward to penalize using motors

        Args:
            action ([np.ndarray]): agent action
            scale (float, optional): reward scale. Defaults to 0.5.

        Returns:
            [float]: action reward
        """
        motors = self.get_cur_act()
        motors_rew = np.exp(-scale * np.linalg.norm(motors))
        return motors_rew

    def get_cur_act(self):
        """get current action"""
        cur_act = self.cur_act.copy()
        return cur_act.reshape(
            self.act_dim,
        )


class ContinuousVirtualAction(ContinuousAction):
    """ "+" configuration
    action channel
    0: row
    1: pitch
    2: yaw
    3: thrust
    """

    def mixer(self, action: np.array):
        row, pitch, yaw, thrust = action
        m0 = -pitch - yaw + thrust
        m1 = row + yaw + thrust
        m2 = pitch - yaw + thrust
        m3 = -row + yaw + thrust
        return np.array([m0, m1, m2, m3])

    def process_action(self, action: np.array, noise_stdv: float = 0):
        """convert action to motor command

        Args:
            action (np.array): [row, pitch, yaw, thrust]
            noise_stdv (float, optional): action noise. Defaults to 0.

        Returns:
            proc (np.array): processed action command
        """
        actuator = self.mixer(action)
        actuator += np.random.normal(0, noise_stdv, actuator.shape)
        proc = np.clip(actuator, -1, self.max_thrust)
        proc = utils.lmap(proc, [-1, 1], self.act_range)
        proc = proc.reshape(self.act_dim, 1)
        return proc

    def act(self, action: np.ndarray) -> None:
        """publish action

        Args:
            action (np.ndarray): agent action
        """
        processed_action = self.process_action(action, self.act_noise_stdv)
        self.cur_act = processed_action

        act_msg = Actuators()
        act_msg.header.stamp = rospy.Time.now()
        act_msg.angular_velocities = processed_action

        self.action_publisher.publish(act_msg)

        if self.dbg_act:
            print("[ Action ] act: action:", action)
            print("[ Action ] act: processed_action:", processed_action)


class ContinuousDifferentialAction(ContinuousAction):
    """an accumulative action space to hard constraint the maximum change of the actuators

    actuator channel:
    0: m0
    1: m1
    2: m2
    3: m3
    """

    DIFF_ACT_SCALE = np.array([0.05, 0.05, 0.05, 0.05])
    ACT_DIM = 4

    def __init__(
        self,
        env: "AbstractEnv",
        **kwargs: dict,
    ) -> None:
        super().__init__(env, **kwargs)
        self.act_dim = self.ACT_DIM
        self.diff_act_scale = self.DIFF_ACT_SCALE

        self.init_act = np.zeros(self.act_dim)
        self.cur_act = np.zeros(self.act_dim)

    def space(self):
        return spaces.Box(
            low=np.full((self.act_dim), -1),
            high=np.full((self.act_dim), 1),
            dtype=np.float32,
        )

    def act(self, action: np.ndarray) -> None:
        """process action and publish updated actuator state to the robot

        Args:
            action (np.ndarray): agent action in range [-1, 1] with shape (4,)
        """
        self.cur_act = self.process_action(action, self.cur_act)
        proc_actuator = self.process_actuator_state(
            self.cur_act.copy(), self.act_noise_stdv
        )

        act_msg = Actuators()
        act_msg.header.stamp = rospy.Time.now()
        act_msg.angular_velocities = proc_actuator

        self.action_publisher.publish(act_msg)

        if self.dbg_act:
            print("[ Action ] agent action:", action)
            print("[ Action ] current actuator:", self.cur_act)
            print("[ Action ] process actuator:", proc_actuator)

    def process_action(self, action: np.array, cur_act: np.array) -> np.array:
        """update current actuator state by processed agent action

        Args:
            action ([np.array]): agent action (4,) in [-1,1]

        Returns:
            [np.array]: actuator state (4,) in [-1,1]
        """
        cur_act += self.diff_act_scale * action

        return cur_act

    def process_actuator_state(
        self, act_state: np.array, noise_level: float = 0.0
    ) -> np.array:
        """map agent action to actuator specification

        Args:
            act_state (np.array): agent actuator state [-1, 1] with shape (4,)
            noise_level (float, optional): noise level [0, 1]. Defaults to 0.0.

        Returns:
            [type]: processed actuator state [-1000, 1000] with shape (4,1)
        """
        proc = act_state + np.random.normal(0, noise_level, act_state.shape)
        proc = np.clip(proc, 0, self.max_thrust)
        proc = utils.lmap(proc, [-1, 1], self.act_range)
        proc = proc.reshape(self.act_dim, 1)
        return proc


def action_factory(  # pylint: disable=too-many-return-statements
    env: "AbstractEnv",
    config: dict,
) -> ActionType:
    """control action type"""
    if config["type"] == "ContinuousAction":
        return ContinuousAction(env, **config)
    elif config["type"] == "ContinuousVirtualAction":
        return ContinuousVirtualAction(env, **config)
    elif config["type"] == "ContinuousDifferentialAction":
        return ContinuousDifferentialAction(env, **config)
    else:
        raise ValueError("Unknown action type")
