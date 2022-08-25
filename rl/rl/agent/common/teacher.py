from re import A
from .agent import AbstractAgent
import numpy as np
import torch
from agent.common.util import np2ts, ts2np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPS = 1e-3


class PIDController:
    def __init__(
        self,
        pid_param: np.ndarray = np.array([1.0, 0.2, 0.05]),
        gain: float = 1.0,
        offset: float = 0.0,
        delta_t: float = 0.01,
        i_from_sensor: bool = False,
        d_from_sensor: bool = False,
    ):
        self.pid_param = (
            torch.tensor(pid_param, dtype=torch.float32).to(device).reshape([1, -1])
        )
        self.gain = torch.FloatTensor([gain]).to(device)
        self.offset = torch.FloatTensor([offset]).to(device)
        self.delta_t = torch.FloatTensor([delta_t]).to(device)
        self.i_from_sensor = i_from_sensor
        self.d_from_sensor = d_from_sensor

        self.err_sum = torch.FloatTensor([0.0]).to(device).reshape(1, -1)
        self.prev_err = torch.FloatTensor([0.0]).to(device).reshape(1, -1)

    def action(self, err, err_i_sensor=0, err_d_sensor=0, windup=1):
        n_sample = err.shape[0]
        err = err[:, None]

        if n_sample == 1:
            err_i, err_d = self.one_sample_act(err, err_i_sensor, err_d_sensor, windup)
        else:
            err_i, err_d = self.mul_sample_act(err, err_i_sensor, err_d_sensor, windup)

        err_cat = torch.cat([err, err_i, err_d], 1)
        ctrl = self.gain * torch.sum(self.pid_param * err_cat, 1) + self.offset
        return ctrl

    def one_sample_act(self, err, err_i_sensor, err_d_sensor, windup):
        if not self.i_from_sensor:
            self.err_sum += err * self.delta_t
            self.err_sum = torch.clip(self.err_sum, -windup, windup)
            err_i = self.err_sum
        else:
            err_i = err_i_sensor

        if not self.d_from_sensor:
            err_d = (err - self.prev_err) / (self.delta_t + EPS)
            self.prev_err = err
        else:
            err_d = err_d_sensor

        return err_i, err_d

    def mul_sample_act(self, err, err_i_sensor, err_d_sensor, windup):
        if not self.i_from_sensor:
            # sample uniform random [-1,1]
            err_i = torch.rand(err.shape).to(device) * 2 - 1
            err_i = err_i
        else:
            err_i = err_i_sensor

        if not self.d_from_sensor:
            # sample gaussian (err, 1)
            prev_err = torch.normal(err, torch.ones_like(err)).to(device)
            err_d = (err - prev_err) / (self.delta_t + EPS)
        else:
            err_d = err_d_sensor

        return err_i, err_d

    def clear(self):
        self.err_sum = torch.FloatTensor([0.0]).to(device).reshape(1, -1)
        self.prev_err = torch.FloatTensor([0.0]).to(device).reshape(1, -1)


class AttitudePID:
    CONFIG = {
        "ctrl_config": {
            "roll": {
                "pid_param": np.array([1.0, 0.7, 0.2]),
                "gain": 0.3,
                "d_from_sensor": True,
            },
            "pitch": {
                "pid_param": np.array([1.0, 0.7, 0.2]),
                "gain": 0.3,
                "d_from_sensor": True,
            },
            "yaw": {
                "pid_param": np.array([1.0, 0.5, 0.1]),
                "gain": 0.2,
                "d_from_sensor": True,
            },
        },
    }

    # TODO: get rid of magic number
    OBS_IDX = [
        [24, 25, 26],
        [4, 5, 6],
    ]  # angvel, ang_diff
    OBS_SCALE = [50, 1]

    def __init__(self, delta_t: float = 1 / 50) -> None:
        self.delta_t = delta_t
        self.ctrl_roll = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["roll"],
        )
        self.ctrl_pitch = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["pitch"],
        )
        self.ctrl_yaw = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["yaw"],
        )

    def act(self, obs):
        tobs = self.unpack_teacher_obs_as_tensor(obs)
        action = self.get_action(*tobs).squeeze()
        return ts2np(action)

    def get_action(self, ang_vel, ang_diff):
        """
        generate base control signal
        """
        roll_ctrl = self.ctrl_roll.action(
            err=-ang_diff[:, 0], err_d_sensor=-ang_vel[:, 0]
        )
        pitch_ctrl = self.ctrl_pitch.action(
            err=-ang_diff[:, 1], err_d_sensor=-ang_vel[:, 1]
        )
        yaw_ctrl = self.ctrl_yaw.action(err=ang_diff[:, 2], err_d_sensor=ang_vel[:, 2])

        ctrl = torch.cat([roll_ctrl, pitch_ctrl, yaw_ctrl, torch.zeros([1, 1])])
        action = torch.clip(ctrl, -1, 1)

        return ts2np(action)

    def unpack_teacher_obs(self, obs: np.ndarray):
        return [
            np.array(obs[self.OBS_IDX[i]]) * self.OBS_SCALE[i]
            for i in range(len(self.OBS_IDX))
        ]

    def unpack_teacher_obs_as_tensor(self, obs: np.ndarray):
        tobs = self.unpack_teacher_obs(obs)
        for i in range(len(tobs)):
            tobs[i] = np2ts(tobs[i].reshape([1, -1]))
        return tobs

    def __call__(self, obs: torch.Tensor):
        tobs = [
            obs[:, self.OBS_IDX[i]] * self.OBS_SCALE[i]
            for i in range(len(self.OBS_IDX))
        ]
        return self.get_action(*tobs), None, None

    def clear_ctrl_param(self):
        self.ctrl_roll.clear()
        self.ctrl_pitch.clear()
        self.ctrl_yaw.clear()

    def reset(self):
        self.clear_ctrl_param()


class PositionPID(AttitudePID):
    CONFIG = {
        "ctrl_config": {
            "roll": {
                "pid_param": np.array([1.0, 0.5, 0.1]),
                "gain": 1.0,
                "d_from_sensor": False,
            },
            "pitch": {
                "pid_param": np.array([1.0, 0.5, 0.1]),
                "gain": 1.0,
                "d_from_sensor": False,
            },
            "yaw": {
                "pid_param": np.array([1.0, 0.1, 0.4]),
                "gain": 0.1,
                "d_from_sensor": False,
            },
            "x": {
                "pid_param": np.array([1.0, 0.2, 0.5]),
                "gain": 0.2,
                "d_from_sensor": False,
            },
            "y": {
                "pid_param": np.array([1.0, 0.2, 0.5]),
                "gain": 0.2,
                "d_from_sensor": False,
            },
            "z": {
                "pid_param": np.array([1.0, 0.4, 3]),
                "gain": 1.5,
                "d_from_sensor": False,
            },
        },
    }

    # TODO: get rid of magic numbers here
    OBS_IDX = [
        [21, 22, 23],
        [24, 25, 26],
        [4, 5, 6],
        [10, 11, 12],
    ]  # ang, angvel, ang_diff, pos_diff
    OBS_SCALE = [np.pi, 50, 1, 1]

    def __init__(self, delta_t=1 / 50) -> None:
        super().__init__(delta_t)

        self.ctrl_x = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["x"],
        )
        self.ctrl_y = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["y"],
        )
        self.ctrl_z = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["z"],
        )

    def get_action(self, ang, ang_vel, ang_diff, pos_diff):
        x_ctrl = self.ctrl_x.action(err=pos_diff[:, 0])
        y_ctrl = self.ctrl_y.action(err=pos_diff[:, 1])
        z_ctrl = self.ctrl_z.action(err=pos_diff[:, 2]).unsqueeze(1)

        yaw = -ang[:, 2]
        roll_cmd = x_ctrl * torch.cos(yaw) + y_ctrl * torch.sin(yaw)
        pitch_cmd = -x_ctrl * torch.sin(yaw) + y_ctrl * torch.cos(yaw)

        roll_ctrl = self.ctrl_roll.action(
            err=roll_cmd - ang_diff[:, 0], err_d_sensor=-ang_vel[:, 0]
        ).unsqueeze(1)
        pitch_ctrl = self.ctrl_pitch.action(
            err=-pitch_cmd - ang_diff[:, 1], err_d_sensor=-ang_vel[:, 1]
        ).unsqueeze(1)
        yaw_ctrl = self.ctrl_yaw.action(
            err=ang_diff[:, 2], err_d_sensor=ang_vel[:, 2]
        ).unsqueeze(1)

        ctrl = torch.cat([roll_ctrl, pitch_ctrl, yaw_ctrl, z_ctrl], 1)
        return torch.clip(ctrl, -1, 1)

    def clear_ctrl_param(self):
        self.ctrl_x.clear()
        self.ctrl_y.clear()
        self.ctrl_z.clear()
        super().clear_ctrl_param()
