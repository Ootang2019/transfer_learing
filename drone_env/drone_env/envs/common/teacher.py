import numpy as np


class PIDController:
    def __init__(
        self,
        pid_param=np.array([1.0, 0.2, 0.05]),
        gain=1.0,
        offset=0.0,
        delta_t=0.01,
        i_from_sensor=False,
        d_from_sensor=False,
    ):
        self.pid_param = pid_param
        self.gain = gain
        self.offset = offset
        self.delta_t = delta_t
        self.i_from_sensor = i_from_sensor
        self.d_from_sensor = d_from_sensor

        self.err_sum, self.prev_err = 0.0, 0.0

    def action(self, err, err_i=0, err_d=0, windup=1):
        if not self.i_from_sensor:
            self.err_sum += err * self.delta_t
            self.err_sum = np.clip(self.err_sum, -windup, windup)
            err_i = self.err_sum

        if not self.d_from_sensor:
            err_d = (err - self.prev_err) / (self.delta_t)
            self.prev_err = err

        ctrl = self.gain * np.dot(self.pid_param, np.array([err, err_i, err_d]))
        return ctrl + self.offset

    def clear(self):
        self.err_sum, self.prev_err = 0, 0


class AttitudePID:
    CONFIG = {
        "ctrl_config": {
            "row": {
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

    def __init__(self, delta_t=1 / 50) -> None:
        self.delta_t = delta_t
        self.ctrl_row = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["row"],
        )
        self.ctrl_pitch = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["pitch"],
        )
        self.ctrl_yaw = PIDController(
            delta_t=delta_t,
            **self.CONFIG["ctrl_config"]["yaw"],
        )

    def act(self, obs_info: dict):
        """
        generate base control signal
        """
        ang_diff, ang_vel, pos_diff = (
            obs_info["proc_dict"]["ang_diff"],
            obs_info["obs_dict"]["angular_velocity"],
            obs_info["proc_dict"]["pos_diff"],
        )

        row_ctrl = self.ctrl_row.action(err=-ang_diff[0], err_d=-ang_vel[0])
        pitch_ctrl = self.ctrl_pitch.action(err=-ang_diff[1], err_d=-ang_vel[1])
        yaw_ctrl = self.ctrl_yaw.action(err=ang_diff[2], err_d=ang_vel[2])

        return np.clip(np.array([row_ctrl, pitch_ctrl, yaw_ctrl, 0]), -1, 1)

    def clear_ctrl_param(self):
        self.ctrl_row.clear()
        self.ctrl_pitch.clear()
        self.ctrl_yaw.clear()

    def reset(self):
        self.clear_ctrl_param()


class PositionPID(AttitudePID):
    CONFIG = {
        "ctrl_config": {
            "row": {
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

    def act(self, obs_info: dict):
        ang_diff, ang_vel, pos_diff, ang = (
            obs_info["proc_dict"]["ang_diff"],
            obs_info["obs_dict"]["angular_velocity"],
            obs_info["proc_dict"]["pos_diff"],
            obs_info["obs_dict"]["angle"],
        )
        x_ctrl = self.ctrl_x.action(err=pos_diff[0])
        y_ctrl = self.ctrl_y.action(err=pos_diff[1])
        z_ctrl = self.ctrl_z.action(err=pos_diff[2])

        yaw = -ang[2]
        row_cmd = x_ctrl * np.cos(yaw) + y_ctrl * np.sin(yaw)
        pitch_cmd = -x_ctrl * np.sin(yaw) + y_ctrl * np.cos(yaw)

        row_ctrl = self.ctrl_row.action(err=row_cmd - ang_diff[0], err_d=-ang_vel[0])
        pitch_ctrl = self.ctrl_pitch.action(
            err=-pitch_cmd - ang_diff[1], err_d=-ang_vel[1]
        )
        yaw_ctrl = self.ctrl_yaw.action(err=ang_diff[2], err_d=ang_vel[2])

        return np.clip(np.array([row_ctrl, pitch_ctrl, yaw_ctrl, z_ctrl]), -1, 1)

    def clear_ctrl_param(self):
        self.ctrl_z.clear()

    def reset(self):
        self.clear_ctrl_param()
