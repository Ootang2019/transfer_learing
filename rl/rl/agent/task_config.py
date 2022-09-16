import numpy as np
import copy

# note: all values must be positive

base_task = {
    "tracking": {
        "ori_diff": np.array([0.0, 0.0, 0.0, 0.0]),
        "ang_diff": np.array([0.0, 0.0, 0.0]),
        "angvel_diff": np.array([0.0, 0.0, 0.0]),
        "pos_diff": np.array([0.0, 0.0, 0.0]),
        "vel_diff": np.array([0.0, 0.0, 0.0]),
        "vel_norm_diff": np.array([0.0]),
    },
    "constraint": {
        "survive": np.array([0]),
        "action_cost": np.array([0, 0]),
        "pos_ubnd_cost": np.array([0.0, 0.0, 0.0]),
        "pos_lbnd_cost": np.array([0.0, 0.0, 0.0]),
    },
    "success": {
        "att": np.array([0.0, 0.0, 0.0]),
        "pos": np.array([0.0, 0.0, 0.0]),
        "fail": np.array([0.0]),
    },
}

attitude_ctrl_task = copy.deepcopy(base_task)
attitude_ctrl_task["tracking"].update({"ang_diff": np.array([1.0, 1.0, 1.0])})
attitude_ctrl_task["constraint"].update(
    {
        "action_cost": np.array([0.01, 0.01]),
        "pos_ubnd_cost": np.array([0.5, 0.5, 0.5]),
        "pos_lbnd_cost": np.array([0.5, 0.5, 0.5]),
    }
)
attitude_ctrl_task["success"].update({"att": np.array([10.0, 10.0, 10.0])})

roll_ctrl_task = copy.deepcopy(attitude_ctrl_task)
roll_ctrl_task["tracking"].update({"ang_diff": np.array([1.0, 0.0, 0.0])})
roll_ctrl_task["success"].update({"att": np.array([10.0, 0.0, 0.0])})

pitch_ctrl_task = copy.deepcopy(attitude_ctrl_task)
pitch_ctrl_task["tracking"].update({"ang_diff": np.array([0.0, 1.0, 0.0])})
pitch_ctrl_task["success"].update({"att": np.array([0.0, 10.0, 0.0])})

yaw_ctrl_task = copy.deepcopy(attitude_ctrl_task)
yaw_ctrl_task["tracking"].update({"ang_diff": np.array([0.0, 0.0, 1.0])})
yaw_ctrl_task["success"].update({"att": np.array([0.0, 0.0, 10.0])})

xyz_ctrl_task = copy.deepcopy(base_task)
xyz_ctrl_task["tracking"].update(
    {
        "ang_diff": np.array([0.0, 0.0, 1.0]),
        "pos_diff": np.array([1.0, 1.0, 1.0]),
    },
)
xyz_ctrl_task["constraint"].update(
    {
        "survive": np.array([1]),
        "action_cost": np.array([0.01, 0.01]),
        "pos_ubnd_cost": np.array([0.5, 0.5, 0.5]),
        "pos_lbnd_cost": np.array([0.5, 0.5, 0.5]),
    }
)
xyz_ctrl_task["success"].update(
    {
        "pos": np.array([10.0, 10.0, 10.0]),
        "fail": np.array([10.0]),
    }
)

z_ctrl_task = copy.deepcopy(xyz_ctrl_task)
z_ctrl_task["tracking"].update(
    {
        "ang_diff": np.array([0.0, 0.0, 0.0]),
        "pos_diff": np.array([0.0, 0.0, 1.0]),
    },
)
z_ctrl_task["success"].update(
    {
        "pos": np.array([0.0, 0.0, 10.0]),
    }
)

xz_ctrl_task = copy.deepcopy(xyz_ctrl_task)
xz_ctrl_task["tracking"].update(
    {
        "ang_diff": np.array([0.0, 0.0, 0.0]),
        "pos_diff": np.array([1.0, 0.0, 1.0]),
    },
)
xz_ctrl_task["success"].update(
    {
        "pos": np.array([10.0, 0.0, 10.0]),
    }
)

yz_ctrl_task = copy.deepcopy(xyz_ctrl_task)
yz_ctrl_task["tracking"].update(
    {
        "ang_diff": np.array([0.0, 0.0, 0.0]),
        "pos_diff": np.array([0.0, 1.0, 1.0]),
    },
)
yz_ctrl_task["success"].update(
    {
        "pos": np.array([0.0, 10.0, 10.0]),
    }
)


def get_task_schedule(tasks=["roll", "pitch", "yaw", "att", "z", "xz", "yz", "xyz"]):
    schedule = []
    if "roll" in tasks:
        schedule.append(roll_ctrl_task)
    if "pitch" in tasks:
        schedule.append(pitch_ctrl_task)
    if "yaw" in tasks:
        schedule.append(yaw_ctrl_task)
    if "att" in tasks:
        schedule.append(attitude_ctrl_task)
    if "z" in tasks:
        schedule.append(z_ctrl_task)
    if "xz" in tasks:
        schedule.append(xz_ctrl_task)
    if "yz" in tasks:
        schedule.append(yz_ctrl_task)
    if "xyz" in tasks:
        schedule.append(xyz_ctrl_task)

    return schedule
