""" script """
import errno
import os
import pathlib
import socket
import subprocess
import time
from typing import Tuple

import rospy
from drone_env.envs.common.utils import timeout

path = pathlib.Path(__file__).parent.resolve()

DEFAULT_ROSPORT = 11311
DEFAULT_GAZPORT = 11351


# ============ check screen ============#


def find_screen_session(name: str):
    try:
        screen_name = subprocess.check_output(
            f"ls /var/run/screen/S-* | grep {name}",
            shell=True,
        )
    except subprocess.CalledProcessError:
        screen_name = None
    return screen_name


def check_screen_sessions_exist(names: list = ["ROSMASTER_", "WORLD_", "DRONE_"]):
    all_exist = True
    for name in names:
        all_exist *= find_screen_session(name) is not None
    return bool(all_exist)


# ============ Spawn Script ============#


def spawn_ros_master(
    robot_id: int = 0, ros_port: int = DEFAULT_ROSPORT, gaz_port: int = DEFAULT_GAZPORT
) -> int:
    """spawn ros master at specified port number"""

    names = ["ROSMASTER_" + str(robot_id)]
    while check_screen_sessions_exist(names=names) is not True:
        call_reply = subprocess.check_call(
            str(path)
            + f"/spawn_rosmaster.sh -i {robot_id} -p {gaz_port} -r {ros_port}",
            shell=True,
        )
        time.sleep(3)
    return int(call_reply)


def spawn_world(
    robot_id: int = 0,
    world: str = "basic",
    gui: bool = False,
    gaz_port: int = DEFAULT_GAZPORT,
    ros_port: int = DEFAULT_ROSPORT,
) -> int:
    """spawn gazebo world"""
    names = ["WORLD_" + str(robot_id)]
    while check_screen_sessions_exist(names=names) is not True:
        call_reply = subprocess.check_call(
            str(path)
            + f"/spawn_world.sh -i {robot_id} -g {gui} -d {world} -p {gaz_port} -r {ros_port}",
            shell=True,
        )
        time.sleep(3)

    return int(call_reply)


def spawn_drone(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    position: tuple = (0, 0, 100),
    namespace: str = "machine_",
    mav_name: str = "hummingbird",
) -> int:
    """spawn drone software in-the-loop"""

    names = ["DRONE_" + str(robot_id)]
    # while check_screen_sessions_exist(names=names) is not True:
    kill_screens(robot_id=robot_id, screen_names=names, sleep_times=[1])
    call_reply = subprocess.check_call(
        str(path)
        + f"/spawn_drone.sh -i {robot_id} -r {ros_port} -p {gaz_port} -px {position[0]} -py {position[1]} -pz {position[2]} -n {namespace} -m {mav_name}",
        shell=True,
    )
    time.sleep(3)

    return int(call_reply)


# ============ Composite Spawn Script ============#


def spawn_simulation_on_different_port(
    robot_id: int = 0,
    gui: bool = True,
    world: str = "basic",
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    position: tuple = (0, 0, 100),
    **kwargs,  # pylint: disable=unused-argument
) -> dict:
    """start drone simulator on different ros or gazbo port"""

    ros_reply = spawn_ros_master(
        robot_id=robot_id, ros_port=ros_port, gaz_port=gaz_port
    )
    world_reply = spawn_world(
        robot_id=robot_id, world=world, gui=gui, ros_port=ros_port, gaz_port=gaz_port
    )
    drone_reply = spawn_drone(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        position=position,
    )
    proc_result = {
        "ros_reply": ros_reply,
        "world_reply": world_reply,
        "drone_reply": drone_reply,
    }
    rospy.loginfo("spawn process result:", proc_result)
    return proc_result


# ============ Kill Script ============#


def close_simulation() -> int:
    """kill all simulators"""
    reply = int(subprocess.check_call(str(path) + "/cleanup.sh"))
    return reply


def kill_screen(screen_name, sleep_time=1):
    reply = 1
    while find_screen_session(screen_name) is not None:
        try:
            reply = subprocess.check_call(
                f'for session in $(screen -ls | grep {screen_name}); do screen -S "${{session}}" -X quit; done',
                shell=True,
            )
            time.sleep(sleep_time)
        except:
            print(f"screen {screen_name} not found, skip kill")
    return reply


def kill_screens(
    robot_id: int,
    screen_names: list = ["DRONE_", "WORLD_", "ROSMASTER_"],
    sleep_times: list = [3, 1, 10, 5],
) -> Tuple[int]:
    """kill screen session by specifying screen name and robot_id

    Args:
        robot_id ([str]): [number of the robot]
        screen_names ([list]): [screen name]
        sleep_time ([list]): [sleep time after kill]

    Returns:
        [Tuple[int]]: [status of the script]
    """
    reply = {}
    for screen_name, sleep_time in zip(screen_names, sleep_times):
        kill_reply = kill_screen(screen_name + str(robot_id), sleep_time)
        reply.update({"kill_" + screen_name + str(robot_id): kill_reply})
    return reply


def kill_rosnode(node_name: str) -> str:
    reply = subprocess.check_call(
        f"for node in $(rosnode list | grep {node_name}) | xargs rosnode kill",
        shell=True,
    )
    return reply


def remove_drone(robot_id: int) -> int:
    """remove drone model from gazebo world"""
    reply = int(
        subprocess.check_call(
            f"rosservice call /gazebo/delete_model \"model_name: 'machine_{robot_id}' \"",
            shell=True,
        )
    )
    time.sleep(5)
    return reply


# ============ Respawn Script ============#


@timeout(50, os.strerror(errno.ETIMEDOUT))
def respawn_model(
    robot_id: int = 0,
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    position: tuple = (0, 0, 100),
    **kwargs,  # pylint: disable=unused-argument
) -> dict:
    """respawn model
    first kill the screen session and then remove model from gazebo
    lastly spawn model again
    """
    kill_model_reply = kill_screens(
        robot_id=robot_id, screen_names=["DRONE_"], sleep_times=[1]
    )
    remove_model_reply = remove_drone(robot_id=robot_id)

    drone_reply = spawn_drone(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        position=position,
    )

    return {
        "kill_model": kill_model_reply,
        "remove_model": remove_model_reply,
        "spawn_model": drone_reply,
    }


def resume_simulation(
    robot_id: int = 0,
    gui: bool = True,
    world: str = "basic",
    ros_port: int = DEFAULT_ROSPORT,
    gaz_port: int = DEFAULT_GAZPORT,
    position: tuple = (0, 0, 100),
    **kwargs,  # pylint: disable=unused-argument
) -> dict:
    """resume simulation
    first kill all screens
    then spawn gazebo world and blimp SITL

    Args:
        robot_id (int, optional): [description]. Defaults to 0.
        gui (bool, optional): [description]. Defaults to True.
        world (str, optional): [description]. Defaults to "basic".
        ros_port ([type], optional): [description]. Defaults to DEFAULT_ROSPORT.
        gaz_port ([type], optional): [description]. Defaults to DEFAULT_GAZPORT.
        enable_meshes (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [success]
    """
    kill_reply = kill_screens(robot_id=robot_id)
    master_reply = spawn_ros_master(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
    )
    world_reply = spawn_world(
        robot_id=robot_id,
        world=world,
        gui=gui,
        ros_port=ros_port,
        gaz_port=gaz_port,
    )

    drone_reply = spawn_drone(
        robot_id=robot_id,
        ros_port=ros_port,
        gaz_port=gaz_port,
        position=position,
    )

    proc_result = {
        "master_reply": master_reply,
        "world_reply": world_reply,
        "drone_reply": drone_reply,
    }
    rospy.loginfo("spawn process result:", proc_result)

    return {
        "kill_all_screen": kill_reply,
        "proc_result": proc_result,
    }
