#!/bin/bash

# read params
robotID=0

MAV_NAME=hummingbird
NAMESPACE=machine_

ROS_PORT=11311
GAZ_PORT=11351
ROSIP=$(hostname -I | cut -d' ' -f1)

X=0
Y=0
Z=10

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--robotID) robotID="$2"; shift ;;
        -p|--gaz_port) GAZ_PORT="$2"; shift ;;
        -r|--ros_port) ROS_PORT="$2"; shift ;;
        -m|--mav_name) MAV_NAME="$2"; shift ;;
        -n|--namespace) NAMESPACE="$2"; shift ;;
        -px|--pos_x) Xs="$2"; shift ;;
        -py|--pos_y) Ys="$2"; shift ;;
        -pz|--pos_z) Zs="$2"; shift ;;

        *) echo "Unknown parameter passed: $1";; 
    esac
    shift
done

# start business logics

echo "---- Spawn DRONE_${robotID} ----"
echo "robotID:$robotID"
echo "X:=${X} Y:=${Y} Z:=${Z}"

echo "Spawning DRONE_${robotID}"
screen -dmS DRONE_${robotID} screen bash -c "\
    export ROS_MASTER_URI=http://$ROSIP:$ROS_PORT;\
    export GAZEBO_MASTER_URI=http://$ROSIP:$GAZ_PORT;\
    export ROS_IP=$ROSIP;\
    export ROS_HOSTNAME=$ROSIP;\
	source ~/catkin_ws/devel/setup.bash;\
	roslaunch rotors_gazebo spawn_mav.launch mav_name:=${MAV_NAME} namespace:=${NAMESPACE}${robotID} robotID:=${robotID} x:=${X} y:=${Y} z:=${Z};"
sleep 10


echo "---- Spawn Drone_${robotID} Complete ----"
