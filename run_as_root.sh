#!/bin/bash

export FASTRTPS_DEFAULT_PROFILES_FILE=/home/steve/ros2_ws/src/pi3hat_hardware_interface/fastrtps_profile_no_shmem.xml
source /opt/ros/humble/setup.bash
source /home/steve/ros2_ws/install/setup.bash

exec "$@"