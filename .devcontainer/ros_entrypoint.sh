#!/bin/bash
# shellcheck disable=SC1090,SC1091
set -e

# setup ros2 environment
source /opt/ros/"$ROS_DISTRO"/setup.bash --
source ~/eurobot_2026_ws/install/setup.bash --
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=1
export LIBGL_ALWAYS_SOFTWARE=1

# add sourcing to .bashrc
echo "source '/opt/ros/$ROS_DISTRO/setup.bash'" >> ~/.bashrc
echo "source '~/eurobot_2026_ws/install/setup.bash'" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc
echo "export LIBGL_ALWAYS_SOFTWARE=1" >> ~/.bashrc

exec "$@"
