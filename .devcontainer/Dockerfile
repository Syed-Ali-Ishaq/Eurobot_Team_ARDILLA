FROM ros:humble-ros-base

ARG USERNAME=USERNAME
ARG UID=UID
ARG GID=$UID

# Install some dependencies packages
RUN apt update -q \
    && apt upgrade -q -y \
    && apt install -y --no-install-recommends \
    software-properties-common \
    python3-pip \
    xauth \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
  
# Create and switch to user
RUN groupadd -g $GID $USERNAME \
&& useradd -lm -u $UID -g $USERNAME -s /bin/bash $USERNAME \
&& echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER $USERNAME

# Install Gazebo
RUN sudo apt update && \
    sudo apt-get install -y ros-${ROS_DISTRO}-ros-gz

# Create workspace so that user own this directory
RUN mkdir -p /home/$USERNAME/eurobot_2026_ws/src
WORKDIR /home/$USERNAME/eurobot_2026_ws

# Copy configuration files
RUN echo 'source /opt/ros/'$ROS_DISTRO'/setup.bash' >> /home/$USERNAME/.bashrc
RUN echo 'source /home/'$USERNAME'/eurobot_2026_ws/install/setup.bash' >> /home/$USERNAME/.bashrc
RUN echo 'export ROS_DOMAIN_ID=0' >> /home/$USERNAME/.bashrc
RUN echo 'export ROS_LOCALHOST_ONLY=1' >> /home/$USERNAME/.bashrc
RUN echo 'export LIBGL_ALWAYS_SOFTWARE=1' >> ~/.bashrc

# Setup entrypoint
COPY ./ros_entrypoint.sh /
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
