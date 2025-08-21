# Dockerfile for the Aloha real environment.

# Build the container:
# docker build . -t aloha_real -f examples/aloha_real/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v /dev:/dev -v .:/app --privileged aloha_real /bin/bash

FROM ros:noetic-robot@sha256:7cf0b9f6546abeba308ea42cb7ad3453f3e520e1af57cdf179fe915c939674bc
SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    curl \
    libffi-dev \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    whiptail \
    git \
    wget \
    openssh-client \
    ros-noetic-cv-bridge \
    ros-noetic-usb-cam \
    ros-noetic-realsense2-camera \
    keyboard-configuration

WORKDIR /root
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh
RUN chmod +x xsarm_amd64_install.sh
RUN export TZ='America/Los_Angeles' && ./xsarm_amd64_install.sh -d noetic -n

COPY ./third_party/aloha /root/interbotix_ws/src/aloha
RUN cd /root/interbotix_ws && source /opt/ros/noetic/setup.sh && source /root/interbotix_ws/devel/setup.sh && catkin_make

# Install python 3.10 because this ROS image comes with 3.8
RUN mkdir /python && \
    cd /python && \
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar -zxvf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ls -lhR && \
    ./configure --enable-optimizations && \
    make install && \
    echo 'alias python3="/usr/local/bin/python3.10"' >> ~/.bashrc && \
    echo 'alias python="/usr/local/bin/python3.10"' >> ~/.bashrc && \
    cd ~ && rm -rf /python && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.6 /uv /bin/uv
ENV UV_HTTP_TIMEOUT=120
ENV UV_LINK_MODE=copy
COPY ./examples/aloha_real/requirements.txt /tmp/requirements.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml
RUN uv pip sync --python 3.10 --system /tmp/requirements.txt /tmp/openpi-client/pyproject.toml

ENV PYTHONPATH=/app:/app/src:/app/packages/openpi-client/src:/root/interbotix_ws/src/aloha/aloha_scripts:/root/interbotix_ws/src/aloha
WORKDIR /app

# Create an entrypoint script to run the setup commands, followed by the command passed in.
RUN cat <<'EOF' > /usr/local/bin/entrypoint.sh
#!/bin/bash
source /opt/ros/noetic/setup.sh && source /root/interbotix_ws/devel/setup.sh && "$@"
EOF
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python3", "/app/examples/aloha_real/main.py"]
