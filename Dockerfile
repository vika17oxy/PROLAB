# ── Stage 1: build ──────────────────────────────────────────────────────────
FROM ros:humble AS builder

# Install build tools and Eigen
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-colcon-common-extensions \
        python3-rosdep \
        libeigen3-dev \
        ros-humble-eigen3-cmake-module \
        ros-humble-tf2 \
        ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs \
        ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# Create workspace and copy source
WORKDIR /ros2_ws/src
COPY prol_filters/ prol_filters/
COPY map.pgm prol_filters/map/map.pgm
COPY map.yaml prol_filters/map/map.yaml

# Build the package
WORKDIR /ros2_ws
RUN . /opt/ros/humble/setup.sh && \
    colcon build \
        --packages-select prol_filters \
        --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && rm -rf build/prol_filters/CMakeFiles

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM ros:humble AS runtime

# Runtime Python deps + RViz2 + X11 client libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-matplotlib \
        python3-numpy \
        ros-humble-tf2 \
        ros-humble-tf2-ros \
        ros-humble-tf2-geometry-msgs \
        ros-humble-visualization-msgs \
        ros-humble-rviz2 \
        libgl1-mesa-glx \
        libx11-6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# X11 display forwarding — set DISPLAY at runtime (e.g. -e DISPLAY=$DISPLAY)
ENV QT_X11_NO_MITSHM=1

# Copy built workspace from builder
COPY --from=builder /ros2_ws/install /ros2_ws/install

# CSV logs land here — mount a host volume to persist them
VOLUME ["/data"]

# Source overlay on every shell/exec
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo "source /ros2_ws/install/setup.bash" >> /root/.bashrc

# Entrypoint sources both setups then runs the given command
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# Default: launch all three filters
CMD ["ros2", "launch", "prol_filters", "filters.launch.py"]
