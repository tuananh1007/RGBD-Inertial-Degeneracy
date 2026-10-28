# The estimator (rgbd_vio) on Ubuntu 20.04 and ROS Noetic, CPU only, for amd64 and arm64 (e.g. Jetson Orin under
# JetPack 5 or 6; the container needs no NVIDIA runtime).  Third-party sources are pinned to the commits used in the
# paper: FAST-LIVO2 0d2c034 (with the two patches in patches/fastlivo2), rpg_vikit 6c886c8, Sophus a621ff.
#
#   docker build -t rgbd-inertial:v1.0 .
FROM ros:noetic-ros-base
ARG FASTLIVO2_COMMIT=0d2c0346107b75b59934975adec9a6eeeb913c64
ARG VIKIT_COMMIT=6c886c8e5d83997806e00294826d528cea3581dd
# parallel compile jobs; 4 keeps the build within 8 GB of memory
ARG BUILD_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git python3-catkin-tools \
        libboost-all-dev libeigen3-dev libopencv-dev libpcl-dev libfmt-dev \
        ros-noetic-pcl-ros ros-noetic-cv-bridge ros-noetic-image-transport \
        ros-noetic-tf ros-noetic-tf2-ros ros-noetic-eigen-conversions \
        ros-noetic-nav-msgs ros-noetic-sensor-msgs ros-noetic-geometry-msgs \
        ros-noetic-visualization-msgs ros-noetic-message-generation ros-noetic-message-runtime \
        python3-rosbag python3-numpy psmisc procps \
    && rm -rf /var/lib/apt/lists/*

# Sophus, non-templated double-only commit a621ff (as FAST-LIVO2 requires)
RUN git clone https://github.com/strasdat/Sophus.git /opt/Sophus && cd /opt/Sophus && git checkout a621ff && \
    sed -i 's/unit_complex_.real() = 1\.;/unit_complex_ = std::complex<double>(1.0, 0.0);/' sophus/so2.cpp && \
    sed -i '/unit_complex_.imag() = 0\.;/d' sophus/so2.cpp && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j${BUILD_JOBS} && make install && ldconfig

# vikit (FAST-LIVO2 fork) and the FAST-LIVO2 sources (sources only: rgbd_vio compiles the files it shares)
RUN mkdir -p /ws/src && cd /ws/src && \
    git clone https://github.com/xuankuzcr/rpg_vikit.git && cd rpg_vikit && git checkout ${VIKIT_COMMIT} && cd .. && \
    git clone https://github.com/hku-mars/FAST-LIVO2.git FAST-LIVO2-main && \
    cd FAST-LIVO2-main && git checkout ${FASTLIVO2_COMMIT} && touch CATKIN_IGNORE

# Photometric consistency gate (vio/gate_chi2) and local photometric bundle adjustment (vio/lba_en), both off by default
COPY patches/fastlivo2/02-photometric-gate.patch patches/fastlivo2/03-local-ba.patch /tmp/
RUN cd /ws/src/FAST-LIVO2-main && git apply /tmp/02-photometric-gate.patch && git apply /tmp/03-local-ba.patch

COPY rgbd_vio /ws/src/rgbd_vio

RUN cd /ws && . /opt/ros/noetic/setup.sh && \
    catkin build rgbd_vio -j${BUILD_JOBS} --no-status --cmake-args -DCMAKE_BUILD_TYPE=Release > /tmp/ws_build.log 2>&1 \
        || (grep -n -E 'error|Error|Killed|fatal' /tmp/ws_build.log | head -40; tail -40 /tmp/ws_build.log; exit 1) && \
    /ws/devel/lib/rgbd_vio/test_inertial && \
    /ws/devel/lib/rgbd_vio/test_dtf

WORKDIR /ws
