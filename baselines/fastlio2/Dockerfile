# Upstream FAST-LIO2 (hku-mars/FAST_LIO at 7cc4175, ikd-Tree submodule) with the restriction module of
# e14_restriction.hpp and e14_laserMapping.patch, on RGB-D bags through depth_bridge (Supplementary S12).  Built on the
# estimator image for ROS Noetic and PCL; livox_ros_driver holds message definitions only.
#
#   docker build -t rgbd-inertial:v1.1 .
#   docker build -t fastlio2-depth:7cc4175 baselines/fastlio2
FROM rgbd-inertial:v1.1

COPY livox_ros_driver /ws/src/livox_ros_driver
COPY depth_bridge /ws/src/depth_bridge

RUN git clone https://github.com/hku-mars/FAST_LIO.git /ws/src/fast_lio && \
    cd /ws/src/fast_lio && git checkout 7cc4175de6f8ba2edf34bab02a42195b141027e9 && git submodule update --init

# e14_check verifies the covariance forms against the two-dimensional example of the paper
COPY e14_restriction.hpp e14_laserMapping.patch e14_check.cpp /ws/src/fast_lio/
RUN cd /ws/src/fast_lio && mv e14_restriction.hpp include/ && git apply e14_laserMapping.patch && \
    g++ -O2 -I/usr/include/eigen3 -Iinclude e14_check.cpp -o /tmp/e14_check && /tmp/e14_check

RUN cd /ws && . /opt/ros/noetic/setup.sh && \
    catkin build livox_ros_driver depth_bridge && . /ws/devel/setup.sh && \
    catkin build fast_lio --cmake-args -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20 && \
    test -x /ws/devel/lib/fast_lio/fastlio_mapping && mkdir -p /ws/src/fast_lio/Log/result
