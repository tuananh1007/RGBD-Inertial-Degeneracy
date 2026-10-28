cmake_minimum_required(VERSION 2.8.3)
project(rgbd_vio)

set(CMAKE_BUILD_TYPE "Release")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Architecture detection (same as FAST-LIVO2)
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|aarch64|ARM|AARCH64)")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -ffast-math")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mcpu=native -mtune=native -mfpu=neon -ffast-math")
  endif()
  add_definitions(-DARM_ARCH)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops")
  add_definitions(-DX86_ARCH)
endif()

# Root directory for log/config paths
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

# ---- Path to shared FAST-LIVO2 sources ----
set(FASTLIVO2_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../FAST-LIVO2-main)

# Multithreading
option(FORCE_SINGLE_THREAD "Build without OpenMP parallel loops (determinism test)" OFF)
include(ProcessorCount)
ProcessorCount(N)
if(FORCE_SINGLE_THREAD)
  set(PROC_NUM 1)
  add_definitions(-DMP_PROC_NUM=1)
elseif(N GREATER 4)
  math(EXPR PROC_NUM "4")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
elseif(N GREATER 1)
  math(EXPR PROC_NUM "${N}")
  add_definitions(-DMP_EN -DMP_PROC_NUM=${PROC_NUM})
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()
message(STATUS "Multithreading: ${PROC_NUM} cores")

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  add_compile_options(${OpenMP_CXX_FLAGS})
endif()

find_package(mimalloc QUIET)

# ROS + required libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  std_msgs
  pcl_ros
  tf
  message_generation
  eigen_conversions
  vikit_common
  vikit_ros
  cv_bridge
  image_transport
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

set(Sophus_LIBRARIES libSophus.so)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
                 message_runtime cv_bridge vikit_common vikit_ros image_transport
  DEPENDS EIGEN3 PCL OpenCV Sophus
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Sophus_INCLUDE_DIRS}
  include                          # our headers
  ${FASTLIVO2_DIR}/include         # reuse FAST-LIVO2 headers (VIO, LIO, IMU, common)
)

# ----- Shared FAST-LIVO2 source libraries -----
# Compiled from the FAST-LIVO2 sources (vio.cpp carries patches/fastlivo2/02-photometric-gate.patch and
# 03-local-ba.patch).
# Prefixed names: the ocs_vio package (Oursv2) builds unprefixed fl2_* libraries from the same sources,
# and devel/lib on LD_LIBRARY_PATH would otherwise shadow ours at runtime.
add_library(rgbd_vio_fl2_vio
  ${FASTLIVO2_DIR}/src/vio.cpp
  ${FASTLIVO2_DIR}/src/frame.cpp
  ${FASTLIVO2_DIR}/src/visual_point.cpp
)

add_library(rgbd_vio_fl2_lio
  ${FASTLIVO2_DIR}/src/voxel_map.cpp
)

add_library(rgbd_vio_fl2_imu
  ${FASTLIVO2_DIR}/src/IMU_Processing.cpp
)

# ----- Our new sources -----
add_library(depth_proc src/depth_processor.cpp)
add_library(rgbd_mapper src/RGBDMapper.cpp src/inertial_init.cpp src/dtf.cpp)

# Unit tests for the staged-IMU maths (Eigen only): rosrun rgbd_vio test_inertial
add_executable(test_inertial test/test_inertial.cpp src/inertial_init.cpp)

# Unit tests for the degeneracy-aware ICP update (Eigen only): rosrun rgbd_vio test_dtf
add_executable(test_dtf test/test_dtf.cpp src/dtf.cpp)

add_executable(rgbd_mapping_node src/main.cpp)

set(COMMON_LINK_LIBS
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Sophus_LIBRARIES}
  ${Boost_LIBRARIES}
)

target_link_libraries(rgbd_mapping_node
  rgbd_mapper
  depth_proc
  rgbd_vio_fl2_vio
  rgbd_vio_fl2_lio
  rgbd_vio_fl2_imu
  ${COMMON_LINK_LIBS}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(rgbd_mapping_node OpenMP::OpenMP_CXX)
  target_link_libraries(test_inertial OpenMP::OpenMP_CXX)
  target_link_libraries(test_dtf OpenMP::OpenMP_CXX)
endif()

if(mimalloc_FOUND)
  target_link_libraries(rgbd_mapping_node mimalloc)
endif()
