#!/usr/bin/env python3
"""
Convert TUM RGB-D benchmark sequences to ROS bags for our RGB-D VIO system.

TUM RGB-D format:
  rgb/               — colour PNGs (640x480 BGR8)
  depth/             — depth PNGs (640x480 16UC1, factor=5000 → m = val/5000)
  groundtruth.txt    — timestamp tx ty tz qx qy qz qw
  rgb.txt            — timestamp filename
  depth.txt          — timestamp filename
  accelerometer.txt  — timestamp ax ay az (Kinect accel, not used)

Output ROS bag topics:
  /camera/depth/image_rect_raw   sensor_msgs/Image     (16UC1, original TUM encoding)
  /camera/depth/camera_info      sensor_msgs/CameraInfo
  /camera/color/image_raw        sensor_msgs/Image     (BGR8)
  /camera/imu                    sensor_msgs/Imu       (synthesised from GT)

IMU synthesis:
  imu_synthesis.synthesise_imu(): ground-truth poses are resampled to 200 Hz,
  low-pass filtered (10 Hz, zero phase) and differentiated (central-difference
  body rate, analytic spline acceleration), gravity-compensated, and corrupted
  with Gaussian noise matching the D455 BMI055 IMU.

Usage:
  python3 convert_tum_rgbd.py /path/to/rgbd_dataset_freiburg1_desk \
      --output fr1_desk_d455.bag --freiburg 1

Dependencies: rosbags, numpy, scipy, opencv-python, PIL/Pillow
"""

import argparse
import os
import sys
import zlib
import numpy as np
from pathlib import Path
from imu_synthesis import synthesise_imu, IMU_RATE  # shared, tested IMU generator

# ── rosbags (ROS-version-independent bag writer) ───────────────────────────
from rosbags.rosbag1 import Writer as Bag1Writer
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS1_NOETIC)
serialize_ros1 = typestore.serialize_ros1

# Get types from the typestore
Time       = typestore.types['builtin_interfaces/msg/Time']
Header     = typestore.types['std_msgs/msg/Header']
Image      = typestore.types['sensor_msgs/msg/Image']
CameraInfo = typestore.types['sensor_msgs/msg/CameraInfo']
Imu        = typestore.types['sensor_msgs/msg/Imu']
Vector3    = typestore.types['geometry_msgs/msg/Vector3']
Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
ROI        = typestore.types['sensor_msgs/msg/RegionOfInterest']

import cv2

# ── TUM RGB-D camera intrinsics per freiburg setup ────────────────────────
INTRINSICS = {
    1: dict(fx=517.3, fy=516.5, cx=318.6, cy=255.3, w=640, h=480),
    2: dict(fx=520.9, fy=521.0, cx=325.1, cy=249.7, w=640, h=480),
    3: dict(fx=535.4, fy=539.2, cx=320.1, cy=247.6, w=640, h=480),
}

# RGB lens distortion (k1 k2 p1 p2 k3, TUM calibration).  The registered depth
# shares the raw RGB pixel grid, so fr1/fr2 colour and depth are undistorted
# together to the pinhole above; fr3 images are already undistorted.
DISTORTION = {
    1: [0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
    2: [0.2312, -0.7849, -0.0033, -0.0001, 0.9172],
    3: None,
}

# TUM depth factor: pixel_value / DEPTH_FACTOR = depth in metres
DEPTH_FACTOR = 5000.0


def read_file_list(path):
    """Read TUM-format text file: lines of 'timestamp data...'"""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            lines.append(parts)
    return lines


def associate(list_a, list_b, max_dt=0.02):
    """Associate two timestamp lists by nearest neighbour (max_dt seconds)."""
    matches = []
    b_times = np.array([float(b[0]) for b in list_b])
    for a in list_a:
        t = float(a[0])
        idx = np.argmin(np.abs(b_times - t))
        if abs(b_times[idx] - t) < max_dt:
            matches.append((a, list_b[idx]))
    return matches


def read_groundtruth(path):
    """Read groundtruth.txt → (N,8) array: [t, tx, ty, tz, qx, qy, qz, qw]"""
    lines = read_file_list(path)
    data = np.array([[float(x) for x in l] for l in lines])
    # Remove duplicate timestamps (required for interpolation)
    _, unique_idx = np.unique(data[:, 0], return_index=True)
    data = data[unique_idx]
    return data


def ts_to_ros(t_sec):
    """Convert float seconds to (sec, nsec) pair."""
    sec = int(t_sec)
    nsec = int((t_sec - sec) * 1e9)
    return sec, nsec


def main():
    parser = argparse.ArgumentParser(description='Convert TUM RGB-D to ROS bag')
    parser.add_argument('dataset_dir', help='Path to TUM sequence directory')
    parser.add_argument('--output', '-o', required=True, help='Output bag path')
    parser.add_argument('--freiburg', '-f', type=int, choices=[1, 2, 3],
                        required=True, help='Freiburg camera setup (1, 2, or 3)')
    parser.add_argument('--max-frames', type=int, default=0,
                        help='Limit number of frames (0 = all)')
    parser.add_argument('--imu-noise-scale', type=float, default=1.0,
                        help='Multiplier on BMI055 IMU noise std (default 1.0). '
                             'Used by the synth-IMU sigma-sweep experiment.')
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    intr = INTRINSICS[args.freiburg]

    # ── Read file lists ──────────────────────────────────────────────────
    print("[1/5] Reading file lists...")
    rgb_list = read_file_list(dataset / 'rgb.txt')
    depth_list = read_file_list(dataset / 'depth.txt')
    gt = read_groundtruth(dataset / 'groundtruth.txt')

    # ── Associate RGB ↔ depth ────────────────────────────────────────────
    print("[2/5] Associating RGB and depth frames...")
    pairs = associate(rgb_list, depth_list, max_dt=0.02)
    print(f"  {len(pairs)} associated pairs (of {len(rgb_list)} RGB, {len(depth_list)} depth)")

    # Filter pairs to be within GT time range (required for IMU synthesis)
    gt_t_min, gt_t_max = gt[0, 0], gt[-1, 0]
    pairs_filtered = []
    for rgb_entry, depth_entry in pairs:
        t = (float(rgb_entry[0]) + float(depth_entry[0])) / 2.0
        if gt_t_min <= t <= gt_t_max:
            pairs_filtered.append((rgb_entry, depth_entry))
    if len(pairs_filtered) < len(pairs):
        print(f"  Filtered {len(pairs) - len(pairs_filtered)} pairs outside GT range "
              f"[{gt_t_min:.2f}, {gt_t_max:.2f}] → {len(pairs_filtered)} pairs")
    pairs = pairs_filtered

    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]

    # ── Synthesise IMU ───────────────────────────────────────────────────
    print("[3/5] Synthesising IMU from ground truth...")
    imu_data = synthesise_imu(gt, noise_scale=args.imu_noise_scale,
                              seed=zlib.crc32(Path(args.output).stem.encode()))
    print(f"  {len(imu_data)} IMU samples at {IMU_RATE} Hz "
          f"({imu_data[-1, 0] - imu_data[0, 0]:.1f} s)")

    # ── Build camera info ────────────────────────────────────────────────
    K = [intr['fx'], 0.0, intr['cx'],
         0.0, intr['fy'], intr['cy'],
         0.0, 0.0, 1.0]
    undistort = None
    if DISTORTION[args.freiburg]:
        Km = np.array(K).reshape(3, 3)
        undistort = cv2.initUndistortRectifyMap(Km, np.array(DISTORTION[args.freiburg]), None, Km,
                                                (intr['w'], intr['h']), cv2.CV_32FC1)

    # ── Write bag ────────────────────────────────────────────────────────
    print(f"[4/5] Writing bag to {args.output}...")
    out_path = Path(args.output)
    if out_path.exists():
        out_path.unlink()

    with Bag1Writer(out_path) as writer:
        # Create connections (pass typestore for msgdef generation)
        conn_depth = writer.add_connection(
            '/camera/depth/image_rect_raw', 'sensor_msgs/msg/Image',
            typestore=typestore)
        conn_info = writer.add_connection(
            '/camera/depth/camera_info', 'sensor_msgs/msg/CameraInfo',
            typestore=typestore)
        conn_rgb = writer.add_connection(
            '/camera/color/image_raw', 'sensor_msgs/msg/Image',
            typestore=typestore)
        conn_imu = writer.add_connection(
            '/camera/imu', 'sensor_msgs/msg/Imu',
            typestore=typestore)

        def make_header(t_sec, frame_id='camera'):
            sec, nsec = ts_to_ros(t_sec)
            return Header(seq=0, stamp=Time(sec=sec, nanosec=nsec),
                          frame_id=frame_id)

        # ── Write IMU messages ───────────────────────────────────────────
        imu_count = 0
        for row in imu_data:
            t = row[0]
            imu_msg = Imu(
                header=make_header(t, 'imu'),
                angular_velocity=Vector3(x=row[1], y=row[2], z=row[3]),
                linear_acceleration=Vector3(x=row[4], y=row[5], z=row[6]),
                orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                orientation_covariance=np.array([-1.0] + [0.0]*8),
                angular_velocity_covariance=np.zeros(9),
                linear_acceleration_covariance=np.zeros(9),
            )
            raw = serialize_ros1(imu_msg, 'sensor_msgs/msg/Imu')
            writer.write(conn_imu, int(t * 1e9), raw)
            imu_count += 1

        print(f"  Wrote {imu_count} IMU messages")

        # ── Write image pairs ────────────────────────────────────────────
        frame_count = 0
        for rgb_entry, depth_entry in pairs:
            # Each image keeps its own capture time: the Kinect colour and
            # depth streams are not synchronised (an associated pair can be
            # up to ~16 ms apart), and a shared mean stamp misaligns them.
            t_rgb = float(rgb_entry[0])
            t_depth = float(depth_entry[0])

            # Read images
            rgb_path = str(dataset / rgb_entry[1])
            depth_path = str(dataset / depth_entry[1])

            rgb_img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)  # BGR8
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # 16UC1

            if rgb_img is None or depth_img is None:
                print(f"  WARNING: skipping frame {frame_count}, "
                      f"cannot read {rgb_path} or {depth_path}")
                continue
            if undistort is not None:
                rgb_img = cv2.remap(rgb_img, *undistort, cv2.INTER_LINEAR)
                depth_img = cv2.remap(depth_img, *undistort, cv2.INTER_NEAREST)

            hdr = make_header(t_depth, 'depth_camera')

            # Depth image message
            depth_msg = Image(
                header=hdr,
                height=depth_img.shape[0],
                width=depth_img.shape[1],
                encoding='16UC1',
                is_bigendian=0,
                step=depth_img.shape[1] * 2,
                data=np.frombuffer(np.ascontiguousarray(depth_img).tobytes(), dtype=np.uint8),
            )
            raw = serialize_ros1(depth_msg, 'sensor_msgs/msg/Image')
            writer.write(conn_depth, int(t_depth * 1e9), raw)

            # CameraInfo message
            P_mat = np.array([K[0], K[1], K[2], 0.0,
                              K[3], K[4], K[5], 0.0,
                              K[6], K[7], K[8], 0.0])
            info_msg = CameraInfo(
                header=hdr,
                height=intr['h'],
                width=intr['w'],
                distortion_model='plumb_bob',
                D=np.zeros(5),
                K=np.array(K),
                R=np.eye(3).flatten(),
                P=P_mat,
                binning_x=0, binning_y=0,
                roi=ROI(x_offset=0, y_offset=0, height=0, width=0,
                        do_rectify=False),
            )
            raw = serialize_ros1(info_msg, 'sensor_msgs/msg/CameraInfo')
            writer.write(conn_info, int(t_depth * 1e9), raw)

            # RGB image message
            rgb_msg = Image(
                header=make_header(t_rgb, 'color_camera'),
                height=rgb_img.shape[0],
                width=rgb_img.shape[1],
                encoding='bgr8',
                is_bigendian=0,
                step=rgb_img.shape[1] * 3,
                data=np.ascontiguousarray(rgb_img).view(np.uint8).ravel(),
            )
            raw = serialize_ros1(rgb_msg, 'sensor_msgs/msg/Image')
            writer.write(conn_rgb, int(t_rgb * 1e9), raw)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"  {frame_count}/{len(pairs)} frames...")

        print(f"  Wrote {frame_count} image pairs")

    # ── Save ground truth in TUM format for evo ──────────────────────────
    gt_out = out_path.with_suffix('.gt.txt')
    print(f"[5/5] Saving ground truth to {gt_out}")
    np.savetxt(gt_out, gt, fmt='%.6f',
               header='timestamp tx ty tz qx qy qz qw')

    print(f"\nDone! Bag: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Ground truth: {gt_out}")
    print(f"\nConfig notes for this sequence (freiburg{args.freiburg}):")
    print(f"  depth_scale: {1.0/DEPTH_FACTOR}  (TUM factor={DEPTH_FACTOR})")
    print(f"  intrinsics: [{intr['fx']}, {intr['fy']}, {intr['cx']}, {intr['cy']}]")
    print(f"  resolution: [{intr['w']}, {intr['h']}]")


if __name__ == '__main__':
    main()
