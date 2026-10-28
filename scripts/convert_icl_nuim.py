#!/usr/bin/env python3
"""
Convert ICL-NUIM benchmark sequences to ROS bags for our RGB-D VIO system.

ICL-NUIM format (living room / office room trajectories):
  Sequence directories (e.g., living_room_traj0_frei_png/):
    rgb/               — colour PNGs (640x480 RGB)
    depth/             — depth PNGs (640x480 16UC1, factor=5000 → m = val/5000)
    associations.txt   — or generated from filenames
  Ground truth:  livingRoom0.gt.freiburg  (TUM format: t tx ty tz qx qy qz qw)

ICL-NUIM camera intrinsics (same for all sequences):
  fx=481.20, fy=-480.00, cx=319.50, cy=239.50
  Note: fy is negative in their convention (image rows run upward in the camera frame of the
  ground-truth poses); we flip the image rows and use |fy|, which keeps that camera frame.

Download:
  https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html

Usage:
  python3 convert_icl_nuim.py /path/to/living_room_traj0_frei_png \
      --gt livingRoom0.gt.freiburg \
      --output icl_lr0.bag

Dependencies: rosbags, numpy, scipy, opencv-python
"""

import argparse
import os
import sys
import zlib
import numpy as np
from pathlib import Path
from imu_synthesis import synthesise_imu, IMU_RATE  # shared, tested IMU generator

from rosbags.rosbag1 import Writer as Bag1Writer
from rosbags.typesys import Stores, get_typestore

typestore = get_typestore(Stores.ROS1_NOETIC)
serialize_ros1 = typestore.serialize_ros1

Time       = typestore.types['builtin_interfaces/msg/Time']
Header     = typestore.types['std_msgs/msg/Header']
Image      = typestore.types['sensor_msgs/msg/Image']
CameraInfo = typestore.types['sensor_msgs/msg/CameraInfo']
Imu        = typestore.types['sensor_msgs/msg/Imu']
Vector3    = typestore.types['geometry_msgs/msg/Vector3']
Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
ROI        = typestore.types['sensor_msgs/msg/RegionOfInterest']

import cv2

# ICL-NUIM camera intrinsics (all sequences)
ICL_INTRINSICS = dict(fx=481.20, fy=480.00, cx=319.50, cy=239.50, w=640, h=480)

# Depth factor: pixel_value / 5000 = metres (same as TUM convention)
DEPTH_FACTOR = 5000.0


def read_gt_freiburg(path):
    """Read ICL-NUIM ground truth in TUM/freiburg format.

    ICL-NUIM GT uses frame indices as timestamps (1,2,3,...).
    We rescale to 30 Hz so that timestamps match the image bag.
    """
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [float(x) for x in line.split()]
            lines.append(parts)
    data = np.array(lines)
    # Deduplicate timestamps
    _, unique_idx = np.unique(data[:, 0], return_index=True)
    data = data[unique_idx]

    # ICL-NUIM timestamps are frame indices (1,2,3...).
    # Rescale to 30 Hz: frame_idx -> (frame_idx - 1) / 30.
    fps = 30.0
    t = data[:, 0]
    if len(t) > 1 and np.allclose(np.diff(t), 1.0, atol=0.1):
        data[:, 0] = (t - t[0]) / fps
    return data


def discover_frames(seq_dir):
    """Discover RGB + depth frame pairs from directory listing."""
    rgb_dir = seq_dir / 'rgb'
    depth_dir = seq_dir / 'depth'

    # Try associations.txt first
    assoc_file = seq_dir / 'associations.txt'
    if assoc_file.exists():
        pairs = []
        with open(assoc_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    # ICL-NUIM associations: t_depth depth_path t_rgb rgb_path
                    # Detect format by checking if 2nd col contains 'depth' or 'rgb'
                    if 'depth' in parts[1]:
                        pairs.append((float(parts[2]), parts[3], float(parts[0]), parts[1]))
                    else:
                        pairs.append((float(parts[0]), parts[1], float(parts[2]), parts[3]))

        # ICL-NUIM "timestamps" are frame indices (0,1,2,...), not seconds.
        # Rescale to 30 Hz so IMU synthesis and playback work correctly.
        if pairs and pairs[-1][0] > 0:
            t_max = max(p[0] for p in pairs)
            n_frames = len(pairs)
            if abs(t_max - (n_frames - 1)) < 1.0:  # integer frame indices
                fps = 30.0
                pairs = [(i / fps, p[1], i / fps, p[3])
                         for i, p in enumerate(pairs)]
        return pairs

    # Fall back to filename enumeration.  ICL-NUIM filenames are sequential integers (0.png, 1.png, ...): sort them
    # numerically.  (Until 2026-10-26 they were sorted as strings, 0, 1, 10, 100, 1000, ..., which scrambled the
    # frame order of every ICL bag built without an associations.txt.)
    frame_index = lambda f: int(f.stem) if f.stem.isdigit() else f.stem
    rgb_files = sorted(rgb_dir.glob('*.png'), key=frame_index)
    depth_files = sorted(depth_dir.glob('*.png'), key=frame_index)

    # Synthesise timestamps at 30 Hz
    fps = 30.0
    pairs = []
    for i, (rgb_f, depth_f) in enumerate(zip(rgb_files, depth_files)):
        t = i / fps
        pairs.append((t, str(rgb_f.relative_to(seq_dir)),
                       t, str(depth_f.relative_to(seq_dir))))
    return pairs


def ts_to_ros(t_sec):
    sec = int(t_sec)
    nsec = int((t_sec - sec) * 1e9)
    return sec, nsec


def main():
    parser = argparse.ArgumentParser(description='Convert ICL-NUIM to ROS bag')
    parser.add_argument('dataset_dir', help='Path to ICL-NUIM sequence directory')
    parser.add_argument('--gt', required=True, help='Ground truth file (.gt.freiburg)')
    parser.add_argument('--output', '-o', required=True, help='Output bag path')
    parser.add_argument('--max-frames', type=int, default=0, help='Limit frames (0=all)')
    args = parser.parse_args()

    dataset = Path(args.dataset_dir)
    intr = ICL_INTRINSICS

    print("[1/5] Reading ground truth...")
    gt = read_gt_freiburg(args.gt)
    print(f"  {len(gt)} GT poses")

    print("[2/5] Discovering frames...")
    pairs = discover_frames(dataset)
    print(f"  {len(pairs)} frame pairs")

    if args.max_frames > 0:
        pairs = pairs[:args.max_frames]

    print("[3/5] Synthesising IMU from ground truth...")
    imu_data = synthesise_imu(gt, seed=zlib.crc32(Path(args.output).stem.encode()))
    print(f"  {len(imu_data)} IMU samples at {IMU_RATE} Hz")

    K = [intr['fx'], 0.0, intr['cx'],
         0.0, intr['fy'], intr['cy'],
         0.0, 0.0, 1.0]

    print(f"[4/5] Writing bag to {args.output}...")
    out_path = Path(args.output)
    if out_path.exists():
        out_path.unlink()

    with Bag1Writer(out_path) as writer:
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

        # Write IMU
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

        print(f"  Wrote {len(imu_data)} IMU messages")

        # Write image pairs
        frame_count = 0
        for entry in pairs:
            t_rgb, rgb_path_rel, t_depth, depth_path_rel = entry
            t = (t_rgb + t_depth) / 2.0

            rgb_img = cv2.imread(str(dataset / rgb_path_rel), cv2.IMREAD_COLOR)
            depth_img = cv2.imread(str(dataset / depth_path_rel), cv2.IMREAD_UNCHANGED)

            if rgb_img is None or depth_img is None:
                continue

            # ICL-NUIM renders with fy = -480.  Flipping the rows and using |fy| reproduces the camera frame of
            # the ground-truth poses; without the flip the depth geometry is the y-mirror of the reference motion,
            # so an IMU synthesised from the reference cannot agree with it (every bag before 2026-10-26 had this).
            rgb_img = np.ascontiguousarray(rgb_img[::-1])
            depth_img = np.ascontiguousarray(depth_img[::-1])

            hdr = make_header(t, 'depth_camera')

            depth_msg = Image(
                header=hdr,
                height=depth_img.shape[0], width=depth_img.shape[1],
                encoding='16UC1', is_bigendian=0,
                step=depth_img.shape[1] * 2,
                data=np.frombuffer(np.ascontiguousarray(depth_img).tobytes(),
                                   dtype=np.uint8),
            )
            raw = serialize_ros1(depth_msg, 'sensor_msgs/msg/Image')
            writer.write(conn_depth, int(t * 1e9), raw)

            P_mat = np.array([K[0], K[1], K[2], 0.0,
                              K[3], K[4], K[5], 0.0,
                              K[6], K[7], K[8], 0.0])
            info_msg = CameraInfo(
                header=hdr, height=intr['h'], width=intr['w'],
                distortion_model='plumb_bob',
                D=np.zeros(5), K=np.array(K),
                R=np.eye(3).flatten(), P=P_mat,
                binning_x=0, binning_y=0,
                roi=ROI(x_offset=0, y_offset=0, height=0, width=0,
                        do_rectify=False),
            )
            raw = serialize_ros1(info_msg, 'sensor_msgs/msg/CameraInfo')
            writer.write(conn_info, int(t * 1e9), raw)

            rgb_msg = Image(
                header=make_header(t, 'color_camera'),
                height=rgb_img.shape[0], width=rgb_img.shape[1],
                encoding='bgr8', is_bigendian=0,
                step=rgb_img.shape[1] * 3,
                data=np.ascontiguousarray(rgb_img).view(np.uint8).ravel(),
            )
            raw = serialize_ros1(rgb_msg, 'sensor_msgs/msg/Image')
            writer.write(conn_rgb, int(t * 1e9), raw)

            frame_count += 1
            if frame_count % 200 == 0:
                print(f"  {frame_count}/{len(pairs)} frames...")

        print(f"  Wrote {frame_count} image pairs")

    gt_out = out_path.with_suffix('.gt.txt')
    print(f"[5/5] Saving ground truth to {gt_out}")
    np.savetxt(gt_out, gt, fmt='%.6f',
               header='timestamp tx ty tz qx qy qz qw')

    print(f"\nDone! Bag: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Ground truth: {gt_out}")
    print(f"\nConfig notes for ICL-NUIM:")
    print(f"  depth_scale: {1.0/DEPTH_FACTOR}")
    print(f"  intrinsics: [{intr['fx']}, {intr['fy']}, {intr['cx']}, {intr['cy']}]")
    print(f"  resolution: [{intr['w']}, {intr['h']}]")


if __name__ == '__main__':
    main()
