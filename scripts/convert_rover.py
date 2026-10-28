#!/usr/bin/env python3
"""
Convert one ROVER recording (Schmidt et al., T-RO 2025; huggingface.co/datasets/iis-esslingen/ROVER, MIT licence)
to our bag format, read straight from its zip:

    /camera/color/image_raw       RealSense D435i colour, 640x480 bgr8 (realsense_D435i/rgb.txt)
    /camera/depth/image_rect_raw  D435i depth, 16UC1 millimetres (depth.txt, *_raw.png), registered to the colour
                                  image, as the dataset's own RGB-D baselines use it (one set of intrinsics)
    /camera/depth/camera_info     the Kalibr colour intrinsics of calib_d435i.yaml, no distortion
    /camera/imu                   D435i IMU rows of imu/imu.txt (t, acc xyz [m/s^2], gyro xyz [rad/s])

The total-station reference (groundtruth.txt: prism position, no orientation) is written in TUM format with an
identity quaternion.  Score with evaluate_runs.py --lever X Y Z, the prism in the IMU frame, which this script
prints from calib_d435i.yaml (Cam-To-Prism, IMU-To-Cam; "A-To-B" is T_B_A).

Usage: convert_rover.py RECORDING.zip CALIB_D435I.yaml OUT_BAG OUT_GT [--max-seconds S]
"""
import argparse
import zipfile

import cv2
import numpy as np
import yaml
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore

ts = get_typestore(Stores.ROS1_NOETIC)
T = ts.types
Time, Header = T['builtin_interfaces/msg/Time'], T['std_msgs/msg/Header']
Image, CameraInfo, Imu = T['sensor_msgs/msg/Image'], T['sensor_msgs/msg/CameraInfo'], T['sensor_msgs/msg/Imu']
Vector3, Quaternion, ROI = T['geometry_msgs/msg/Vector3'], T['geometry_msgs/msg/Quaternion'], T['sensor_msgs/msg/RegionOfInterest']


def header(t, seq, frame='camera'):
    s = int(t)
    return Header(seq=seq, stamp=Time(sec=s, nanosec=int(round((t - s) * 1e9)) % 1000000000), frame_id=frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zip'); ap.add_argument('calib'); ap.add_argument('out_bag'); ap.add_argument('out_gt')
    ap.add_argument('--max-seconds', type=float, default=None, help='convert only the start (for testing)')
    a = ap.parse_args()
    cal = yaml.safe_load(open(a.calib))
    fx, fy, cx, cy = cal['Cam_Intrinsics']['intrinsics']
    T_prism_cam, T_cam_imu = np.array(cal['Cam-To-Prism']), np.array(cal['IMU-To-Cam'])
    lever = (np.linalg.inv(T_cam_imu) @ np.linalg.inv(T_prism_cam) @ np.array([0, 0, 0, 1.0]))[:3]
    print('prism in the IMU frame (--lever): %.6f %.6f %.6f' % tuple(lever))

    z = zipfile.ZipFile(a.zip)
    root = z.namelist()[0].split('/')[0]
    D = f'{root}/realsense_D435i'
    idx = lambda f: [l.split() for l in z.read(f'{D}/{f}').decode().splitlines() if l.strip() and not l.startswith('#')]
    imu = np.loadtxt(z.open(f'{D}/imu/imu.txt'), delimiter=',', ndmin=2)
    imu = imu[np.isfinite(imu).all(1)]
    gt = np.loadtxt(z.open(f'{root}/groundtruth.txt'), ndmin=2)
    t0 = min(float(idx('rgb.txt')[0][0]), imu[0, 0])
    tmax = t0 + a.max_seconds if a.max_seconds else np.inf
    ev = [(float(t), 0, f'{D}/{p}') for t, p in idx('rgb.txt') if float(t) <= tmax]
    ev += [(float(t), 1, f'{D}/{p}') for t, p in idx('depth.txt') if float(t) <= tmax]
    ev += [(r[0], 2, i) for i, r in enumerate(imu) if r[0] <= tmax]
    ev.sort(key=lambda e: (e[0], e[1]))

    K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1.0]); P = np.array([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0.0])
    n = [0, 0, 0]
    with Writer(a.out_bag) as w:
        cc = w.add_connection('/camera/color/image_raw', 'sensor_msgs/msg/Image', typestore=ts)
        cd = w.add_connection('/camera/depth/image_rect_raw', 'sensor_msgs/msg/Image', typestore=ts)
        ci = w.add_connection('/camera/depth/camera_info', 'sensor_msgs/msg/CameraInfo', typestore=ts)
        cm = w.add_connection('/camera/imu', 'sensor_msgs/msg/Imu', typestore=ts)
        for t, kind, ref in ev:
            tn = int(round(t * 1e9))
            if kind == 2:
                r = imu[ref]
                m = Imu(header=header(t, n[2], 'imu'), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                        orientation_covariance=np.full(9, -1.0), angular_velocity=Vector3(x=r[4], y=r[5], z=r[6]),
                        angular_velocity_covariance=np.zeros(9), linear_acceleration=Vector3(x=r[1], y=r[2], z=r[3]),
                        linear_acceleration_covariance=np.zeros(9))
                w.write(cm, tn, ts.serialize_ros1(m, 'sensor_msgs/msg/Imu'))
            else:
                img = cv2.imdecode(np.frombuffer(z.read(ref), np.uint8), cv2.IMREAD_UNCHANGED)
                h, wd = img.shape[:2]
                if kind == 0:
                    m = Image(header=header(t, n[0]), height=h, width=wd, encoding='bgr8', is_bigendian=0, step=wd * 3,
                              data=np.ascontiguousarray(img).reshape(-1))
                    w.write(cc, tn, ts.serialize_ros1(m, 'sensor_msgs/msg/Image'))
                else:
                    img = img.astype(np.uint16)
                    m = Image(header=header(t, n[1]), height=h, width=wd, encoding='16UC1', is_bigendian=0, step=wd * 2,
                              data=np.ascontiguousarray(img).view(np.uint8).reshape(-1))
                    w.write(cd, tn, ts.serialize_ros1(m, 'sensor_msgs/msg/Image'))
                    info = CameraInfo(header=header(t, n[1]), height=h, width=wd, distortion_model='plumb_bob',
                                      D=np.zeros(5), K=K, R=np.eye(3).reshape(-1), P=P, binning_x=0, binning_y=0,
                                      roi=ROI(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False))
                    w.write(ci, tn, ts.serialize_ros1(info, 'sensor_msgs/msg/CameraInfo'))
            n[kind] += 1
    g = gt[gt[:, 0] <= tmax]
    np.savetxt(a.out_gt, np.column_stack([g[:, :4], np.zeros((len(g), 3)), np.ones(len(g))]), fmt='%.9f',
               header='ROVER total-station prism position (identity orientation: position only)\ntimestamp tx ty tz qx qy qz qw')
    print(f'colour {n[0]}  depth {n[1]}  imu {n[2]}  reference {len(g)}  -> {a.out_bag}')


if __name__ == '__main__':
    main()
