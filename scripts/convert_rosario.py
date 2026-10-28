#!/usr/bin/env python3
"""
Convert a Rosario dataset v2 base rosbag (RealSense D435i on a weeding robot in soybean fields) to our
pipeline format, streaming (the bags are 30-100 GB):

    /camera/color/image_raw    left infrared image (infra1, the frame the D435i computes depth in), bgr8
    /camera/depth/image_rect_raw  depth, 16UC1 millimetres (depth_scale 0.001)
    /camera/depth/camera_info  pinhole intrinsics of the output grid, no distortion
    /camera/imu                fused D435i IMU (200 Hz), unchanged

Depth and infra1 share one optical frame and one timestamp, so the depth needs no registration and the
photometric update sees the image the depth was computed from.  Both are decimated by --stride (pixel
(u, v) -> (u/s, v/s), intrinsics divided by s).  Messages are stamped with their header stamps (the
camera's global time domain, the clock of the reference); camera stamps get + --timeshift, Kalibr's
timeshift_cam_imu (t_imu = t_cam + shift).  The reference (MINS: PPK GNSS fused with this D435i IMU, in the
D435i IMU frame) is copied to --gt-out.

Usage: convert_rosario.py IN.compressed.bag OUT_d455.bag --gt-in X_mins_tum.csv --gt-out X_gt.txt
       [--stride 2] [--timeshift 0.006482764254944565] [--max-seconds N]
"""
import argparse
import shutil

import cv2
import numpy as np
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore

ts = get_typestore(Stores.ROS1_NOETIC)
T = ts.types
IN_DEPTH, IN_IR, IN_IMU, IN_INFO = ('/realsense/depth/image_rect_raw', '/realsense/infra1/image_rect_raw',
                                    '/realsense/imu', '/realsense/depth/camera_info')
OUT_COLOR, OUT_DEPTH, OUT_INFO, OUT_IMU = ('/camera/color/image_raw', '/camera/depth/image_rect_raw',
                                           '/camera/depth/camera_info', '/camera/imu')


def stamp(m):
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def header(t, frame):
    ns = int(round(t * 1e9))
    return ns, T['std_msgs/msg/Header'](seq=0, stamp=T['builtin_interfaces/msg/Time'](
        sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000), frame_id=frame)


def image(t, img, enc):
    ns, h = header(t, 'camera')
    img = np.ascontiguousarray(img)
    return ns, T['sensor_msgs/msg/Image'](header=h, height=img.shape[0], width=img.shape[1], encoding=enc,
                                          is_bigendian=0, step=img.strides[0], data=img.view(np.uint8).reshape(-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag')
    ap.add_argument('out')
    ap.add_argument('--gt-in', required=True)
    ap.add_argument('--gt-out', required=True)
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--timeshift', type=float, default=0.006482764254944565)
    ap.add_argument('--max-seconds', type=float, default=0)
    a = ap.parse_args()
    s = a.stride

    with Reader(a.bag) as r:
        print('topics:', sorted({c.topic for c in r.connections}))
        info = next(r.messages(connections=[c for c in r.connections if c.topic == IN_INFO]))
        ci = ts.deserialize_ros1(info[2], info[0].msgtype)
        K = np.array(ci.k if hasattr(ci, 'k') else ci.K, dtype=float).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0] / s, K[1, 1] / s, K[0, 2] / s, K[1, 2] / s
        w, h = ci.width // s, ci.height // s
        print(f'depth intrinsics {ci.width}x{ci.height} fx {K[0, 0]:.4f} fy {K[1, 1]:.4f} cx {K[0, 2]:.4f} '
              f'cy {K[1, 2]:.4f}; output {w}x{h} fx {fx:.4f} fy {fy:.4f} cx {cx:.4f} cy {cy:.4f}')

        conns = [c for c in r.connections if c.topic in (IN_DEPTH, IN_IR, IN_IMU)]
        n = dict(depth=0, ir=0, imu=0)
        t0 = None
        with Writer(a.out) as wr:
            c_col = wr.add_connection(OUT_COLOR, 'sensor_msgs/msg/Image', typestore=ts)
            c_dep = wr.add_connection(OUT_DEPTH, 'sensor_msgs/msg/Image', typestore=ts)
            c_inf = wr.add_connection(OUT_INFO, 'sensor_msgs/msg/CameraInfo', typestore=ts)
            c_imu = wr.add_connection(OUT_IMU, 'sensor_msgs/msg/Imu', typestore=ts)
            for conn, _, raw in r.messages(connections=conns):
                m = ts.deserialize_ros1(raw, conn.msgtype)
                t = stamp(m)
                t0 = t if t0 is None else t0
                if a.max_seconds and t - t0 > a.max_seconds:
                    break
                if conn.topic == IN_IMU:
                    ns, hd = header(t, 'imu')
                    m.header = hd
                    wr.write(c_imu, ns, ts.serialize_ros1(m, 'sensor_msgs/msg/Imu'))
                    n['imu'] += 1
                    continue
                t += a.timeshift
                if conn.topic == IN_DEPTH:
                    assert m.encoding in ('16UC1', 'mono16'), m.encoding
                    d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width)[::s, ::s]
                    ns, msg = image(t, d, '16UC1')
                    wr.write(c_dep, ns, ts.serialize_ros1(msg, 'sensor_msgs/msg/Image'))
                    _, hd = header(t, 'camera')
                    cinfo = T['sensor_msgs/msg/CameraInfo'](
                        header=hd, height=h, width=w, distortion_model='plumb_bob', D=np.zeros(5),
                        K=np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1.0]), R=np.eye(3).reshape(-1),
                        P=np.array([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1.0, 0]), binning_x=0, binning_y=0,
                        roi=T['sensor_msgs/msg/RegionOfInterest'](x_offset=0, y_offset=0, height=0, width=0,
                                                                   do_rectify=False))
                    wr.write(c_inf, ns, ts.serialize_ros1(cinfo, 'sensor_msgs/msg/CameraInfo'))
                    n['depth'] += 1
                else:
                    assert m.encoding == 'mono8', m.encoding
                    g = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width][::s, ::s]
                    ns, msg = image(t, cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 'bgr8')
                    wr.write(c_col, ns, ts.serialize_ros1(msg, 'sensor_msgs/msg/Image'))
                    n['ir'] += 1
                if (n['depth'] + n['ir']) % 3000 == 0:
                    print(f'  t={t - t0:.0f} s {n}', flush=True)
    print('written', n)
    shutil.copyfile(a.gt_in, a.gt_out)


if __name__ == '__main__':
    main()
