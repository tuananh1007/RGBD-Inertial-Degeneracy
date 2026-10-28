#!/usr/bin/env python3
"""
Convert a UT-MM bag (Sun et al., IROS 2024; RealSense D455 + Microstrain IMU +
Vicon) to the topic layout played by run_sequences.sh.

  /realsense/depth/image_rect_raw       -> /camera/depth/image_rect_raw  (copied)
  /realsense/depth/camera_info          -> /camera/depth/camera_info     (copied)
  /realsense/color/image_raw/compressed -> /camera/color/image_raw       (JPEG -> bgr8)
  /microstrain/imu/data                 -> /camera/imu                   (copied)

Copied messages keep their bytes, connection headers and bag timestamps.
Also written next to OUT.bag:
  OUT.gt.txt        Vicon pose of the Jackal body (TUM format, header stamps)
  OUT.tf_static.txt every /tf_static transform (parent child tx ty tz qx qy qz qw)

Usage:
  python3 convert_utmm.py IN.bag OUT_d455.bag
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

typestore = get_typestore(Stores.ROS1_NOETIC)
Image = typestore.types['sensor_msgs/msg/Image']

COPY = {
    '/realsense/depth/image_rect_raw': '/camera/depth/image_rect_raw',
    '/realsense/depth/camera_info': '/camera/depth/camera_info',
    '/microstrain/imu/data': '/camera/imu',
}
COLOR_IN, COLOR_OUT = '/realsense/color/image_raw/compressed', '/camera/color/image_raw'
GT_TOPIC, TF_STATIC = '/vrpn_client_node/Jackal_Latest/pose', '/tf_static'


def stamp(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('input')
    ap.add_argument('output')
    args = ap.parse_args()
    src, dst = Path(args.input), Path(args.output)
    if dst.exists():
        dst.unlink()

    gt_rows, tf_rows, counts = [], {}, {}
    with Reader(src) as reader, Writer(dst) as writer:
        conns = {c.topic: c for c in reader.connections}
        if TF_STATIC in conns:   # tf2_msgs is not in the stock Noetic typestore
            typestore.register(get_types_from_msg(conns[TF_STATIC].msgdef.data,
                                                  conns[TF_STATIC].msgtype))
        missing = [t for t in list(COPY) + [COLOR_IN, GT_TOPIC] if t not in conns]
        if missing:
            raise SystemExit(f'missing topics in {src.name}: {missing}')
        out = {t: writer.add_connection(COPY[t], c.msgtype, msgdef=c.msgdef.data,
                                        md5sum=c.digest)
               for t, c in conns.items() if t in COPY}
        out[COLOR_IN] = writer.add_connection(COLOR_OUT, 'sensor_msgs/msg/Image',
                                              typestore=typestore)
        wanted = [c for c in reader.connections
                  if c.topic in COPY or c.topic in (COLOR_IN, GT_TOPIC, TF_STATIC)]
        for conn, t_bag, raw in reader.messages(connections=wanted):
            if conn.topic in COPY:
                writer.write(out[conn.topic], t_bag, raw)
            elif conn.topic == COLOR_IN:
                msg = typestore.deserialize_ros1(raw, conn.msgtype)
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                rgb = Image(header=msg.header, height=img.shape[0], width=img.shape[1],
                            encoding='bgr8', is_bigendian=0, step=img.shape[1] * 3,
                            data=np.ascontiguousarray(img).reshape(-1))
                writer.write(out[COLOR_IN], t_bag,
                             typestore.serialize_ros1(rgb, 'sensor_msgs/msg/Image'))
            elif conn.topic == GT_TOPIC:
                m = typestore.deserialize_ros1(raw, conn.msgtype)
                p, q = m.pose.position, m.pose.orientation
                gt_rows.append([stamp(m.header), p.x, p.y, p.z, q.x, q.y, q.z, q.w])
            else:
                for tf in typestore.deserialize_ros1(raw, conn.msgtype).transforms:
                    tr, q = tf.transform.translation, tf.transform.rotation
                    tf_rows[(tf.header.frame_id, tf.child_frame_id)] = (
                        tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)
            counts[conn.topic] = counts.get(conn.topic, 0) + 1

    gt = np.array(sorted(gt_rows))
    np.savetxt(dst.with_suffix('.gt.txt'), gt, fmt='%.9f',
               header='timestamp tx ty tz qx qy qz qw (Vicon Jackal_Latest)')
    with open(dst.with_suffix('.tf_static.txt'), 'w') as f:
        f.write('# parent child tx ty tz qx qy qz qw\n')
        for (parent, child), v in sorted(tf_rows.items()):
            f.write(f'{parent} {child} ' + ' '.join(f'{x:.9f}' for x in v) + '\n')
    print(f'{dst.name}: ' + ', '.join(f'{t}={n}' for t, n in sorted(counts.items())) +
          f'; gt {len(gt)} poses; tf_static {len(tf_rows)} transforms')


if __name__ == '__main__':
    main()
