#!/usr/bin/env python3
"""
Copy of a <seq>_d455.bag with a lower IMU rate: every STRIDE-th message of each sensor_msgs/Imu topic, all other
messages unchanged, at their original receive times.  For the IMU-rate test of Supplementary S10 (does the per-sample h^2 noise form depend on the IMU
rate?): the per-sample variances of e1_h2match_*.yaml were matched to our densities at the recorded rate, so on
a decimated bag they no longer correspond to them, whereas the continuous-time densities of Ours need no change.

Usage (in an image with ROS Noetic): decimate_imu.py IN.bag OUT.bag [--stride 2]
"""
import argparse

import rosbag


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('inp'); ap.add_argument('out'); ap.add_argument('--stride', type=int, default=2)
    a = ap.parse_args()
    with rosbag.Bag(a.inp) as bag:
        topics = bag.get_type_and_topic_info().topics
        imu = {t for t, info in topics.items() if info.msg_type == 'sensor_msgs/Imu'}
        if not imu:
            raise SystemExit('no sensor_msgs/Imu topic in ' + a.inp)
        seen = dict.fromkeys(imu, 0); n = dict.fromkeys(topics, 0)
        with rosbag.Bag(a.out, 'w') as out:
            for topic, msg, t in bag.read_messages():
                if topic in imu:
                    seen[topic] += 1
                    if (seen[topic] - 1) % a.stride:
                        continue
                out.write(topic, msg, t); n[topic] += 1
    for t in sorted(n):
        print(f'{t:36s} {topics[t].message_count:6d} -> {n[t]:6d}')


if __name__ == '__main__':
    main()
