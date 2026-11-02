#!/usr/bin/env python3
"""
Sensor-to-pose latency inside the benchmark container (ROS 1, Python 3).

Subscribes to the depth image topic and to the estimator's pose topic as raw messages (no deserialization beyond
the header), records the wall-clock arrival time of every depth frame and of every pose, and matches a pose to the
depth frame with the same header stamp (the pose is stamped with the depth frame it closes).  One CSV row per pose:
  stamp, t_img_wall, t_pose_wall, latency_s, backlog
where latency_s = t_pose_wall - t_img_wall (the time from the depth frame's arrival to the pose being published)
and backlog = depth frames received so far minus poses published so far (queue growth).

Usage (inside the container): python3 latency_logger.py OUT.csv [--depth /camera/depth/image_rect_raw] [--pose /aft_mapped_to_init]
"""
import argparse
import struct
import sys
import time

import rospy


def header_stamp(msg):
    # ROS 1 serialization of std_msgs/Header: uint32 seq, uint32 sec, uint32 nsec, string frame_id
    _, sec, nsec = struct.unpack_from('<III', msg._buff, 0)
    return sec + 1e-9 * nsec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out'); ap.add_argument('--depth', default='/camera/depth/image_rect_raw'); ap.add_argument('--pose', default='/aft_mapped_to_init')
    a = ap.parse_args()
    rospy.init_node('latency_logger', anonymous=True, disable_signals=True)
    arrivals, n_img, n_pose = {}, [0], [0]
    out = open(a.out, 'w'); out.write('stamp,t_img_wall,t_pose_wall,latency_s,backlog\n')

    def on_depth(msg):
        arrivals[round(header_stamp(msg), 4)] = time.time(); n_img[0] += 1

    def on_pose(msg):
        now = time.time(); n_pose[0] += 1
        s = header_stamp(msg); key = round(s, 4)
        t_img = arrivals.get(key)
        if t_img is None and arrivals:                     # nearest within 2 ms
            k = min(arrivals, key=lambda x: abs(x - key))
            t_img = arrivals[k] if abs(k - key) < 0.002 else None
        out.write(f'{s:.6f},{t_img if t_img else ""},{now:.6f},{now - t_img if t_img else ""},{n_img[0] - n_pose[0]}\n'); out.flush()

    rospy.Subscriber(a.depth, rospy.AnyMsg, on_depth, queue_size=100)
    rospy.Subscriber(a.pose, rospy.AnyMsg, on_pose, queue_size=100)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    out.close()


if __name__ == '__main__':
    main()
