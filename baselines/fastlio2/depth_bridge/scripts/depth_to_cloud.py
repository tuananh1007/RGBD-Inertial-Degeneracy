#!/usr/bin/env python3
"""Depth image -> sensor_msgs/PointCloud2 (x, y, z, rgb) in the depth optical frame, for FAST-LIVO2's L515 input type.
Every pixel with min_depth <= z <= max_depth is back-projected with the K of the depth camera_info (the
depth images are rectified); rgb is left zero (FAST-LIVO2 colours its map from the camera image).  The header, and so
the scan time, is the depth image's.  Parameters, global, from the FAST-LIVO2 config:
depth_bridge/{depth_topic, info_topic, cloud_topic, depth_scale, min_depth, max_depth}; fourth_field names the zero
fourth field (default rgb; intensity for FAST-LIO2's point-cloud input type 4)."""
import numpy as np
import rospy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField


def main():
    rospy.init_node('depth_to_cloud')
    g = lambda k, d: rospy.get_param('depth_bridge/' + k, d)
    scale, zmin, zmax = g('depth_scale', 0.001), g('min_depth', 0.1), g('max_depth', 10.0)
    fields = [PointField(n, 4 * i, PointField.FLOAT32, 1) for i, n in enumerate(('x', 'y', 'z', g('fourth_field', 'rgb')))]
    info = rospy.wait_for_message(g('info_topic', '/camera/depth/camera_info'), CameraInfo)
    fx, fy, cx, cy = info.K[0], info.K[4], info.K[2], info.K[5]
    pub = rospy.Publisher(g('cloud_topic', '/depth_cloud'), PointCloud2, queue_size=100)
    rays = {}

    def cb(m):
        if (m.height, m.width) not in rays:
            v, u = np.mgrid[0:m.height, 0:m.width]
            rays[(m.height, m.width)] = ((u - cx) / fx).astype(np.float32).ravel(), ((v - cy) / fy).astype(np.float32).ravel()
        a, b = rays[(m.height, m.width)]
        d = np.frombuffer(m.data, dtype='>u2' if m.is_bigendian else '<u2').reshape(m.height, m.step // 2)[:, :m.width]
        d = d.astype(np.float32).ravel() * scale
        ok = (d >= zmin) & (d <= zmax)
        z = d[ok]
        pts = np.zeros((len(z), 4), np.float32)
        pts[:, 0], pts[:, 1], pts[:, 2] = a[ok] * z, b[ok] * z, z
        pub.publish(PointCloud2(header=m.header, height=1, width=len(z), fields=fields, is_bigendian=False,
                                point_step=16, row_step=16 * len(z), data=pts.tobytes(), is_dense=True))

    rospy.Subscriber(g('depth_topic', '/camera/depth/image_rect_raw'), Image, cb, queue_size=100, buff_size=2 ** 26)
    rospy.spin()


if __name__ == '__main__':
    main()
