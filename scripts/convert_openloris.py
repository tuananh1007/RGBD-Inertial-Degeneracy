#!/usr/bin/env python3
"""
Convert OpenLORIS-Scene bags to our RGB-D VIO pipeline format.

OpenLORIS-Scene provides real D435i data with:
  Topics:
    /d400/color/image_raw           (848x480, bgr8/rgb8, 30 Hz)
    /d400/aligned_depth_to_color/image_raw  (848x480, 16UC1, factor=1000)
    /d400/accel/sample              (sensor_msgs/Imu, 250 Hz)
    /d400/gyro/sample               (sensor_msgs/Imu, 400 Hz)
    /gt                             (geometry_msgs/PoseStamped, ground truth)

Our pipeline expects:
    /camera/color/image_raw         (640x480, bgr8, 30 Hz)
    /camera/depth/image_rect_raw    (640x480, 16UC1, factor=5000)
    /camera/depth/camera_info       (sensor_msgs/CameraInfo)
    /camera/imu                     (sensor_msgs/Imu, combined accel+gyro)

Conversion:
  1. Resize 848x480 → 640x480 (scale intrinsics accordingly)
  2. Rescale depth from factor=1000 to factor=5000 (multiply by 5)
  3. Merge separate accel/gyro into combined IMU at accel rate (250 Hz)
  4. Extract ground truth to TUM format file

Usage:
  python3 convert_openloris.py /path/to/office1-1.bag \
      --output office1-1_d455.bag --gt office1-1_gt.txt

Dependencies: rosbags, numpy, opencv-python
"""

import argparse
import os
import sys
import numpy as np

from rosbags.rosbag1 import Reader as Bag1Reader, Writer as Bag1Writer
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

typestore = get_typestore(Stores.ROS1_NOETIC)

# Register tf2_msgs which may not be in the default Noetic store
try:
    tf2_msg_text = 'geometry_msgs/TransformStamped[] transforms\n'
    typestore.register(get_types_from_msg(tf2_msg_text, 'tf2_msgs/msg/TFMessage'))
except Exception:
    pass  # already registered

deserialize_ros1 = typestore.deserialize_ros1
serialize_ros1 = typestore.serialize_ros1

import cv2


# OpenLORIS D435i intrinsics (from camera_info topic, typical values)
# Will be read from the bag's camera_info topic if available
OPENLORIS_INTRINSICS = dict(fx=616.0, fy=616.0, cx=424.0, cy=240.0,
                            w=848, h=480)
# Target resolution
TARGET_W, TARGET_H = 640, 480

# Depth scale conversion: OpenLORIS factor=1000, ours factor=5000
DEPTH_RESCALE = 5  # multiply pixel values by 5


def merge_accel_gyro(accel_msgs, gyro_msgs):
    """
    Merge separate accelerometer and gyroscope messages into combined IMU.

    Strategy: For each accel message (250 Hz), find the nearest gyro message
    (400 Hz) and combine them. This gives us ~250 Hz combined IMU.
    """
    if not accel_msgs or not gyro_msgs:
        return []

    gyro_times = np.array([t for t, _ in gyro_msgs])
    combined = []

    for accel_t, accel_msg in accel_msgs:
        # Find nearest gyro
        idx = np.searchsorted(gyro_times, accel_t)
        if idx == 0:
            gyro_msg = gyro_msgs[0][1]
        elif idx >= len(gyro_msgs):
            gyro_msg = gyro_msgs[-1][1]
        else:
            # Pick the closer one
            if abs(gyro_times[idx] - accel_t) < abs(gyro_times[idx-1] - accel_t):
                gyro_msg = gyro_msgs[idx][1]
            else:
                gyro_msg = gyro_msgs[idx-1][1]

        combined.append((accel_t, accel_msg, gyro_msg))

    return combined


def stream_events(input_bag, color_topic, depth_topic, imu_events):
    """Colour and depth events read again from the bag in time order, with the (time-sorted) IMU events merged in."""
    kinds = {color_topic: 'color', depth_topic: 'depth'}
    k = 0
    with Bag1Reader(input_bag) as reader:
        conns = [c for c in reader.connections if c.topic in kinds]
        for conn, timestamp, rawdata in reader.messages(connections=conns):
            t_sec = timestamp * 1e-9
            while k < len(imu_events) and imu_events[k][1] < t_sec:
                yield imu_events[k]
                k += 1
            yield (kinds[conn.topic], t_sec, timestamp, rawdata, conn.msgtype)
    yield from imu_events[k:]


def convert_bag(input_bag, output_bag, gt_file, max_frames=None, stream=False):
    """Convert a single OpenLORIS bag to our format."""

    # Phase 1: Read all messages
    print(f"Reading {input_bag}...")

    color_msgs = []
    depth_msgs = []
    accel_msgs = []
    gyro_msgs = []
    gt_msgs = []
    camera_info = None

    # Topic mapping
    COLOR_TOPIC = '/d400/color/image_raw'
    DEPTH_TOPIC = '/d400/aligned_depth_to_color/image_raw'
    ACCEL_TOPIC = '/d400/accel/sample'
    GYRO_TOPIC = '/d400/gyro/sample'
    GT_TOPIC = '/gt'
    COLOR_INFO_TOPIC = '/d400/color/camera_info'
    DEPTH_INFO_TOPIC = '/d400/aligned_depth_to_color/camera_info'

    topics_of_interest = {COLOR_TOPIC, DEPTH_TOPIC, ACCEL_TOPIC, GYRO_TOPIC,
                          GT_TOPIC, COLOR_INFO_TOPIC, DEPTH_INFO_TOPIC}

    with Bag1Reader(input_bag) as reader:
        # Print available topics
        print("Available topics:")
        for conn in reader.connections:
            print(f"  {conn.topic}: {conn.msgtype}")

        msg_count = 0
        for conn, timestamp, rawdata in reader.messages():
            if conn.topic not in topics_of_interest:
                continue

            t_sec = timestamp * 1e-9  # nanoseconds to seconds
            if stream and conn.topic in (COLOR_TOPIC, DEPTH_TOPIC):
                continue  # --stream: images are read again while writing

            if conn.topic == COLOR_TOPIC:
                color_msgs.append((t_sec, timestamp, rawdata, conn.msgtype))
                if max_frames and len(color_msgs) >= max_frames:
                    break
            elif conn.topic == DEPTH_TOPIC:
                depth_msgs.append((t_sec, timestamp, rawdata, conn.msgtype))
            elif conn.topic == ACCEL_TOPIC:
                accel_msgs.append((t_sec, timestamp, rawdata, conn.msgtype))
            elif conn.topic == GYRO_TOPIC:
                gyro_msgs.append((t_sec, timestamp, rawdata, conn.msgtype))
            elif conn.topic == GT_TOPIC:
                gt_msgs.append((t_sec, timestamp, rawdata, conn.msgtype))
            elif conn.topic in (COLOR_INFO_TOPIC, DEPTH_INFO_TOPIC):
                if camera_info is None:
                    camera_info = (rawdata, conn.msgtype)

            msg_count += 1
            if msg_count % 5000 == 0:
                print(f"  Read {msg_count} messages "
                      f"(color={len(color_msgs)}, depth={len(depth_msgs)}, "
                      f"accel={len(accel_msgs)}, gyro={len(gyro_msgs)}, "
                      f"gt={len(gt_msgs)})")

    print(f"Total: color={len(color_msgs)}, depth={len(depth_msgs)}, "
          f"accel={len(accel_msgs)}, gyro={len(gyro_msgs)}, gt={len(gt_msgs)}")

    # Extract intrinsics from camera_info
    intr = OPENLORIS_INTRINSICS.copy()
    if camera_info is not None:
        try:
            ci_msg = deserialize_ros1(camera_info[0], camera_info[1])
            K = ci_msg.K
            intr['fx'] = K[0]
            intr['fy'] = K[4]
            intr['cx'] = K[2]
            intr['cy'] = K[5]
            intr['w'] = ci_msg.width
            intr['h'] = ci_msg.height
            print(f"Camera intrinsics from bag: fx={intr['fx']:.1f}, fy={intr['fy']:.1f}, "
                  f"cx={intr['cx']:.1f}, cy={intr['cy']:.1f}, {intr['w']}x{intr['h']}")
        except Exception as e:
            print(f"Warning: Could not parse camera_info: {e}")

    # Compute scaled intrinsics for 640x480 target
    sx = TARGET_W / intr['w']
    sy = TARGET_H / intr['h']
    scaled_intr = dict(
        fx=intr['fx'] * sx,
        fy=intr['fy'] * sy,
        cx=intr['cx'] * sx,
        cy=intr['cy'] * sy,
        w=TARGET_W,
        h=TARGET_H
    )
    print(f"Scaled intrinsics: fx={scaled_intr['fx']:.1f}, fy={scaled_intr['fy']:.1f}, "
          f"cx={scaled_intr['cx']:.1f}, cy={scaled_intr['cy']:.1f}")

    # Phase 2: Write ground truth to TUM format
    # GT is tf2_msgs/TFMessage with transforms[0] containing the pose
    if gt_msgs and gt_file:
        print(f"Writing ground truth to {gt_file}...")
        gt_count = 0
        with open(gt_file, 'w') as f:
            f.write("# ground truth trajectory (TUM format)\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")
            for t_sec, timestamp, rawdata, msgtype in gt_msgs:
                try:
                    msg = deserialize_ros1(rawdata, msgtype)
                    if hasattr(msg, 'transforms') and msg.transforms:
                        # tf2_msgs/TFMessage format
                        t = msg.transforms[0]
                        tr = t.transform.translation
                        ro = t.transform.rotation
                        f.write(f"{t_sec:.6f} {tr.x:.6f} {tr.y:.6f} {tr.z:.6f} "
                                f"{ro.x:.6f} {ro.y:.6f} {ro.z:.6f} {ro.w:.6f}\n")
                    elif hasattr(msg, 'pose'):
                        # geometry_msgs/PoseStamped format
                        p = msg.pose.position
                        q = msg.pose.orientation
                        f.write(f"{t_sec:.6f} {p.x:.6f} {p.y:.6f} {p.z:.6f} "
                                f"{q.x:.6f} {q.y:.6f} {q.z:.6f} {q.w:.6f}\n")
                    gt_count += 1
                except Exception as e:
                    print(f"  Warning: could not parse GT at t={t_sec}: {e}")
        print(f"  Wrote {gt_count} GT poses")

    # Phase 3: Merge accel + gyro
    print("Merging accel + gyro into combined IMU...")
    accel_parsed = []
    for t_sec, timestamp, rawdata, msgtype in accel_msgs:
        try:
            msg = deserialize_ros1(rawdata, msgtype)
            accel_parsed.append((t_sec, msg))
        except:
            pass

    gyro_parsed = []
    for t_sec, timestamp, rawdata, msgtype in gyro_msgs:
        try:
            msg = deserialize_ros1(rawdata, msgtype)
            gyro_parsed.append((t_sec, msg))
        except:
            pass

    combined_imu = merge_accel_gyro(accel_parsed, gyro_parsed)
    print(f"  Combined IMU: {len(combined_imu)} messages at ~250 Hz")

    # Phase 4: Write output bag
    print(f"Writing {output_bag}...")

    # Get type definitions
    Time       = typestore.types['builtin_interfaces/msg/Time']
    Header     = typestore.types['std_msgs/msg/Header']
    Image      = typestore.types['sensor_msgs/msg/Image']
    CameraInfo = typestore.types['sensor_msgs/msg/CameraInfo']
    Imu        = typestore.types['sensor_msgs/msg/Imu']
    Vector3    = typestore.types['geometry_msgs/msg/Vector3']
    Quaternion = typestore.types['geometry_msgs/msg/Quaternion']
    ROI        = typestore.types['sensor_msgs/msg/RegionOfInterest']

    OUT_COLOR = '/camera/color/image_raw'
    OUT_DEPTH = '/camera/depth/image_rect_raw'
    OUT_DEPTH_INFO = '/camera/depth/camera_info'
    OUT_IMU = '/camera/imu'

    def make_time(t_sec):
        s = int(t_sec)
        ns = int((t_sec - s) * 1e9)
        return Time(sec=s, nanosec=ns)

    seq_counter = [0]

    def make_header(t_sec, frame_id='camera'):
        seq_counter[0] += 1
        return Header(seq=seq_counter[0], stamp=make_time(t_sec), frame_id=frame_id)

    def make_camera_info(t_sec, intr):
        fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
        w, h = intr['w'], intr['h']
        K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1], dtype=np.float64)
        P = np.array([fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0], dtype=np.float64)
        R = np.eye(3, dtype=np.float64).flatten()
        D = np.zeros(5, dtype=np.float64)
        return CameraInfo(
            header=make_header(t_sec),
            height=h, width=w,
            distortion_model='plumb_bob',
            D=D, K=K, R=R, P=P,
            binning_x=0, binning_y=0,
            roi=ROI(x_offset=0, y_offset=0, height=0, width=0,
                    do_rectify=False)
        )

    written = 0
    with Bag1Writer(output_bag) as writer:
        # Create connections
        conn_color = writer.add_connection(OUT_COLOR, 'sensor_msgs/msg/Image')
        conn_depth = writer.add_connection(OUT_DEPTH, 'sensor_msgs/msg/Image')
        conn_depth_info = writer.add_connection(OUT_DEPTH_INFO, 'sensor_msgs/msg/CameraInfo')
        conn_imu = writer.add_connection(OUT_IMU, 'sensor_msgs/msg/Imu')

        # Collect all events and sort by time
        events = []

        # Color images
        for t_sec, timestamp, rawdata, msgtype in color_msgs:
            events.append(('color', t_sec, timestamp, rawdata, msgtype))

        # Depth images
        for t_sec, timestamp, rawdata, msgtype in depth_msgs:
            events.append(('depth', t_sec, timestamp, rawdata, msgtype))

        # IMU
        for t_sec, accel_msg, gyro_msg in combined_imu:
            events.append(('imu', t_sec, None, accel_msg, gyro_msg))

        # Sort by time
        events.sort(key=lambda x: x[1])
        if stream:
            events = stream_events(input_bag, COLOR_TOPIC, DEPTH_TOPIC, events)

        for evt in events:
            kind = evt[0]
            t_sec = evt[1]
            t_ns = int(t_sec * 1e9)

            if kind == 'color':
                rawdata, msgtype = evt[3], evt[4]
                try:
                    msg = deserialize_ros1(rawdata, msgtype)
                    # msg.data is already np.ndarray (uint8)
                    img = np.array(msg.data, dtype=np.uint8).reshape(
                        msg.height, msg.width, -1)
                    # Resize
                    if msg.width != TARGET_W or msg.height != TARGET_H:
                        img = cv2.resize(img, (TARGET_W, TARGET_H),
                                        interpolation=cv2.INTER_LINEAR)
                    # Ensure bgr8
                    encoding = msg.encoding
                    if encoding in ('rgb8', 'RGB8'):
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        encoding = 'bgr8'
                    elif encoding in ('bgr8', 'BGR8'):
                        encoding = 'bgr8'

                    out_msg = Image(
                        header=make_header(t_sec),
                        height=TARGET_H, width=TARGET_W,
                        encoding=encoding,
                        is_bigendian=0,
                        step=TARGET_W * 3,
                        data=np.ascontiguousarray(img).flatten()
                    )
                    writer.write(conn_color, t_ns, serialize_ros1(out_msg, 'sensor_msgs/msg/Image'))
                    written += 1
                except Exception as e:
                    print(f"  Warning: could not process color at t={t_sec:.3f}: {e}")

            elif kind == 'depth':
                rawdata, msgtype = evt[3], evt[4]
                try:
                    msg = deserialize_ros1(rawdata, msgtype)
                    # msg.data is np.ndarray; interpret as uint16
                    depth = np.frombuffer(msg.data.tobytes(), dtype=np.uint16).reshape(
                        msg.height, msg.width)
                    # Resize with nearest-neighbor (preserve depth values)
                    if msg.width != TARGET_W or msg.height != TARGET_H:
                        depth = cv2.resize(depth, (TARGET_W, TARGET_H),
                                          interpolation=cv2.INTER_NEAREST)
                    # Rescale: factor 1000 → factor 5000
                    depth = (depth.astype(np.uint32) * DEPTH_RESCALE).clip(0, 65535).astype(np.uint16)

                    # Image.data expects np.ndarray[uint8], so view depth as uint8
                    out_msg = Image(
                        header=make_header(t_sec),
                        height=TARGET_H, width=TARGET_W,
                        encoding='16UC1',
                        is_bigendian=0,
                        step=TARGET_W * 2,
                        data=depth.view(np.uint8).flatten()
                    )
                    writer.write(conn_depth, t_ns, serialize_ros1(out_msg, 'sensor_msgs/msg/Image'))

                    # Also write camera_info
                    ci_msg = make_camera_info(t_sec, scaled_intr)
                    writer.write(conn_depth_info, t_ns,
                                serialize_ros1(ci_msg, 'sensor_msgs/msg/CameraInfo'))
                    written += 1
                except Exception as e:
                    print(f"  Warning: could not process depth at t={t_sec:.3f}: {e}")

            elif kind == 'imu':
                accel_msg, gyro_msg = evt[3], evt[4]
                try:
                    imu_msg = Imu(
                        header=make_header(t_sec, 'imu'),
                        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                        orientation_covariance=np.full(9, -1.0, dtype=np.float64),
                        angular_velocity=Vector3(
                            x=gyro_msg.angular_velocity.x,
                            y=gyro_msg.angular_velocity.y,
                            z=gyro_msg.angular_velocity.z
                        ),
                        angular_velocity_covariance=np.zeros(9, dtype=np.float64),
                        linear_acceleration=Vector3(
                            x=accel_msg.linear_acceleration.x,
                            y=accel_msg.linear_acceleration.y,
                            z=accel_msg.linear_acceleration.z
                        ),
                        linear_acceleration_covariance=np.zeros(9, dtype=np.float64)
                    )
                    writer.write(conn_imu, t_ns, serialize_ros1(imu_msg, 'sensor_msgs/msg/Imu'))
                    written += 1
                except Exception as e:
                    print(f"  Warning: could not write IMU at t={t_sec:.3f}: {e}")

            if written % 2000 == 0 and written > 0:
                print(f"  Written {written} messages...")

    print(f"Done. Wrote {written} messages to {output_bag}")
    return scaled_intr


def main():
    parser = argparse.ArgumentParser(
        description='Convert OpenLORIS-Scene bags to our RGB-D VIO format')
    parser.add_argument('input', help='Input OpenLORIS bag file')
    parser.add_argument('--output', '-o', required=True,
                        help='Output bag file')
    parser.add_argument('--gt', '-g', default=None,
                        help='Output ground truth TUM file')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Max color frames to convert (for testing)')
    parser.add_argument('--stream', action='store_true',
                        help='Do not hold the images in memory (bags larger than RAM)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input bag not found: {args.input}")
        sys.exit(1)

    gt_file = args.gt
    if gt_file is None:
        gt_file = os.path.splitext(args.output)[0] + '_gt.txt'

    scaled_intr = convert_bag(args.input, args.output, gt_file, args.max_frames, args.stream)

    print(f"\nScaled camera intrinsics for config file:")
    print(f"  fx={scaled_intr['fx']:.2f}, fy={scaled_intr['fy']:.2f}, "
          f"cx={scaled_intr['cx']:.2f}, cy={scaled_intr['cy']:.2f}")
    print(f"  resolution: [{scaled_intr['w']}, {scaled_intr['h']}]")
    print(f"  depth_scale: 0.0002  (factor=5000)")


if __name__ == '__main__':
    main()
