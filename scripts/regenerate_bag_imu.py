#!/usr/bin/env python3
"""
Regenerate the synthetic /camera/imu topic of an existing converted bag.

Every other message is copied byte-for-byte with its original connection
header (message definition and MD5) and bag timestamp, so the depth/RGB
input is unchanged.  IMU messages keep their stamps and are recomputed from
the companion ground truth with imu_synthesis.synthesise_imu().  The output
is verified against the input before the script reports success, and one
provenance row per bag is appended to <output dir>/imu_regeneration_manifest.csv.

Usage:
  python3 regenerate_bag_imu.py IN.bag OUT.bag [--gt IN.gt.txt] [--seed N]
"""
import argparse
import csv
import hashlib
import shutil
import zlib
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation, Slerp

import imu_synthesis
from imu_synthesis import synthesise_imu

IMU_TOPIC = '/camera/imu'
typestore = get_typestore(Stores.ROS1_NOETIC)
Imu = typestore.types['sensor_msgs/msg/Imu']
Vector3 = typestore.types['geometry_msgs/msg/Vector3']


def gyro_scale(gt, imu):
    """Median ratio of gyro-integrated to GT rotation angle over 1-s windows
    with > 0.1 rad of GT rotation (0.5 for the original defect, ~1 if correct)."""
    rot = Slerp(gt[:, 0], Rotation.from_quat(gt[:, 4:8]))
    ratios = []
    for t0 in np.arange(imu[0, 0] + 0.5, imu[-1, 0] - 1.5, 1.0):
        seg = imu[(imu[:, 0] >= t0) & (imu[:, 0] < t0 + 1.0)]
        if len(seg) < 2:
            continue
        R = Rotation.identity()
        for k in range(1, len(seg)):
            R = R * Rotation.from_rotvec(0.5 * (seg[k - 1, 1:4] + seg[k, 1:4])
                                         * (seg[k, 0] - seg[k - 1, 0]))
        ang_gt = (rot([seg[0, 0]])[0].inv() * rot([seg[-1, 0]])[0]).magnitude()
        if ang_gt > 0.1:
            ratios.append(R.magnitude() / ang_gt)
    return float(np.median(ratios)) if ratios else float('nan')


def payload_digest(path):
    """Per-topic message counts, plus one hash over all non-IMU payloads and stamps."""
    counts, h, imu_stamps = {}, hashlib.sha256(), []
    with Reader(path) as reader:
        for conn, stamp, raw in reader.messages():
            counts[conn.topic] = counts.get(conn.topic, 0) + 1
            if conn.topic == IMU_TOPIC:
                imu_stamps.append(stamp)
            else:
                h.update(conn.topic.encode()); h.update(stamp.to_bytes(8, 'little')); h.update(raw)
        headers = {c.topic: (c.msgtype, c.digest) for c in reader.connections}
    return counts, h.hexdigest(), imu_stamps, headers


def generator_hash():
    """Short SHA-256 of imu_synthesis.py, identifying the generator version."""
    return hashlib.sha256(Path(imu_synthesis.__file__).read_bytes()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('input')
    ap.add_argument('output')
    ap.add_argument('--gt', help='ground truth (default: INPUT with .gt.txt suffix)')
    ap.add_argument('--seed', type=int, help='noise seed (default: CRC32 of output stem)')
    ap.add_argument('--noise-scale', type=float, default=1.0)
    ap.add_argument('--cutoff-hz', type=float, default=imu_synthesis.CUTOFF_HZ)
    args = ap.parse_args()

    src, dst = Path(args.input), Path(args.output)
    gt_path = Path(args.gt) if args.gt else src.with_suffix('.gt.txt')
    seed = args.seed if args.seed is not None else zlib.crc32(dst.stem.encode())
    if dst.resolve() == src.resolve():
        raise SystemExit('refusing to overwrite the input bag')
    gt = np.loadtxt(gt_path)

    with Reader(src) as reader:
        imu_conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
        headers = []
        for _, stamp, raw in reader.messages(connections=imu_conns):
            headers.append(typestore.deserialize_ros1(raw, 'sensor_msgs/msg/Imu').header)
        stamps = np.array([h.stamp.sec + h.stamp.nanosec * 1e-9 for h in headers])
        imu = synthesise_imu(gt, noise_scale=args.noise_scale, seed=seed,
                             cutoff_hz=args.cutoff_hz, imu_t=stamps)

        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        with Writer(dst) as writer:
            out_conns = {c.id: writer.add_connection(
                c.topic, c.msgtype, msgdef=c.msgdef.data, md5sum=c.digest,
                callerid=c.ext.callerid, latching=c.ext.latching) for c in reader.connections}
            k = 0
            for conn, stamp, raw in reader.messages():
                if conn.topic == IMU_TOPIC:
                    row = imu[k]
                    msg = Imu(header=headers[k],
                              orientation=typestore.types['geometry_msgs/msg/Quaternion'](
                                  x=0.0, y=0.0, z=0.0, w=1.0),
                              orientation_covariance=np.array([-1.0] + [0.0] * 8),
                              angular_velocity=Vector3(x=row[1], y=row[2], z=row[3]),
                              angular_velocity_covariance=np.zeros(9),
                              linear_acceleration=Vector3(x=row[4], y=row[5], z=row[6]),
                              linear_acceleration_covariance=np.zeros(9))
                    raw = typestore.serialize_ros1(msg, 'sensor_msgs/msg/Imu')
                    k += 1
                writer.write(out_conns[conn.id], stamp, raw)

    # ── Verify: same connections, counts, non-IMU payloads and IMU stamps ──
    c_in, h_in, s_in, hd_in = payload_digest(src)
    c_out, h_out, s_out, hd_out = payload_digest(dst)
    assert c_in == c_out, f'message counts differ: {c_in} vs {c_out}'
    assert h_in == h_out, 'non-IMU payloads or stamps differ'
    assert s_in == s_out, 'IMU bag timestamps differ'
    assert hd_in == hd_out, f'connection headers differ: {hd_in} vs {hd_out}'

    shutil.copyfile(gt_path, dst.with_suffix('.gt.txt'))
    gaps = np.diff(np.sort(gt[:, 0]))
    row = dict(bag=dst.name, source=str(src), generator_sha256=generator_hash(), seed=seed,
               noise_scale=args.noise_scale, cutoff_hz=args.cutoff_hz, imu_msgs=len(stamps),
               gyro_scale_before=round(gyro_scale(gt, _read_imu(src)), 4),
               gyro_scale_after=round(gyro_scale(gt, imu), 4),
               gt_max_gap_s=round(float(gaps.max()), 3),
               gt_gaps_over_0p05s=int((gaps > 0.05).sum()))
    manifest = dst.parent / 'imu_regeneration_manifest.csv'
    new_file = not manifest.exists()
    with open(manifest, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(', '.join(f'{k}={v}' for k, v in row.items()))


def _read_imu(path):
    rows = []
    with Reader(path) as reader:
        conns = [c for c in reader.connections if c.topic == IMU_TOPIC]
        for _, _, raw in reader.messages(connections=conns):
            m = typestore.deserialize_ros1(raw, 'sensor_msgs/msg/Imu')
            rows.append([m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                         m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
    return np.array(rows)


if __name__ == '__main__':
    main()
