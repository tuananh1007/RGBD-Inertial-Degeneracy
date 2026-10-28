#!/usr/bin/env python3
"""
convert_fastlivo2_bag.py
UCLA UAV Pest Control — RGB-D VIO

Converts a FAST-LIVO2 bag (Livox LiDAR + left-camera + IMU) into a
synthetic D455-format bag (depth-image + colour + IMU) that can be
played directly into our RGBDMapper.

Pipeline
--------
  /livox/lidar  (livox_ros_driver/CustomMsg)
       │
       ▼  project through LiDAR→camera extrinsics + camera intrinsics
  /camera/depth/image_rect_raw  (16UC1, mm)
  /camera/depth/camera_info
       │
  /left_camera/image[/compressed]  ──►  /camera/color/image_raw
  /livox/imu                        ──►  /camera/imu

Usage
-----
  python3 convert_fastlivo2_bag.py  \\
      --input  Bright_Screen_Wall.bag  \\
      --output Bright_Screen_Wall_d455.bag  \\
      --seq    Bright_Screen_Wall

Calibration presets (from calibration.yaml shipped with FAST-LIVO2):
  Bright_Screen_Wall / Retail_Street / CBD_Building_01 → "retail"
  HKU_* / CBD_Building_02/03                           → "hku"

  Override with --rcl / --pcl / --fx / --fy / --cx / --cy /
                --width / --height / --scale

The benchmark bags of the paper (Retail_Street, CBD_Building_01: retail; CBD_Building_02: hku) are the sparse projection
with a 40 m cap:  --seq PRESET --no-dense --max-depth 40
"""

import argparse, struct, time
import numpy as np
import cv2
from pathlib import Path
from rosbags.rosbag1 import Reader, Writer
from rosbags.typesys import get_typestore, Stores
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# Calibration presets
# ---------------------------------------------------------------------------
PRESETS = {
    "retail": dict(
        # Livox→IMU  (identity rotation, small offset)
        extR = np.eye(3),
        extT = np.array([0.04165, 0.02326, -0.0284]),
        # Colour-camera → LiDAR  (from calibration.yaml)
        Rcl  = np.array([
             0.00610193, -0.999863, -0.0154172,
            -0.00615449,  0.0153796, -0.999863,
             0.999962,    0.00619598, -0.0060598]).reshape(3,3),
        Pcl  = np.array([0.0194384, 0.104689, -0.0251952]),
        # Camera intrinsics (at 0.5 scale → effective 640×512)
        cam_w=640, cam_h=512,
        fx=1293.56944*0.5, fy=1293.3155*0.5,
        cx=626.91359*0.5,  cy=522.799224*0.5,
    ),
    "hku": dict(
        extR = np.eye(3),
        extT = np.array([0.04165, 0.02326, -0.0284]),
        Rcl  = np.array([
            -0.00200, -0.99975, -0.02211,
            -0.00366,  0.02212, -0.99975,
             0.99999, -0.00192, -0.00371]).reshape(3,3),
        Pcl  = np.array([0.00260, 0.05057, -0.00587]),
        cam_w=640, cam_h=512,
        fx=1176.2874292149932*0.5, fy=1176.21585445307*0.5,
        cx=592.1187382755453*0.5,  cy=509.0864309628322*0.5,
    ),
    "hit": dict(
        extR = np.eye(3),
        extT = np.array([0.04165, 0.02326, -0.0284]),
        Rcl  = np.array([
            -0.0036250, -0.9998907, -0.0143360,
             0.0075568,  0.0143083, -0.9998690,
             0.9999649, -0.0037329,  0.0075041]).reshape(3,3),
        Pcl  = np.array([0.00549469, 0.0712101, 0.0322054]),
        cam_w=640, cam_h=512,
        fx=1311.89517127580*0.5, fy=1311.36488586115*0.5,
        cx=656.523841857393*0.5, cy=504.136322840350*0.5,
    ),
    "red_sculpture": dict(
        extR = np.eye(3),
        extT = np.array([0.04165, 0.02326, -0.0284]),
        Rcl  = np.array([
            -0.00668, -0.99965, -0.02543,
            -0.01151,  0.02550, -0.99961,
             0.99991, -0.00638, -0.01168]).reshape(3,3),
        Pcl  = np.array([-0.00077, 0.04809, -0.00133]),
        cam_w=640, cam_h=512,
        fx=1294.7265716372897*0.5, fy=1294.8678078910468*0.5,
        cx=626.663267153558*0.5,   cy=531.0334324363173*0.5,
    ),
}

# ---------------------------------------------------------------------------
# Livox CustomMsg binary parser (pure Python, no livox_ros_driver needed)
# ---------------------------------------------------------------------------
POINT_DTYPE = np.dtype([
    ('offset_time', '<u4'),
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('reflectivity', 'u1'), ('tag', 'u1'), ('line', 'u1'),
])
POINT_BYTES = 19   # 4+4+4+4+1+1+1

def parse_livox(data: bytes):
    """Return (stamp_sec, stamp_nsec, Nx3 float32 xyz array in LiDAR frame)."""
    pos = 0
    _seq             = struct.unpack_from('<I', data, pos)[0]; pos += 4
    sec              = struct.unpack_from('<I', data, pos)[0]; pos += 4
    nsec             = struct.unpack_from('<I', data, pos)[0]; pos += 4
    fid_len          = struct.unpack_from('<I', data, pos)[0]; pos += 4 + fid_len
    _timebase        = struct.unpack_from('<Q', data, pos)[0]; pos += 8
    point_num        = struct.unpack_from('<I', data, pos)[0]; pos += 4
    pos += 4         # lidar_id (1) + rsvd (3)
    _arr_len         = struct.unpack_from('<I', data, pos)[0]; pos += 4  # array length prefix
    pts = np.frombuffer(data[pos:pos + point_num * POINT_BYTES], dtype=POINT_DTYPE)
    xyz = np.column_stack([pts['x'], pts['y'], pts['z']]).astype(np.float32)
    return sec, nsec, xyz

# ---------------------------------------------------------------------------
# IMU raw parser
# ---------------------------------------------------------------------------
def parse_imu_raw(data: bytes, store, conn):
    """Deserialise sensor_msgs/Imu via rosbags store."""
    return store.deserialize_ros1(data, conn.msgtype)

# ---------------------------------------------------------------------------
# Projection: LiDAR points → depth image
# ---------------------------------------------------------------------------
def project_lidar_to_depth(xyz_lidar, Rcl, Pcl, fx, fy, cx, cy, W, H,
                            min_depth=0.3, max_depth=15.0, dense=True):
    """
    Project Livox points in LiDAR frame to a 16-bit depth image.

    Coordinate transform:
        p_cam = Rcl @ p_lidar + Pcl

    When dense=True (default), fills sparse projected pixels via nearest-
    neighbour depth completion so the resulting depth image resembles a real
    D455 and provides enough point density for voxel plane fitting.
    Without completion the Livox scan fills only ~4 % of pixels, giving
    too few points per 0.3 m voxel for ICP plane fitting to succeed.

    Returns uint16 depth image (millimetres, 0 = invalid).
    """
    # Filter zero points (invalid Livox returns)
    valid = (xyz_lidar[:,0]!=0)|(xyz_lidar[:,1]!=0)|(xyz_lidar[:,2]!=0)
    pts   = xyz_lidar[valid]
    if len(pts) == 0:
        return np.zeros((H, W), dtype=np.uint16)

    # Transform to camera frame:  p_cam = Rcl @ p_lidar + Pcl
    # Rcl rotates vectors from LiDAR frame into camera frame (Rci = Rcl * Rli)
    p_cam = (Rcl @ pts.T).T + Pcl   # (N,3)

    # Keep points in front of camera
    in_front = p_cam[:, 2] > 0
    p_cam    = p_cam[in_front]

    depth = p_cam[:, 2]
    range_ok = (depth >= min_depth) & (depth <= max_depth)
    p_cam = p_cam[range_ok]
    depth = depth[range_ok]

    # Project to image plane
    u = (fx * p_cam[:, 0] / p_cam[:, 2] + cx).astype(np.int32)
    v = (fy * p_cam[:, 1] / p_cam[:, 2] + cy).astype(np.int32)

    in_frame = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, depth = u[in_frame], v[in_frame], depth[in_frame]

    depth_img = np.zeros((H, W), dtype=np.uint16)
    # For overlapping pixels keep the nearest point
    order = np.argsort(-depth)   # process far→near so near overwrites
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
    depth_img[v[order], u[order]] = depth_mm[order]

    if dense and np.count_nonzero(depth_img) > 0:
        # Nearest-neighbour depth completion:  fill every empty pixel with
        # the depth value of the spatially closest projected Livox point.
        # This raises the ~4 % fill rate to ~100 %, giving the depth processor
        # enough points per 0.3 m voxel for ICP plane fitting to work.
        # Quality is Voronoi-like: accurate on surfaces, approximate at edges
        # (depth discontinuities are handled downstream by flying-pixel rejection).
        mask_empty = depth_img == 0
        inds = distance_transform_edt(mask_empty, return_distances=False,
                                      return_indices=True)
        depth_img = depth_img[inds[0], inds[1]]

    return depth_img

# ---------------------------------------------------------------------------
# Serialise depth image as sensor_msgs/Image (16UC1)
# ---------------------------------------------------------------------------
def make_image_msg(img_np, sec, nsec, frame_id, encoding):
    """Pack a numpy image into a rosbags Image message."""
    H, W = img_np.shape[:2]
    if encoding == '16UC1':
        raw = img_np.astype(np.uint16).tobytes()
        step = W * 2
    elif encoding == 'bgr8':
        raw = img_np.astype(np.uint8).tobytes()
        step = W * 3
    else:
        raise ValueError(f"Unknown encoding: {encoding}")
    return dict(
        header=dict(seq=0, stamp=dict(sec=sec, nanosec=nsec), frame_id=frame_id),
        height=H, width=W, encoding=encoding,
        is_bigendian=0, step=step, data=np.frombuffer(raw, dtype=np.uint8),
    )

def make_camera_info_msg(sec, nsec, W, H, fx, fy, cx, cy, frame_id):
    K = [fx,0,cx,  0,fy,cy,  0,0,1.0]
    P = [fx,0,cx,0,  0,fy,cy,0,  0,0,1,0]
    return dict(
        header=dict(seq=0, stamp=dict(sec=sec, nanosec=nsec), frame_id=frame_id),
        height=H, width=W,
        distortion_model='plumb_bob',
        d=[0.0,0.0,0.0,0.0,0.0],
        k=K, r=[1,0,0,0,1,0,0,0,1.0], p=P,
        binning_x=0, binning_y=0,
        roi=dict(x_offset=0,y_offset=0,height=0,width=0,do_rectify=False),
    )

# ---------------------------------------------------------------------------
# Serialise messages manually (rosbags write path)
# ---------------------------------------------------------------------------
def ser_header(seq, sec, nsec, frame_id):
    fid = frame_id.encode()
    return struct.pack('<III', seq, sec, nsec) + struct.pack('<I', len(fid)) + fid

def ser_image(seq, sec, nsec, frame_id, H, W, encoding, step, img_bytes):
    hdr  = ser_header(seq, sec, nsec, frame_id)
    enc  = encoding.encode()
    data = struct.pack('<II', H, W)
    data += struct.pack('<I', len(enc)) + enc
    data += struct.pack('<B', 0)         # is_bigendian
    data += struct.pack('<I', step)
    data += struct.pack('<I', len(img_bytes)) + img_bytes
    return hdr + data

def ser_camera_info(seq, sec, nsec, frame_id, H, W, fx, fy, cx, cy):
    hdr = ser_header(seq, sec, nsec, frame_id)
    dm  = b'plumb_bob'
    K   = [fx,0,cx, 0,fy,cy, 0,0,1.0]
    P   = [fx,0,cx,0, 0,fy,cy,0, 0,0,1,0]
    R   = [1.0,0,0, 0,1,0, 0,0,1.0]
    out  = hdr
    out += struct.pack('<II', H, W)
    out += struct.pack('<I', len(dm)) + dm
    out += struct.pack('<I', 5) + struct.pack('<5d', *[0.0]*5)
    out += struct.pack('<9d', *K)
    out += struct.pack('<9d', *R)
    out += struct.pack('<12d', *P)
    out += struct.pack('<II', 0, 0)      # binning
    out += struct.pack('<IIIII', 0, 0, 0, 0, 0)  # roi
    return out

def ser_imu(rawdata):
    """IMU is a standard msg — pass raw bytes through unchanged."""
    return rawdata

# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------
def convert(args):
    calib = PRESETS[args.seq]
    # Allow CLI overrides
    Rcl = np.array(args.rcl).reshape(3,3) if args.rcl else calib['Rcl']
    Pcl = np.array(args.pcl)              if args.pcl else calib['Pcl']
    fx  = args.fx  or calib['fx']
    fy  = args.fy  or calib['fy']
    cx  = args.cx  or calib['cx']
    cy  = args.cy  or calib['cy']
    W   = args.width  or calib['cam_w']
    H   = args.height or calib['cam_h']

    print(f"Calibration:")
    print(f"  Camera: {W}×{H}  fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")
    print(f"  Pcl (cam origin in LiDAR): {Pcl}")

    in_path  = Path(args.input)
    out_path = Path(args.output)
    store    = get_typestore(Stores.ROS1_NOETIC)

    # Message type strings for the output bag
    DEPTH_TYPE = 'sensor_msgs/msg/Image'
    INFO_TYPE  = 'sensor_msgs/msg/CameraInfo'
    COLOR_TYPE = 'sensor_msgs/msg/Image'
    IMU_TYPE   = 'sensor_msgs/msg/Imu'

    # ROS1 message definitions (needed for bag connection records)
    IMAGE_MSGDEF = """
uint32 height
uint32 width
string encoding
uint8 is_bigendian
uint32 step
uint8[] data
"""
    CAM_INFO_MSGDEF = """
std_msgs/Header header
uint32 height
uint32 width
string distortion_model
float64[] d
float64[9] k
float64[9] r
float64[12] p
uint32 binning_x
uint32 binning_y
sensor_msgs/RegionOfInterest roi
"""

    frame_id_depth = "camera_depth_optical_frame"
    frame_id_color = "camera_color_optical_frame"
    frame_id_imu   = "camera_imu_frame"

    img_topic   = None
    img_compressed = False

    n_lidar = n_img = n_imu = 0
    t0 = time.time()

    with Reader(in_path) as reader:
        # Detect image topic
        for conn in reader.connections:
            if 'image' in conn.topic and 'camera' in conn.topic:
                img_topic = conn.topic
                img_compressed = 'compressed' in conn.msgtype.lower()
                print(f"Image topic: {img_topic}  compressed={img_compressed}")
                break

        lidar_conns = [c for c in reader.connections if c.topic == '/livox/lidar']
        imu_conns   = [c for c in reader.connections if c.topic == '/livox/imu']
        img_conns   = [c for c in reader.connections if c.topic == img_topic] if img_topic else []

        print(f"Input: {in_path.name}  duration={reader.duration/1e9:.1f}s")

        with Writer(out_path) as writer:
            # Register output connections (msgdef/md5sum inherited from reader connections)
            imu_conn_src = imu_conns[0] if imu_conns else None
            c_depth = writer.add_connection('/camera/depth/image_rect_raw', DEPTH_TYPE)
            c_info  = writer.add_connection('/camera/depth/camera_info',    INFO_TYPE)
            c_color = writer.add_connection('/camera/color/image_raw',      COLOR_TYPE)
            imu_msgdef = None
            if imu_conn_src:
                raw_msgdef = imu_conn_src.msgdef
                imu_msgdef = raw_msgdef[1] if isinstance(raw_msgdef, (list, tuple)) else raw_msgdef
            c_imu   = writer.add_connection('/camera/imu', IMU_TYPE,
                                            msgdef=imu_msgdef)

            for conn, ts_ns, rawdata in reader.messages():
                topic = conn.topic

                if topic == '/livox/lidar':
                    try:
                        sec, nsec, xyz = parse_livox(rawdata)
                        depth_img = project_lidar_to_depth(
                            xyz, Rcl, Pcl, fx, fy, cx, cy, W, H,
                            min_depth=0.3, max_depth=getattr(args, 'max_depth', 15.0),
                            dense=args.dense)

                        img_bytes = depth_img.tobytes()
                        depth_raw = ser_image(n_lidar, sec, nsec,
                                              frame_id_depth,
                                              H, W, '16UC1', W*2, img_bytes)
                        info_raw  = ser_camera_info(n_lidar, sec, nsec,
                                                    frame_id_depth,
                                                    H, W, fx, fy, cx, cy)
                        writer.write(c_depth, ts_ns, depth_raw)
                        writer.write(c_info,  ts_ns, info_raw)
                        n_lidar += 1

                        if n_lidar % 50 == 0:
                            pts_proj = int(np.count_nonzero(depth_img))
                            elapsed  = time.time() - t0
                            print(f"  [{elapsed:5.1f}s] lidar={n_lidar} img={n_img} "
                                  f"imu={n_imu}  proj_pts={pts_proj}")
                    except Exception as e:
                        print(f"  [WARN] Lidar parse error: {e}")

                elif topic == img_topic:
                    try:
                        if img_compressed:
                            # Decompress JPEG/PNG to BGR
                            arr = np.frombuffer(rawdata, dtype=np.uint8)
                            # Skip ROS Header to get to compressed data bytes
                            # CompressedImage: Header + string format + uint8[] data
                            pos = 0
                            _seq = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            sec2 = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            nsec2= struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            flen = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4+flen
                            fmt_len = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4+fmt_len
                            dlen = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            compressed_bytes = np.frombuffer(rawdata[pos:pos+dlen], dtype=np.uint8)
                            bgr = cv2.imdecode(compressed_bytes, cv2.IMREAD_COLOR)
                            if bgr is None: continue
                            bgr = cv2.resize(bgr, (W, H))
                            img_bytes = bgr.tobytes()
                            color_raw = ser_image(n_img, sec2, nsec2,
                                                  frame_id_color,
                                                  H, W, 'bgr8', W*3, img_bytes)
                            writer.write(c_color, ts_ns, color_raw)
                        else:
                            # Raw Image: Header + height + width + encoding + bigendian + step + data
                            pos = 0
                            _seq = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            sec2 = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            nsec2= struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            flen = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4+flen
                            img_h= struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            img_w= struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            enc_len=struct.unpack_from('<I',rawdata,pos)[0]; pos+=4
                            enc = rawdata[pos:pos+enc_len].decode(); pos+=enc_len
                            _be = rawdata[pos]; pos+=1
                            step = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            dlen = struct.unpack_from('<I', rawdata, pos)[0]; pos+=4
                            raw_px = rawdata[pos:pos+dlen]
                            # Decode to BGR and resize
                            if enc == 'bgr8':
                                bgr = np.frombuffer(raw_px, dtype=np.uint8).reshape(img_h, img_w, 3)
                            elif enc == 'rgb8':
                                rgb = np.frombuffer(raw_px, dtype=np.uint8).reshape(img_h, img_w, 3)
                                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            elif enc == 'mono8':
                                mono= np.frombuffer(raw_px, dtype=np.uint8).reshape(img_h, img_w)
                                bgr = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
                            else:
                                bgr = np.frombuffer(raw_px, dtype=np.uint8).reshape(img_h, img_w, -1)
                            bgr = cv2.resize(bgr, (W, H))
                            img_bytes = bgr.tobytes()
                            color_raw = ser_image(n_img, sec2, nsec2,
                                                  frame_id_color,
                                                  H, W, 'bgr8', W*3, img_bytes)
                            writer.write(c_color, ts_ns, color_raw)
                        n_img += 1
                    except Exception as e:
                        print(f"  [WARN] Image error: {e}")

                elif topic == '/livox/imu':
                    # IMU: pass raw bytes through (same format, just remap topic)
                    writer.write(c_imu, ts_ns, rawdata)
                    n_imu += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s:")
    print(f"  Depth frames : {n_lidar}")
    print(f"  Colour frames: {n_img}")
    print(f"  IMU messages : {n_imu}")
    print(f"  Output       : {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input',  required=True,  help='Input FAST-LIVO2 bag')
    ap.add_argument('--output', required=True,  help='Output D455-format bag')
    ap.add_argument('--seq',    required=True,
                    choices=['retail', 'hku', 'hit', 'red_sculpture'],
                    help='Calibration preset: "retail" for Bright_Screen_Wall/Retail_Street/CBD_01, '
                         '"hku" for HKU_*/CBD_02/03, "hit" for HIT_*/SYSU_*, '
                         '"red_sculpture" for Red_Sculpture')
    # Optional overrides
    ap.add_argument('--rcl',    nargs=9,   type=float, default=None, metavar='R',
                    help='9 values of camera→LiDAR rotation matrix (row-major)')
    ap.add_argument('--pcl',    nargs=3,   type=float, default=None, metavar='T',
                    help='Camera origin in LiDAR frame (3 values, metres)')
    ap.add_argument('--fx',     type=float, default=None)
    ap.add_argument('--fy',     type=float, default=None)
    ap.add_argument('--cx',     type=float, default=None)
    ap.add_argument('--cy',     type=float, default=None)
    ap.add_argument('--width',  type=int,   default=None)
    ap.add_argument('--height', type=int,   default=None)
    ap.add_argument('--dense', action='store_true', default=True,
                    help='Fill sparse projected depth via nearest-neighbour '
                         'completion (default: on; needed for voxel plane fitting)')
    ap.add_argument('--no-dense', dest='dense', action='store_false',
                    help='Disable depth completion (raw sparse projection only)')
    ap.add_argument('--max-depth', type=float, default=15.0,
                    help='Drop LiDAR returns beyond this depth (m); the bags of the paper use 40')
    args = ap.parse_args()
    convert(args)
