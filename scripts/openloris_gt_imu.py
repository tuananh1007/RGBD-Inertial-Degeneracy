#!/usr/bin/env python3
"""
Express the OpenLORIS references (pose of base_link) in the D435i IMU frame, the frame of every estimate, with the
static transforms recorded in the dataset's bags (market has its own mount and calibration): writes
DATA_DIR/<seq>_gt_imu.txt next to DATA_DIR/<seq>_gt.txt.  Score with evaluate_runs.py ... --gt-pattern '{seq}_gt_imu.txt'.

Usage: openloris_gt_imu.py DATA_DIR [SEQ ...]   (default: office1-1 ... office1-7)
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_runs import load_tum  # noqa: E402


def pose(t, q):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q).as_matrix()
    T[:3, 3] = t
    return T


# T(base_link <- d400_imu) from the bags' /tf_static
T_BODY = {'openloris': pose([0.226484, -0.051142, 0.916], [-0.496762, 0.49988, -0.495107, 0.50815])
          @ pose([0.020313, -0.005103, -0.011201], [0.001745, 0.002931, -0.001919, 0.999992]),
          'openloris_market': np.array([[0.053362, -0.141152, 0.988549, 0.388942], [-0.998532, -0.016784, 0.051504, 0.008422],
                                        [0.009321, -0.989846, -0.14184, 1.205217], [0.0, 0.0, 0.0, 1.0]])}


def in_body(gt, T):
    """Reference poses T_w_ref -> T_w_ref @ T (TUM rows)."""
    R = Rotation.from_quat(gt[:, 4:8])
    return np.column_stack([gt[:, 0], gt[:, 1:4] + R.apply(T[:3, 3]), (R * Rotation.from_matrix(T[:3, :3])).as_quat()])


data = Path(sys.argv[1])
for seq in sys.argv[2:] or [f'office1-{k}' for k in range(1, 8)]:
    gt = load_tum(data / f'{seq}_gt.txt')
    np.savetxt(data / f'{seq}_gt_imu.txt', in_body(gt, T_BODY['openloris_market' if seq.startswith('market') else 'openloris']), fmt='%.9f',
               header='OpenLORIS reference in the D435i IMU frame (openloris_gt_imu.py)\ntimestamp tx ty tz qx qy qz qw')
    print(seq, len(gt))
