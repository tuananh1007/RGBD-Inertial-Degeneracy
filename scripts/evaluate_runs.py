#!/usr/bin/env python3
"""
Score every run listed in OUT_ROOT/runs.csv (written by run_sequences.sh).

Per run: pose count, finite fraction, temporal coverage of the GT span,
ATE RMSE after rigid SE(3) alignment (primary) and after Sim(3) alignment
(with the fitted scale and scale error |1-s|, for comparison with earlier
tables), the share of time-associated poses within 10 cm after SE(3)
alignment, and RPE over 1-s intervals (translation m, rotation deg).  A run with no output, any
non-finite pose, or output that stops more than 1 s before the end of the GT
span is flagged as not complete; the start delay (initialization) is reported.
Coverage, start delay and completion use the part of the GT span that overlaps
the recording (SEQ_d455.bag in GT_DIR, when present).

Usage:
  python3 evaluate_runs.py OUT_ROOT GT_DIR [--out OUT_ROOT/metrics.csv]
                                [--gt-pattern '{seq}_d455.gt.txt']
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader
from scipy.spatial.transform import Rotation

ASSOC_MAX_DT = 0.02    # s, est-to-GT timestamp association (at least; see associate)
RPE_DT = 1.0           # s
RPE_TOL = 0.1          # s
END_TOL_S = 1.0       # s, a complete run must reach the end of the GT span
INLIER_M = 0.10       # m, SE(3)-aligned per-pose error threshold


def load_tum(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            vals = [float(x) for x in line.split()[:8]]
            if len(vals) == 8:
                rows.append(vals)
    return np.array(rows).reshape(-1, 8)


def associate(t_est, t_gt):
    idx = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    left, right = t_gt[idx - 1], t_gt[idx]
    idx = np.where(np.abs(t_est - left) <= np.abs(t_est - right), idx - 1, idx)
    # half a GT period for low-rate references (10 Hz FAST-LIVO2 pseudo-GT), else ASSOC_MAX_DT
    max_dt = max(ASSOC_MAX_DT, 0.5 * float(np.median(np.diff(t_gt))))
    ok = np.abs(t_gt[idx] - t_est) <= max_dt
    return ok, idx


def umeyama(src, dst, with_scale):
    """Least-squares s, R, t with dst ~= s R src + t (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum() if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def ate(est_p, gt_p, with_scale):
    """RMSE, fitted scale, and per-pose error norms after alignment."""
    s, R, t = umeyama(est_p, gt_p, with_scale)
    err = np.linalg.norm(gt_p - (s * (R @ est_p.T).T + t), axis=1)
    return float(np.sqrt((err ** 2).mean())), float(s), err


def rpe(t, est, gt):
    """RMSE of 1-s relative pose errors over associated pose pairs."""
    j = np.searchsorted(t, t + RPE_DT)
    ok = j < len(t)
    j = np.where(ok, j, 0)
    ok &= np.abs(t[j] - (t + RPE_DT)) <= RPE_TOL
    if not ok.any():
        return np.nan, np.nan
    i, j = np.nonzero(ok)[0], j[ok]

    def rel(poses):
        R = Rotation.from_quat(poses[:, 3:7])
        dR = R[i].inv() * R[j]
        dp = R[i].inv().apply(poses[j, :3] - poses[i, :3])
        return dR, dp

    dR_e, dp_e = rel(est)
    dR_g, dp_g = rel(gt)
    t_err = np.linalg.norm(dR_g.inv().apply(dp_e - dp_g), axis=1)
    r_err = np.degrees((dR_g.inv() * dR_e).magnitude())
    return float(np.sqrt((t_err ** 2).mean())), float(np.sqrt((r_err ** 2).mean()))


def score(traj_path, gt_path, span=None, lever=None):
    out = dict(poses=0, finite=0.0, coverage=0.0, start_delay_s=np.nan, ate_se3=np.nan, ate_sim3=np.nan,
               sim3_scale=np.nan, scale_err_pct=np.nan, inlier_10cm=np.nan, rpe_t=np.nan, rpe_r=np.nan, complete=False)
    if not Path(traj_path).is_file():
        return out
    est = load_tum(traj_path)
    gt = load_tum(gt_path)
    gt = gt[np.argsort(gt[:, 0])]
    out['poses'] = len(est)
    if len(est) == 0:
        return out
    finite = np.isfinite(est).all(1)
    out['finite'] = float(finite.mean())
    est = est[finite]
    if lever is not None:     # position-only reference of a point fixed on the body (ROVER: total-station prism)
        est[:, 1:4] += Rotation.from_quat(est[:, 4:8]).apply(lever)
    ok, idx = associate(est[:, 0], gt[:, 0])
    if ok.sum() < 10:
        return out
    est, gt_a = est[ok], gt[idx[ok]]
    if span is not None:      # coverage/completion relative to the GT that overlaps the recording
        inside = (gt[:, 0] >= span[0]) & (gt[:, 0] <= span[1])
        if inside.sum() > 10:
            gt = gt[inside]
    out['coverage'] = float((est[-1, 0] - est[0, 0]) / (gt[-1, 0] - gt[0, 0]))
    out['start_delay_s'] = float(est[0, 0] - gt[0, 0])
    out['ate_se3'], _, err = ate(est[:, 1:4], gt_a[:, 1:4], with_scale=False)
    out['inlier_10cm'] = float(100.0 * (err < INLIER_M).mean())
    out['ate_sim3'], out['sim3_scale'], _ = ate(est[:, 1:4], gt_a[:, 1:4], with_scale=True)
    out['scale_err_pct'] = float(100.0 * abs(1.0 - out['sim3_scale']))
    out['rpe_t'], out['rpe_r'] = rpe(est[:, 0], est[:, 1:], gt_a[:, 1:])
    out['complete'] = bool(out['finite'] == 1.0 and gt[-1, 0] - est[-1, 0] <= END_TOL_S)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out_root')
    ap.add_argument('gt_dir')
    ap.add_argument('--out')
    ap.add_argument('--gt-pattern', default='{seq}_d455.gt.txt',
                    help='GT file name inside GT_DIR (OpenLORIS: {seq}_gt.txt)')
    ap.add_argument('--lever', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                    help='score the point at X Y Z in the estimate\'s body frame (position-only reference, e.g. ROVER prism)')
    args = ap.parse_args()
    root = Path(args.out_root)
    runs = list(csv.DictReader(open(root / 'runs.csv')))
    latest = {(r['seq'], r['variant'], r['run']): r for r in runs}   # last attempt wins

    rows, spans = [], {}
    for (seq, variant, run), r in sorted(latest.items()):
        if seq not in spans:
            bag = Path(args.gt_dir) / f'{seq}_d455.bag'
            spans[seq] = None
            if bag.exists():
                with Reader(bag) as b:
                    spans[seq] = (b.start_time * 1e-9, b.end_time * 1e-9)
        m = score(root / f'{variant}_r{run}' / f'{seq}_traj.txt',
                  Path(args.gt_dir) / args.gt_pattern.format(seq=seq), spans[seq], args.lever)
        rows.append(dict(seq=seq, variant=variant, run=run, **m))
    out = Path(args.out) if args.out else root / 'metrics.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for row in rows:
            w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in row.items()})

    by_var = defaultdict(list)
    for row in rows:
        by_var[row['variant']].append(row)
    print(f"{'variant':28s} {'runs':>4s} {'complete':>8s} | {'SE3 ATE mean':>12s} {'median':>7s} | "
          f"{'Sim3 ATE mean':>13s} {'scale err%':>10s} {'<10cm %':>7s} | {'RPE t':>6s} {'RPE r':>6s}   (means over complete runs)")
    for v, rs in sorted(by_var.items()):
        c = [r for r in rs if r['complete']]
        f = lambda k: np.array([r[k] for r in c]) if c else np.array([np.nan])
        print(f"{v:28s} {len(rs):4d} {len(c):8d} | {np.mean(f('ate_se3')):12.3f} "
              f"{np.median(f('ate_se3')):7.3f} | {np.mean(f('ate_sim3')):13.3f} {np.mean(f('scale_err_pct')):10.1f} {np.mean(f('inlier_10cm')):7.1f} | "
              f"{np.mean(f('rpe_t')):6.3f} {np.mean(f('rpe_r')):6.2f}")
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
