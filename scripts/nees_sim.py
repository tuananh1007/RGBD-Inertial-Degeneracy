#!/usr/bin/env python3
"""
Consistency of the reported covariance in the E12 simulation (ICL-NUIM, exact depth, synthesized IMU with known
noise, 2026-10-26).  The estimator's world frame is the body frame at its first depth update, so the reference is
expressed in that frame with the reference pose at the first output time, and no alignment is estimated.

Per run (<variant>_r<k>/<seq>_traj.txt + <seq>_cov.txt, reference <seq>_d455.gt.txt in GT_DIR):
  e_p   position error (m) in the estimator's world frame, p_ref = p_est + e_p
  e_th  attitude error (rad), right perturbation: R_ref = R_est Exp(e_th)
  (both reference minus estimate, the convention of the filter's error state and of the logged cross block P_rp)
  NEES_p = e_p^T P_pp^-1 e_p (chi^2_3: mean 3), NEES_6 with the 6x6 attitude-position block (chi^2_6: mean 6)
Per variant, sequence, and 1 s bin: the average NEES over the runs (independent IMU noise realizations) and frames in
the bin, and the share of frames inside the two-sided 95 % interval of the per-frame statistic.  The covariance file
holds P_pp (upper triangle), P_rr (upper triangle), P_rp (3x3), written by the estimator (evo/cov_output_en).

Usage: nees_sim.py OUT_ROOT GT_DIR [--bin 1.0] [--out OUT_ROOT/nees.csv]
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import chi2


def sym3(u):
    a, b, c, d, e, f = u
    return np.array([[a, b, c], [b, d, e], [c, e, f]])


def load_cov(path):
    rows = np.loadtxt(path)
    t = rows[:, 0]
    P = np.zeros((len(rows), 6, 6))
    for i, r in enumerate(rows):
        P[i, 3:, 3:] = sym3(r[1:7]); P[i, :3, :3] = sym3(r[7:13]); P[i, :3, 3:] = r[13:22].reshape(3, 3)
        P[i, 3:, :3] = P[i, :3, 3:].T
    return t, P


def reference_in_first_frame(gt, t0):
    """Reference poses relative to the reference pose at t0 (interpolated)."""
    tg, pg = gt[:, 0], gt[:, 1:4]
    rg = Rotation.from_quat(gt[:, 4:8])
    slerp = Slerp(tg, rg)
    R0 = slerp([t0])[0]; p0 = np.array([np.interp(t0, tg, pg[:, k]) for k in range(3)])
    return tg, R0.inv().apply(pg - p0), R0.inv() * rg


def run_nees(traj, cov, gt):
    d = np.loadtxt(traj); te, pe = d[:, 0], d[:, 1:4]; Re = Rotation.from_quat(d[:, 4:8])
    tc, P = load_cov(cov)
    tg, pr, Rr = reference_in_first_frame(gt, te[0])
    slerp = Slerp(tg, Rr)
    out = []
    j = np.searchsorted(tc, te)   # covariance rows at the same stamps
    for i in range(len(te)):
        if j[i] >= len(tc) or abs(tc[j[i]] - te[i]) > 1e-3 or te[i] < tg[0] or te[i] > tg[-1]:
            continue
        p_ref = np.array([np.interp(te[i], tg, pr[:, k]) for k in range(3)])
        e_p = p_ref - pe[i]
        e_th = (Re[i].inv() * slerp([te[i]])[0]).as_rotvec()
        Pi = P[j[i]]
        try:
            n3 = float(e_p @ np.linalg.solve(Pi[3:, 3:], e_p))
            e6 = np.r_[e_th, e_p]; n6 = float(e6 @ np.linalg.solve(Pi, e6))
        except np.linalg.LinAlgError:
            continue
        out.append((te[i] - te[0], np.linalg.norm(e_p), np.linalg.norm(e_th), math.sqrt(np.trace(Pi[3:, 3:]) / 3), n3, n6))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('out_root'); ap.add_argument('gt_dir'); ap.add_argument('--bin', type=float, default=1.0); ap.add_argument('--out')
    a = ap.parse_args()
    root, gtd = Path(a.out_root), Path(a.gt_dir)
    lo3, hi3 = chi2.ppf([0.025, 0.975], 3); lo6, hi6 = chi2.ppf([0.025, 0.975], 6)
    per_bin = defaultdict(list)   # (variant, seq, bin) -> rows
    summary = []
    for traj in sorted(root.glob('*_r*/*_traj.txt')):
        var, run = traj.parent.name.rsplit('_r', 1); seq = traj.name[:-9]
        cov = traj.with_name(f'{seq}_cov.txt'); gt = gtd / f'{seq}_d455.gt.txt'
        if not cov.exists() or not gt.exists():
            continue
        r = run_nees(traj, cov, np.loadtxt(gt))
        if not len(r):
            continue
        for row in r:
            per_bin[(var, seq[:-2] if seq[-2] == 's' else seq, int(row[0] // a.bin))].append(row)
        summary.append((var, seq, int(run), len(r), np.median(r[:, 1]), np.sqrt(np.mean(r[:, 1] ** 2)), np.median(r[:, 3]),
                        np.median(r[:, 4]), np.mean((r[:, 4] >= lo3) & (r[:, 4] <= hi3)), np.median(r[:, 5]),
                        np.mean((r[:, 5] >= lo6) & (r[:, 5] <= hi6))))
    out = Path(a.out or root / 'nees.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['variant', 'seq', 'bin_s', 'frames', 'runs', 'rmse_p', 'med_sigma_p', 'anees_p', 'in95_p', 'anees_6', 'in95_6'])
        for (var, seq, b), rows in sorted(per_bin.items()):
            r = np.array(rows)
            w.writerow([var, seq, b * a.bin, len(r), '', f'{np.sqrt(np.mean(r[:, 1] ** 2)):.4f}', f'{np.median(r[:, 3]):.5f}',
                        f'{np.mean(r[:, 4]):.2f}', f'{np.mean((r[:, 4] >= lo3) & (r[:, 4] <= hi3)):.3f}',
                        f'{np.mean(r[:, 5]):.2f}', f'{np.mean((r[:, 5] >= lo6) & (r[:, 5] <= hi6)):.3f}'])
    print(f"{'variant':18s} {'seq':8s} run frames | med|e_p| rmse_p med_sigma_p | med NEES_p in95 | med NEES_6 in95")
    for s in summary:
        print(f'{s[0]:18s} {s[1]:8s} {s[2]:3d} {s[3]:6d} | {s[4]:8.4f} {s[5]:7.4f} {s[6]:11.5f} | {s[7]:10.1f} {s[8]:4.2f} | {s[9]:10.1f} {s[10]:4.2f}')
    print('wrote', out)


if __name__ == '__main__':
    main()
