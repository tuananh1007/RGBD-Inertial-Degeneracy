#!/usr/bin/env python3
"""
Score field runs without ground truth (OUT_ROOT/runs.csv from run_sequences.sh).

Per run: finite fraction, coverage of the bag span (complete = finite and
reaching within 1 s of the bag end), path length, peak and 95th-percentile
speed over 1-s windows, the share of 1-s windows faster than VMAX (the
platforms move at walking pace, so faster windows mean the estimate has
diverged), and the end-to-start distance, which is the drift for sequences
that return to their start (pass them with --loops).

Usage: evaluate_field_runs.py OUT_ROOT DATA_DIR [--loops SEQ ...] [--vmax 3.0]
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from rosbags.rosbag1 import Reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_root'); ap.add_argument('data_dir')
    ap.add_argument('--loops', nargs='*', default=[]); ap.add_argument('--vmax', type=float, default=3.0)
    a = ap.parse_args()
    root = Path(a.out_root)
    span = {}
    rows = []
    latest = {(r['seq'], r['variant'], r['run']): r for r in csv.DictReader(open(root / 'runs.csv'))}  # one row per run
    for seq, var, run in sorted(latest):
        if seq not in span:
            with Reader(Path(a.data_dir) / f'{seq}_d455.bag') as b:
                span[seq] = (b.start_time * 1e-9, b.end_time * 1e-9)
        p = root / f'{var}_r{run}' / f'{seq}_traj.txt'
        m = dict(seq=seq, variant=var, run=run, poses=0, finite=0.0, coverage=0.0, path_m=np.nan, vmax=np.nan,
                 v95=np.nan, fast_pct=np.nan, end_start_m=np.nan, complete=False)
        if p.exists() and p.stat().st_size:
            x = np.loadtxt(p, ndmin=2)
            ok = np.isfinite(x[:, 1:4]).all(axis=1)
            m['poses'], m['finite'] = len(x), float(ok.mean())
            x = x[ok]
            if len(x) > 2:
                t, q = x[:, 0], x[:, 1:4]
                m['coverage'] = (t[-1] - t[0]) / (span[seq][1] - span[seq][0])
                m['complete'] = bool(m['finite'] == 1.0 and span[seq][1] - t[-1] < 1.0)
                m['path_m'] = float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())
                tw = np.arange(t[0], t[-1] - 1.0, 0.5)
                if len(tw):
                    i0, i1 = np.searchsorted(t, tw), np.searchsorted(t, tw + 1.0)
                    i1 = np.minimum(i1, len(t) - 1)
                    v = np.linalg.norm(q[i1] - q[i0], axis=1) / np.maximum(t[i1] - t[i0], 1e-3)
                    m['vmax'], m['v95'] = float(v.max()), float(np.percentile(v, 95))
                    m['fast_pct'] = float(100 * np.mean(v > a.vmax))
                m['end_start_m'] = float(np.linalg.norm(q[-1] - q[0]))
        rows.append(m)
    keys = list(rows[0].keys())
    with open(root / 'field_metrics.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    by = defaultdict(list)
    for m in rows:
        by[m['variant']].append(m)
    print(f"{'variant':16s} runs complete | path m  peak v  v95 | >{a.vmax:g} m/s % | end-start drift on loops (m)")
    for var, ms in sorted(by.items()):
        loop = {s: [m['end_start_m'] for m in ms if m['seq'] == s] for s in a.loops}
        loop_txt = '  '.join(f'{s} {np.nanmean(v):.2f}' for s, v in loop.items() if v)
        print(f"{var:16s} {len(ms):4d} {sum(m['complete'] for m in ms):8d} | {np.nanmean([m['path_m'] for m in ms]):6.1f} "
              f"{np.nanmedian([m['vmax'] for m in ms]):7.2f} {np.nanmedian([m['v95'] for m in ms]):5.2f} | "
              f"{np.nanmean([m['fast_pct'] for m in ms]):10.1f} | {loop_txt}")
    print('wrote', root / 'field_metrics.csv')


if __name__ == '__main__':
    main()
