#!/usr/bin/env python3
"""
Per-frame processing time from the timing runs (the estimator
adds a "[Timing] depth selection" line per depth frame to the usual LIO and VIO
time tables in the rgbd_vio log).

For each log: depth selection (ITPS or depth bins), LIO update (Total (current)
of the LIO table), VIO update (Current Total Time of the VIO table), and their
per-frame sum, as mean / 95th percentile / max in ms, and the share of frames
whose sum exceeds the camera frame period.

When the run folder also holds <seq>.latency.csv (latency_logger.py), <seq>.mem.csv (mem_logger.sh) and
<seq>.power.csv (tegrastats_monitor.sh), as the Jetson benchmark writes them, a second table reports the
sensor-to-pose latency (p50 / p95 / p99 / max ms, share of poses later than one frame period, largest queue
backlog), peak resident memory, mean and peak board power, and peak SoC temperature.

Usage: summarize_timing.py TIMING_ROOT   (TIMING_ROOT/<dataset>/<variant>_r<k>/<seq>.log)
"""
import re
import sys
from pathlib import Path

import numpy as np

ANSI = re.compile(r'\x1b\[[0-9;]*m')
PERIOD_S = {'utmm': 1 / 30, 'openloris': 1 / 30, 'fastlivo2_retail': 0.1, 'fastlivo2_hku': 0.1,
            'site_a': 1 / 30, 'site_b': 1 / 30}


def parse(log):
    sel, lio, vio = [], [], []
    for line in open(log, errors='replace'):
        line = ANSI.sub('', line)
        if line.startswith('[Timing] depth selection'):
            sel.append(float(line.split()[3]))
        elif '| Total (current)' in line:
            lio.append(float(line.split('|')[2]))
        elif '| Current Total Time' in line:
            vio.append(float(line.split('|')[2]))
    return np.array(sel), np.array(lio), np.array(vio)


def ms(x):
    return f'{1e3 * x.mean():6.1f} {1e3 * np.percentile(x, 95):6.1f} {1e3 * x.max():7.1f}' if len(x) else f"{'--':>6s} {'--':>6s} {'--':>7s}"


def column(path, name):
    """One numeric CSV column (empty cells skipped), or an empty array."""
    if not path.exists():
        return np.array([])
    vals = []
    with open(path) as fh:
        head = fh.readline().strip().split(',')
        if name not in head:
            return np.array([])
        k = head.index(name)
        for line in fh:
            f = line.rstrip('\n').split(',')
            if len(f) > k and f[k] not in ('', 'nan'):
                vals.append(float(f[k]))
    return np.array(vals)


def main():
    root = Path(sys.argv[1])
    print(f"{'dataset':17s} {'variant':16s} {'sequence':16s} | {'selection mean/p95/max ms':>26s} | "
          f"{'LIO mean/p95/max':>22s} | {'VIO mean/p95/max':>22s} | {'sum mean/p95':>13s} | >period %")
    extra = []
    for log in sorted(root.glob('*/*_r*/*.log')):
        if log.name.endswith('.latency.log'):
            continue
        ds, var, seq = log.parent.parent.name, log.parent.name.rsplit('_r', 1)[0], log.stem
        sel, lio, vio = parse(log)
        n = min(len(sel), len(lio))
        if n == 0:
            continue
        # one depth frame = selection + LIO; each image frame adds a VIO update (same rate here)
        tot = sel[:n] + lio[:n] + (np.resize(vio, n) if len(vio) else 0.0)
        period = PERIOD_S.get(ds, 1 / 30)
        over = 100 * np.mean(tot > period)
        print(f'{ds:17s} {var:16s} {seq:16s} | {ms(sel):>26s} | {ms(lio):>22s} | {ms(vio):>22s} | '
              f'{1e3 * tot.mean():6.1f} {1e3 * np.percentile(tot, 95):6.1f} | {over:6.1f}')
        lat = column(log.with_name(f'{seq}.latency.csv'), 'latency_s')
        if len(lat):
            back = column(log.with_name(f'{seq}.latency.csv'), 'backlog')
            rss = column(log.with_name(f'{seq}.mem.csv'), 'rss_mb')
            pw = column(log.with_name(f'{seq}.power.csv'), 'power_total_mw')
            tc = column(log.with_name(f'{seq}.power.csv'), 'soc_temp_c')
            extra.append(f'{ds:17s} {var:16s} {seq:16s} | {1e3 * np.percentile(lat, 50):6.1f} {1e3 * np.percentile(lat, 95):6.1f} '
                         f'{1e3 * np.percentile(lat, 99):6.1f} {1e3 * lat.max():7.1f} | {100 * np.mean(lat > period):6.1f} | '
                         f"{int(back.max()) if len(back) else '--':>7} | {rss.max() if len(rss) else float('nan'):8.0f} | "
                         f"{pw.mean() / 1e3 if len(pw) else float('nan'):6.1f} {pw.max() / 1e3 if len(pw) else float('nan'):6.1f} | "
                         f"{tc.max() if len(tc) else float('nan'):5.1f}")
    if extra:
        print()
        print(f"{'dataset':17s} {'variant':16s} {'sequence':16s} | {'latency p50/p95/p99/max ms':>28s} | {'late %':>6s} | "
              f"{'backlog':>7s} | {'RSS MB':>8s} | {'power mean/peak W':>13s} | {'T max':>5s}")
        print('\n'.join(extra))


if __name__ == '__main__':
    main()
