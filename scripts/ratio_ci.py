#!/usr/bin/env python3
"""
Effect sizes with intervals for the paired comparisons of the paper.

For every variant of the given metrics.csv files (written by evaluate_runs.py) against a reference variant:
the per-sequence median SE(3) ATE over complete runs, the number of
sequences on which the variant has the lower median, the geometric-mean ratio variant / reference with a 95 %
confidence interval (t interval on the per-sequence log ratios, and a percentile bootstrap over sequences), the exact
Wilcoxon signed-rank p-value on the log ratios, and the runs with ATE above --thr metres or incomplete.  Sequences are
the units, so the interval describes this benchmark set.  A sequence enters a comparison only if both variants have a
majority of complete runs on it.

  ratio_ci.py --ref s_ours_lin SWEEP/utmm/metrics.csv SWEEP/openloris/metrics.csv
  ratio_ci.py --ref s_ours_lin --exclude CBD_Building_01 CBD_Building_02 Retail_Street  ...   # dense depth only
  ratio_ci.py --ref s_ours_lin --variants u_ours_lin ours_lin --tex out.tex ...

For the main comparison pass the utmm, openloris, fastlivo2_bench/retail and fastlivo2_bench/hku metrics.csv;
with --exclude of the three FAST-LIVO2 sequences the sampling rows are tested on the 15 dense-depth sequences only
(on sparse depth they run the fallback and are the same configuration as Ours).
"""
import argparse
import collections
import csv

import numpy as np
from scipy import stats


def load(paths):
    runs = collections.defaultdict(list)
    for p in paths:
        for x in csv.DictReader(open(p)):
            ok = x['complete'] == 'True' and x['ate_se3'] not in ('', 'nan')
            runs[(x['seq'], x['variant'])].append(float(x['ate_se3']) if ok else np.nan)
    return runs


def median(a):
    a = np.asarray(a, float)
    return np.nanmedian(a) if np.isfinite(a).sum() * 2 > len(a) else np.nan


def compare(runs, ref, var, seqs, thr, n_boot, rng):
    pairs = [(s, median(runs[(s, ref)]), median(runs[(s, var)])) for s in seqs if runs.get((s, ref)) and runs.get((s, var))]
    pairs = [p for p in pairs if np.isfinite(p[1]) and np.isfinite(p[2])]
    n = len(pairs)
    if n < 2:
        return None
    d = np.log([v / r for _, r, v in pairs])
    lo, hi = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d)) if d.std() > 0 else (d.mean(), d.mean())
    boot = np.exp(np.percentile([rng.choice(d, n).mean() for _ in range(n_boot)], [2.5, 97.5]))
    p = stats.wilcoxon(d).pvalue if n >= 6 and np.any(d != 0) else np.nan
    allr = np.concatenate([np.asarray(runs[(s, var)], float) for s in seqs if runs.get((s, var))])
    return dict(n=n, lower=int((d < 0).sum()), ratio=float(np.exp(d.mean())), lo=float(np.exp(lo)), hi=float(np.exp(hi)),
                blo=float(boot[0]), bhi=float(boot[1]), p=float(p), bad=int((~(allr <= thr)).sum()), runs=len(allr))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('metrics', nargs='+')
    ap.add_argument('--ref', required=True)
    ap.add_argument('--variants', nargs='*')
    ap.add_argument('--exclude', nargs='*', default=[], help='sequences left out')
    ap.add_argument('--only', nargs='*', help='sequences kept (default: all)')
    ap.add_argument('--thr', type=float, default=1.0, help='run ATE threshold in metres for the failure count')
    ap.add_argument('--boot', type=int, default=10000)
    ap.add_argument('--tex', help='also write the rows as a LaTeX tabular body')
    a = ap.parse_args()
    runs = load(a.metrics)
    seqs = sorted({s for s, _ in runs} - set(a.exclude))
    if a.only:
        seqs = [s for s in seqs if s in a.only]
    variants = a.variants or sorted({v for _, v in runs} - {a.ref})
    rng = np.random.default_rng(1)
    r0 = np.concatenate([np.asarray(runs[(s, a.ref)], float) for s in seqs if runs.get((s, a.ref))])
    print(f'reference {a.ref}: {len(seqs)} sequences, runs above {a.thr:g} m or incomplete: {int((~(r0 <= a.thr)).sum())} of {len(r0)}')
    print(f"{'variant':30s} {'lower':>7s} {'ratio':>6s} {'95% CI (t)':>14s} {'95% CI (boot)':>14s} {'Wilcoxon':>8s} {'runs > thr':>11s}")
    tex = []
    for v in variants:
        c = compare(runs, a.ref, v, seqs, a.thr, a.boot, rng)
        if c is None:
            print(f'{v:30s} fewer than two sequences in common'); continue
        print(f"{v:30s} {c['lower']:3d}/{c['n']:<3d} {c['ratio']:6.2f} [{c['lo']:5.2f},{c['hi']:6.2f}] [{c['blo']:5.2f},{c['bhi']:6.2f}] "
              f"{c['p']:8.3f} {c['bad']:5d}/{c['runs']:<5d}")
        tex.append(f"{v.replace('_', chr(92) + '_')} & {c['lower']}/{c['n']} & {c['ratio']:.2f} & [{c['lo']:.2f}, {c['hi']:.2f}] & "
                   f"{c['p']:.2g} & {c['bad']}/{c['runs']} \\\\")
    if a.tex:
        open(a.tex, 'w').write('\n'.join(tex) + '\n')


if __name__ == '__main__':
    main()
