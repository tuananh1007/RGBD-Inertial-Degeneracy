#!/usr/bin/env python3
"""
Points per frame at each stage of the depth update, per sampling policy: does the voxel
filter after selection leave the policies with the same registered set?

Reads the run logs OUT_ROOT/<variant>_r<k>/<seq>.log (written by run_sequences.sh) and, per variant and sequence, takes
the median over frames and runs of
  selected    points returned by the selector            ("[Timing] depth selection ... s, N points")
  voxel       points after the voxel filter              ("[ LIO ] ... downsampled feature num:N ...", last iteration)
  matched     points matched to a map plane              ("... effective feature num: N ...", last iteration)
  rho_t       translation conditioning ratio             ("[DTF] ... ratio_trans=R ...")
The "[ LIO ]" line is printed once per filter iteration; the last one before the next "[Timing]" line is the frame's.

  point_counts.py SWEEP/utmm SWEEP/openloris --variants s_ours_lin u_ours_lin ours_lin lin_noitps
"""
import argparse
import collections
import re
from pathlib import Path

import numpy as np

SEL = re.compile(r'\[Timing\] depth selection [\d.eE+-]+ s, (\d+) points')
LIO = re.compile(r'downsampled feature num:\s*(\d+)\s+effective feature num:\s*(\d+)')
DTF = re.compile(r'\[DTF\].*ratio_trans=([\d.eE+-]+)')


def frames(path):
    sel, vox, mat, rho = [], [], [], []
    last = None
    for line in open(path, errors='replace'):
        m = SEL.search(line)
        if m:
            if last:
                vox.append(last[0]); mat.append(last[1]); last = None
            sel.append(int(m.group(1))); continue
        m = LIO.search(line)
        if m:
            last = (int(m.group(1)), int(m.group(2))); continue
        m = DTF.search(line)
        if m:
            rho.append(float(m.group(1)))
    if last:
        vox.append(last[0]); mat.append(last[1])
    return sel, vox, mat, rho


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('roots', nargs='+')
    ap.add_argument('--variants', nargs='*')
    a = ap.parse_args()
    data = collections.defaultdict(lambda: [[], [], [], []])
    for root in a.roots:
        for log in sorted(Path(root).glob('*_r*/*.log')):
            variant = re.sub(r'_r\d+$', '', log.parent.name)
            if a.variants and variant not in a.variants:
                continue
            for acc, vals in zip(data[(variant, log.stem)], frames(log)):
                acc.extend(vals)
    med = lambda v: float(np.median(v)) if len(v) else float('nan')   # noqa: E731
    print(f"{'sequence':28s} {'variant':26s} {'selected':>9s} {'voxel':>7s} {'matched':>8s} {'rho_t':>7s} {'frames':>7s}")
    per_var = collections.defaultdict(list)
    for (variant, seq), (sel, vox, mat, rho) in sorted(data.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = (med(sel), med(vox), med(mat), med(rho))
        per_var[variant].append(row)
        print(f'{seq:28s} {variant:26s} {row[0]:9.0f} {row[1]:7.0f} {row[2]:8.0f} {row[3]:7.3f} {len(vox):7d}')
    print('\nmedian over sequences')
    for variant, rows in sorted(per_var.items()):
        r = np.nanmedian(np.array(rows), axis=0)
        print(f"{'':28s} {variant:26s} {r[0]:9.0f} {r[1]:7.0f} {r[2]:8.0f} {r[3]:7.3f} {len(rows):7d}")


if __name__ == '__main__':
    main()
