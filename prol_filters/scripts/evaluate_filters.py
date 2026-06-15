#!/usr/bin/env python3
"""
evaluate_filters.py  —  Quantitative evaluation of KF / EKF / PF filter results.

Reads kf_log.csv, ekf_log.csv, pf_log.csv from a data directory and prints
a metrics table + saves evaluation_summary.csv + generates rmse_bar.png.

Metrics:
  RMSE          — root mean square 2D position error [m]
  MAE           — mean absolute 2D position error [m]
  MaxErr        — maximum position error [m]
  FinalErr      — position error at last timestep [m]
  HeadingRMSE   — heading RMSE [rad] and [deg]
  CovTrace/ESS  — mean covariance trace (KF/EKF) or mean ESS (PF)
  UpdateMs      — mean per-step computation time [ms] (PF only)

Usage:
  python3 evaluate_filters.py
  python3 evaluate_filters.py --data-dir /path/to/data/01_baseline
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_KF  = 'tab:blue'    # Elias scheme: KF blue / EKF green / PF red
C_EKF = 'tab:green'
C_PF  = 'tab:red'


def load_csv(path):
    """Load a CSV log file into a dict of numpy arrays, or None if missing/empty."""
    if not os.path.isfile(path):
        return None
    d = np.genfromtxt(path, delimiter=',', names=True, invalid_raise=False)
    if d is None or d.size == 0 or d.dtype.names is None:
        return None
    return {n: np.atleast_1d(d[n]) for n in d.dtype.names}


def pos_error(d):
    """Euclidean 2D position error at each timestep."""
    xc = 'x' if 'x' in d else 'px'
    yc = 'y' if 'y' in d else 'py'
    return np.sqrt((d[xc] - d['gt_x'])**2 + (d[yc] - d['gt_y'])**2)


def heading_rmse(d):
    """Heading RMSE [rad], wrapped to [-π, π]."""
    if d is None or 'theta' not in d or 'gt_theta' not in d:
        return None
    diff = (d['theta'] - d['gt_theta'] + np.pi) % (2 * np.pi) - np.pi
    return float(np.sqrt(np.mean(diff**2)))


def compute_metrics(d, name):
    """Return a dict of metric strings for one filter."""
    if d is None or 'gt_x' not in d:
        return {k: 'N/A' for k in
                ['filter', 'RMSE', 'MAE', 'MaxErr', 'FinalErr',
                 'HeadingRMSE_deg', 'CovTrace/ESS', 'UpdateMs']}

    err  = pos_error(d)
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(err))
    me   = float(np.max(err))
    fe   = float(err[-1])
    he   = heading_rmse(d)

    if 'cov_trace' in d:
        unc = f'{float(np.mean(d["cov_trace"])):.5f} (cov_trace)'
    elif 'ess' in d:
        n_p = len(d['ess'])
        unc = f'{float(np.mean(d["ess"])):.1f} / {n_p} (ESS/N)'
    else:
        unc = 'N/A'

    upd = f'{float(np.mean(d["update_ms"])):.4f}' if 'update_ms' in d else 'N/A'
    he_deg = f'{float(np.degrees(he)):.3f}' if he is not None else 'N/A'

    return {
        'filter':         name,
        'RMSE':           f'{rmse:.5f}',
        'MAE':            f'{mae:.5f}',
        'MaxErr':         f'{me:.5f}',
        'FinalErr':       f'{fe:.5f}',
        'HeadingRMSE_deg': he_deg,
        'CovTrace/ESS':   unc,
        'UpdateMs':       upd,
    }


def print_table(rows):
    cols = ['filter', 'RMSE', 'MAE', 'MaxErr', 'FinalErr',
            'HeadingRMSE_deg', 'CovTrace/ESS', 'UpdateMs']
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    header = '  '.join(c.ljust(widths[c]) for c in cols)
    sep    = '  '.join('-' * widths[c] for c in cols)
    print(header)
    print(sep)
    for r in rows:
        print('  '.join(str(r[c]).ljust(widths[c]) for c in cols))


def save_csv(rows, path):
    cols = ['filter', 'RMSE', 'MAE', 'MaxErr', 'FinalErr',
            'HeadingRMSE_deg', 'CovTrace/ESS', 'UpdateMs']
    with open(path, 'w') as f:
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')


def save_rmse_bar(rows, out_dir, title=''):
    """Save rmse_bar.png with RMSE values for each filter."""
    cmap  = {'KF': C_KF, 'EKF': C_EKF, 'PF': C_PF}
    names, vals, cols = [], [], []
    for r in rows:
        if r['RMSE'] != 'N/A':
            names.append(r['filter'])
            vals.append(float(r['RMSE']))
            cols.append(cmap.get(r['filter'], '#888888'))
    if not names:
        return

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(names, vals, color=cols, edgecolor='black', linewidth=0.8, width=0.55)
    vmax = max(vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vmax * 0.02,
                f'{v:.4f} m', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE [m]')
    ax.set_title('Position RMSE: KF vs EKF vs PF', fontweight='bold')
    ax.set_ylim(0, vmax * 1.25)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.join(out_dir, 'plots'), exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'plots', 'rmse_bar.png'), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Evaluate PROL filter results')
    parser.add_argument('--data-dir', default='.', help='Directory containing *_log.csv files')
    args  = parser.parse_args()
    ddir  = args.data_dir
    exp   = os.path.basename(os.path.abspath(ddir))

    kf  = load_csv(os.path.join(ddir, 'kf_log.csv'))
    ekf = load_csv(os.path.join(ddir, 'ekf_log.csv'))
    pf  = load_csv(os.path.join(ddir, 'pf_log.csv'))

    sep = '=' * 65
    print(f'\n{sep}')
    print(f'  PROL Filter Evaluation  —  {exp}')
    print(sep)

    rows = [compute_metrics(kf, 'KF'), compute_metrics(ekf, 'EKF'), compute_metrics(pf, 'PF')]

    print('\nPosition and Heading Metrics:')
    print_table(rows)

    print(f'\nSample counts:')
    for d, name in [(kf, 'KF'), (ekf, 'EKF'), (pf, 'PF')]:
        xc = 'x' if (d and 'x' in d) else ('px' if (d and 'px' in d) else None)
        n  = len(d[xc]) if d and xc else 0
        print(f'  {name:4s}  {n} timesteps')

    out_csv = os.path.join(ddir, 'evaluation_summary.csv')
    save_csv(rows, out_csv)
    print(f'\nSaved: {out_csv}')

    save_rmse_bar(rows, ddir, exp)
    print(f'Saved: {os.path.join(ddir, "plots", "rmse_bar.png")}')


if __name__ == '__main__':
    main()
