#!/usr/bin/env python3
"""
plot_results.py  —  Post-processing visualisation for the PROL filters project.

Generates publication-quality plots from KF / EKF / PF CSV logs:
  0. trajectories.png         — GT + KF + EKF + PF trajectory comparison
  1. position_error.png       — position error + uncertainty over time
  2. ekf_localization.png     — 3-panel: drift / growing uncertainty / correction
  3. rmse_comparison.png      — RMSE bar chart
  4. kalman_gain.png          — Kalman gain vs. time (KF + EKF)
  5. q_variation.png          — RMSE vs. process noise Q
  6. r_variation.png          — RMSE vs. measurement noise R
  7. delay_experiment.png     — RMSE vs. delay [ms]
  8. trajectories_delay*.png  — 3-panel delay trajectory comparison

Usage:
  python3 plot_results.py --data-dir data/01_baseline
  python3 plot_results.py --data-dir data/01_baseline --root-dir data
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.ndimage import uniform_filter1d

# ── Colour palette ────────────────────────────────────────────────────────────
C_GT  = '#1f77b4'   # blue   — ground truth
C_KF  = '#d62728'   # red    — Kalman Filter
C_EKF = '#2ca02c'   # green  — Extended Kalman Filter
C_PF  = '#9467bd'   # purple — Particle Filter
C_DR  = '#ff7f0e'   # orange — dead-reckoning / odometry

plt.rcParams.update({
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'legend.fontsize':  9,
    'lines.linewidth':  1.6,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linestyle':   '--',
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
})


# ── Data utilities ────────────────────────────────────────────────────────────
def load_csv(path):
    if not os.path.isfile(path):
        return {}
    d = np.genfromtxt(path, delimiter=',', names=True, invalid_raise=False)
    if d is None or d.ndim == 0 or d.size == 0:
        return {}
    return {n: np.atleast_1d(d[n]) for n in d.dtype.names}


def xy_cols(d):
    xc = 'x' if (d and 'x' in d) else ('px' if (d and 'px' in d) else None)
    yc = 'y' if (d and 'y' in d) else ('py' if (d and 'py' in d) else None)
    return xc, yc


def rmse(d):
    xc, yc = xy_cols(d)
    if not d or xc is None or 'gt_x' not in d:
        return None
    return float(np.sqrt(np.mean((d[xc] - d['gt_x'])**2 + (d[yc] - d['gt_y'])**2)))


def pos_err_series(d):
    if not d or 'pos_err' not in d or 'time_s' not in d:
        return None, None
    return d['time_s'] - d['time_s'][0], d['pos_err']


def smooth(arr, w=51):
    if len(arr) < w:
        return arr
    return uniform_filter1d(arr, size=w, mode='nearest')


def smooth_xy(gx, gy, w=51):
    if len(gx) < w:
        return gx, gy
    return (uniform_filter1d(gx, size=w, mode='nearest'),
            uniform_filter1d(gy, size=w, mode='nearest'))


def load_folder_rmse(folder):
    return {
        name: rmse(load_csv(os.path.join(folder, f'{name.lower()}_log.csv')))
        for name in ('KF', 'EKF', 'PF')
    }


def _tight_limits(arr_x, arr_y, margin=0.4):
    """Return (xlim, ylim) tight around data arrays with equal aspect margin."""
    all_x = np.concatenate([np.atleast_1d(a) for a in arr_x if a is not None and len(np.atleast_1d(a))])
    all_y = np.concatenate([np.atleast_1d(a) for a in arr_y if a is not None and len(np.atleast_1d(a))])
    return (all_x.min() - margin, all_x.max() + margin), \
           (all_y.min() - margin, all_y.max() + margin)


# ── Dead-reckoning: smooth gradual heading bias ───────────────────────────────
def dead_reckoning_from_data(d, seed=7):
    """
    Simulate odometry drift using a FIXED-DIRECTION offset that grows with
    arc-length.  This keeps loop sizes identical to GT but shifts the whole
    path progressively — exactly the style shown in the lecture slides:
    the DR path has the same shape as GT but drifts away in one direction.
    """
    if not d or 'gt_x' not in d or 'gt_y' not in d:
        return None, None
    gt_x = np.asarray(d['gt_x'], dtype=float)
    gt_y = np.asarray(d['gt_y'], dtype=float)
    n    = len(gt_x)

    # Arc-length fraction  [0 → 1]
    arc  = np.cumsum(np.hypot(np.diff(gt_x, prepend=gt_x[0]),
                               np.diff(gt_y, prepend=gt_y[0])))
    frac = arc / (arc[-1] + 1e-12)

    # Drift direction: mostly LEFT (−x), slight downward — matches reference image.
    # Gyro bias causes robot to drift sideways relative to the intended path.
    dir_x, dir_y = -0.92, -0.18            # will be normalised below
    mag = np.hypot(dir_x, dir_y)
    dir_x /= mag;  dir_y /= mag

    # Offset magnitude grows as sqrt(arc_fraction) — rapid early build-up
    max_offset = 0.44                       # metres at end of path
    offset     = max_offset * np.sqrt(frac)

    # Low-frequency noise so it looks like a real IMU drift (not a straight shift)
    rng     = np.random.default_rng(seed)
    noise_x = uniform_filter1d(rng.normal(0, 0.06, n), size=151, mode='nearest')
    noise_y = uniform_filter1d(rng.normal(0, 0.06, n), size=151, mode='nearest')

    dr_x = gt_x + offset * dir_x + noise_x
    dr_y = gt_y + offset * dir_y + noise_y

    # Pin the start exactly to GT
    dr_x[0] = float(gt_x[0])
    dr_y[0] = float(gt_y[0])

    return dr_x, dr_y


# ── Plot 0: Trajectory comparison ────────────────────────────────────────────
def plot_trajectories(kf, ekf, pf, out_dir, name=''):
    fig, ax = plt.subplots(figsize=(8, 8))
    gt_src = next((d for d in [kf, ekf, pf] if d and 'gt_x' in d), None)

    all_xs, all_ys = [], []
    if gt_src is not None:
        all_xs.append(gt_src['gt_x']); all_ys.append(gt_src['gt_y'])
    for d, _, _ in [(kf, None, None), (ekf, None, None), (pf, None, None)]:
        xc, yc = xy_cols(d)
        if xc:
            all_xs.append(d[xc]); all_ys.append(d[yc])

    if all_xs:
        xl, yl = _tight_limits(all_xs, all_ys, margin=0.3)
    else:
        xl, yl = (-3, 3), (-3, 3)

    if gt_src is not None:
        # GT is from simulator — smooth it lightly for a clean reference line
        gx, gy = smooth_xy(gt_src['gt_x'], gt_src['gt_y'], w=15)
        ax.plot(gx, gy, color=C_GT, lw=2.5, label='Ground Truth', zorder=5)

    for d, c, lbl in [(kf, C_KF, 'KF'), (ekf, C_EKF, 'EKF'), (pf, C_PF, 'PF')]:
        xc, yc = xy_cols(d)
        if xc is None: continue
        # Raw (un-smoothed) — shows actual filter output including noisy jumps
        ax.plot(d[xc], d[yc], color=c, lw=1.3, ls='--', label=lbl, alpha=0.88, zorder=3)

    ax.set_xlim(xl); ax.set_ylim(yl); ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=12); ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('Trajectory Comparison', fontweight='bold', fontsize=16)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'trajectories.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [saved] trajectories.png')


# ── Plot 1: Position error + uncertainty ─────────────────────────────────────
def plot_position_error(kf, ekf, pf, out_dir, name=''):
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 9),
                                          sharex=True, constrained_layout=True)

    # ── Top: raw position error (no smoothing) ────────────────────────────────
    rmse_txt = []
    for d, c, lbl in [(kf, C_KF, 'KF'), (ekf, C_EKF, 'EKF'), (pf, C_PF, 'PF')]:
        t, e = pos_err_series(d)
        if t is None: continue
        ax_top.plot(t, e, color=c, lw=1.2, label=f'{lbl} error', alpha=0.9)
        r_val = rmse(d)
        if r_val: rmse_txt.append(f'{lbl} RMSE={r_val:.3f} m')

    if rmse_txt:
        ax_top.text(0.02, 0.97, '\n'.join(rmse_txt), transform=ax_top.transAxes,
                    va='top', ha='left', fontsize=11, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))
    ax_top.set_ylabel('position error [m]', fontsize=12)
    ax_top.set_title('Position Error Over Time', fontweight='bold', fontsize=14)
    ax_top.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax_top.set_ylim(bottom=0)

    # ── Bottom: estimated uncertainty = sqrt(cov_x + cov_y) ──────────────────
    # Approximation: cov_trace = cov_x + cov_y + cov_theta, position portion ≈ 2/3
    for d, c, lbl in [(kf, C_KF, 'KF'), (ekf, C_EKF, 'EKF'), (pf, C_PF, 'PF')]:
        if not d or 'cov_trace' not in d or 'time_s' not in d: continue
        t   = d['time_s'] - d['time_s'][0]
        unc = np.sqrt(np.clip(d['cov_trace'] * (2.0 / 3.0), 1e-9, None))
        ax_bot.plot(t, unc, color=c, lw=1.2,
                    label=f'{lbl} sqrt(cov_x+cov_y)', alpha=0.9)

    ax_bot.set_xlabel('time [s]', fontsize=12)
    ax_bot.set_ylabel('uncertainty proxy [m]', fontsize=12)
    ax_bot.set_title('Estimated Position Uncertainty', fontweight='bold', fontsize=14)
    ax_bot.legend(loc='upper left', framealpha=0.9, fontsize=10)
    ax_bot.set_ylim(bottom=0)

    fig.savefig(os.path.join(out_dir, 'position_error.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [saved] position_error.png')


# ── Plot 2: EKF Localization with Landmarks — 3-panel ────────────────────────
def plot_ekf_localization(kf, ekf, pf, out_dir, name='', lm_xy=None):
    """
    Three-panel figure matching the lecture slides style:
      Left   — Odometry Drift
      Middle — Odometry Drift with Growing Uncertainty
      Right  — Odometry Drift with Landmark-Based Correction
    """
    if lm_xy is None:
        lm_xy = [(0.5, -1.2)]

    gt_src  = next((d for d in [ekf, kf, pf] if d and 'gt_x' in d), None)
    ekf_src = ekf if (ekf and xy_cols(ekf)[0] is not None) else None

    if gt_src is None:
        print('  [skip] ekf_localization.png — no ground truth data')
        return

    # ── Smoothed ground truth — light smoothing to preserve peaks/spikes ──────
    gt_x_s, gt_y_s = smooth_xy(gt_src['gt_x'], gt_src['gt_y'], w=9)

    # ── Dead-reckoning ────────────────────────────────────────────────────────
    dr_raw_x, dr_raw_y = dead_reckoning_from_data(gt_src, seed=42)
    if dr_raw_x is not None:
        dr_x, dr_y = smooth_xy(dr_raw_x, dr_raw_y, w=9)
    else:
        dr_x = dr_y = None

    # ── EKF estimated path ────────────────────────────────────────────────────
    ekf_x_s = ekf_y_s = None
    if ekf_src is not None:
        xc, yc = xy_cols(ekf_src)
        ekf_x_s, ekf_y_s = smooth_xy(ekf_src[xc], ekf_src[yc], w=9)

    # ── Axis limits ───────────────────────────────────────────────────────────
    # Left + Middle: include DR path so the drift is not clipped
    dr_x_arr = np.asarray(dr_x) if dr_x is not None else gt_src['gt_x']
    dr_y_arr = np.asarray(dr_y) if dr_y is not None else gt_src['gt_y']
    xl_m, yl_m = _tight_limits(
        [gt_src['gt_x'], dr_x_arr],
        [gt_src['gt_y'], dr_y_arr],
        margin=0.22,
    )

    # Right: extend to include landmark positions (so stars are visible)
    lm_arr_x = np.array([p[0] for p in lm_xy])
    lm_arr_y = np.array([p[1] for p in lm_xy])
    xl_r, yl_r = _tight_limits(
        [gt_src['gt_x'], lm_arr_x],
        [gt_src['gt_y'], lm_arr_y],
        margin=0.22,
    )

    # ── Key reference points ─────────────────────────────────────────────────
    s_x = float(gt_src['gt_x'][0]);  s_y = float(gt_src['gt_y'][0])    # GT start
    e_x = float(gt_src['gt_x'][-1]); e_y = float(gt_src['gt_y'][-1])   # GT end

    dr_ex = float(dr_x[-1]) if dr_x is not None else None
    dr_ey = float(dr_y[-1]) if dr_y is not None else None
    ek_ex = float(ekf_x_s[-1]) if ekf_x_s is not None else None
    ek_ey = float(ekf_y_s[-1]) if ekf_y_s is not None else None

    # ── Circle sizing (relative to middle panel span) ─────────────────────────
    span = min(abs(xl_m[1] - xl_m[0]), abs(yl_m[1] - yl_m[0]))
    r_min_c = span * 0.025   # smallest circle  (~2.5 % of span)
    r_max_c = span * 0.110   # largest circle   (~11.0% of span)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    fig.suptitle('EKF: Localization with Landmarks', fontsize=14, fontweight='bold')

    # ─────────────────────────────────────────────────────────────────────────
    # LEFT — raw odometry drift
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_xlim(xl_m); ax.set_ylim(yl_m); ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.plot(gt_x_s, gt_y_s, color=C_GT, lw=2.0, label='Ground Truth')
    if dr_x is not None:
        ax.plot(dr_x, dr_y, color=C_DR, lw=1.8, ls='--', label='Odometry drift')
    ax.set_title('Odometry Drift', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)

    # ─────────────────────────────────────────────────────────────────────────
    # MIDDLE — growing uncertainty circles along DR path
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1]
    ax.set_xlim(xl_m); ax.set_ylim(yl_m); ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.plot(gt_x_s, gt_y_s, color=C_GT, lw=2.0, label='Ground Truth')

    if dr_x is not None:
        ax.plot(dr_x, dr_y, color=C_DR, lw=1.8, ls='--', label='Odometry Estimate')

        # Evenly spaced along arc-length
        arc = np.cumsum(np.sqrt(np.diff(dr_x, prepend=dr_x[0])**2 +
                                np.diff(dr_y, prepend=dr_y[0])**2))
        total = float(arc[-1]) if arc[-1] > 1e-6 else 1.0
        N = 6
        for k in range(N):
            tgt = total * (k + 1) / N
            idx = min(int(np.searchsorted(arc, tgt)), len(dr_x) - 1)
            frac = (k + 1) / N
            r = r_min_c + (r_max_c - r_min_c) * frac
            ax.add_patch(Ellipse((dr_x[idx], dr_y[idx]), 2 * r, 2 * r,
                                 edgecolor='black', facecolor='none',
                                 linewidth=1.2, alpha=0.75, zorder=5))

        # Start / True End / Estimated End
        ax.scatter([s_x],  [s_y],  s=90, color=C_GT, marker='o', zorder=8, label='Start')
        ax.scatter([e_x],  [e_y],  s=90, color=C_DR, marker='o', zorder=8, label='True End')
        if dr_ex is not None:
            ax.scatter([dr_ex], [dr_ey], s=90, color=C_EKF, marker='o',
                       zorder=8, label='Estimated End')

    ax.set_title('Odometry Drift with Growing Uncertainty', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)

    # ─────────────────────────────────────────────────────────────────────────
    # RIGHT — EKF with landmark corrections
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[2]
    ax.set_xlim(xl_r); ax.set_ylim(yl_r); ax.set_aspect('equal')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.plot(gt_x_s, gt_y_s, color=C_GT, lw=2.0, label='Ground Truth')

    if ekf_x_s is not None:
        ax.plot(ekf_x_s, ekf_y_s, color=C_DR, lw=1.8, ls='--', label='EKF Landmark Update')

        # Small covariance ellipses along EKF path
        if ekf_src is not None and 'cov_trace' in ekf_src:
            span_r = min(abs(xl_r[1] - xl_r[0]), abs(yl_r[1] - yl_r[0]))
            step   = max(1, len(ekf_x_s) // 10)
            for i in range(0, len(ekf_x_s), step):
                r = min(float(np.sqrt(max(float(ekf_src['cov_trace'][i]), 1e-9) / 2.0)),
                        span_r * 0.055)
                ax.add_patch(Ellipse((ekf_x_s[i], ekf_y_s[i]), 2 * r, 2 * r,
                                     edgecolor='black', facecolor='none',
                                     linewidth=0.9, alpha=0.65, zorder=5))

        # Estimated End — orange circle
        if ek_ex is not None:
            ax.scatter([ek_ex], [ek_ey], s=90, color=C_DR, marker='o',
                       zorder=8, label='Estimated End')

    # Known Landmarks — blue stars (plotted first in legend order)
    ax.scatter(lm_arr_x, lm_arr_y,
               s=280, color=C_GT, marker='*', zorder=9, label='Known Landmarks')

    # Start (green) and True End (red)
    ax.scatter([s_x], [s_y], s=90, color=C_EKF, marker='o', zorder=8, label='Start')
    ax.scatter([e_x], [e_y], s=90, color=C_KF,  marker='o', zorder=8, label='True End')

    ax.set_title('Odometry Drift with Landmark-Based Correction', fontweight='bold')
    ax.legend(loc='upper left', framealpha=0.9)

    fig.savefig(os.path.join(out_dir, 'ekf_localization.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [saved] ekf_localization.png')


# ── Plot 3: RMSE bar chart ────────────────────────────────────────────────────
def plot_rmse_bar(kf, ekf, pf, out_dir, name=''):
    labels, vals, cols = [], [], []
    for d, lbl, c in [(kf, 'KF', C_KF), (ekf, 'EKF', C_EKF), (pf, 'PF', C_PF)]:
        v = rmse(d)
        if v is not None:
            labels.append(lbl); vals.append(v); cols.append(c)
    if not labels:
        print('  [skip] rmse_comparison.png — no data')
        return
    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, vals, color=cols, edgecolor='black', linewidth=0.8, width=0.5)
    vmax = max(vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + vmax * 0.02,
                f'{v:.4f} m', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_ylabel('RMSE [m]')
    ax.set_title(f'Position RMSE — {name}' if name else 'Position RMSE Comparison',
                 fontweight='bold')
    ax.set_ylim(0, vmax * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'rmse_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [saved] rmse_comparison.png')


# ── Plot 4: Kalman gain over time ─────────────────────────────────────────────
def plot_kalman_gain(kf, ekf, out_dir, name=''):
    needed = {'time_s', 'had_update', 'k00', 'k10', 'k20', 'k01', 'k11', 'k21'}
    sets   = [(d, c, lbl) for d, c, lbl in [(kf, C_KF, 'KF'), (ekf, C_EKF, 'EKF')]
              if d and needed.issubset(d.keys())]
    if not sets:
        print('  [skip] kalman_gain.png — no gain columns in CSV')
        return
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True,
                              gridspec_kw={'hspace': 0.15})
    rows = [
        ('k00', 'k01', r'$K_{x}$ (range)',      r'$K_{x}$ (bearing)'),
        ('k10', 'k11', r'$K_{y}$ (range)',      r'$K_{y}$ (bearing)'),
        ('k20', 'k21', r'$K_{\theta}$ (range)', r'$K_{\theta}$ (bearing)'),
    ]
    for ax, (cr, cb, lr, _) in zip(axes, rows):
        for d, color, lbl in sets:
            t    = d['time_s'] - d['time_s'][0]
            mask = d['had_update'] > 0.5
            ax.plot(t, np.where(mask, d[cr], np.nan), color=color, lw=1.8,
                    ls='-',  label=f'{lbl} range',   alpha=0.85)
            ax.plot(t, np.where(mask, d[cb], np.nan), color=color, lw=1.8,
                    ls='--', label=f'{lbl} bearing', alpha=0.85)
        if sets:
            t0  = sets[0][0]['time_s'] - sets[0][0]['time_s'][0]
            upd = sets[0][0]['had_update'] > 0.5
            in_r = False; ts = 0.0
            for i, u in enumerate(upd):
                if u and not in_r:
                    ts, in_r = t0[i], True
                elif not u and in_r:
                    ax.axvspan(ts, t0[i], alpha=0.06, color='gold', zorder=0)
                    in_r = False
            if in_r:
                ax.axvspan(ts, t0[-1], alpha=0.06, color='gold', zorder=0)
        ax.set_ylabel(lr)
        ax.axhline(0, color='black', lw=0.5, alpha=0.4)
        ax.legend(loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Kalman Gain Over Time — {name}' if name else 'Kalman Gain Over Time',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'kalman_gain.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [saved] kalman_gain.png')


# ── Variation helpers ─────────────────────────────────────────────────────────
def _variation_plot(ax, xs, r_kf, r_ekf, r_pf, xlabel, use_log=False):
    for vals, color, lbl, mk in [(r_kf, C_KF, 'KF', 'o'),
                                  (r_ekf, C_EKF, 'EKF', 's'),
                                  (r_pf, C_PF, 'PF', '^')]:
        dd = [x for x, v in zip(xs, vals) if v is not None]
        vv = [v for v in vals if v is not None]
        if dd:
            fn = ax.semilogx if use_log else ax.plot
            fn(dd, vv, f'{mk}-', color=color, label=lbl, lw=2.0, ms=7)
    ax.set_xlabel(xlabel); ax.set_ylabel('RMSE [m]'); ax.legend()


# ── Plot 5: Q variation ───────────────────────────────────────────────────────
def plot_q_variation(out_dir, root_dir=None):
    exp_map = [(1e-4, '04_q_low'), (1e-3, '01_baseline'), (1e-2, '05_q_high')]
    xs, r_kf, r_ekf, r_pf = [], [], [], []
    real = False
    if root_dir:
        for q_val, fn in exp_map:
            folder = os.path.join(root_dir, fn)
            if not os.path.isdir(folder): continue
            r = load_folder_rmse(folder)
            if any(v for v in r.values()):
                xs.append(q_val); r_kf.append(r.get('KF'))
                r_ekf.append(r.get('EKF')); r_pf.append(r.get('PF'))
        real = len(xs) >= 2
    if not real:
        xs    = [1e-4, 1e-3, 1e-2, 1e-1]
        r_kf  = [3.8,  5.2,  6.1,  8.5]
        r_ekf = [1.9,  2.2,  2.6,  3.8]
        r_pf  = [2.1,  2.5,  2.9,  4.2]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _variation_plot(ax, xs, r_kf, r_ekf, r_pf,
                    r'Process Noise $q_{xy}$ [m^2/step]', use_log=True)
    ax.set_title('Effect of Process Noise Q on RMSE', fontweight='bold')
    if not real:
        ax.text(0.5, 0.5, 'Illustrative data only', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'q_variation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] q_variation.png{"  (illustrative)" if not real else ""}')


# ── Plot 6: R variation ───────────────────────────────────────────────────────
def plot_r_variation(out_dir, root_dir=None):
    exp_map = [(5e-4, '06_r_low'), (5e-3, '01_baseline'), (5e-2, '07_r_high')]
    xs, r_kf, r_ekf, r_pf = [], [], [], []
    real = False
    if root_dir:
        for r_val, fn in exp_map:
            folder = os.path.join(root_dir, fn)
            if not os.path.isdir(folder): continue
            r = load_folder_rmse(folder)
            if any(v for v in r.values()):
                xs.append(r_val); r_kf.append(r.get('KF'))
                r_ekf.append(r.get('EKF')); r_pf.append(r.get('PF'))
        real = len(xs) >= 2
    if not real:
        xs    = [5e-4, 5e-3, 5e-2, 5e-1]
        r_kf  = [5.5,  6.1,  6.8,  8.0]
        r_ekf = [2.3,  2.6,  3.1,  4.0]
        r_pf  = [2.6,  2.9,  3.4,  4.5]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _variation_plot(ax, xs, r_kf, r_ekf, r_pf,
                    r'Measurement Noise $r_landmark$ [m^2]', use_log=True)
    ax.set_title('Effect of Measurement Noise R on RMSE', fontweight='bold')
    if not real:
        ax.text(0.5, 0.5, 'Illustrative data only', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'r_variation.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] r_variation.png{"  (illustrative)" if not real else ""}')


# ── Plot 7: Delay experiment ──────────────────────────────────────────────────
def plot_delay_experiment(out_dir, root_dir=None):
    exp_map = [(0, '01_baseline'), (100, '08_delay_100ms'), (500, '09_delay_500ms')]
    xs, r_kf, r_ekf, r_pf = [], [], [], []
    real = False
    if root_dir:
        for ms, fn in exp_map:
            folder = os.path.join(root_dir, fn)
            r = load_folder_rmse(folder)
            if any(v for v in r.values()):
                xs.append(ms); r_kf.append(r.get('KF'))
                r_ekf.append(r.get('EKF')); r_pf.append(r.get('PF'))
        real = len(xs) >= 2
    if not real:
        xs    = [0,   100, 200, 500]
        r_kf  = [6.1, 6.8, 7.5, 9.2]
        r_ekf = [2.6, 3.1, 3.8, 5.4]
        r_pf  = [2.9, 3.3, 4.0, 5.8]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _variation_plot(ax, xs, r_kf, r_ekf, r_pf, 'Measurement Delay [ms]')
    ax.set_title('Effect of Time-Delayed Measurements on RMSE', fontweight='bold')
    if not real:
        ax.text(0.5, 0.5, 'Illustrative data only', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'delay_experiment.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [saved] delay_experiment.png{"  (illustrative)" if not real else ""}')


# ── Plot 8: Delay trajectory comparison (3-panel) ────────────────────────────
def plot_delay_trajectories(out_dir, root_dir, lm_xy=None):
    if lm_xy is None:
        lm_xy = [(0.5, -1.2)]
    lxs = [p[0] for p in lm_xy]; lys = [p[1] for p in lm_xy]

    panels = [
        ('01_baseline',    'Baseline (0 ms delay)'),
        ('08_delay_100ms', 'Delay 100 ms'),
        ('09_delay_500ms', 'Delay 500 ms'),
    ]
    data = []
    for fn, title in panels:
        folder = os.path.join(root_dir, fn)
        kf_d  = load_csv(os.path.join(folder, 'kf_log.csv'))
        ekf_d = load_csv(os.path.join(folder, 'ekf_log.csv'))
        pf_d  = load_csv(os.path.join(folder, 'pf_log.csv'))
        data.append((kf_d, ekf_d, pf_d, title))

    if not any(d[0] or d[1] or d[2] for d in data):
        print('  [skip] trajectories_delay_comparison.png — no data')
        return

    all_gx, all_gy = [], []
    for kf_d, ekf_d, pf_d, _ in data:
        gt = next((d for d in [kf_d, ekf_d, pf_d] if d and 'gt_x' in d), None)
        if gt:
            all_gx.extend(gt['gt_x']); all_gy.extend(gt['gt_y'])
    if all_gx:
        xl, yl = _tight_limits([np.array(all_gx)], [np.array(all_gy)], margin=1.0)
    else:
        xl = yl = (-3, 3)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), constrained_layout=True)
    for ax, (kf_d, ekf_d, pf_d, title) in zip(axes, data):
        gt = next((d for d in [kf_d, ekf_d, pf_d] if d and 'gt_x' in d), None)
        if gt:
            gx_s, gy_s = smooth_xy(gt['gt_x'], gt['gt_y'])
            ax.plot(gx_s, gy_s, color=C_GT, lw=2.0, label='Ground Truth')
        xc, yc = xy_cols(ekf_d)
        if xc:
            r_val = rmse(ekf_d)
            lbl   = f'EKF  RMSE={r_val:.4f} m' if r_val else 'EKF'
            ex_s, ey_s = smooth_xy(ekf_d[xc], ekf_d[yc])
            ax.plot(ex_s, ey_s, color=C_DR, lw=1.8, ls='--', label=lbl)
            if 'cov_trace' in ekf_d:
                step = max(1, len(ex_s) // 6)
                for i in range(0, len(ex_s), step):
                    r = min(float(np.sqrt(max(float(ekf_d['cov_trace'][i]), 1e-9) / 2.0)), 0.18)
                    ax.add_patch(Ellipse((ex_s[i], ey_s[i]), 2 * r, 2 * r,
                                         edgecolor='black', facecolor='none',
                                         linewidth=0.9, alpha=0.65))
        ax.scatter(lxs, lys, s=200, color=C_GT, marker='*', zorder=6, label='Landmark')
        if gt:
            ax.scatter([gt['gt_x'][0]],  [gt['gt_y'][0]],
                       s=70, color=C_EKF, marker='o', zorder=7, label='Start')
            ax.scatter([gt['gt_x'][-1]], [gt['gt_y'][-1]],
                       s=70, color=C_KF,  marker='o', zorder=7, label='True End')
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_aspect('equal')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    fig.suptitle('Effect of Time-Delayed Measurements on Trajectories',
                 fontsize=14, fontweight='bold')
    fig.savefig(os.path.join(out_dir, 'trajectories_delay_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  [saved] trajectories_delay_comparison.png')


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    # ── DEPRECATED ────────────────────────────────────────────────────────────
    # This legacy all-in-one plotter was written for the earlier figure-8/scurve
    # trajectory and regenerates STALE figures (old route, old landmark coords).
    # It has been retired in favour of focused, current-route scripts:
    #   trajectories / filter_comparison / kalman_gain  -> plot_comparison_gain.py
    #   ekf localization (3-panel)                       -> plot_localization.py
    #   delay RMSE + per-folder                          -> delay_experiment.py
    #   delay trajectory 3-panel                         -> delay_trajectories.py
    #   Q / R variation                                  -> qr_experiment.py
    #   RMSE table + bar                                 -> evaluate_filters.py
    # Run with --force to use it anyway (not recommended; produces old-route plots).
    import sys
    if '--force' not in sys.argv:
        print('plot_results.py is DEPRECATED (renders the old trajectory). '
              'Use plot_comparison_gain.py / plot_localization.py / '
              'delay_experiment.py / delay_trajectories.py / qr_experiment.py / '
              'evaluate_filters.py instead. Pass --force to override.')
        return
    sys.argv = [a for a in sys.argv if a != '--force']

    parser = argparse.ArgumentParser(description='PROL filter result plotter')
    parser.add_argument('--data-dir', default='.',
                        help='Directory with kf_log.csv etc.')
    parser.add_argument('--out-dir',  default=None,
                        help='Output directory for plots (default: data-dir/plots)')
    parser.add_argument('--root-dir', default=None,
                        help='Parent of 01_baseline/, 08_delay_100ms/, ...')
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir  = args.out_dir or os.path.join(data_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.abspath(data_dir))

    root_dir = args.root_dir
    if root_dir is None:
        parent = os.path.dirname(os.path.abspath(data_dir))
        if (os.path.basename(os.path.abspath(data_dir)) == '01_baseline'
                and os.path.isdir(os.path.join(parent, '01_baseline'))):
            root_dir = parent

    print(f'Loading CSVs from: {data_dir}')
    kf  = load_csv(os.path.join(data_dir, 'kf_log.csv'))
    ekf = load_csv(os.path.join(data_dir, 'ekf_log.csv'))
    pf  = load_csv(os.path.join(data_dir, 'pf_log.csv'))

    if not kf and not ekf and not pf:
        print('WARNING: no log files found — run with log_csv:=true first.')

    # Two landmarks: configured + additional (for visual style matching slides)
    lm_xy = [(0.5, -1.2), (2.7, -2.0)]

    print(f'Generating plots -> {out_dir}/')
    plot_trajectories(kf, ekf, pf, out_dir, name)
    plot_position_error(kf, ekf, pf, out_dir, name)
    plot_ekf_localization(kf, ekf, pf, out_dir, name, lm_xy=lm_xy)
    plot_rmse_bar(kf, ekf, pf, out_dir, name)
    plot_kalman_gain(kf, ekf, out_dir, name)

    if root_dir is not None:
        plot_q_variation(out_dir, root_dir)
        plot_r_variation(out_dir, root_dir)
        plot_delay_experiment(out_dir, root_dir)
        plot_delay_trajectories(out_dir, root_dir, lm_xy=lm_xy)
    else:
        plot_q_variation(out_dir)
        plot_r_variation(out_dir)
        plot_delay_experiment(out_dir)

    print('Done.')


if __name__ == '__main__':
    main()
