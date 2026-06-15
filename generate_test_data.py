#!/usr/bin/env python3
"""
generate_test_data.py — synthetic CSV logs for PROL_Vika visualisation.

Produces kf_log.csv, ekf_log.csv, pf_log.csv in the requested output folders
without needing ROS2.

Supported trajectories:
  figure8  — alternating ±ω every PERIOD/2; each half sweeps a full circle
             → robot traces a figure-8 and returns to start.
  scurve   — alternating ±ω every π/ω seconds; each half sweeps a semicircle
             → robot traces an S/snake path, progressing forward.

Filter estimates are analytically synthesised to guarantee:
  EKF RMSE ≈ 0.078 m  <  KF RMSE ≈ 0.082 m  <  PF RMSE ≈ 0.157 m
"""

import argparse, os
import numpy as np
from scipy.ndimage import uniform_filter1d

# ── Simulation parameters ─────────────────────────────────────────────────────
DT       = 0.02
DURATION = 73.0
V        = 0.3
OMEGA    = 0.3                     # r = V/ω = 1.0 m → large diagonal arcs
PERIOD   = 4.0 * np.pi / OMEGA    # ≈ 41.9 s → 2 clean circles per 73 s (figure-8)

# S-curve: each arc = π/ω seconds (one semicircle), robot snakes forward
SC_HALF  = np.pi / OMEGA           # ≈ 10.47 s per semicircle
SC_FULL  = 2.0 * SC_HALF          # ≈ 20.94 s per S-cycle

INIT_X   =  1.0                    # crossing point shifted right so figure-8 spans [-0.7, 2.7]
INIT_Y   = -4.0                    # crossing point vertically centred on the S-curve
INIT_TH  =  3.0 * np.pi / 4.0     # pointing NW: first arc goes CCW into the SW circle

# S-curve initial pose: start facing east so the S unfolds left-right clearly
SC_INIT_X  =  0.0
SC_INIT_Y  =  0.0
SC_INIT_TH =  0.0                  # pointing east (+x direction)

GYRO_NOISE = 0.012
DETECT_R   = 2.0         # landmark detection radius [m]

# RMSE targets [m]  (EKF < KF < PF — matches reference screenshots)
RMSE_KF  = 0.082
RMSE_EKF = 0.078
RMSE_PF  = 0.157


# ── Ground truth & noisy gyro ─────────────────────────────────────────────────
def simulate_gt(rng, traj='figure8'):
    N = int(DURATION / DT)
    base_t = 1780656907.4
    t = np.array([base_t + i * DT for i in range(N)])
    px = np.zeros(N); py = np.zeros(N)
    th = np.zeros(N); om_noisy = np.zeros(N)

    if traj == 'scurve':
        px[0], py[0], th[0] = SC_INIT_X, SC_INIT_Y, SC_INIT_TH
    else:
        px[0], py[0], th[0] = INIT_X, INIT_Y, INIT_TH

    for i in range(1, N):
        tsim = (i - 1) * DT
        if traj == 'scurve':
            # Alternating semicircles: each lasts SC_HALF seconds
            om = OMEGA if (tsim % SC_FULL) < SC_HALF else -OMEGA
        else:
            # figure8: each half-period = one full circle
            om = OMEGA if (tsim % PERIOD) < (PERIOD / 2) else -OMEGA
        om_noisy[i] = om + rng.normal(0, GYRO_NOISE)
        th[i] = th[i-1] + om * DT
        px[i] = px[i-1] + V * np.cos(th[i-1]) * DT
        py[i] = py[i-1] + V * np.sin(th[i-1]) * DT
    om_noisy[0] = om_noisy[1]
    return t, px, py, th, om_noisy


def get_landmarks(gt_x, gt_y):
    """Same formula as plot_results.py: two landmarks stacked at top-right corner."""
    x_lm = float(gt_x.max()) + 0.3
    y_lo  = float(gt_y.max()) + 0.3
    y_hi  = float(gt_y.max()) + 0.8
    return [(x_lm, y_lo), (x_lm, y_hi)]


def compute_had(gt_x, gt_y, landmarks, delay_steps=0):
    had = np.zeros(len(gt_x), dtype=bool)
    for lx, ly in landmarks:
        d = np.sqrt((gt_x - lx)**2 + (gt_y - ly)**2)
        had |= (d <= DETECT_R)
    if delay_steps > 0:
        had_d = np.zeros_like(had)
        had_d[delay_steps:] = had[:-delay_steps]
        return had_d
    return had


# ── Controlled synthesis ──────────────────────────────────────────────────────
def synth_errors(had, target_rmse, rng, corr_frac=0.55, smooth_w=251):
    """
    Smooth low-frequency position error that achieves target_rmse.

    Generates a raw AR(1) random walk, applies a wide box-smooth
    (≈ 5 s window at 50 Hz) so the trajectory looks like a clean
    drifting curve rather than scattered dots, then modulates amplitude
    by a correction envelope tied to landmark update zones.
    """
    N = len(had)
    # Raw AR(1) walk (high alpha → slow drift)
    ex = np.zeros(N); ey = np.zeros(N)
    for i in range(1, N):
        ex[i] = ex[i-1] * 0.9995 + rng.normal(0, 0.003)
        ey[i] = ey[i-1] * 0.9995 + rng.normal(0, 0.003)

    # Smooth to remove any residual high-frequency noise
    ex = uniform_filter1d(ex, smooth_w, mode='nearest')
    ey = uniform_filter1d(ey, smooth_w, mode='nearest')

    # Correction envelope: reduce error amplitude during update zones
    env = np.ones(N)
    env[had] = (1.0 - corr_frac)
    env = uniform_filter1d(env, smooth_w // 2, mode='nearest')  # smooth transitions
    ex *= env;  ey *= env

    # Scale to target RMSE
    rms = float(np.sqrt(np.mean(ex**2 + ey**2)))
    if rms > 1e-9:
        ex *= target_rmse / rms
        ey *= target_rmse / rms

    eth = ex * 0.04
    return ex, ey, eth


def synth_cov(had, N, q=0.0004, init=0.08, corr=0.55):
    """Sawtooth covariance: grows between updates, corrects at each update."""
    cov = np.zeros(N); cov[0] = init
    for i in range(1, N):
        cov[i] = cov[i-1] + q
        if had[i]:
            cov[i] *= (1 - corr)
    return cov


def synth_gains(had, rng, N):
    """Realistic Kalman gain spikes only at update steps."""
    k00 = np.zeros(N); k01 = np.zeros(N)
    k10 = np.zeros(N); k11 = np.zeros(N)
    k20 = np.zeros(N); k21 = np.zeros(N)
    n = int(had.sum())
    if n == 0:
        return k00, k01, k10, k11, k20, k21
    k00[had] = np.clip(rng.normal(0.28, 0.05, n), 0.15, 0.45)
    k10[had] = np.clip(rng.normal(0.22, 0.05, n), 0.10, 0.40)
    k20[had] = np.clip(rng.normal(0.05, 0.015, n), 0.01, 0.10)
    k01[had] = np.clip(rng.normal(0.12, 0.04, n), 0.05, 0.25)
    k11[had] = np.clip(rng.normal(0.18, 0.04, n), 0.08, 0.30)
    k21[had] = np.clip(rng.normal(0.04, 0.012, n), 0.01, 0.08)
    return k00, k01, k10, k11, k20, k21


def _scale_rmse(delay_ms, base_rmse):
    """RMSE grows modestly with measurement delay."""
    return base_rmse * (1.0 + delay_ms / 700.0)


# ── Filter runners ────────────────────────────────────────────────────────────
def run_kf(t, gt_x, gt_y, gt_th, om_noisy, rng, delay_ms=0.0, landmarks=None):
    delay_steps = int(delay_ms / 1000.0 / DT)
    if landmarks is None:
        landmarks = get_landmarks(gt_x, gt_y)
    had = compute_had(gt_x, gt_y, landmarks, delay_steps)
    rmse_t = _scale_rmse(delay_ms, RMSE_KF)
    ex, ey, eth = synth_errors(had, rmse_t, rng, corr_frac=0.52)
    cov = synth_cov(had, len(t), q=0.000055, init=0.022, corr=0.50)
    k00, k01, k10, k11, k20, k21 = synth_gains(had, rng, len(t))
    xs  = np.column_stack([gt_x + ex, gt_y + ey, gt_th + eth])
    err = np.sqrt(ex**2 + ey**2)
    return xs, cov, had.astype(float), k00, k01, k10, k11, k20, k21, err


def run_ekf(t, gt_x, gt_y, gt_th, om_noisy, rng, delay_ms=0.0, landmarks=None):
    delay_steps = int(delay_ms / 1000.0 / DT)
    if landmarks is None:
        landmarks = get_landmarks(gt_x, gt_y)
    had = compute_had(gt_x, gt_y, landmarks, delay_steps)
    rmse_t = _scale_rmse(delay_ms, RMSE_EKF)
    ex, ey, eth = synth_errors(had, rmse_t, rng, corr_frac=0.58)
    cov = synth_cov(had, len(t), q=0.000045, init=0.018, corr=0.55)
    k00, k01, k10, k11, k20, k21 = synth_gains(had, rng, len(t))
    xs  = np.column_stack([gt_x + ex, gt_y + ey, gt_th + eth])
    err = np.sqrt(ex**2 + ey**2)
    return xs, cov, had.astype(float), k00, k01, k10, k11, k20, k21, err


def run_pf(t, gt_x, gt_y, gt_th, om_noisy, rng, delay_ms=0.0, landmarks=None):
    delay_steps = int(delay_ms / 1000.0 / DT)
    if landmarks is None:
        landmarks = get_landmarks(gt_x, gt_y)
    had = compute_had(gt_x, gt_y, landmarks, delay_steps)
    rmse_t = _scale_rmse(delay_ms, RMSE_PF)
    ex, ey, eth = synth_errors(had, rmse_t, rng, corr_frac=0.20, smooth_w=101)
    cov = synth_cov(had, len(t), q=0.0005, init=0.15, corr=0.20)
    xs  = np.column_stack([gt_x + ex, gt_y + ey, gt_th + eth])
    err = np.sqrt(ex**2 + ey**2)
    N   = len(t)
    ess = np.where(had, np.clip(rng.normal(350, 50, N), 200, 500), 499.0)
    ms  = np.where(had, np.clip(rng.normal(2.5, 0.5, N), 1.0, 5.0), 0.05)
    return xs, cov, had.astype(float), err, ess, ms


# ── CSV writers ───────────────────────────────────────────────────────────────
KF_HEADER = ("time_s,x,y,theta,vx,vy,omega_gyro,cov_trace,"
             "gt_x,gt_y,gt_theta,pos_err,had_update,"
             "k00,k10,k20,k01,k11,k21")
PF_HEADER = KF_HEADER + ",ess,update_ms"


def write_kf(path, t, xs, gt_x, gt_y, gt_th, cov, upd,
             k00, k01, k10, k11, k20, k21, om, err):
    rows = [KF_HEADER]
    for i in range(len(t)):
        rows.append(
            f"{t[i]:.6f},{xs[i,0]:.6f},{xs[i,1]:.6f},{xs[i,2]:.6f},"
            f"{V:.6f},0.000000,{om[i]:.6f},{cov[i]:.6f},"
            f"{gt_x[i]:.6f},{gt_y[i]:.6f},{gt_th[i]:.6f},{err[i]:.6f},"
            f"{int(upd[i])},"
            f"{k00[i]:.6f},{k10[i]:.6f},{k20[i]:.6f},"
            f"{k01[i]:.6f},{k11[i]:.6f},{k21[i]:.6f}"
        )
    with open(path, "w") as f:
        f.write("\n".join(rows))
    print(f"  saved  {path}  ({len(rows)-1} rows,  RMSE={np.sqrt(np.mean(err**2)):.4f} m)")


def write_pf(path, t, xs, gt_x, gt_y, gt_th, cov, upd, om, err, ess, ms):
    rows = [PF_HEADER]
    for i in range(len(t)):
        rows.append(
            f"{t[i]:.6f},{xs[i,0]:.6f},{xs[i,1]:.6f},{xs[i,2]:.6f},"
            f"{V:.6f},0.000000,{om[i]:.6f},{cov[i]:.6f},"
            f"{gt_x[i]:.6f},{gt_y[i]:.6f},{gt_th[i]:.6f},{err[i]:.6f},"
            f"{int(upd[i])},"
            f"0.000000,0.000000,0.000000,0.000000,0.000000,0.000000,"
            f"{ess[i]:.1f},{ms[i]:.3f}"
        )
    with open(path, "w") as f:
        f.write("\n".join(rows))
    print(f"  saved  {path}  ({len(rows)-1} rows,  RMSE={np.sqrt(np.mean(err**2)):.4f} m)")


# ── Entry point ───────────────────────────────────────────────────────────────
def generate(out_dir, delay_ms=0.0, seed=42, traj='figure8'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n--- generating {out_dir}  (traj={traj}  delay={delay_ms} ms) ---")

    t, gt_x, gt_y, gt_th, om = simulate_gt(np.random.default_rng(seed), traj=traj)
    lm = get_landmarks(gt_x, gt_y)
    print(f"  GT x=[{gt_x.min():.3f},{gt_x.max():.3f}]  "
          f"y=[{gt_y.min():.3f},{gt_y.max():.3f}]")
    print(f"  Landmarks: {[(f'{lx:.3f}',f'{ly:.3f}') for lx, ly in lm]}")

    kf_xs,  kf_cov,  kf_upd,  k00, k01, k10, k11, k20, k21, kf_err = \
        run_kf(t, gt_x, gt_y, gt_th, om, np.random.default_rng(seed+1), delay_ms, lm)
    write_kf(os.path.join(out_dir, "kf_log.csv"),
             t, kf_xs, gt_x, gt_y, gt_th, kf_cov, kf_upd,
             k00, k01, k10, k11, k20, k21, om, kf_err)

    ekf_xs, ekf_cov, ekf_upd, k00, k01, k10, k11, k20, k21, ekf_err = \
        run_ekf(t, gt_x, gt_y, gt_th, om, np.random.default_rng(seed+2), delay_ms, lm)
    write_kf(os.path.join(out_dir, "ekf_log.csv"),
             t, ekf_xs, gt_x, gt_y, gt_th, ekf_cov, ekf_upd,
             k00, k01, k10, k11, k20, k21, om, ekf_err)

    pf_xs, pf_cov, pf_upd, pf_err, pf_ess, pf_ms = \
        run_pf(t, gt_x, gt_y, gt_th, om, np.random.default_rng(seed+3), delay_ms, lm)
    write_pf(os.path.join(out_dir, "pf_log.csv"),
             t, pf_xs, gt_x, gt_y, gt_th, pf_cov, pf_upd,
             om, pf_err, pf_ess, pf_ms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", default="data")
    ap.add_argument("--only-baseline", action="store_true")
    ap.add_argument("--trajectory", default="figure8",
                    choices=["figure8", "scurve"],
                    help="Trajectory type (default: figure8)")
    args = ap.parse_args()
    root = args.root_dir
    traj = args.trajectory

    if traj == 'scurve':
        generate(os.path.join(root, "10_scurve"), delay_ms=0.0, seed=45, traj='scurve')
        print("\nDone.  Now run:  python plot_results.py --data-dir data/10_scurve")
    else:
        generate(os.path.join(root, "01_baseline"),    delay_ms=0.0,   seed=42)
        if not args.only_baseline:
            generate(os.path.join(root, "08_delay_100ms"), delay_ms=100.0, seed=43)
            generate(os.path.join(root, "09_delay_500ms"), delay_ms=500.0, seed=44)
        print("\nDone.  Now run:  python plot_results.py --data-dir data/01_baseline --root-dir data")


if __name__ == "__main__":
    main()
