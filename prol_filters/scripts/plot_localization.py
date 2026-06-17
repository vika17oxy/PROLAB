#!/usr/bin/env python3
"""
plot_localization.py — "EKF: Localization with Landmarks" figure, in the exact
style of the PRO Localization lecture slide, but on the current S-slalom route
(lower-left -> upper-right, same waypoints the simulator drives in RViz):

  1. Odometry Drift                              (Ground Truth vs Odometry drift)
  2. Odometry Drift with Growing Uncertainty     (+ growing covariance ellipses)
  3. Odometry Drift with Landmark-Based Correction (+ known landmarks, EKF snaps back)

Run:  python3 plot_localization.py --out /data/.../ekf_localization.png
"""
import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ── Current route (must match imu_simulator.py waypoint defaults) ─────────────
WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]
WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LANDMARKS = [(1.8, 3.0)]    # single known landmark, off the path (matches the rest of the study)
V, DT = 0.12, 0.02
GYRO_BIAS = 0.007            # systematic heading drift (odometry error source)
GYRO_NOISE = 0.004
Q_XY, Q_TH = 1.5e-5, 8e-6      # process noise (sizes the growing ellipses)
R_RANGE, R_BEARING = 0.01, 0.01
LM_RADIUS = 12.0             # always in view
UPDATE_EVERY = 5            # landmark sensor at 10 Hz
RNG = np.random.default_rng(7)

# NOTE: this 3-panel figure is a didactic localization explainer (lecture-slide
# style). Q is kept small and a small gyro bias is injected so the odometry drift
# and the growing 2-sigma ellipses read clearly; these deviate ON PURPOSE from the
# node defaults (q_xy=0.001, gyro_noise=0.012) used by the quantitative figures.


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def build_path(xs, ys):
    pts = np.array(list(zip(xs, ys)), float)
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    al, dense = 0.5, []
    for i in range(1, len(P) - 2):
        P0, P1, P2, P3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(P1 - P0) ** al, 1e-6)
        t2 = t1 + max(np.linalg.norm(P2 - P1) ** al, 1e-6)
        t3 = t2 + max(np.linalg.norm(P3 - P2) ** al, 1e-6)
        for k in range(80):
            t = t1 + (t2 - t1) * k / 80.0
            A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
            A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
            A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            dense.append((t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2)
    dense.append(pts[-1])
    D = np.array(dense)
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    return D, np.concatenate([[0.0], np.cumsum(seg)])


def state_at(D, cum, s):
    s = min(max(s, 0.0), cum[-1])
    x = float(np.interp(s, cum, D[:, 0])); y = float(np.interp(s, cum, D[:, 1]))
    i = min(max(int(np.searchsorted(cum, s)), 1), len(D) - 2)
    return x, y, math.atan2(D[i + 1, 1] - D[i - 1, 1], D[i + 1, 0] - D[i - 1, 0])


def simulate(delay_steps=0):
    D, cum = build_path(WPX, WPY)
    n = int(cum[-1] / (V * DT))
    gt = np.zeros((n, 3)); odo = np.zeros((n, 3)); ekf = np.zeros((n, 3))
    P_odo = np.eye(3) * 0.01; P_ekf = np.eye(3) * 0.01
    cov_odo, cov_ekf = [], []
    pending = []   # buffered landmark obs (delay between measurement and processing)
    x0, y0, th0 = state_at(D, cum, 0.0)
    gt[0] = odo[0] = ekf[0] = [x0, y0, th0]
    prev_th = th0
    Q = np.diag([Q_XY, Q_XY, Q_TH]); R = np.diag([R_RANGE, R_BEARING])
    for k in range(1, n):
        gx, gy, gth = state_at(D, cum, k * V * DT); gt[k] = [gx, gy, gth]
        omega = wrap(gth - prev_th) / DT + GYRO_BIAS + RNG.normal(0, GYRO_NOISE)
        prev_th = gth

        def predict(state, Pm):
            th = state[2]; s = state.copy()
            s[0] += V * math.cos(th) * DT; s[1] += V * math.sin(th) * DT
            s[2] = wrap(s[2] + omega * DT)
            G = np.eye(3); G[0, 2] = -V * math.sin(th) * DT; G[1, 2] = V * math.cos(th) * DT
            return s, G @ Pm @ G.T + Q

        odo[k], P_odo = predict(odo[k - 1], P_odo)
        ekf[k], P_ekf = predict(ekf[k - 1], P_ekf)

        # Observe landmarks -> buffer (measurement)
        if k % UPDATE_EVERY == 0:
            for lx, ly in LANDMARKS:
                if math.hypot(lx - gx, ly - gy) < LM_RADIUS:
                    z_r = math.hypot(lx - gx, ly - gy) + RNG.normal(0, math.sqrt(R_RANGE))
                    z_b = wrap(math.atan2(ly - gy, lx - gx) - gth) + RNG.normal(0, math.sqrt(R_BEARING))
                    pending.append((k, lx, ly, z_r, z_b))
        # Process buffered measurements only after measurement_delay has elapsed
        while pending and (k - pending[0][0]) >= delay_steps:
            _, lx, ly, z_r, z_b = pending.pop(0)
            ex, ey, eth = ekf[k]; dx, dy = lx - ex, ly - ey; r = math.hypot(dx, dy)
            if r >= 1e-6:
                H = np.array([[-dx / r, -dy / r, 0.0], [dy / r / r, -dx / r / r, -1.0]])
                S = H @ P_ekf @ H.T + R; K = P_ekf @ H.T @ np.linalg.inv(S)
                innov = np.array([z_r - r, wrap(z_b - wrap(math.atan2(dy, dx) - eth))])
                new = ekf[k] + K @ innov; new[2] = wrap(new[2]); ekf[k] = new
                IKH = np.eye(3) - K @ H; P_ekf = IKH @ P_ekf @ IKH.T + K @ R @ K.T
        cov_odo.append(P_odo[:2, :2].copy()); cov_ekf.append(P_ekf[:2, :2].copy())
    return gt, odo, ekf, cov_odo, cov_ekf


def draw_ellipse(ax, mean, cov, nsig=2.0):
    # 2σ covariance ellipse via eigendecomposition.
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]; vals = vals[order]; vecs = vecs[:, order]
    w = min(2 * nsig * math.sqrt(vals[0]), 6.0)
    h = min(2 * nsig * math.sqrt(vals[1]), 6.0)
    ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    ax.add_patch(Ellipse(mean, w, h, angle=ang, fill=False, ec="black", lw=0.9, alpha=0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/localization_plot.png")
    ap.add_argument("--delay-ms", type=float, default=0.0,
                    help="measurement->processing delay [ms] for the EKF panel")
    args = ap.parse_args()
    gt, odo, ekf, cov_odo, cov_ekf = simulate(int(round(args.delay_ms / 1000.0 / DT)))
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    TRUTH, ODO, EKFC, LM = "k", "#ff7f0e", "#2ca02c", "#1f77b4"
    M_START, M_TRUE, M_EST = "#2ca02c", "#ff7f0e", "#d62728"

    def setup(a, title):
        a.set_aspect("equal"); a.grid(True, alpha=0.3)
        a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.set_title(title)

    # Panel 1 — Odometry drift (dead reckoning, no correction)
    ax[0].plot(gt[:, 0], gt[:, 1], "-", color=TRUTH, lw=2.2, label="Ground Truth")
    ax[0].plot(odo[:, 0], odo[:, 1], "--", color=ODO, lw=2.0, label="Odometry (dead reckoning)")
    setup(ax[0], "Odometry drift\n(dead reckoning, no correction)"); ax[0].legend(loc="upper left", fontsize=10)

    # Panel 2 — Odometry drift with growing uncertainty
    ax[1].plot(gt[:, 0], gt[:, 1], "-", color=TRUTH, lw=2.2, label="Ground Truth")
    ax[1].plot(odo[:, 0], odo[:, 1], "--", color=ODO, lw=2.0, label="Odometry (dead reckoning)")
    step = max(1, len(cov_odo) // 12)
    for k in range(step, len(cov_odo), step):
        draw_ellipse(ax[1], odo[k + 1, :2], cov_odo[k])
    ax[1].plot(*gt[0, :2], "o", color=M_START, ms=10, label="Start")
    ax[1].plot(*gt[-1, :2], "o", color=M_TRUE, ms=10, label="True End")
    ax[1].plot(*odo[-1, :2], "o", color=M_EST, ms=10, label="Estimated End")
    setup(ax[1], "Odometry drift with growing uncertainty"); ax[1].legend(loc="upper left", fontsize=10)

    # Panel 3 — EKF: with landmark-based correction
    ax[2].plot(gt[:, 0], gt[:, 1], "-", color=TRUTH, lw=2.2, label="Ground Truth")
    ax[2].plot(ekf[:, 0], ekf[:, 1], "--", color=EKFC, lw=2.0, label="EKF estimate")
    step = max(1, len(cov_ekf) // 12)
    for k in range(step, len(cov_ekf), step):
        draw_ellipse(ax[2], ekf[k + 1, :2], cov_ekf[k])
    lx = [p[0] for p in LANDMARKS]; ly = [p[1] for p in LANDMARKS]
    ax[2].plot(lx, ly, "*", color=LM, ms=20, label="Known Landmark", linestyle="None")
    ax[2].plot(*gt[0, :2], "o", color=M_START, ms=10, label="Start")
    ax[2].plot(*gt[-1, :2], "o", color=M_TRUE, ms=10, label="True End")
    ax[2].plot(*ekf[-1, :2], "o", color=M_EST, ms=10, label="Estimated End")
    setup(ax[2], "EKF: with landmark-based correction"); ax[2].legend(loc="upper left", fontsize=10)

    dly = f"  (measurement delay {args.delay_ms:.0f} ms)" if args.delay_ms > 0 else ""
    fig.suptitle("EKF localization: odometry drift → growing uncertainty → "
                 "landmark-based correction" + dly, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print("saved", args.out,
          "| odo end err %.2f m, EKF end err %.2f m" %
          (np.hypot(*(odo[-1, :2] - gt[-1, :2])), np.hypot(*(ekf[-1, :2] - gt[-1, :2]))))


if __name__ == "__main__":
    main()
