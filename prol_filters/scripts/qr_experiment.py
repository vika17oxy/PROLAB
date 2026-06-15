#!/usr/bin/env python3
"""
qr_experiment.py — Process-noise (Q) and measurement-noise (R) variation
experiments (text.md mandatory experiments), deterministic offline EKF on the
current S-slalom trajectory.

A small gyro bias is injected as a real motion-model error, so the Q/R trade-off
is meaningful:
  * Q = process-noise / "model confidence":
      Q too small  -> filter over-trusts its (biased) motion model -> drifts.
      Q too large  -> filter chases the noisy landmark measurements -> jittery.
  * R = measurement-noise / "sensor trust":
      R too small  -> over-trusts noisy sensor -> jittery.
      R too large  -> ignores sensor, relies on biased model -> drifts.
Both show a U-shaped RMSE with an optimum near the true noise level.

Writes:
  <data>/02_q_variation/plots/q_variation.png
  <data>/03_r_variation/plots/r_variation.png

Run:  python3 qr_experiment.py --data /data
"""
import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]
WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LANDMARK = (1.8, 3.0)
LM_RADIUS = 6.0
V, DT = 0.4, 0.02
# NOTE: GYRO_BIAS (model error) and R_TRUE are deliberately larger than the node
# defaults so the Q/R sweeps have a non-trivial interior optimum; without model
# error a smaller Q/R would always win and the trade-off would not appear.
# Q_DEFAULT matches the node (q_xy=0.001, q_theta=0.0005); the swept variable is
# the one under test in each panel.
GYRO_BIAS = 0.05         # injected motion-model error (rad/s) — large enough to need the sensor
GYRO_NOISE = 0.006
R_TRUE = 0.02            # actual landmark range/bearing noise variance (sensor really is noisy)
Q_DEFAULT = (0.001, 0.0005)
R_DEFAULT = 0.02


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


def make_truth():
    D, cum = build_path(WPX, WPY)
    n = int(cum[-1] / (V * DT))
    gt = np.zeros((n, 3))
    for k in range(n):
        s = min(k * V * DT, cum[-1])
        x = float(np.interp(s, cum, D[:, 0])); y = float(np.interp(s, cum, D[:, 1]))
        i = min(max(int(np.searchsorted(cum, s)), 1), len(D) - 2)
        gt[k] = [x, y, math.atan2(D[i + 1, 1] - D[i - 1, 1], D[i + 1, 0] - D[i - 1, 0])]
    rng = np.random.default_rng(1)
    omega = np.zeros(n)
    for k in range(1, n):
        omega[k] = wrap(gt[k, 2] - gt[k - 1, 2]) / DT + GYRO_BIAS + rng.normal(0, GYRO_NOISE)
    return gt, omega


def run_ekf(gt, omega, q_xy, q_th, r_var, seed=42):
    rng = np.random.default_rng(seed)
    lx, ly = LANDMARK
    Q = np.diag([q_xy, q_xy, q_th]); R = np.diag([r_var, r_var])
    x = gt[0].copy().astype(float); P = np.eye(3) * 0.1
    est = np.zeros_like(gt); est[0] = x
    for k in range(1, len(gt)):
        th = x[2]
        x = x + [V * math.cos(th) * DT, V * math.sin(th) * DT, omega[k] * DT]
        x[2] = wrap(x[2])
        G = np.eye(3); G[0, 2] = -V * math.sin(th) * DT; G[1, 2] = V * math.cos(th) * DT
        P = G @ P @ G.T + Q
        gx, gy, gth = gt[k]
        if math.hypot(lx - gx, ly - gy) < LM_RADIUS:
            z_r = math.hypot(lx - gx, ly - gy) + rng.normal(0, math.sqrt(R_TRUE))
            z_b = wrap(math.atan2(ly - gy, lx - gx) - gth) + rng.normal(0, math.sqrt(R_TRUE))
            dx, dy = lx - x[0], ly - x[1]; r = math.hypot(dx, dy)
            if r > 1e-6:
                H = np.array([[-dx / r, -dy / r, 0.0], [dy / r / r, -dx / r / r, -1.0]])
                S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
                innov = np.array([z_r - r, wrap(z_b - wrap(math.atan2(dy, dx) - x[2]))])
                x = x + K @ innov; x[2] = wrap(x[2])
                IKH = np.eye(3) - K @ H; P = IKH @ P @ IKH.T + K @ R @ K.T
        est[k] = x
    return est


def rmse(est, gt):
    return float(np.sqrt(np.mean((est[:, 0] - gt[:, 0]) ** 2 + (est[:, 1] - gt[:, 1]) ** 2)))


def sweep_figure(gt, omega, values, run_fn, label, picks, title, fname, outdir):
    rmses = [rmse(run_fn(v), gt) for v in values]
    opt_i = int(np.argmin(rmses))
    fig, (axr, axt) = plt.subplots(1, 2, figsize=(14, 5.4))

    axr.plot(values, rmses, "o-", color="#2980b9", lw=2, ms=8)
    axr.plot(values[opt_i], rmses[opt_i], "o", color="#27ae60", ms=13,
             label=f"optimum ({label}={values[opt_i]:g}, {rmses[opt_i]:.3f} m)")
    axr.set_xscale("log"); axr.set_xlabel(f"{label}"); axr.set_ylabel("position RMSE [m]")
    axr.set_title(f"RMSE vs {label}", fontweight="bold")
    axr.grid(True, which="both", alpha=0.3); axr.legend()

    axt.plot(gt[:, 0], gt[:, 1], "-", color="black", lw=2.4, label="Ground Truth", zorder=5)
    cols = ["#e74c3c", "#27ae60", "#8e44ad"]
    tags = ["too small", "good", "too large"]
    for v, c, tg in zip(picks, cols, tags):
        e = run_fn(v)
        axt.plot(e[:, 0], e[:, 1], "--", color=c, lw=1.6, alpha=0.9,
                 label=f"{label}={v:g} ({tg}, RMSE {rmse(e, gt):.2f} m)")
    axt.plot(*LANDMARK, "*", color="#1f77b4", ms=18, label="Known Landmark")
    axt.plot(gt[0, 0], gt[0, 1], "o", color="0.4", ms=9, label="Start")
    axt.set_aspect("equal"); axt.grid(True, alpha=0.3)
    axt.set_xlabel("x [m]"); axt.set_ylabel("y [m]")
    axt.set_title("EKF estimate for representative values", fontweight="bold")
    axt.legend(loc="best", fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.01); fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(title)
    for v, r in zip(values, rmses):
        print(f"   {label}={v:<8g}  RMSE={r:.4f} m")
    print("   saved", out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="/data")
    a = ap.parse_args()
    gt, omega = make_truth()

    # Q-variation (sweep q_xy, q_theta scaled with it; R fixed at default)
    qvals = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    sweep_figure(
        gt, omega, qvals,
        lambda q: run_ekf(gt, omega, q, q * 0.5, R_DEFAULT),
        "q_xy", picks=[1e-6, 1e-5, 1e-1],
        title="Process-Noise (Q) Variation — model confidence",
        fname="q_variation.png",
        outdir=os.path.join(a.data, "02_q_variation", "plots"))

    # R-variation (sweep r_landmark; Q fixed small so the high-R side really drifts)
    rvals = [5e-4, 5e-3, 5e-2, 5e-1, 5e0, 5e1]
    sweep_figure(
        gt, omega, rvals,
        lambda r: run_ekf(gt, omega, 1e-5, 5e-6, r),
        "r_landmark", picks=[5e-4, 5e-2, 5e1],
        title="Measurement-Noise (R) Variation — sensor trust",
        fname="r_variation.png",
        outdir=os.path.join(a.data, "03_r_variation", "plots"))


if __name__ == "__main__":
    main()
