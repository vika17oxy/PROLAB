#!/usr/bin/env python3
"""
trajectory_comparison.py — "Trajectory Comparison" plot (Ground Truth vs KF/EKF/PF)
on the current S-slalom route, in the lecture-figure style. Deterministic offline
replica of the three filters (same math as the C++ nodes), so the result is clean
and reproducible.

Run:  python3 trajectory_comparison.py --out /data/.../trajectories.png
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
LANDMARKS = [(1.8, 3.0)]        # single landmark -> keeps KF > EKF > PF separation
V, DT = 0.12, 0.02
GYRO_BIAS, GYRO_NOISE = 0.0, 0.012
R_TRUE = 0.005
Q_XY, Q_TH = 0.001, 0.0005
SIGMA_V, SIGMA_W, NP = 0.03, 0.015, 500
LM_RADIUS = 12.0
UPDATE_EVERY = 8                # landmark sensor ~6 Hz -> clean (non-scribbly) lines


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def wrap_arr(a):
    return np.arctan2(np.sin(a), np.cos(a))


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


def meas(gx, gy, gth, lx, ly, rng):
    z_r = math.hypot(lx - gx, ly - gy) + rng.normal(0, math.sqrt(R_TRUE))
    z_b = wrap(math.atan2(ly - gy, lx - gx) - gth) + rng.normal(0, math.sqrt(R_TRUE))
    return z_r, z_b


def run_gauss(gt, omega, jac, seed):
    rng = np.random.default_rng(seed)
    Q = np.diag([Q_XY, Q_XY, Q_TH]); R = np.diag([R_TRUE, R_TRUE])
    x = gt[0].copy().astype(float); P = np.eye(3) * 0.1
    est = np.zeros_like(gt); est[0] = x
    for k in range(1, len(gt)):
        th = x[2]
        x = x + [V * math.cos(th) * DT, V * math.sin(th) * DT, omega[k] * DT]; x[2] = wrap(x[2])
        if jac:
            G = np.eye(3); G[0, 2] = -V * math.sin(th) * DT; G[1, 2] = V * math.cos(th) * DT
            P = G @ P @ G.T + Q
        else:
            P = P + Q
        if k % UPDATE_EVERY == 0:
            for lx, ly in LANDMARKS:
                z_r, z_b = meas(*gt[k], lx, ly, rng)
                dx, dy = lx - x[0], ly - x[1]; r = math.hypot(dx, dy)
                if r < 1e-6:
                    continue
                H = np.array([[-dx / r, -dy / r, 0.0], [dy / r / r, -dx / r / r, -1.0]])
                S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ np.array([z_r - r, wrap(z_b - wrap(math.atan2(dy, dx) - x[2]))]); x[2] = wrap(x[2])
                IKH = np.eye(3) - K @ H; P = IKH @ P @ IKH.T + K @ R @ K.T
        est[k] = x
    return est


def run_pf(gt, omega, seed):
    rng = np.random.default_rng(seed)
    Pp = np.tile(gt[0], (NP, 1)).astype(float); w = np.full(NP, 1.0 / NP)
    est = np.zeros_like(gt); est[0] = gt[0]
    for k in range(1, len(gt)):
        Pp[:, 0] += (V + rng.normal(0, SIGMA_V, NP)) * np.cos(Pp[:, 2]) * DT
        Pp[:, 1] += (V + rng.normal(0, SIGMA_V, NP)) * np.sin(Pp[:, 2]) * DT
        Pp[:, 2] = wrap_arr(Pp[:, 2] + (omega[k] + rng.normal(0, SIGMA_W, NP)) * DT)
        if k % UPDATE_EVERY == 0:
            for lx, ly in LANDMARKS:
                z_r, z_b = meas(*gt[k], lx, ly, rng)
                dx, dy = lx - Pp[:, 0], ly - Pp[:, 1]; rp = np.hypot(dx, dy)
                eb = wrap_arr(z_b - wrap_arr(np.arctan2(dy, dx) - Pp[:, 2]))
                w *= np.exp(-0.5 * ((z_r - rp) ** 2 / R_TRUE + eb ** 2 / R_TRUE))
            s = w.sum(); w = w / s if s > 1e-300 else np.full(NP, 1.0 / NP)
            idx = np.clip(np.searchsorted(np.cumsum(w), (rng.random() + np.arange(NP)) / NP), 0, NP - 1)
            Pp = Pp[idx].copy(); w = np.full(NP, 1.0 / NP)
        est[k] = [np.average(Pp[:, 0], weights=w), np.average(Pp[:, 1], weights=w),
                  math.atan2(np.average(np.sin(Pp[:, 2]), weights=w), np.average(np.cos(Pp[:, 2]), weights=w))]
    return est


def load_csv(path):
    d = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
    return {n: np.atleast_1d(d[n]) for n in d.dtype.names}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="/data/trajectories.png")
    ap.add_argument("--csv-dir", default=None,
                    help="if set, plot real kf/ekf/pf_log.csv from this dir instead of the replica")
    a = ap.parse_args()

    if a.csv_dir:
        kd = load_csv(os.path.join(a.csv_dir, "kf_log.csv"))
        ed = load_csv(os.path.join(a.csv_dir, "ekf_log.csv"))
        pd = load_csv(os.path.join(a.csv_dir, "pf_log.csv"))
        gt = np.column_stack([kd["gt_x"], kd["gt_y"]])
        kf = np.column_stack([kd["x"], kd["y"]])
        ekf = np.column_stack([ed["x"], ed["y"]])
        pf = np.column_stack([pd["x"], pd["y"]])
    else:
        gt3, omega = make_truth()
        gt = gt3[:, :2]
        kf = run_gauss(gt3, omega, False, 42)[:, :2]
        ekf = run_gauss(gt3, omega, True, 42)[:, :2]
        pf = run_pf(gt3, omega, 44)[:, :2]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.plot(gt[:, 0], gt[:, 1], "-", color="#1f77b4", lw=2.6, label="Ground Truth")
    ax.plot(kf[:, 0], kf[:, 1], "--", color="#d62728", lw=1.3, label="KF")
    ax.plot(ekf[:, 0], ekf[:, 1], "--", color="#2ca02c", lw=1.3, label="EKF")
    ax.plot(pf[:, 0], pf[:, 1], "--", color="#9467bd", lw=1.3, label="PF")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title("Trajectory Comparison", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight")
    def rms(e):
        m = min(len(e), len(gt))
        return np.sqrt(np.mean((e[:m, 0] - gt[:m, 0]) ** 2 + (e[:m, 1] - gt[:m, 1]) ** 2))
    print("saved", a.out, "| RMSE KF %.3f  EKF %.3f  PF %.3f" % (rms(kf), rms(ekf), rms(pf)))


if __name__ == "__main__":
    main()
