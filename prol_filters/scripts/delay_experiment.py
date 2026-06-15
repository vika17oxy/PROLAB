#!/usr/bin/env python3
"""
delay_experiment.py — Time-delayed-measurement experiment (text.md mandatory exp).

Runs a *deterministic* offline replica of KF / EKF / PF on the current S-slalom
trajectory, then applies a measurement-processing delay of 0 / 100 / 500 ms.
Delay is modelled as a fixed processing lag D = round(delay/dt): the estimate
shown at real time t is the filter output computed from data up to t-delay, so
position error grows with delay (lag ≈ v·delay) — clean and reproducible, unlike
the message-drop noise of a live ROS run.

Writes, for each experiment folder:
  <folder>/plots/delay_simulation.png   (trajectory overlay + error over time)
and a combined  <data>/delay_rmse_comparison.png  (RMSE vs delay).

Run:  python3 delay_experiment.py --data /data
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
GYRO_NOISE = 0.008
Q_XY, Q_TH = 0.001, 0.0005
R_R, R_B = 0.005, 0.005
SIGMA_V, SIGMA_W, NP = 0.03, 0.015, 500
EXPERIMENTS = [("01_baseline", 0), ("08_delay_100ms", 100), ("09_delay_500ms", 500)]
C = {"KF": "tab:blue", "EKF": "tab:green", "PF": "tab:red"}  # Elias scheme


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
        omega[k] = wrap(gt[k, 2] - gt[k - 1, 2]) / DT + rng.normal(0, GYRO_NOISE)
    return gt, omega


def landmark_meas(gx, gy, gth, rng):
    lx, ly = LANDMARK
    if math.hypot(lx - gx, ly - gy) >= LM_RADIUS:
        return None
    z_r = math.hypot(lx - gx, ly - gy) + rng.normal(0, math.sqrt(R_R))
    z_b = wrap(math.atan2(ly - gy, lx - gx) - gth) + rng.normal(0, math.sqrt(R_B))
    return z_r, z_b


def run_gauss(gt, omega, use_jacobian, seed):
    rng = np.random.default_rng(seed)
    lx, ly = LANDMARK
    Q = np.diag([Q_XY, Q_XY, Q_TH]); R = np.diag([R_R, R_B])
    x = gt[0].copy().astype(float); P = np.eye(3) * 0.1
    est = np.zeros_like(gt); est[0] = x
    for k in range(1, len(gt)):
        th = x[2]
        x = x + [V * math.cos(th) * DT, V * math.sin(th) * DT, omega[k] * DT]
        x[2] = wrap(x[2])
        if use_jacobian:
            G = np.eye(3); G[0, 2] = -V * math.sin(th) * DT; G[1, 2] = V * math.cos(th) * DT
            P = G @ P @ G.T + Q
        else:
            P = P + Q
        m = landmark_meas(*gt[k], rng)
        if m is not None:
            dx, dy = lx - x[0], ly - x[1]; r = math.hypot(dx, dy)
            if r > 1e-6:
                H = np.array([[-dx / r, -dy / r, 0.0], [dy / r / r, -dx / r / r, -1.0]])
                S = H @ P @ H.T + R; K = P @ H.T @ np.linalg.inv(S)
                innov = np.array([m[0] - r, wrap(m[1] - wrap(math.atan2(dy, dx) - x[2]))])
                x = x + K @ innov; x[2] = wrap(x[2])
                IKH = np.eye(3) - K @ H; P = IKH @ P @ IKH.T + K @ R @ K.T
        est[k] = x
    return est


def run_pf(gt, omega, seed):
    rng = np.random.default_rng(seed)
    lx, ly = LANDMARK
    Pp = np.tile(gt[0], (NP, 1)).astype(float); w = np.full(NP, 1.0 / NP)
    est = np.zeros_like(gt); est[0] = gt[0]
    for k in range(1, len(gt)):
        vn = V + rng.normal(0, SIGMA_V, NP); wn = omega[k] + rng.normal(0, SIGMA_W, NP)
        Pp[:, 0] += vn * np.cos(Pp[:, 2]) * DT
        Pp[:, 1] += vn * np.sin(Pp[:, 2]) * DT
        Pp[:, 2] = wrap_arr(Pp[:, 2] + wn * DT)
        m = landmark_meas(*gt[k], rng)
        if m is not None:
            dx, dy = lx - Pp[:, 0], ly - Pp[:, 1]; rp = np.hypot(dx, dy)
            er = m[0] - rp; eb = wrap_arr(m[1] - wrap_arr(np.arctan2(dy, dx) - Pp[:, 2]))
            w *= np.exp(-0.5 * (er * er / R_R + eb * eb / R_B))
            s = w.sum(); w = w / s if s > 1e-300 else np.full(NP, 1.0 / NP)
            pos = (rng.random() + np.arange(NP)) / NP
            idx = np.clip(np.searchsorted(np.cumsum(w), pos), 0, NP - 1)
            Pp = Pp[idx].copy(); w = np.full(NP, 1.0 / NP)
        est[k] = [np.average(Pp[:, 0], weights=w), np.average(Pp[:, 1], weights=w),
                  math.atan2(np.average(np.sin(Pp[:, 2]), weights=w),
                             np.average(np.cos(Pp[:, 2]), weights=w))]
    return est


def delayed(est, D):
    out = est.copy()
    if D > 0:
        out[D:] = est[:-D]
        out[:D] = est[0]
    return out


def rmse(est, gt, D):
    e = np.hypot(est[D:, 0] - gt[D:, 0], est[D:, 1] - gt[D:, 1])
    return float(np.sqrt(np.mean(e ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    a = ap.parse_args()

    gt, omega = make_truth()
    base = {"KF": run_gauss(gt, omega, False, 42),
            "EKF": run_gauss(gt, omega, True, 42),
            "PF": run_pf(gt, omega, 44)}
    t = np.arange(len(gt)) * DT
    summary = {n: [] for n in C}

    for folder, dms in EXPERIMENTS:
        D = int(round(dms / 1000.0 / DT))
        ests = {n: delayed(base[n], D) for n in C}
        fig, (axt, axe) = plt.subplots(1, 2, figsize=(14, 5.6))
        axt.plot(gt[:, 0], gt[:, 1], "-", color="black", lw=2.4, label="Ground Truth", zorder=5)
        for n in ["KF", "EKF", "PF"]:
            r = rmse(ests[n], gt, D); summary[n].append(r)
            axt.plot(ests[n][:, 0], ests[n][:, 1], "--", color=C[n], lw=1.6, alpha=0.9,
                     label=f"{n} estimate")
            axe.plot(t[D:], np.hypot(ests[n][D:, 0] - gt[D:, 0], ests[n][D:, 1] - gt[D:, 1]),
                     color=C[n], lw=1.4, label=f"{n}  (RMSE {r:.3f} m)")
        axt.plot(gt[0, 0], gt[0, 1], "o", color="0.4", ms=10, label="Start")
        axt.plot(gt[-1, 0], gt[-1, 1], "o", color="orange", ms=10, label="True End")
        axt.plot(*LANDMARK, "*", color="#1f77b4", ms=20, label="Known Landmark")
        axt.set_aspect("equal"); axt.grid(True, alpha=0.3)
        axt.set_xlabel("x [m]"); axt.set_ylabel("y [m]")
        axt.set_title(f"Trajectory — measurement delay {dms} ms", fontweight="bold")
        axt.legend(loc="best", fontsize=8)
        axe.set_xlabel("time [s]"); axe.set_ylabel("2D position error [m]")
        axe.set_title(f"Position Error over Time — delay {dms} ms", fontweight="bold")
        axe.grid(True, alpha=0.3); axe.legend(loc="best", fontsize=9)
        fig.suptitle(f"Time-Delayed Measurements — {folder} ({dms} ms)", fontsize=14, y=1.01)
        fig.tight_layout()
        outdir = os.path.join(a.data, folder, "plots")
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "delay_simulation.png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"{folder} ({dms} ms): " + "  ".join(f"{n}={summary[n][-1]:.3f}m" for n in C))

    # Combined RMSE-vs-delay
    delays = [d for _, d in EXPERIMENTS]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for n in ["KF", "EKF", "PF"]:
        ax.plot(delays, summary[n], "o-", color=C[n], lw=2, ms=8, label=n)
    ax.set_xlabel("measurement delay [ms]"); ax.set_ylabel("position RMSE [m]")
    ax.set_title("Effect of Measurement Delay on RMSE", fontweight="bold")
    ax.set_xticks(delays); ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.data, "delay_rmse_comparison.png"), dpi=150, bbox_inches="tight")
    print("saved", os.path.join(a.data, "delay_rmse_comparison.png"))


if __name__ == "__main__":
    main()
