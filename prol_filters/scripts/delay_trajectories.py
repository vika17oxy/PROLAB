#!/usr/bin/env python3
"""
delay_trajectories.py — "Effect of Time-Delayed Measurements on Trajectories"
3-panel EKF figure (Baseline / 100 ms / 500 ms) on the current S-slalom route.
Deterministic offline EKF (own code) with the delay modelled as in the node:
each landmark measurement is buffered and applied D = round(delay/dt) steps late.

Output: <out>   (default /data/10_scurve/plots/trajectories_delay_comparison.png)
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Plotting conventions: filter colours, legend outside, clean rc.
plt.rcParams.update({
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.edgecolor": "0.4", "axes.linewidth": 1.0,
    "font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
    "legend.framealpha": 0.92,
})

WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]
WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LM = (1.8, 3.0); LM_R = 6.0
V, DT = 0.4, 0.02       # v=0.4 makes the processing-lag visible (lag = v*delay)
GYRO_NOISE = 0.012      # = node gyro_noise_std
R_R, R_B = 0.005, 0.01  # = node r_landmark, r_bearing
Q_XY, Q_TH = 0.001, 0.0005    # = node q_xy, q_theta
DELAYS = [(0, "Baseline (0 ms delay)"), (100, "Delay 100 ms"),
          (500, "Delay 500 ms"), (5000, "Delay 5000 ms (5 s)")]


def wrap(a): return math.atan2(math.sin(a), math.cos(a))


def build_path(xs, ys):
    pts = np.array(list(zip(xs, ys)), float)
    P = np.vstack([2*pts[0]-pts[1], pts, 2*pts[-1]-pts[-2]]); al = 0.5; dense = []
    for i in range(1, len(P)-2):
        P0, P1, P2, P3 = P[i-1], P[i], P[i+1], P[i+2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(P1-P0)**al, 1e-6)
        t2 = t1 + max(np.linalg.norm(P2-P1)**al, 1e-6)
        t3 = t2 + max(np.linalg.norm(P3-P2)**al, 1e-6)
        for k in range(80):
            t = t1 + (t2-t1)*k/80.0
            A1 = (t1-t)/(t1-t0)*P0 + (t-t0)/(t1-t0)*P1
            A2 = (t2-t)/(t2-t1)*P1 + (t-t1)/(t2-t1)*P2
            A3 = (t3-t)/(t3-t2)*P2 + (t-t2)/(t3-t2)*P3
            B1 = (t2-t)/(t2-t0)*A1 + (t-t0)/(t2-t0)*A2
            B2 = (t3-t)/(t3-t1)*A2 + (t-t1)/(t3-t1)*A3
            dense.append((t2-t)/(t2-t1)*B1 + (t-t1)/(t2-t1)*B2)
    dense.append(pts[-1]); D = np.array(dense)
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    return D, np.concatenate([[0.0], np.cumsum(seg)])


def make_truth(seed=1):
    D, cum = build_path(WPX, WPY); n = int(cum[-1] / (V*DT))
    gt = np.zeros((n, 3))
    for k in range(n):
        s = min(k*V*DT, cum[-1])
        x = float(np.interp(s, cum, D[:, 0])); y = float(np.interp(s, cum, D[:, 1]))
        i = min(max(int(np.searchsorted(cum, s)), 1), len(D)-2)
        gt[k] = [x, y, math.atan2(D[i+1, 1]-D[i-1, 1], D[i+1, 0]-D[i-1, 0])]
    rng = np.random.default_rng(seed); omega = np.zeros(n)
    for k in range(1, n):
        omega[k] = wrap(gt[k, 2]-gt[k-1, 2])/DT + rng.normal(0, GYRO_NOISE)
    return gt, omega


def run_ekf(gt, omega, delay_steps, seed=42):
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    x = gt[0].copy(); P = np.eye(3)*0.1; R = np.diag([R_R, R_B]); Q = np.diag([Q_XY, Q_XY, Q_TH])
    est = np.zeros((n, 2)); est[0] = gt[0, :2]; cov = [P[:2, :2].copy()]; pending = []
    for k in range(1, n):
        th = x[2]; x = x + [V*math.cos(th)*DT, V*math.sin(th)*DT, omega[k]*DT]; x[2] = wrap(x[2])
        G = np.eye(3); G[0, 2] = -V*math.sin(th)*DT; G[1, 2] = V*math.cos(th)*DT; P = G@P@G.T + Q
        gx, gy, gth = gt[k]
        if math.hypot(lx-gx, ly-gy) < LM_R:
            z_r = math.hypot(lx-gx, ly-gy) + rng.normal(0, math.sqrt(R_R))
            z_b = wrap(math.atan2(ly-gy, lx-gx)-gth) + rng.normal(0, math.sqrt(R_B))
            pending.append((k, z_r, z_b))
        while pending and (k - pending[0][0]) >= delay_steps:
            _, z_r, z_b = pending.pop(0); dx = lx-x[0]; dy = ly-x[1]; r = math.hypot(dx, dy)
            if r >= 1e-6:
                H = np.array([[-dx/r, -dy/r, 0.0], [dy/r/r, -dx/r/r, -1.0]])
                S = H@P@H.T + R; K = P@H.T@np.linalg.inv(S)
                x = x + K@np.array([z_r-r, wrap(z_b-wrap(math.atan2(dy, dx)-x[2]))]); x[2] = wrap(x[2])
                IKH = np.eye(3)-K@H; P = IKH@P@IKH.T + K@R@K.T   # Joseph form (matches C++)
        est[k] = [x[0], x[1]]; cov.append(P[:2, :2].copy())
    return est, cov


def draw_ellipse(ax, mean, c, nsig=2.0):
    vals, vecs = np.linalg.eigh(c); vals = np.maximum(vals, 1e-12)
    o = vals.argsort()[::-1]; vals = vals[o]; vecs = vecs[:, o]
    ax.add_patch(Ellipse(mean, min(2*nsig*math.sqrt(vals[0]), 4.0),
                         min(2*nsig*math.sqrt(vals[1]), 4.0),
                         angle=math.degrees(math.atan2(vecs[1, 0], vecs[0, 0])),
                         fill=False, ec="tab:green", lw=0.9, alpha=0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/10_scurve/plots/trajectories_delay_comparison.png")
    a = ap.parse_args()
    gt, omega = make_truth(0)
    truths = [make_truth(s) for s in range(30)]            # vary gyro + meas noise
    fig, axs = plt.subplots(1, len(DELAYS), figsize=(4.6*len(DELAYS), 5.2), sharey=True)
    for ax, (dms, title) in zip(axs, DELAYS):
        D = int(round(dms/1000.0/DT))
        est, cov = run_ekf(gt, omega, D, seed=0)          # representative path
        # RMSE averaged over gyro+measurement realisations -> stable, monotonic in delay
        vals = []
        for s, (g, om) in enumerate(truths):
            e = run_ekf(g, om, D, seed=1000 + s)[0]
            vals.append(np.sqrt(np.mean((e[:, 0]-g[:, 0])**2 + (e[:, 1]-g[:, 1])**2)))
        rmse = float(np.mean(vals))
        ax.plot(gt[:, 0], gt[:, 1], "-", color="k", lw=2.2, label="Ground Truth", zorder=5)
        ax.plot(est[:, 0], est[:, 1], "--", color="tab:green", lw=1.8,
                label=f"EKF estimate  (RMSE {rmse:.3f} m)", zorder=6)
        step = max(1, len(cov)//14)
        for k in range(step, len(cov), step):
            draw_ellipse(ax, est[k, :2], cov[k])
        ax.plot(*LM, "*", color="tab:blue", ms=18, markeredgecolor="k",
                label="Known Landmark", zorder=7)
        ax.plot(gt[0, 0], gt[0, 1], "o", color="green", ms=9, label="Start", zorder=8)
        ax.plot(gt[-1, 0], gt[-1, 1], "o", color="orange", ms=9, label="True End", zorder=8)
        ax.plot(est[-1, 0], est[-1, 1], "o", color="red", ms=9, label="Estimated End", zorder=8)
        ax.set_aspect("equal"); ax.set_xlabel("x [m]")
        ax.set_title(title); ax.legend(loc="upper left", fontsize=8)
    axs[0].set_ylabel("y [m]")
    fig.suptitle("Effect of Time-Delayed Measurements on Trajectories",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight"); print("saved", a.out)


if __name__ == "__main__":
    main()
