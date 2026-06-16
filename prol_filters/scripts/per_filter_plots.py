#!/usr/bin/env python3
"""
per_filter_plots.py — one plot per filter (KF, EKF, PF) on the S-slalom.

Each panel shows ground truth, that filter's estimate, the landmark and
start/end markers, plus the filter's own uncertainty representation:
  • KF / EKF : 2-sigma covariance ellipses (eigendecomposition of P)
  • PF        : the particle cloud (snapshots along the run)

Deterministic offline replicas of the C++ filters with node-matched parameters.
Outputs <out> (default /data/10_scurve/plots/per_filter.png).
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]
WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LM = (1.8, 3.0); LM_R = 6.0
V, DT = 0.3, 0.02
GYRO_NOISE = 0.012
R_R, R_B = 0.005, 0.01
Q_XY, Q_TH = 0.001, 0.0005
SIGMA_V, SIGMA_W, NP = 0.03, 0.015, 500
C = {"KF": "tab:blue", "EKF": "tab:green", "PF": "tab:red"}

plt.rcParams.update({
    "axes.grid": True, "grid.color": "0.85", "axes.axisbelow": True,
    "axes.edgecolor": "0.4", "font.size": 12, "axes.titlesize": 13,
    "axes.titleweight": "bold", "legend.framealpha": 0.92,
})


def wrap(a): return math.atan2(math.sin(a), math.cos(a))
def wrap_arr(a): return np.arctan2(np.sin(a), np.cos(a))


def build_path(xs, ys):
    pts = np.array(list(zip(xs, ys)), float)
    P = np.vstack([2*pts[0]-pts[1], pts, 2*pts[-1]-pts[-2]]); al = 0.5; dense = []
    for i in range(1, len(P)-2):
        P0, P1, P2, P3 = P[i-1], P[i], P[i+1], P[i+2]
        t0 = 0.0
        t1 = t0+max(np.linalg.norm(P1-P0)**al, 1e-6)
        t2 = t1+max(np.linalg.norm(P2-P1)**al, 1e-6); t3 = t2+max(np.linalg.norm(P3-P2)**al, 1e-6)
        for k in range(80):
            t = t1+(t2-t1)*k/80.0
            A1 = (t1-t)/(t1-t0)*P0+(t-t0)/(t1-t0)*P1
            A2 = (t2-t)/(t2-t1)*P1+(t-t1)/(t2-t1)*P2
            A3 = (t3-t)/(t3-t2)*P2+(t-t2)/(t3-t2)*P3
            B1 = (t2-t)/(t2-t0)*A1+(t-t0)/(t2-t0)*A2
            B2 = (t3-t)/(t3-t1)*A2+(t-t1)/(t3-t1)*A3
            dense.append((t2-t)/(t2-t1)*B1+(t-t1)/(t2-t1)*B2)
    dense.append(pts[-1]); D = np.array(dense)
    seg = np.linalg.norm(np.diff(D, axis=0), axis=1)
    return D, np.concatenate([[0.0], np.cumsum(seg)])


def make_truth(seed=0):
    D, cum = build_path(WPX, WPY); n = int(cum[-1]/(V*DT)); gt = np.zeros((n, 3))
    for k in range(n):
        s = min(k*V*DT, cum[-1])
        x = float(np.interp(s, cum, D[:, 0])); y = float(np.interp(s, cum, D[:, 1]))
        i = min(max(int(np.searchsorted(cum, s)), 1), len(D)-2)
        gt[k] = [x, y, math.atan2(D[i+1, 1]-D[i-1, 1], D[i+1, 0]-D[i-1, 0])]
    rng = np.random.default_rng(seed); omega = np.zeros(n)
    for k in range(1, n):
        omega[k] = wrap(gt[k, 2]-gt[k-1, 2])/DT + rng.normal(0, GYRO_NOISE)
    return gt, omega


def run_gauss(jac, gt, omega, seed):
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    x = gt[0].copy(); P = np.eye(3)*0.1; R = np.diag([R_R, R_B]); Q = np.diag([Q_XY, Q_XY, Q_TH])
    est = np.zeros((n, 2)); est[0] = gt[0, :2]; cov = [P[:2, :2].copy()]
    for k in range(1, n):
        th = x[2]; x = x + [V*math.cos(th)*DT, V*math.sin(th)*DT, omega[k]*DT]; x[2] = wrap(x[2])
        if jac:
            G = np.eye(3); G[0, 2] = -V*math.sin(th)*DT; G[1, 2] = V*math.cos(th)*DT; P = G@P@G.T+Q
        else:
            P = P+Q
        gx, gy, gth = gt[k]
        if math.hypot(lx-gx, ly-gy) < LM_R:
            z_r = math.hypot(lx-gx, ly-gy)+rng.normal(0, math.sqrt(R_R))
            z_b = wrap(math.atan2(ly-gy, lx-gx)-gth)+rng.normal(0, math.sqrt(R_B))
            dx = lx-x[0]; dy = ly-x[1]; r = math.hypot(dx, dy)
            if r >= 1e-6:
                H = np.array([[-dx/r, -dy/r, 0.0], [dy/r/r, -dx/r/r, -1.0]])
                S = H@P@H.T+R; K = P@H.T@np.linalg.inv(S)
                x = x+K@np.array([z_r-r, wrap(z_b-wrap(math.atan2(dy, dx)-x[2]))]); x[2] = wrap(x[2])
                IKH = np.eye(3)-K@H; P = IKH@P@IKH.T+K@R@K.T
        est[k] = x[:2]; cov.append(P[:2, :2].copy())
    return est, cov


def run_pf(gt, omega, seed):
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    Pp = np.tile(gt[0], (NP, 1)).astype(float); w = np.full(NP, 1.0/NP)
    est = np.zeros((n, 2)); est[0] = gt[0, :2]; snaps = []
    snap_at = set(np.linspace(0, n-1, 9).astype(int))
    for k in range(1, n):
        vn = V+rng.normal(0, SIGMA_V, NP); wn = omega[k]+rng.normal(0, SIGMA_W, NP)
        Pp[:, 0] += vn*np.cos(Pp[:, 2])*DT; Pp[:, 1] += vn*np.sin(Pp[:, 2])*DT
        Pp[:, 2] = wrap_arr(Pp[:, 2]+wn*DT)
        gx, gy, gth = gt[k]
        if math.hypot(lx-gx, ly-gy) < LM_R:
            z_r = math.hypot(lx-gx, ly-gy)+rng.normal(0, math.sqrt(R_R))
            z_b = wrap(math.atan2(ly-gy, lx-gx)-gth)+rng.normal(0, math.sqrt(R_B))
            dx = lx-Pp[:, 0]; dy = ly-Pp[:, 1]; rp = np.hypot(dx, dy)
            er = z_r-rp; eb = wrap_arr(z_b-wrap_arr(np.arctan2(dy, dx)-Pp[:, 2]))
            w *= np.exp(-0.5*(er*er/R_R+eb*eb/R_B))
            s = w.sum(); w = w/s if s > 1e-300 else np.full(NP, 1.0/NP)
            pos = (rng.random()+np.arange(NP))/NP
            idx = np.clip(np.searchsorted(np.cumsum(w), pos), 0, NP-1)
            Pp = Pp[idx].copy(); w = np.full(NP, 1.0/NP)
        est[k] = [np.average(Pp[:, 0], weights=w), np.average(Pp[:, 1], weights=w)]
        if k in snap_at:
            snaps.append(Pp[:, :2].copy())
    return est, snaps


def draw_ellipse(ax, mean, c, col, nsig=2.0):
    vals, vecs = np.linalg.eigh(c); vals = np.maximum(vals, 1e-12)
    o = vals.argsort()[::-1]; vals = vals[o]; vecs = vecs[:, o]
    ax.add_patch(Ellipse(mean, min(2*nsig*math.sqrt(vals[0]), 4.0),
                         min(2*nsig*math.sqrt(vals[1]), 4.0),
                         angle=math.degrees(math.atan2(vecs[1, 0], vecs[0, 0])),
                         fill=False, ec=col, lw=0.9, alpha=0.55))


def setup(ax, gt, est, col, title):
    ax.plot(gt[:, 0], gt[:, 1], "-", color="k", lw=2.4, label="Ground Truth", zorder=5)
    ax.plot(est[:, 0], est[:, 1], "--", color=col, lw=1.8, label="estimate", zorder=6)
    ax.plot(*LM, "*", color="tab:blue", ms=16, markeredgecolor="k", label="Landmark", zorder=8)
    ax.plot(gt[0, 0], gt[0, 1], "o", color="green", ms=8, label="Start", zorder=9)
    ax.plot(gt[-1, 0], gt[-1, 1], "o", color="orange", ms=8, label="True End", zorder=9)
    ax.plot(est[-1, 0], est[-1, 1], "o", color="red", ms=8, label="Est. End", zorder=9)
    rmse = float(np.sqrt(np.mean((est[:, 0]-gt[:, 0])**2 + (est[:, 1]-gt[:, 1])**2)))
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_title(f"{title}\nRMSE {rmse:.3f} m")
    ax.legend(loc="upper left", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/10_scurve/plots/per_filter.png")
    a = ap.parse_args()
    gt, om = make_truth(0)
    kf, kcov = run_gauss(False, gt, om, 42)
    ekf, ecov = run_gauss(True, gt, om, 42)
    pf, snaps = run_pf(gt, om, 44)

    fig, axs = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
    # KF
    step = max(1, len(kcov)//12)
    for k in range(step, len(kcov), step): draw_ellipse(axs[0], kf[k], kcov[k], C["KF"])
    setup(axs[0], gt, kf, C["KF"], "KF  (F = I,  2σ ellipses)")
    # EKF
    step = max(1, len(ecov)//12)
    for k in range(step, len(ecov), step): draw_ellipse(axs[1], ekf[k], ecov[k], C["EKF"])
    setup(axs[1], gt, ekf, C["EKF"], "EKF  (Jacobian,  2σ ellipses)")
    # PF
    for s in snaps:
        axs[2].plot(s[:, 0], s[:, 1], ".", color=C["PF"], ms=1.5, alpha=0.12, zorder=2)
    setup(axs[2], gt, pf, C["PF"], "PF  (500 particles, cloud)")

    axs[0].set_ylabel("y [m]")
    fig.suptitle("Per-filter localization on the S-slalom", fontsize=15, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight"); print("saved", a.out)


if __name__ == "__main__":
    main()
