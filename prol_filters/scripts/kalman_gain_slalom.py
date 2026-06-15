#!/usr/bin/env python3
"""
kalman_gain_slalom.py — Kalman gain over time on the *S-slalom route*, in Elias's
plotting style, with the measurement->processing delay applied as in the node.

KF (F=I covariance) and EKF (full Jacobian), gain components K_x / K_y / K_theta
for the range and bearing channels. The landmark sits off the path so it is seen
only as the robot passes the two crests of the S -> clean detection windows
(shaded), rather than a continuously-shaded plot. Own code; Elias colour/legend
conventions.

Usage:  python3 kalman_gain_slalom.py --out <png> [--delay-ms 0|100|500]
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]
WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LM = (1.8, 3.0); LM_R = 12.0         # off-path, continuously visible -> smooth gain
V, DT = 0.3, 0.02
GYRO_NOISE = 0.01
R_R, R_B = 0.005, 0.01
Q_XY, Q_TH = 0.001, 0.0005
C = {"KF": "tab:blue", "EKF": "tab:green"}

plt.rcParams.update({
    "axes.grid": True, "grid.color": "0.85", "axes.axisbelow": True,
    "axes.edgecolor": "0.4", "font.size": 12, "axes.titlesize": 14,
    "axes.titleweight": "bold", "legend.framealpha": 0.92,
})


def wrap(a): return math.atan2(math.sin(a), math.cos(a))


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


def make_truth():
    D, cum = build_path(WPX, WPY); n = int(cum[-1]/(V*DT)); gt = np.zeros((n, 3))
    for k in range(n):
        s = min(k*V*DT, cum[-1])
        x = float(np.interp(s, cum, D[:, 0])); y = float(np.interp(s, cum, D[:, 1]))
        i = min(max(int(np.searchsorted(cum, s)), 1), len(D)-2)
        gt[k] = [x, y, math.atan2(D[i+1, 1]-D[i-1, 1], D[i+1, 0]-D[i-1, 0])]
    rng = np.random.default_rng(1); omega = np.zeros(n)
    for k in range(1, n):
        omega[k] = wrap(gt[k, 2]-gt[k-1, 2])/DT + rng.normal(0, GYRO_NOISE)
    return gt, omega


def run(jac, gt, delay_steps):
    # Nominal gain trajectory: K = P H^T (H P H^T + R)^-1 depends only on the
    # covariance P, the measurement Jacobian H and R — not on the noise realisation.
    # H is evaluated on the smooth ground-truth viewing geometry, so the gain shows
    # its true convergence + geometry dependence along the route. The delay defers
    # the first D updates (covariance grows until measurements start being processed).
    n = len(gt); lx, ly = LM
    P = np.eye(3)*0.1; R = np.diag([R_R, R_B]); Q = np.diag([Q_XY, Q_XY, Q_TH])
    Kc = np.full((n, 3, 2), np.nan); upd = np.zeros(n, bool)
    for k in range(1, n):
        gx, gy, gth = gt[k]
        if jac:
            G = np.eye(3); G[0, 2] = -V*math.sin(gth)*DT; G[1, 2] = V*math.cos(gth)*DT; P = G@P@G.T + Q
        else:
            P = P + Q                     # KF: F = I covariance
        if k >= max(1, delay_steps):
            dx = lx-gx; dy = ly-gy; r = math.hypot(dx, dy)
            H = np.array([[-dx/r, -dy/r, 0.0], [dy/r/r, -dx/r/r, -1.0]])
            S = H@P@H.T + R; K = P@H.T@np.linalg.inv(S); P = (np.eye(3)-K@H)@P
            Kc[k] = K; upd[k] = True
    return Kc, upd


def shade(ax, t, upd):
    inr = False; ts = 0.0
    for i, u in enumerate(upd):
        if u and not inr: ts, inr = t[i], True
        elif not u and inr: ax.axvspan(ts, t[i], color="gold", alpha=0.10, zorder=0); inr = False
    if inr: ax.axvspan(ts, t[-1], color="gold", alpha=0.10, zorder=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/10_scurve/plots/kalman_gain.png")
    ap.add_argument("--delay-ms", type=float, default=0.0)
    a = ap.parse_args()
    gt, _ = make_truth(); t = np.arange(len(gt))*DT
    D = int(round(a.delay_ms/1000.0/DT))
    res = {"KF": run(False, gt, D), "EKF": run(True, gt, D)}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"hspace": 0.18})
    rows = [(0, r"$K_x$"), (1, r"$K_y$"), (2, r"$K_\theta$")]
    for ax, (gi, lbl) in zip(axes, rows):
        for name in ("KF", "EKF"):
            Kc, _ = res[name]
            ax.plot(t, Kc[:, gi, 0], "-",  color=C[name], lw=1.8, alpha=0.9, label=f"{name} range")
            ax.plot(t, Kc[:, gi, 1], "--", color=C[name], lw=1.8, alpha=0.9, label=f"{name} bearing")
        if np.mean(res["EKF"][1]) < 0.9:          # only shade if intermittent
            shade(ax, t, res["EKF"][1])
        ax.axhline(0, color="k", lw=0.5, alpha=0.4); ax.set_ylabel(f"{lbl} gain")
        ax.set_xlim(t[0], t[-1])
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=9)
    axes[-1].set_xlabel("time [s]")
    dly = f"  (measurement delay {a.delay_ms:.0f} ms)" if a.delay_ms > 0 else ""
    fig.suptitle("Kalman gain over time on the S-slalom  "
                 "(continuous landmark updates → gain converges)" + dly,
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=140, bbox_inches="tight"); print("saved", a.out)


if __name__ == "__main__":
    main()
