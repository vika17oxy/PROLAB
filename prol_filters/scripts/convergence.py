#!/usr/bin/env python3
"""
convergence.py — filter convergence rate from a wrong initial pose.

Each filter is initialised with a deliberate initial-pose error (default 1.0 m +
0.3 rad). With the landmark in view the measurement updates pull the estimate
back onto the true track; this script measures HOW FAST each filter converges.
Deterministic offline replicas with node-matched parameters, averaged over noise
seeds. Outputs the position-error-vs-time curves and the convergence time
(first instant the error drops below a threshold and stays there).

Outputs:  <data>/convergence.png   and   <data>/convergence.csv
Usage:    python3 convergence.py --data /data [--v 0.3] [--offset 1.0]
          [--thr 0.10] [--seeds 12]
"""
import argparse, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]
WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LM = (1.8, 3.0); LM_R = 6.0
DT = 0.02
GYRO_NOISE = 0.012
R_R, R_B = 0.005, 0.01
Q_XY, Q_TH = 0.001, 0.0005
SIGMA_V, SIGMA_W, NP = 0.03, 0.015, 500
C = {"KF": "tab:blue", "EKF": "tab:green", "PF": "tab:red"}

plt.rcParams.update({
    "axes.grid": True, "grid.color": "0.85", "axes.axisbelow": True,
    "axes.edgecolor": "0.4", "font.size": 12, "axes.titlesize": 14,
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


def make_truth(V, seed):
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


def run_gauss(jac, gt, omega, V, off, seed):
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    x = gt[0] + off; P = np.diag([1.0, 1.0, 0.3]); R = np.diag([R_R, R_B]); Q = np.diag([Q_XY, Q_XY, Q_TH])
    err = np.zeros(n); err[0] = math.hypot(off[0], off[1])
    for k in range(1, n):
        th = x[2]; x = x + [V*math.cos(th)*DT, V*math.sin(th)*DT, omega[k]*DT]; x[2] = wrap(x[2])
        if jac:
            G = np.eye(3); G[0, 2] = -V*math.sin(th)*DT; G[1, 2] = V*math.cos(th)*DT; P = G@P@G.T+Q
        else:
            P = P + Q
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
        err[k] = math.hypot(x[0]-gx, x[1]-gy)
    return err


def run_pf(gt, omega, V, off, seed):
    rng = np.random.default_rng(seed); n = len(gt); lx, ly = LM
    # spread the initial cloud around the (wrong) initial guess
    Pp = gt[0] + off + rng.normal(0, [0.5, 0.5, 0.08], (NP, 3)); w = np.full(NP, 1.0/NP)
    err = np.zeros(n); err[0] = math.hypot(off[0], off[1])
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
        mx = np.average(Pp[:, 0], weights=w); my = np.average(Pp[:, 1], weights=w)
        err[k] = math.hypot(mx-gx, my-gy)
    return err


def conv_time(err, t, thr):
    # initial-recovery time: first instant the error drops below the threshold
    below = np.where(err < thr)[0]
    return float(t[below[0]]) if len(below) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--v", type=float, default=0.3)
    ap.add_argument("--offset", type=float, default=1.0, help="initial position error [m]")
    ap.add_argument("--thr", type=float, default=0.10, help="convergence threshold [m]")
    ap.add_argument("--seeds", type=int, default=12)
    a = ap.parse_args()
    off = np.array([a.offset/math.sqrt(2), a.offset/math.sqrt(2), 0.0])  # position-only offset

    n0 = len(make_truth(a.v, 0)[0]); t = np.arange(n0)*DT
    curves = {}
    for name in ("KF", "EKF", "PF"):
        acc = np.zeros(n0)
        for s in range(a.seeds):
            gt, om = make_truth(a.v, s)
            if name == "KF":  e = run_gauss(False, gt, om, a.v, off, 100+s)
            elif name == "EKF": e = run_gauss(True, gt, om, a.v, off, 100+s)
            else: e = run_pf(gt, om, a.v, off, 200+s)
            acc += e[:n0]
        curves[name] = acc/a.seeds

    ct = {n: conv_time(curves[n], t, a.thr) for n in curves}

    fig, ax = plt.subplots(figsize=(10, 6))
    for n in ("KF", "EKF", "PF"):
        rec = f"recovers {ct[n]:.2f} s" if ct[n] is not None else "no recovery"
        ax.plot(t, curves[n], color=C[n], lw=2,
                label=f"{n}  ({rec}, final {curves[n][-1]:.2f} m)")
    ax.axhline(a.thr, color="0.3", ls=":", lw=1.4, label=f"recovery threshold {a.thr:.2f} m")
    ax.set_xlabel("time [s]"); ax.set_ylabel("2D position error [m]")
    ax.set_title(f"Filter convergence from a {a.offset:.1f} m initial-pose error")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=10)
    fig.tight_layout()
    out = os.path.join(a.data, "convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")

    with open(os.path.join(a.data, "convergence.csv"), "w") as f:
        f.write("time_s,err_KF,err_EKF,err_PF\n")
        for i in range(n0):
            f.write(f"{t[i]:.3f},{curves['KF'][i]:.4f},{curves['EKF'][i]:.4f},{curves['PF'][i]:.4f}\n")

    print(f"\n=== Convergence ({a.offset} m initial error, v={a.v} m/s, {a.seeds} seeds) ===")
    for n in ("KF", "EKF", "PF"):
        c = f"{ct[n]:.2f} s" if ct[n] is not None else "never"
        print(f"  {n:4s}: recovers <{a.thr} m in {c}   final error {curves[n][-1]:.3f} m   "
              f"min {curves[n].min():.3f} m")
    print("saved", out)


if __name__ == "__main__":
    main()
