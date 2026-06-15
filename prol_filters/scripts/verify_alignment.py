#!/usr/bin/env python3
"""
verify_alignment.py — sim<->plot alignment check.
Runs the offline KF/EKF replicas (the same equations the C++ nodes use, with the
node's default noise: q_xy=0.001 q_theta=0.0005 r_range=0.005 r_bearing=0.01) on
the SAME scenario as a real logged run, and prints offline RMSE next to the real
run's RMSE. Close numbers + identical ordering == the offline-replica figures
faithfully represent the simulation.

Usage:  python3 verify_alignment.py --real /data/11_runtime --v 0.12
"""
import argparse, math, os
import numpy as np

WPX = [-0.3, 0.94, 2.02, 3.10, 4.2]; WPY = [-0.5, 1.05, -0.01, 1.05, 1.3]
LM = (1.8, 3.0); LM_R = 6.0
DT = 0.02; GYRO_NOISE = 0.012
R_R, R_B = 0.005, 0.01; Q_XY, Q_TH = 0.001, 0.0005


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


def make_truth(V):
    D, cum = build_path(WPX, WPY); n = int(cum[-1]/(V*DT)); gt = np.zeros((n, 3))
    for k in range(n):
        s = min(k*V*DT, cum[-1])
        x = float(np.interp(s, cum, D[:, 0])); y = float(np.interp(s, cum, D[:, 1]))
        i = min(max(int(np.searchsorted(cum, s)), 1), len(D)-2)
        gt[k] = [x, y, math.atan2(D[i+1, 1]-D[i-1, 1], D[i+1, 0]-D[i-1, 0])]
    rng = np.random.default_rng(0); omega = np.zeros(n)
    for k in range(1, n):
        omega[k] = wrap(gt[k, 2]-gt[k-1, 2])/DT + rng.normal(0, GYRO_NOISE)
    return gt, omega


def run(jac, gt, omega, V):
    rng = np.random.default_rng(42); n = len(gt); lx, ly = LM
    x = gt[0].copy(); P = np.eye(3)*0.1; R = np.diag([R_R, R_B]); Q = np.diag([Q_XY, Q_XY, Q_TH])
    est = np.zeros((n, 2)); est[0] = gt[0, :2]
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
                P = (np.eye(3)-K@H)@P
        est[k] = x[:2]
    return float(np.sqrt(np.mean((est[:, 0]-gt[:, 0])**2+(est[:, 1]-gt[:, 1])**2)))


def real_rmse(path):
    if not os.path.isfile(path):
        return None
    d = np.genfromtxt(path, delimiter=",", names=True, invalid_raise=False)
    e = d["pos_err"].astype(float); return float(np.sqrt(np.mean(e[np.isfinite(e)]**2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default="/data/11_runtime"); ap.add_argument("--v", type=float, default=0.12)
    a = ap.parse_args()
    gt, omega = make_truth(a.v)
    off = {"KF": run(False, gt, omega, a.v), "EKF": run(True, gt, omega, a.v)}
    real = {n: real_rmse(os.path.join(a.real, f"{n.lower()}_log.csv")) for n in ("KF", "EKF", "PF")}
    print(f"\nScenario: S-slalom, v={a.v} m/s, landmark {LM} r={LM_R} (always visible)\n")
    print(f"{'filter':6s} {'real RMSE [m]':>14s} {'offline RMSE [m]':>17s} {'Δ [m]':>9s}")
    for n in ("KF", "EKF", "PF"):
        rs = real.get(n); os_ = off.get(n)
        rs_s = f"{rs:.4f}" if rs is not None else "  n/a"
        os_s = f"{os_:.4f}" if os_ is not None else "  n/a (no offline PF)"
        dd = f"{abs(rs-os_):.4f}" if (rs is not None and os_ is not None) else ""
        print(f"{n:6s} {rs_s:>14s} {os_s:>17s} {dd:>9s}")


if __name__ == "__main__":
    main()
