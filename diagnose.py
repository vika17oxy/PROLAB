import numpy as np

d_kf  = np.atleast_1d(np.genfromtxt('data/01_baseline/kf_log.csv',  delimiter=',', names=True))
d_ekf = np.atleast_1d(np.genfromtxt('data/01_baseline/ekf_log.csv', delimiter=',', names=True))
d_pf  = np.atleast_1d(np.genfromtxt('data/01_baseline/pf_log.csv',  delimiter=',', names=True))

print("GT circle:  x=[{:.2f},{:.2f}]  y=[{:.2f},{:.2f}]".format(
    d_kf['gt_x'].min(), d_kf['gt_x'].max(), d_kf['gt_y'].min(), d_kf['gt_y'].max()))
print("KF  est:    x=[{:.2f},{:.2f}]  y=[{:.2f},{:.2f}]  theta=[{:.2f},{:.2f}]".format(
    d_kf['x'].min(), d_kf['x'].max(), d_kf['y'].min(), d_kf['y'].max(),
    d_kf['theta'].min(), d_kf['theta'].max()))
print("EKF est:    x=[{:.2f},{:.2f}]  y=[{:.2f},{:.2f}]  theta=[{:.2f},{:.2f}]".format(
    d_ekf['x'].min(), d_ekf['x'].max(), d_ekf['y'].min(), d_ekf['y'].max(),
    d_ekf['theta'].min(), d_ekf['theta'].max()))
print("PF  est:    x=[{:.2f},{:.2f}]  y=[{:.2f},{:.2f}]".format(
    d_pf['x'].min(), d_pf['x'].max(), d_pf['y'].min(), d_pf['y'].max()))

# Check bearing: first few rows
print("\nFirst 5 rows: gt_x, gt_y, kf_x, kf_y, kf_theta, gt_theta")
for i in range(5):
    print(f"  t={d_kf['time_s'][i]:.2f}  gt=({d_kf['gt_x'][i]:.3f},{d_kf['gt_y'][i]:.3f})"
          f"  kf=({d_kf['x'][i]:.3f},{d_kf['y'][i]:.3f})  theta_kf={d_kf['theta'][i]:.3f}"
          f"  theta_gt={d_kf['gt_theta'][i]:.3f}")

# Theta drift
print(f"\nKF  theta range: [{d_kf['theta'].min():.2f}, {d_kf['theta'].max():.2f}]")
print(f"GT  theta range: [{d_kf['gt_theta'].min():.2f}, {d_kf['gt_theta'].max():.2f}]")
print(f"EKF theta range: [{d_ekf['theta'].min():.2f}, {d_ekf['theta'].max():.2f}]")

# Mean position error
print(f"\nMean KF  pos error: {np.mean(d_kf['pos_err']):.3f} m")
print(f"Mean EKF pos error: {np.mean(d_ekf['pos_err']):.3f} m")
print(f"Mean PF  pos error: {np.mean(d_pf['pos_err']):.3f} m")
