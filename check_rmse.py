import numpy as np
for exp in ['01_baseline', '08_delay_100ms', '09_delay_500ms']:
    print(f'--- {exp} ---')
    for tag, f in [('KF','kf_log.csv'),('EKF','ekf_log.csv'),('PF','pf_log.csv')]:
        d = np.atleast_1d(np.genfromtxt(f'data/{exp}/{f}', delimiter=',', names=True))
        rmse = float(np.sqrt(np.mean((d['x']-d['gt_x'])**2 + (d['y']-d['gt_y'])**2)))
        print(f'  {tag:4s}  RMSE={rmse:.4f} m')
