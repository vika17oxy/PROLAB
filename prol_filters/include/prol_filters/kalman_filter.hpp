#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace prol_filters {

/**
 * Linear Kalman Filter (KF) for unicycle robot pose estimation.
 *
 * Implements the Kalman Filter algorithm from Thrun et al. (2006), Ch. 3.2.
 * Course: "PRO KF EKF" slides 25–26.
 *
 * State:   x_t = [x, y, theta]^T     — 2D position + heading
 * Control: u_t = [v, omega]^T        — linear speed [m/s], angular rate [rad/s]
 *
 * ─── PREDICTION (algorithm lines 2–3) ─────────────────────────────────────
 *
 * Line 2:  x̄_t = g(u_t, x_{t-1})    — nonlinear unicycle kinematics
 *              x  += v · cos(θ) · dt
 *              y  += v · sin(θ) · dt
 *              θ  += ω · dt
 *
 * Line 3:  P̄_t = F · P_{t-1} · F^T + Q
 *
 *   *** KF APPROXIMATION: F = I (identity) ***
 *   The true Jacobian of g w.r.t. x is G_t (used in EKF):
 *     G_t = | 1  0  -v·sin(θ)·dt |
 *           | 0  1   v·cos(θ)·dt |
 *           | 0  0   1            |
 *   The KF ignores the off-diagonal terms (θ→position coupling),
 *   so: P̄_t = P_{t-1} + Q
 *   This causes KF to underestimate uncertainty during turns.
 *
 * ─── UPDATE (algorithm lines 4–6) ──────────────────────────────────────────
 *
 * Landmark measurement z = [r, φ]:
 *   r = sqrt((lx-x)^2 + (ly-y)^2)      — range to landmark
 *   φ = atan2(ly-y, lx-x) - θ           — bearing to landmark (robot frame)
 *
 * Measurement Jacobian H = ∂h/∂x:
 *   H = | -(lx-x)/r       -(ly-y)/r       0  |
 *       |  (ly-y)/r²      -(lx-x)/r²     -1  |
 *
 * Line 4:  K_t = P̄ · H^T · (H · P̄ · H^T + R)^{-1}
 * Line 5:  x_t = x̄_t + K_t · (z_t − h(x̄_t))
 * Line 6:  P_t = (I − K_t·H) · P̄  [Joseph form for numerical stability]
 *
 * Key limitation vs. EKF:
 *   F = I misses the θ→(x,y) coupling, so heading errors are not propagated
 *   to position uncertainty. EKF fixes this with the full Jacobian G_t.
 */
class KalmanFilter {
public:
    using Vec3  = Eigen::Vector3d;
    using Mat3  = Eigen::Matrix3d;
    using Vec2  = Eigen::Vector2d;
    using Mat2  = Eigen::Matrix2d;
    using Mat23 = Eigen::Matrix<double, 2, 3>;
    using Mat32 = Eigen::Matrix<double, 3, 2>;

    Vec3  x      = Vec3::Zero();      // state mean  [x, y, θ]
    Mat3  P      = Mat3::Identity();  // state covariance
    Mat3  Q      = Mat3::Zero();      // process noise covariance
    Mat32 last_K = Mat32::Zero();     // last Kalman gain (for CSV logging)

    double R_range   = 0.05;   // landmark range noise variance   [m²]
    double R_bearing = 0.05;   // landmark bearing noise variance  [rad²]

    KalmanFilter() {
        Q.diagonal() << 0.001, 0.001, 0.0005;
    }

    void setProcessNoise(double q_xy, double q_theta) {
        Q = Mat3::Zero();
        Q(0, 0) = q_xy;
        Q(1, 1) = q_xy;
        Q(2, 2) = q_theta;
    }

    void setMeasurementNoise(double r_range, double r_bearing) {
        R_range   = r_range;
        R_bearing = r_bearing;
    }

    // ── KF Prediction  (algorithm lines 2–3) ───────────────────────────────
    void predict(double v, double omega, double dt) {
        // Line 2: nonlinear unicycle kinematics  g(u, x)
        x(0) += v * std::cos(x(2)) * dt;
        x(1) += v * std::sin(x(2)) * dt;
        x(2) += omega * dt;
        x(2)  = wrapAngle(x(2));

        // Line 3: P̄ = F·P·F^T + Q  with  F = I  (KF approximation, no Jacobian)
        P = P + Q;
    }

    // ── KF Update  (algorithm lines 4–6) ───────────────────────────────────
    // z_range   : noisy measured range to landmark  [m]
    // z_bearing : noisy measured bearing to landmark [rad], robot frame
    // lx, ly    : known landmark position in world frame  [m]
    void updateLandmark(double z_range, double z_bearing, double lx, double ly) {
        const double dx = lx - x(0);
        const double dy = ly - x(1);
        const double r  = std::sqrt(dx*dx + dy*dy);
        if (r < 1e-6) return;

        // Predicted measurement  h(x̄)
        const double r_hat   = r;
        const double phi_hat = wrapAngle(std::atan2(dy, dx) - x(2));

        // Measurement Jacobian  H  (2×3)
        Mat23 H;
        H << -dx/r,      -dy/r,      0.0,
               dy/(r*r), -dx/(r*r), -1.0;

        // Measurement noise  R  (2×2)
        Mat2 R_mat;
        R_mat << R_range, 0.0,
                 0.0,     R_bearing;

        // Innovation  ν = z − h(x̄)
        Vec2 innov;
        innov << z_range   - r_hat,
                 wrapAngle(z_bearing - phi_hat);

        // Line 4: Kalman gain  K = P̄·H^T·(H·P̄·H^T + R)^{-1}
        const Mat2  S = H * P * H.transpose() + R_mat;
        const Mat32 K = P * H.transpose() * S.inverse();

        // Line 5: x = x̄ + K·ν
        x += K * innov;
        x(2) = wrapAngle(x(2));

        // Line 6: P = (I−K·H)·P̄  — Joseph form ensures P stays positive-definite
        const Mat3 IKH = Mat3::Identity() - K * H;
        P = IKH * P * IKH.transpose() + K * R_mat * K.transpose();

        last_K = K;
    }

    Vec3  getState()      const { return x; }
    Mat3  getCovariance() const { return P; }
    Mat32 getLastK()      const { return last_K; }
    void  setState(const Vec3& s)      { x = s; }
    void  setCovariance(const Mat3& c) { P = c; }

private:
    static double wrapAngle(double a) {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }
};

}  // namespace prol_filters
