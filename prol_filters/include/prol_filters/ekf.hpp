#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace prol_filters {

/**
 * Extended Kalman Filter (EKF) for unicycle robot pose estimation
 * with landmark-based correction.
 *
 * Implements the EKF Localization algorithm from Thrun et al. (2006), Ch. 3.3.
 * Course: "PRO KF EKF" slides 42–43, "PRO Localization" slides 18–25.
 *
 * State:   x_t = [x, y, θ]^T      — 2D position + heading
 * Control: u_t = [v, ω]^T         — linear speed [m/s], angular rate [rad/s]
 *
 * ─── PREDICTION ─────────────────────────────────────────────────────────────
 *
 * Nonlinear motion model  g(u_t, x_{t-1})  — slides "PRO KF EKF" slide 37:
 *   g_1 = x_{t-1} + v · cos(θ_{t-1}) · dt
 *   g_2 = y_{t-1} + v · sin(θ_{t-1}) · dt
 *   g_3 = θ_{t-1} + ω · dt
 *
 * EKF Jacobian  G_t = ∂g/∂x  — slides "PRO KF EKF" slide 41:
 *   G_t = | 1   0   −v·sin(θ)·dt |
 *          | 0   1    v·cos(θ)·dt |
 *          | 0   0    1            |
 *
 * Covariance prediction:  P̄_t = G_t · P_{t-1} · G_t^T + Q
 *
 *   *** KEY DIFFERENCE vs. KF ***
 *   KF uses F = I, so off-diagonal terms are zero.
 *   EKF uses G_t ≠ I, which correctly propagates heading uncertainty
 *   into position uncertainty. This gives lower RMSE especially during turns.
 *
 * ─── UPDATE (landmark-based correction) ────────────────────────────────────
 *
 * Measurement model  h(x, m)  — slide "PRO Localization" slide 15:
 *   z = [r, φ] = [sqrt((lx-x)^2+(ly-y)^2),  atan2(ly-y, lx-x) − θ]
 *
 * Measurement Jacobian  H = ∂h/∂x:
 *   H = | −(lx-x)/r       −(ly-y)/r       0  |
 *       |  (ly-y)/r²      −(lx-x)/r²     −1  |
 *
 * Update  (slide "PRO Localization" slide 25):
 *   K_t = P̄ · H^T · (H · P̄ · H^T + R)^{-1}
 *   x_t = x̄_t + K_t · (z_t − h(x̄_t, m))
 *   P_t = (I − K_t·H) · P̄  [Joseph form]
 *
 * Localization behavior ("PRO Localization" slide 18):
 *   - Landmark in view:   correction applied → uncertainty shrinks
 *   - No landmark:        prediction only   → uncertainty grows
 */
class ExtendedKalmanFilter {
public:
    using Vec3  = Eigen::Vector3d;
    using Mat3  = Eigen::Matrix3d;
    using Vec2  = Eigen::Vector2d;
    using Mat2  = Eigen::Matrix2d;
    using Mat23 = Eigen::Matrix<double, 2, 3>;
    using Mat32 = Eigen::Matrix<double, 3, 2>;

    Vec3  x      = Vec3::Zero();
    Mat3  P      = Mat3::Identity();
    Mat3  Q      = Mat3::Zero();
    Mat32 last_K = Mat32::Zero();

    double R_range   = 0.05;
    double R_bearing = 0.05;

    ExtendedKalmanFilter() {
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

    // ── EKF Prediction ──────────────────────────────────────────────────────
    void predict(double v, double omega, double dt) {
        const double th = x(2);

        // Nonlinear motion model  g(u, x)
        x(0) += v * std::cos(th) * dt;
        x(1) += v * std::sin(th) * dt;
        x(2) += omega * dt;
        x(2)  = wrapAngle(x(2));

        // Jacobian  G_t = ∂g/∂x  (evaluated at current state before update)
        Mat3 G = Mat3::Identity();
        G(0, 2) = -v * std::sin(th) * dt;
        G(1, 2) =  v * std::cos(th) * dt;

        // Covariance prediction:  P̄ = G·P·G^T + Q
        P = G * P * G.transpose() + Q;
    }

    // ── EKF Update with landmark measurement ────────────────────────────────
    void updateLandmark(double z_range, double z_bearing, double lx, double ly) {
        const double dx = lx - x(0);
        const double dy = ly - x(1);
        const double r  = std::sqrt(dx*dx + dy*dy);
        if (r < 1e-6) return;

        // Predicted measurement  h(x̄, m)
        const double r_hat   = r;
        const double phi_hat = wrapAngle(std::atan2(dy, dx) - x(2));

        // Measurement Jacobian  H  (2×3)
        Mat23 H;
        H << -dx/r,      -dy/r,      0.0,
               dy/(r*r), -dx/(r*r), -1.0;

        Mat2 R_mat;
        R_mat << R_range, 0.0,
                 0.0,     R_bearing;

        Vec2 innov;
        innov << z_range   - r_hat,
                 wrapAngle(z_bearing - phi_hat);

        const Mat2  S = H * P * H.transpose() + R_mat;
        const Mat32 K = P * H.transpose() * S.inverse();

        x += K * innov;
        x(2) = wrapAngle(x(2));

        // Joseph form: P = (I−K·H)·P̄·(I−K·H)^T + K·R·K^T
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
