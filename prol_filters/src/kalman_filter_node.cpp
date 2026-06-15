/**
 * kalman_filter_node.cpp
 *
 * ROS2 node for the linear Kalman Filter (KF).
 *
 * State:   [x, y, θ]   — unicycle, 3-DOF (matches "PRO KF EKF" slides)
 * Control: u = [v, ω]  — v from parameter, ω from IMU gyroscope z-axis
 *
 * KF vs EKF difference:
 *   Covariance propagation uses F = I (identity), omitting the Jacobian G_t.
 *   EKF uses the full G_t which correctly captures θ→position coupling.
 *   See kalman_filter.hpp for the detailed derivation.
 *
 * Time-delayed measurements (specific task):
 *   IMU messages are buffered in a std::deque.  A message is only processed
 *   after (now − stamp) ≥ measurement_delay_ms / 1000 s.
 *   This simulates a delayed sensor pipeline and degrades accuracy.
 *
 * Landmark detection:
 *   A single user-defined landmark at (landmark_x, landmark_y) is used.
 *   The robot "sees" the landmark when the ground-truth distance is < landmark_radius.
 *   Noisy range + bearing measurements are generated and passed to the KF update.
 *
 * Topics published:  /kf/pose, /kf/odom, /kf/rmse, /kf/landmark_marker
 * Parameters:        q_xy, q_theta, r_landmark, r_bearing,
 *                    measurement_delay_ms, landmark_{x,y,radius},
 *                    initial_{x,y,theta,vx}, log_csv
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <deque>
#include <fstream>
#include <iomanip>
#include <random>
#include <cmath>
#include <chrono>

#include "prol_filters/kalman_filter.hpp"

class KalmanFilterNode : public rclcpp::Node {
public:
    KalmanFilterNode() : Node("kalman_filter_node") {
        declareParams();
        loadParams();
        initRos();
        RCLCPP_INFO(get_logger(),
            "KF node ready | v=%.2f m/s | delay=%.0f ms | lm=(%.2f,%.2f) r=%.2f | log=%s",
            v_, delay_ms_, lx_, ly_, l_radius_, log_csv_ ? "on" : "off");
    }

    ~KalmanFilterNode() { if (csv_.is_open()) csv_.close(); }

private:
    // ── Types ────────────────────────────────────────────────────────────────
    struct ImuStamped { rclcpp::Time stamp; double omega; };

    // ── Filter and buffer ────────────────────────────────────────────────────
    prol_filters::KalmanFilter kf_;
    std::deque<ImuStamped>     buf_;        // time-delay buffer
    rclcpp::Time               last_t_;
    bool                       initialized_{false};

    // ── Parameters ───────────────────────────────────────────────────────────
    double v_, delay_ms_, lx_, ly_, l_radius_, r_lm_, r_bearing_;
    bool   log_csv_{false};

    // ── Ground truth ─────────────────────────────────────────────────────────
    double gt_x_{0.0}, gt_y_{0.0}, gt_theta_{0.0};
    bool   gt_received_{false};

    // ── Logging state ─────────────────────────────────────────────────────────
    double omega_cur_{0.0};
    bool   had_update_{false};
    double rmse_sum_{0.0};
    int    rmse_n_{0};
    double update_ms_{0.0};   // per-tick predict+update wall-clock [ms]

    // ── ROS interfaces ───────────────────────────────────────────────────────
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr            imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr          gt_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr             odom_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr              rmse_pub_;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr     lm_pub_;

    std::ofstream csv_;
    std::mt19937  rng_{42};

    // ── Parameter declaration ─────────────────────────────────────────────────
    void declareParams() {
        declare_parameter("q_xy",                 0.001);
        declare_parameter("q_theta",              0.0005);
        declare_parameter("r_landmark",           0.005);
        declare_parameter("r_bearing",            0.01);
        declare_parameter("measurement_delay_ms", 0.0);
        declare_parameter("landmark_x",           0.5);
        declare_parameter("landmark_y",          -1.2);
        declare_parameter("landmark_radius",      1.5);
        declare_parameter("initial_x",            0.0);
        declare_parameter("initial_y",           -2.7);
        declare_parameter("initial_theta",       -1.5708);
        declare_parameter("initial_vx",           0.3);
        declare_parameter("log_csv",              false);
    }

    // ── Load parameters into filter and node members ──────────────────────────
    void loadParams() {
        kf_.setProcessNoise(
            get_parameter("q_xy").as_double(),
            get_parameter("q_theta").as_double());

        r_lm_      = get_parameter("r_landmark").as_double();
        r_bearing_ = get_parameter("r_bearing").as_double();
        kf_.setMeasurementNoise(r_lm_, r_bearing_);

        delay_ms_ = get_parameter("measurement_delay_ms").as_double();
        lx_       = get_parameter("landmark_x").as_double();
        ly_       = get_parameter("landmark_y").as_double();
        l_radius_ = get_parameter("landmark_radius").as_double();
        v_        = get_parameter("initial_vx").as_double();
        log_csv_  = get_parameter("log_csv").as_bool();

        prol_filters::KalmanFilter::Vec3 x0;
        x0 << get_parameter("initial_x").as_double(),
              get_parameter("initial_y").as_double(),
              get_parameter("initial_theta").as_double();
        kf_.setState(x0);
        kf_.setCovariance(prol_filters::KalmanFilter::Mat3::Identity() * 0.1);

        if (log_csv_) {
            csv_.open("kf_log.csv");
            csv_ << "time_s,x,y,theta,vx,vy,omega_gyro,cov_trace,"
                    "gt_x,gt_y,gt_theta,pos_err,had_update,"
                    "k00,k10,k20,k01,k11,k21,update_ms\n";
        }
    }

    // ── ROS subscriptions and publishers ─────────────────────────────────────
    void initRos() {
        auto qos  = rclcpp::SensorDataQoS();
        imu_sub_  = create_subscription<sensor_msgs::msg::Imu>(
            "/imu", qos, [this](sensor_msgs::msg::Imu::SharedPtr m){ onImu(m); });
        gt_sub_   = create_subscription<nav_msgs::msg::Odometry>(
            "/ground_truth", 10, [this](nav_msgs::msg::Odometry::SharedPtr m){ onGT(m); });
        pose_pub_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/kf/pose", 10);
        odom_pub_ = create_publisher<nav_msgs::msg::Odometry>("/kf/odom", 10);
        rmse_pub_ = create_publisher<std_msgs::msg::Float64>("/kf/rmse", 10);
        lm_pub_   = create_publisher<visualization_msgs::msg::Marker>("/kf/landmark_marker", 10);
    }

    // ── IMU callback: enqueue message and drain buffer ────────────────────────
    void onImu(sensor_msgs::msg::Imu::SharedPtr msg) {
        buf_.push_back({rclcpp::Time(msg->header.stamp), msg->angular_velocity.z});
        drainBuffer(rclcpp::Time(msg->header.stamp));
    }

    // ── Drain time-delay buffer ───────────────────────────────────────────────
    // Process all buffered IMU messages older than delay_ms_.
    // When delay_ms_ = 0 this is a pass-through.
    void drainBuffer(const rclcpp::Time& now) {
        const double delay_s = delay_ms_ * 1e-3;
        while (!buf_.empty()) {
            const auto& e = buf_.front();
            if ((now - e.stamp).seconds() < delay_s) break;

            if (!initialized_) {
                last_t_ = e.stamp;
                initialized_ = true;
                buf_.pop_front();
                continue;
            }

            const double dt = (e.stamp - last_t_).seconds();
            buf_.pop_front();
            if (dt <= 0.0 || dt > 1.0) { last_t_ = e.stamp; continue; }

            omega_cur_ = e.omega;

            const auto t_start = std::chrono::steady_clock::now();
            // ── KF Predict ─────────────────────────────────────────────────
            kf_.predict(v_, e.omega, dt);

            // ── KF Update: landmark measurement ────────────────────────────
            bool detected = false;
            if (gt_received_) {
                const double dx   = lx_ - gt_x_;
                const double dy   = ly_ - gt_y_;
                const double dist = std::sqrt(dx*dx + dy*dy);
                if (dist < l_radius_) {
                    // Generate noisy range + bearing from ground truth
                    std::normal_distribution<> nr(0.0, std::sqrt(r_lm_));
                    std::normal_distribution<> nb(0.0, std::sqrt(r_bearing_));
                    double bearing = std::atan2(dy, dx) - gt_theta_;
                    while (bearing >  M_PI) bearing -= 2.0 * M_PI;
                    while (bearing < -M_PI) bearing += 2.0 * M_PI;
                    kf_.updateLandmark(dist + nr(rng_), bearing + nb(rng_), lx_, ly_);
                    detected    = true;
                    had_update_ = true;
                }
            }
            update_ms_ = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_start).count();

            last_t_ = e.stamp;
            publish(e.stamp, detected);
        }
    }

    // ── Ground-truth callback: RMSE computation ───────────────────────────────
    void onGT(nav_msgs::msg::Odometry::SharedPtr msg) {
        gt_x_ = msg->pose.pose.position.x;
        gt_y_ = msg->pose.pose.position.y;
        const auto& q = msg->pose.pose.orientation;
        gt_theta_    = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                                  1.0 - 2.0*(q.y*q.y + q.z*q.z));
        gt_received_ = true;

        if (!initialized_) return;

        const auto   st = kf_.getState();
        const double ex = st(0) - gt_x_;
        const double ey = st(1) - gt_y_;
        rmse_sum_ += ex*ex + ey*ey;
        ++rmse_n_;

        std_msgs::msg::Float64 rmsg;
        rmsg.data = std::sqrt(rmse_sum_ / rmse_n_);
        rmse_pub_->publish(rmsg);
    }

    // ── Publish state and log ─────────────────────────────────────────────────
    void publish(const rclcpp::Time& stamp, bool detected) {
        const auto   st   = kf_.getState();
        const auto   cov  = kf_.getCovariance();
        const double half = st(2) * 0.5;

        // PoseWithCovarianceStamped
        geometry_msgs::msg::PoseWithCovarianceStamped pm;
        pm.header.stamp    = stamp;
        pm.header.frame_id = "odom";
        pm.pose.pose.position.x    = st(0);
        pm.pose.pose.position.y    = st(1);
        pm.pose.pose.orientation.z = std::sin(half);
        pm.pose.pose.orientation.w = std::cos(half);
        pm.pose.covariance.fill(0.0);
        pm.pose.covariance[0]  = cov(0, 0);
        pm.pose.covariance[7]  = cov(1, 1);
        pm.pose.covariance[35] = cov(2, 2);
        pose_pub_->publish(pm);

        // Odometry
        nav_msgs::msg::Odometry om;
        om.header         = pm.header;
        om.child_frame_id = "base_link";
        om.pose           = pm.pose;
        om.twist.twist.linear.x = v_;
        odom_pub_->publish(om);

        // Landmark visualization marker
        visualization_msgs::msg::Marker mk;
        mk.header = pm.header;
        mk.ns   = "kf_landmark"; mk.id = 0;
        mk.type   = visualization_msgs::msg::Marker::CYLINDER;
        mk.action = visualization_msgs::msg::Marker::ADD;
        mk.pose.position.x = lx_; mk.pose.position.y = ly_;
        mk.pose.position.z = 0.5; mk.pose.orientation.w = 1.0;
        mk.scale.x = 0.3; mk.scale.y = 0.3; mk.scale.z = 1.0;
        mk.color.r = detected ? 0.0f : 1.0f;
        mk.color.g = detected ? 1.0f : 0.3f;
        mk.color.b = 0.0f; mk.color.a = 0.9f;
        lm_pub_->publish(mk);

        // CSV log
        if (log_csv_ && csv_.is_open()) {
            const double trace   = cov.trace();
            const double pos_err = gt_received_
                ? std::sqrt((st(0)-gt_x_)*(st(0)-gt_x_) +
                            (st(1)-gt_y_)*(st(1)-gt_y_))
                : 0.0;
            const auto K = kf_.getLastK();
            csv_ << std::fixed << std::setprecision(6)
                 << stamp.seconds() << ","
                 << st(0) << "," << st(1) << "," << st(2) << ","
                 << v_*std::cos(st(2)) << "," << v_*std::sin(st(2)) << ","
                 << omega_cur_ << "," << trace << ","
                 << gt_x_ << "," << gt_y_ << "," << gt_theta_ << ","
                 << pos_err << "," << (had_update_ ? 1 : 0) << ","
                 << K(0,0) << "," << K(1,0) << "," << K(2,0) << ","
                 << K(0,1) << "," << K(1,1) << "," << K(2,1) << ","
                 << update_ms_ << "\n";
            had_update_ = false;
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<KalmanFilterNode>());
    rclcpp::shutdown();
    return 0;
}
