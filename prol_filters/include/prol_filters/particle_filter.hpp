#pragma once
#include <Eigen/Dense>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace prol_filters {

/**
 * Particle Filter (Monte Carlo Localization, MCL) for unicycle robot.
 *
 * Implements the particle_filter algorithm from Thrun et al. (2006), Ch. 4.3.
 * Course: "05 PRO PF" slides 17–22.
 *
 * State per particle:  x^[m] = [x, y, θ]^T   (3-DOF pose)
 *
 * ─── ALGORITHM (slides 17–18) ──────────────────────────────────────────────
 *
 *  Input: χ_{t-1} = {x^[m]_{t-1}, w^[m]_{t-1}},  u_t,  z_t
 *  Output: χ_t
 *
 *  χ̄_t = {} (temporary set)
 *  for m = 1..M:
 *    [PREDICT] Line 4:  x̄^[m]_t ~ p(x_t | x^[m]_{t-1}, u_t)
 *    [WEIGHT ] Line 5:  w̄^[m]_t  = p(z_t | x̄^[m]_t)
 *    χ̄_t += { x̄^[m]_t, w̄^[m]_t }
 *  [RESAMPLE] Draw χ_t from χ̄_t with probability ∝ weight
 *  return χ_t
 *
 * ─── MOTION MODEL (slides "05 PRO PF" slide 20) ────────────────────────────
 *   sample_motion_model(u_t, x^[m]_{t-1}):
 *     v'     = v     + N(0, σ_v)       — add noise to linear velocity
 *     ω'     = ω     + N(0, σ_ω)       — add noise to angular velocity
 *     x^[m] += v' · cos(θ^[m]) · dt
 *     y^[m] += v' · sin(θ^[m]) · dt
 *     θ^[m] += ω' · dt
 *   Note: noise applied to INPUTS (v, ω), not directly to state.
 *   This models wheel slip and encoder noise at their source.
 *
 * ─── MEASUREMENT MODEL (slides "04 PRO Sensor and Motion Model" slide 12) ──
 *   landmark_model(z_t, x^[m]_t, m_j):
 *     r_pred  = sqrt((lx − x^[m])² + (ly − y^[m])²)
 *     φ_pred  = atan2(ly − y^[m], lx − x^[m]) − θ^[m]
 *     w^[m]  *= N(z_r; r_pred, σ_r) · N(z_φ; φ_pred, σ_φ)
 *
 * ─── RESAMPLING (slides "05 PRO PF" slide 16) ──────────────────────────────
 *   Low-variance (systematic) resampler:
 *     r ~ U[0, 1/M],  U_m = r + (m-1)/M
 *     select particle with cumulative weight ≥ U_m
 *   Benefits: lower variance than multinomial, O(M) complexity.
 *
 * ESS = 1/Σ(w_i²) — effective sample size, monitors particle degeneracy.
 * Low ESS → most weight on few particles → consider increasing N.
 */
class ParticleFilter {
public:
    using Vec3 = Eigen::Vector3d;

    struct Particle {
        Vec3   state  = Vec3::Zero();
        double weight = 0.0;
    };

    std::vector<Particle> particles;

    double sigma_v     = 0.05;  // motion noise σ on v     [m/s]
    double sigma_omega = 0.02;  // motion noise σ on ω     [rad/s]
    double R_range     = 0.05;  // landmark range noise variance   [m²]
    double R_bearing   = 0.05;  // landmark bearing noise variance  [rad²]

    explicit ParticleFilter(int N = 500) {
        particles.resize(N);
        resetWeights();
    }

    void setMotionNoise(double sv, double so) {
        sigma_v = sv; sigma_omega = so;
    }

    void setMeasurementNoise(double r_range, double r_bearing) {
        R_range   = r_range;
        R_bearing = r_bearing;
    }

    void setState(const Vec3& init) {
        for (auto& p : particles) p.state = init;
        resetWeights();
    }

    // ── PREDICT: sample from motion model p(x_t | x_{t-1}, u_t) ─────────────
    // slides "05 PRO PF" slide 20 — noise on control inputs, then kinematics
    void predict(double v, double omega, double dt) {
        std::normal_distribution<> nv(0.0, sigma_v);
        std::normal_distribution<> no(0.0, sigma_omega);
        for (auto& p : particles) {
            const double v_n     = v     + nv(rng_);
            const double omega_n = omega + no(rng_);
            p.state(0) += v_n * std::cos(p.state(2)) * dt;
            p.state(1) += v_n * std::sin(p.state(2)) * dt;
            p.state(2) += omega_n * dt;
            p.state(2)  = wrapAngle(p.state(2));
        }
    }

    // ── WEIGHT + RESAMPLE: update importance weights, then resample ──────────
    // slides "05 PRO PF" slides 14–16
    void updateLandmark(double z_range, double z_bearing, double lx, double ly) {
        double w_sum = 0.0;
        for (auto& p : particles) {
            const double dx      = lx - p.state(0);
            const double dy      = ly - p.state(1);
            const double r_pred  = std::sqrt(dx*dx + dy*dy);
            const double phi_pred = wrapAngle(std::atan2(dy, dx) - p.state(2));

            const double e_r   = z_range   - r_pred;
            const double e_phi = wrapAngle(z_bearing - phi_pred);

            // Joint Gaussian likelihood  p(z | x^[m])
            p.weight *= std::exp(-0.5 * (e_r*e_r / R_range + e_phi*e_phi / R_bearing));
            w_sum += p.weight;
        }

        // Normalize weights
        if (w_sum > 1e-300)
            for (auto& p : particles) p.weight /= w_sum;
        else
            resetWeights();

        // Low-variance resampling — Algorithm 4.4 (Thrun 2006)
        resample();
    }

    // Weighted mean: arithmetic mean for x,y; circular mean for θ
    Vec3 getMeanState() const {
        Vec3   mean    = Vec3::Zero();
        double sin_sum = 0.0, cos_sum = 0.0;
        for (const auto& p : particles) {
            mean(0)  += p.weight * p.state(0);
            mean(1)  += p.weight * p.state(1);
            sin_sum  += p.weight * std::sin(p.state(2));
            cos_sum  += p.weight * std::cos(p.state(2));
        }
        mean(2) = std::atan2(sin_sum, cos_sum);
        return mean;
    }

    // ESS = 1/Σ(w_i²) — tracks particle diversity
    double effectiveSampleSize() const {
        double s = 0.0;
        for (const auto& p : particles) s += p.weight * p.weight;
        return (s > 0.0) ? 1.0 / s : 0.0;
    }

    int numParticles() const { return static_cast<int>(particles.size()); }

    void setRng(unsigned seed) { rng_.seed(seed); }

private:
    std::mt19937 rng_{44};

    void resetWeights() {
        const double w = 1.0 / static_cast<double>(particles.size());
        for (auto& p : particles) p.weight = w;
    }

    // Low-variance (systematic) resampler — slides "05 PRO PF" slide 16
    void resample() {
        const int M = static_cast<int>(particles.size());
        std::vector<Particle> new_p(M);
        std::uniform_real_distribution<> u01(0.0, 1.0 / M);
        double r = u01(rng_);
        double c = particles[0].weight;
        int i = 0;
        const double w_new = 1.0 / M;
        for (int m = 0; m < M; ++m) {
            double U = r + static_cast<double>(m) / M;
            while (U > c && i < M - 1) c += particles[++i].weight;
            new_p[m]        = particles[i];
            new_p[m].weight = w_new;
        }
        particles = std::move(new_p);
    }

    static double wrapAngle(double a) {
        while (a >  M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }
};

}  // namespace prol_filters
