/**
 * MPCC Types — Shared data structures for MPCC controller.
 *
 * Contains Config, Result, PathRef, Obstacle, and dynamics models
 * used by both the acados solver and the controller node.
 */

#ifndef MPCC_TYPES_H
#define MPCC_TYPES_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <functional>

namespace mpcc {

struct Config {
    int horizon = 10;  // Match reference (PolyCtrl 2025 K=10). Longer horizons cause oscillation with linearized QP.
    double dt = 0.1;
    double wheelbase = 0.256;
    double max_velocity = 1.2;
    double min_velocity = 0.0;
    double max_steering = 0.45;  // ±25.8° — hardware servo limit (ref uses π/6=30° but hardware clips at 0.45)

    // Legacy rate limits (used only by AckermannModel for simulation tests)
    double max_acceleration = 1.5;
    double max_steering_rate = 1.5;

    double reference_velocity = 0.45;

    // Cost weights — tuned via swerving diagnosis (Feb 2026)
    // Startup phase (3s) handles initial heading alignment with low steering damping.
    // Normal phase uses moderate steering_rate_weight=1.5 (ref 1.1, was 3.0).
    // Velocity tracking applied at k=0 only (ref R_ref at k=0), smoothness at k>0.
    double contour_weight = 15.0;      // ref q_c=1.8 — higher for tighter lateral tracking
    double lag_weight = 10.0;          // ref q_l=7.0
    double velocity_weight = 15.0;     // ref R_ref[0]=17.0 — tracks v_ref (applied at k=0 only)
    double steering_weight = 0.05;     // ref R_ref[1]=0.05 — tracks δ_ref=0 (no feedforward)
    double acceleration_weight = 0.01; // ref R_u[0]=0.005 — smoothness (Δv)²
    double steering_rate_weight = 1.5; // ref R_u[1]=1.1 — moderate damping (was 3.0, caused swerving)
    double jerk_weight = 0.0;          // not in reference
    double heading_weight = 0.0;       // NOT in reference — set to 0 to match. Was 2.0; caused swerving by fighting contour cost on curves (amplified by servo delay + TF latency)
    double progress_weight = 1.0;     // ref gamma=1.0 — rewards forward progress (-gamma*v*dt)

    // Startup ramp — enabled (3.0s). Reference uses different weights during first 3s:
    // Low steering damping + high progress weight for fast heading alignment.
    double startup_ramp_duration_s = 3.0;
    double startup_elapsed_s = 0.0;
    double startup_contour_weight = 1.0;
    double startup_lag_weight = 10.0;
    double startup_velocity_weight = 5.0;
    double startup_heading_weight = 1.0;  // Mild heading correction during startup only (3s). Reference has 0 everywhere, but our linearized QP benefits from explicit heading guidance at large errors. Decays to 0 via ramp.
    double startup_steering_rate_weight = 0.05;
    double startup_progress_weight = 5.0;    // ref gamma=5.0 during startup (aggressive progress)
    double startup_curvature_decay = -5.0;

    // Obstacle
    double robot_radius = 0.13;
    double safety_margin = 0.10;
    double obstacle_weight = 200.0;

    // Boundary — disabled (ref has 0; boundary cost fights contour cost on curves)
    double boundary_weight = 0.0;
    double boundary_default_width = 0.22;

    // Solver
    int max_sqp_iterations = 5;   // PyMPC uses 5; was 3 (insufficient for curve convergence)
    int max_qp_iterations = 20;   // qpOASES iteration limit (acados multiplies by 10 internally)
    double qp_tolerance = 1e-5;
};

struct PathRef {
    double x, y, cos_theta, sin_theta, curvature;
};

/**
 * Callback for re-projecting predicted positions onto the path.
 * Used by test files for setting up path lookup; the acados solver
 * uses theta_A as a state variable instead.
 */
struct PathLookup {
    using LookupFn = std::function<PathRef(double px, double py, double s_min, double* s_out)>;
    LookupFn lookup = nullptr;
    bool valid() const { return lookup != nullptr; }
};

struct Obstacle {
    double x, y, radius;
    double vx = 0.0, vy = 0.0;
};

struct ObstacleHalfspace {
    double nx, ny, b;
    double weight;
    bool active;
};

struct BoundaryConstraint {
    double nx, ny;
    double b_left, b_right;
};

struct Result {
    double v_cmd;
    double delta_cmd;
    double omega_cmd;
    std::vector<double> predicted_x;
    std::vector<double> predicted_y;
    std::vector<double> predicted_theta;
    double solve_time_us;
    bool success;
    double cost;
};

// Solver uses 3D state, 2D control
using Vec3 = Eigen::Vector3d;
using Vec2 = Eigen::Vector2d;
using Mat33 = Eigen::Matrix3d;
using Mat32 = Eigen::Matrix<double, 3, 2>;

// Legacy 5D types for AckermannModel (used by simulation tests)
using VecX = Eigen::Matrix<double, 5, 1>;
using VecU = Eigen::Matrix<double, 2, 1>;
using MatXX = Eigen::Matrix<double, 5, 5>;
using MatXU = Eigen::Matrix<double, 5, 2>;
using MatUX = Eigen::Matrix<double, 2, 5>;
using MatUU = Eigen::Matrix<double, 2, 2>;

/**
 * Kinematic bicycle model — matches reference MPCC.py exactly.
 * State: [x, y, θ] (3D)
 * Control: [v, δ] (direct speed and steering angle)
 *
 *   β = atan(tan(δ) / 2)
 *   ẋ = v · cos(θ + β)
 *   ẏ = v · sin(θ + β)
 *   θ̇ = v / L · tan(δ) · cos(β)
 */
class KinematicModel {
public:
    static constexpr int NX = 3;
    static constexpr int NU = 2;
    double L;

    KinematicModel(double wheelbase = 0.256) : L(wheelbase) {}

    Vec3 dynamics(const Vec3& x, const Vec2& u) const {
        double v = u(0), delta = u(1), theta = x(2);
        double beta = std::atan(std::tan(delta) / 2.0);
        double cos_beta = std::cos(beta);
        return Vec3(
            v * std::cos(theta + beta),
            v * std::sin(theta + beta),
            v / L * std::tan(delta) * cos_beta
        );
    }

    Vec3 rk4_step(const Vec3& x, const Vec2& u, double dt) const {
        auto k1 = dynamics(x, u);
        auto k2 = dynamics(x + 0.5 * dt * k1, u);
        auto k3 = dynamics(x + 0.5 * dt * k2, u);
        auto k4 = dynamics(x + dt * k3, u);
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    void linearize(const Vec3& x, const Vec2& u, double dt,
                   Mat33& A, Mat32& B, Vec3& c) const {
        double v = u(0), delta = u(1), theta = x(2);
        double tan_del = std::tan(delta);
        double beta = std::atan(tan_del / 2.0);
        double cos_del = std::cos(delta);
        double sec2_del = 1.0 / (cos_del * cos_del);
        double dbeta_ddelta = (sec2_del / 2.0) / (1.0 + tan_del * tan_del / 4.0);

        double cos_tb = std::cos(theta + beta);
        double sin_tb = std::sin(theta + beta);
        double sin_beta = std::sin(beta);
        double cos_beta = std::cos(beta);

        // df/dx (3x3)
        Mat33 Ac = Mat33::Zero();
        Ac(0, 2) = -v * sin_tb;
        Ac(1, 2) =  v * cos_tb;

        // df/du (3x2) — controls directly affect dynamics (unlike rate model)
        Mat32 Bc = Mat32::Zero();
        // df/dv
        Bc(0, 0) = cos_tb;
        Bc(1, 0) = sin_tb;
        Bc(2, 0) = tan_del * cos_beta / L;
        // df/dδ
        Bc(0, 1) = -v * sin_tb * dbeta_ddelta;
        Bc(1, 1) =  v * cos_tb * dbeta_ddelta;
        Bc(2, 1) = v / L * (sec2_del * cos_beta - tan_del * sin_beta * dbeta_ddelta);

        A = Mat33::Identity() + dt * Ac;
        B = dt * Bc;
        c = x + dt * dynamics(x, u) - A * x - B * u;
    }
};

/**
 * Legacy Ackermann model for simulation/testing (5D state, rate controls).
 */
class AckermannModel {
public:
    static constexpr int NX = 5;
    static constexpr int NU = 2;
    double L;

    AckermannModel(double wheelbase = 0.256) : L(wheelbase) {}

    VecX dynamics(const VecX& x, const VecU& u) const {
        VecX xdot;
        double v = x(3), delta = x(4), theta = x(2);
        double beta = std::atan(std::tan(delta) / 2.0);
        double cos_beta = std::cos(beta);
        xdot(0) = v * std::cos(theta + beta);
        xdot(1) = v * std::sin(theta + beta);
        xdot(2) = v / L * std::tan(delta) * cos_beta;
        xdot(3) = u(0);
        xdot(4) = u(1);
        return xdot;
    }

    VecX rk4_step(const VecX& x, const VecU& u, double dt) const {
        auto k1 = dynamics(x, u);
        auto k2 = dynamics(x + 0.5 * dt * k1, u);
        auto k3 = dynamics(x + 0.5 * dt * k2, u);
        auto k4 = dynamics(x + dt * k3, u);
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
};

}  // namespace mpcc

#endif  // MPCC_TYPES_H
