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
    double max_velocity = 0.55;  // Hard speed ceiling — close to reference_velocity to prevent overshoot
    double min_velocity = 0.0;
    double max_steering = 0.45;  // ±25.8° — hardware servo limit (ref uses π/6=30° but hardware clips at 0.45)

    // Legacy rate limits (used only by AckermannModel for simulation tests)
    double max_acceleration = 1.5;
    double max_steering_rate = 1.5;

    double reference_velocity = 0.45;

    // Cost weights — intermediate between reference ratio and deployment-tested values.
    // Reference (CasADi IPOPT) uses q_c:q_l = 1.8:7.0 = 0.26, but acados EXTERNAL cost
    // has different scaling. contour=4 was too weak (CTE 0.673m), contour=15 too aggressive
    // (CTE 0.360m + swerving). contour=8 is the compromise.
    double contour_weight = 8.0;       // Intermediate: 4→too weak, 15→too aggressive
    double lag_weight = 15.0;          // Strong progress to prevent circling
    double velocity_weight = 15.0;     // ref R_ref[0]=17.0 — tracks v_ref at all stages
    double steering_weight = 0.05;     // ref R_ref[1]=0.05 — tracks δ_ref=0 (no feedforward)
    double acceleration_weight = 0.01; // ref R_u[0]=0.005 — smoothness (Δv)²
    double steering_rate_weight = 1.1; // ref R_u[1]=1.1 — match reference exactly
    double jerk_weight = 0.0;          // not in reference
    double heading_weight = 0.0;       // NOT in reference — set to 0. Fights contour on curves.
    double progress_weight = 1.0;     // ref gamma=1.0 — rewards forward progress (-gamma*v*dt)

    // Startup ramp — 1.5s. Brief ramp for initial alignment.
    // Startup weights: slightly more conservative, then quickly transition to normal.
    // Steering is INSTANT (direct servo, no PID) — hardware applies commands within 15ms.
    double startup_ramp_duration_s = 1.5;
    double startup_elapsed_s = 0.0;
    double startup_contour_weight = 6.0;      // Slightly less than normal (8) during startup
    double startup_lag_weight = 12.0;          // Slightly less progress at start
    double startup_velocity_weight = 15.0;     // Same as normal
    double startup_heading_weight = 0.5;       // Mild heading correction during startup only
    double startup_steering_rate_weight = 2.0; // Slightly more damping at start (normal=1.1)
    double startup_progress_weight = 1.0;      // Normal
    double startup_curvature_decay = -0.4;     // Match reference exactly (was -1.0)

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

    // Diagnostics logging (deployment + test instrumentation)
    bool diagnostics_enabled = false;    // Extract solver convergence data after each solve
    bool per_stage_logging = false;      // Log per-stage trajectory + references (expensive)
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

struct SolverDiagnostics {
    // Convergence (from acados via ocp_nlp_get)
    int acados_status = -1;     // Raw return code (0=converged, 2=max_iter, 3/4=MINSTEP)
    int sqp_iter = 0;
    double kkt_norm_inf = -1.0;
    int qp_status = -1;

    // Residuals (matching mpc_planner extraction)
    double res_eq = -1.0;       // Dynamics violation
    double res_ineq = -1.0;     // Constraint violation (obstacle/boundary)
    double res_comp = -1.0;
    double res_stat = -1.0;

    // Timing (acados internal)
    double acados_time_tot_ms = 0.0;
    double acados_time_qp_ms = 0.0;

    // Warmstart
    bool warmstart_used = false;
    int warmstart_shift_count = 0;

    // Startup ramp
    double startup_progress = 0.0;  // 0=startup, 1=normal

    // Effective weights (after ramp interpolation)
    double eff_contour_w = 0.0;
    double eff_lag_w = 0.0;
    double eff_vel_w = 0.0;
    double eff_sr_w = 0.0;
    double eff_progress_w = 0.0;
    double eff_v_ref_k0 = 0.0;  // Curvature-adapted v_ref at first stage

    // Per-stage predicted trajectory (4D state, 3D control)
    std::vector<double> stage_x, stage_y, stage_psi, stage_theta_a;  // N+1
    std::vector<double> stage_v, stage_delta, stage_v_theta;          // N

    // Per-stage references sent to solver
    std::vector<double> ref_x, ref_y, ref_v, ref_curv;  // N+1

    // Obstacle data
    double obs_x = 0, obs_y = 0, obs_r = 0, obs_dist = -1;
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
    SolverDiagnostics diag;
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
