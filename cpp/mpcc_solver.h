/**
 * MPCC Solver - Model Predictive Contouring Control for QCar2
 *
 * Matches reference architecture (PolyCtrl 2025 MPCC.py):
 * - 3D kinematic state [x, y, θ] with direct controls [v, δ]
 * - Condensed QP + ADMM solver (from PyMPC/cpp_mpc)
 * - Contouring + lag cost, control smoothness (u[k]-u[k-1])²
 * - Obstacle avoidance via linearized halfspace QP constraints
 *
 * Key difference from previous version:
 * - Direct controls [speed, steering_angle] instead of rate controls [accel, steer_rate]
 * - 3D state instead of 5D — simpler, more robust, matches hardware interface
 * - No steering lag — solver sets steering angle directly each step
 * - Matches reference MPCC.py dynamics exactly
 */

#ifndef MPCC_SOLVER_H
#define MPCC_SOLVER_H

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

    // Cost weights — tuned via full-mission combined simulation (Feb 2026)
    // contour=20 + heading=3.0 gives 0.094m max CTE with 0% steering saturation
    double contour_weight = 20.0;      // ref q_c=1.8 — higher for tighter lateral tracking
    double lag_weight = 10.0;          // ref q_l=7.0
    double velocity_weight = 15.0;     // ref R_ref[0]=17.0 — tracks v_ref
    double steering_weight = 0.05;     // ref R_ref[1]=0.05 — tracks δ_ref=0 (no feedforward)
    double acceleration_weight = 0.01; // ref R_u[0]=0.005 — smoothness (Δv)²
    double steering_rate_weight = 1.0; // ref R_u[1]=1.1 — smooth steering, prevents bang-bang oscillation
    double jerk_weight = 0.0;          // not in reference
    double heading_weight = 3.0;       // not in reference; helps our QP solver align heading faster
    double progress_weight = 1.0;     // ref gamma=1.0 — rewards forward progress (-gamma*v*dt)

    // Startup ramp — disabled (0.0). The reference MPCC (PolyCtrl 2025) uses
    // constant weights from the first iteration. The ramp caused a 24x jump in
    // steering_rate_weight (0.05→1.2) over 3s, triggering oscillation at startup.
    double startup_ramp_duration_s = 0.0;
    double startup_elapsed_s = 0.0;
    double startup_contour_weight = 1.0;
    double startup_lag_weight = 10.0;
    double startup_velocity_weight = 5.0;
    double startup_heading_weight = 3.0;
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
    int max_qp_iterations = 20;   // PyMPC uses 200; *10 internally = 200 ADMM iterations
    double qp_tolerance = 1e-5;
};

struct PathRef {
    double x, y, cos_theta, sin_theta, curvature;
};

/**
 * Callback for re-projecting predicted positions onto the path.
 * This allows the solver to update path references during SQP iterations,
 * matching the reference MPCC's theta-as-decision-variable behavior.
 *
 * Given a predicted position (px, py) and a minimum arc-length (s_min),
 * returns the closest-point PathRef and the arc-length of that point.
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

/**
 * MPCC Solver — 3D kinematic model with direct controls.
 *
 * Matches reference MPCC.py architecture:
 * - Controls u = [v, δ] applied directly (no rate dynamics)
 * - Control smoothness via (u[k] - u[k-1])² (matching R_u)
 * - Velocity/steering tracking via (u[k] - u_ref)² (matching R_ref)
 * - Condensed QP + ADMM solver
 */
class Solver {
public:
    static constexpr int NX = 3;
    static constexpr int NU = 2;

    Config config;
    KinematicModel model;

    // Path lookup for adaptive reference re-projection during SQP
    PathLookup path_lookup;

    // For simulation tests — legacy 5D model
    AckermannModel ackermann_model;

    // Warm-start
    std::vector<Vec3> X_warm;
    std::vector<Vec2> U_warm;
    bool has_warmstart = false;

    // Previous control (for first-step smoothness, matching reference u_prev)
    Vec2 u_prev_ = Vec2(0.0, 0.0);

    // Current arc-length progress (for adaptive re-projection)
    double current_progress_ = 0.0;

    Solver() : model(0.256), ackermann_model(0.256) {}

    void init(const Config& cfg) {
        config = cfg;
        model = KinematicModel(cfg.wheelbase);
        ackermann_model = AckermannModel(cfg.wheelbase);
        has_warmstart = false;
        X_warm.resize(cfg.horizon + 1);
        U_warm.resize(cfg.horizon);
        u_prev_ = Vec2(0.2, 0.0);  // Start with min speed, zero steering
    }

    void reset() {
        has_warmstart = false;
        u_prev_ = Vec2(0.2, 0.0);
    }

    /**
     * Solve MPCC with 3D state [x, y, θ] and direct controls [v, δ].
     *
     * @param x0  Current vehicle state [x, y, θ]
     * @param path_refs  Path reference points (N+1)
     * @param obstacles  Obstacle list
     * @param boundaries  Road boundary constraints
     * @param measured_v  Measured vehicle velocity (used for u_prev, matching
     *                    reference MPC_node.py line 553: u_prev[0] = v)
     * @param measured_delta  Measured steering angle (last commanded)
     */
    Result solve(
        const Vec3& x0,
        const std::vector<PathRef>& path_refs,
        double current_progress,
        double /*path_total_length*/,
        const std::vector<Obstacle>& obstacles,
        const std::vector<BoundaryConstraint>& boundaries,
        double measured_v = -1.0,
        double measured_delta = -999.0)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        int N = config.horizon;
        current_progress_ = current_progress;
        Result result;
        result.success = false;

        // Update u_prev with measured values (matching reference MPC_node.py:553)
        // Reference: self.u_prev[0] = v (measured velocity)
        // This ensures smoothness cost (u[0] - u_prev)² uses actual vehicle state,
        // not the solver's previous command which may differ due to hardware lag.
        if (measured_v >= 0.0) {
            u_prev_(0) = measured_v;
        }
        if (measured_delta > -900.0) {
            u_prev_(1) = measured_delta;
        }

        std::vector<Vec3> X(N + 1);
        std::vector<Vec2> U(N);

        if (has_warmstart) {
            for (int k = 0; k < N; k++) {
                X[k] = (k + 1 < (int)X_warm.size()) ? X_warm[k + 1] : X_warm.back();
                U[k] = (k + 1 < (int)U_warm.size()) ? U_warm[k + 1] : U_warm.back();
            }
            X[N] = X_warm.back();
            X[0] = x0;
            // Clamp controls and re-propagate
            for (int k = 0; k < N; k++) {
                U[k](0) = clamp(U[k](0), config.min_velocity, config.max_velocity);
                U[k](1) = clamp(U[k](1), -config.max_steering, config.max_steering);
                X[k + 1] = model.rk4_step(X[k], U[k], config.dt);
            }
        } else {
            // Cold start: use measured velocity/steering for initial trajectory.
            // After reset between legs, u_prev_ = (0.2, 0.0). But if the vehicle
            // is already moving at 0.5+ m/s, the speed mismatch causes large
            // velocity tracking error in the first solve. Use measured values
            // when available (measured_v >= 0 means valid measurement).
            double v0 = std::max(measured_v >= 0.0 ? measured_v : u_prev_(0), 0.15);
            double d0 = measured_delta > -900.0 ? measured_delta : u_prev_(1);
            X[0] = x0;
            for (int k = 0; k < N; k++) {
                U[k] = Vec2(v0, d0);
                X[k + 1] = model.rk4_step(X[k], U[k], config.dt);
            }
        }

        // Adaptive path references: if path_lookup is available, re-project
        // predicted positions onto the path each SQP iteration.
        // This matches the reference MPCC where theta (path progress) is a
        // decision variable — the solver adapts which path points to track
        // based on the current predicted trajectory. On tight curves, references
        // naturally tighten because predicted positions are closer together.
        std::vector<PathRef> adaptive_refs = path_refs;  // Start with pre-computed

        // SQP iterations
        for (int sqp = 0; sqp < config.max_sqp_iterations; sqp++) {
            // Re-project path references from predicted positions
            if (path_lookup.valid()) {
                double s_min = current_progress_;
                for (int k = 0; k <= N; k++) {
                    double s_out = s_min;
                    adaptive_refs[k] = path_lookup.lookup(
                        X[k](0), X[k](1), s_min, &s_out);
                    s_min = s_out;  // Monotonic: each ref must be ahead of previous
                }
            }

            std::vector<Mat33> As(N);
            std::vector<Mat32> Bs(N);
            std::vector<Vec3> cs(N);
            for (int k = 0; k < N; k++) {
                model.linearize(X[k], U[k], config.dt, As[k], Bs[k], cs[k]);
            }

            auto obs_hs = precompute_obstacle_halfspaces(X, obstacles);

            Eigen::VectorXd delta_u = solve_condensed_qp(
                X, U, As, Bs, adaptive_refs, obs_hs, boundaries);

            if (delta_u.norm() < config.qp_tolerance) break;

            // Line search with nonlinear rollout
            std::vector<Vec2> best_U = U;
            std::vector<Vec3> best_X = X;
            double best_cost = compute_total_cost(X, U, adaptive_refs, obs_hs, boundaries);
            bool improved = false;

            double alpha = 1.0;
            for (int ls = 0; ls < 4; ls++) {
                std::vector<Vec2> trial_U(N);
                for (int k = 0; k < N; k++) {
                    trial_U[k](0) = U[k](0) + alpha * delta_u(NU * k);
                    trial_U[k](1) = U[k](1) + alpha * delta_u(NU * k + 1);
                    trial_U[k](0) = clamp(trial_U[k](0), config.min_velocity, config.max_velocity);
                    trial_U[k](1) = clamp(trial_U[k](1), -config.max_steering, config.max_steering);
                }

                std::vector<Vec3> trial_X(N + 1);
                trial_X[0] = x0;
                for (int k = 0; k < N; k++) {
                    trial_X[k + 1] = model.rk4_step(trial_X[k], trial_U[k], config.dt);
                }

                // Re-project references for trial trajectory too
                std::vector<PathRef> trial_refs = adaptive_refs;
                if (path_lookup.valid()) {
                    double s_min = current_progress_;
                    for (int k = 0; k <= N; k++) {
                        double s_out = s_min;
                        trial_refs[k] = path_lookup.lookup(
                            trial_X[k](0), trial_X[k](1), s_min, &s_out);
                        s_min = s_out;
                    }
                }

                auto trial_hs = precompute_obstacle_halfspaces(trial_X, obstacles);
                double trial_cost = compute_total_cost(trial_X, trial_U, trial_refs, trial_hs, boundaries);

                if (trial_cost < best_cost) {
                    best_cost = trial_cost;
                    best_U = trial_U;
                    best_X = trial_X;
                    adaptive_refs = trial_refs;
                    improved = true;
                    break;
                }
                alpha *= 0.5;
            }

            if (improved) { U = best_U; X = best_X; }
            else break;
        }

        // Store warm-start and u_prev
        X_warm = X;
        U_warm = U;
        has_warmstart = true;
        u_prev_ = U[0];

        // Extract result — direct control output (no one-step delay!)
        result.v_cmd = clamp(U[0](0), config.min_velocity, config.max_velocity);
        result.delta_cmd = clamp(U[0](1), -config.max_steering, config.max_steering);

        if (std::abs(result.v_cmd) > 0.001) {
            result.omega_cmd = result.v_cmd * std::tan(result.delta_cmd) / config.wheelbase;
        } else {
            result.omega_cmd = 0.0;
        }

        result.predicted_x.resize(N + 1);
        result.predicted_y.resize(N + 1);
        result.predicted_theta.resize(N + 1);
        for (int k = 0; k <= N; k++) {
            result.predicted_x[k] = X[k](0);
            result.predicted_y[k] = X[k](1);
            result.predicted_theta[k] = X[k](2);
        }

        auto final_hs = precompute_obstacle_halfspaces(X, obstacles);
        result.cost = compute_total_cost(X, U, adaptive_refs, final_hs, boundaries);
        result.success = true;

        auto t_end = std::chrono::high_resolution_clock::now();
        result.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();

        return result;
    }

    // Overload: accept 5D state for backward compatibility with controller node
    // Extracts measured v and delta from 5D state for u_prev correction
    Result solve(
        const VecX& x0_5d,
        const std::vector<PathRef>& path_refs,
        double current_progress,
        double path_total_length,
        const std::vector<Obstacle>& obstacles,
        const std::vector<BoundaryConstraint>& boundaries)
    {
        Vec3 x0_3d(x0_5d(0), x0_5d(1), x0_5d(2));
        // Pass measured velocity and steering from 5D state
        return solve(x0_3d, path_refs, current_progress, path_total_length,
                     obstacles, boundaries, x0_5d(3), x0_5d(4));
    }

private:
    static double clamp(double v, double lo, double hi) {
        return std::max(lo, std::min(hi, v));
    }

    static double normalize_angle(double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    double startup_progress() const {
        if (config.startup_ramp_duration_s <= 0.0) return 1.0;
        return std::clamp(config.startup_elapsed_s / config.startup_ramp_duration_s, 0.0, 1.0);
    }

    static double lerp_weight(double startup_val, double normal_val, double progress) {
        return startup_val + progress * (normal_val - startup_val);
    }

    std::vector<std::vector<ObstacleHalfspace>> precompute_obstacle_halfspaces(
        const std::vector<Vec3>& X,
        const std::vector<Obstacle>& obstacles)
    {
        int N = config.horizon;
        std::vector<std::vector<ObstacleHalfspace>> halfspaces(N + 1);

        for (const auto& obs : obstacles) {
            double safe_r = obs.radius + config.robot_radius + config.safety_margin;
            for (int k = 0; k <= N; k++) {
                double ok_x = obs.x + k * config.dt * obs.vx;
                double ok_y = obs.y + k * config.dt * obs.vy;
                double dx = X[k](0) - ok_x;
                double dy = X[k](1) - ok_y;
                double dist = std::sqrt(dx * dx + dy * dy);

                if (dist > safe_r + 1.0) continue;

                ObstacleHalfspace hs;
                hs.active = true;
                if (dist < 1e-4) {
                    double pt = std::atan2(
                        X[k](1) - (k > 0 ? X[k-1](1) : X[k](1)),
                        X[k](0) - (k > 0 ? X[k-1](0) : X[k](0)));
                    hs.nx = -std::sin(pt);
                    hs.ny =  std::cos(pt);
                } else {
                    hs.nx = dx / dist;
                    hs.ny = dy / dist;
                }
                hs.b = hs.nx * ok_x + hs.ny * ok_y + safe_r;
                hs.weight = config.obstacle_weight;
                halfspaces[k].push_back(hs);
            }
        }
        return halfspaces;
    }

    PathRef get_path_ref(int k, const std::vector<PathRef>& refs, const Vec3& xk) {
        if (k < (int)refs.size()) return refs[k];
        if (!refs.empty()) return refs.back();
        PathRef r;
        r.x = xk(0); r.y = xk(1);
        r.cos_theta = std::cos(xk(2));
        r.sin_theta = std::sin(xk(2));
        r.curvature = 0.0;
        return r;
    }

    /**
     * Build and solve condensed QP for 3D kinematic model with direct controls.
     *
     * Decision variables: Δu[k] for k=0..N-1 where u=[v, δ]
     * Sensitivity: M[k][j] = ∂x[k]/∂u[j] is 3×2
     *
     * Cost terms:
     * 1. Contouring/lag: through position sensitivity (same as before)
     * 2. Heading: through θ sensitivity (same as before)
     * 3. Velocity tracking: wv*(u[k](0) - v_ref)² — diagonal on Δu
     * 4. Steering tracking: ws*(u[k](1) - δ_ff)² — diagonal on Δu
     * 5. Control smoothness: R_u*(u[k] - u[k-1])² — cross terms in Δu
     */
    Eigen::VectorXd solve_condensed_qp(
        const std::vector<Vec3>& X,
        const std::vector<Vec2>& U,
        const std::vector<Mat33>& As,
        const std::vector<Mat32>& Bs,
        const std::vector<PathRef>& path_refs,
        const std::vector<std::vector<ObstacleHalfspace>>& obs_hs,
        const std::vector<BoundaryConstraint>& boundaries)
    {
        int N = config.horizon;
        int n_dec = NU * N;

        double sp = startup_progress();
        double wc = lerp_weight(config.startup_contour_weight, config.contour_weight, sp);
        double wl = lerp_weight(config.startup_lag_weight, config.lag_weight, sp);
        double wv = lerp_weight(config.startup_velocity_weight, config.velocity_weight, sp);
        double wh = lerp_weight(config.startup_heading_weight, config.heading_weight, sp);
        double sr = lerp_weight(config.startup_steering_rate_weight, config.steering_rate_weight, sp);
        double curv_decay = lerp_weight(config.startup_curvature_decay, -0.4, sp);

        // Sensitivity matrices M_current[j] = M[k][j] (3×2)
        std::vector<Mat32> M_current(N, Mat32::Zero());

        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n_dec, n_dec);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(n_dec);

        // Count obstacle constraints
        int n_obs = 0;
        for (int k = 1; k <= N; k++)
            if (k < (int)obs_hs.size())
                n_obs += (int)obs_hs[k].size();

        Eigen::MatrixXd C_obs = Eigen::MatrixXd::Zero(n_obs, n_dec);
        Eigen::VectorXd d_obs = Eigen::VectorXd::Zero(n_obs);
        int ci = 0;

        // ==== State-dependent costs (contouring, lag, heading, boundaries, obstacles) ====
        for (int k = 1; k <= N; k++) {
            // Update sensitivity: M_new[j] = A[k-1]*M_current[j], M_new[k-1] = B[k-1]
            std::vector<Mat32> M_new(N, Mat32::Zero());
            for (int j = 0; j < k - 1; j++)
                M_new[j] = As[k - 1] * M_current[j];
            M_new[k - 1] = Bs[k - 1];
            M_current = M_new;

            PathRef ref = get_path_ref(k, path_refs, X[k]);

            double dx = X[k](0) - ref.x;
            double dy = X[k](1) - ref.y;
            double e_c0 = -ref.sin_theta * dx + ref.cos_theta * dy;
            double e_l0 =  ref.cos_theta * dx + ref.sin_theta * dy;

            // Build Jacobians through sensitivity matrices
            Eigen::RowVectorXd Jc = Eigen::RowVectorXd::Zero(n_dec);
            Eigen::RowVectorXd Jl = Eigen::RowVectorXd::Zero(n_dec);
            Eigen::RowVectorXd Jh = Eigen::RowVectorXd::Zero(n_dec);

            for (int j = 0; j < k; j++) {
                const auto& Mkj = M_current[j];
                int col = NU * j;
                Jc.segment<NU>(col) = -ref.sin_theta * Mkj.row(0) + ref.cos_theta * Mkj.row(1);
                Jl.segment<NU>(col) =  ref.cos_theta * Mkj.row(0) + ref.sin_theta * Mkj.row(1);
                Jh.segment<NU>(col) = Mkj.row(2);
            }

            double w = (k == N) ? 2.0 : 1.0;

            // Contouring + lag
            H += (2.0 * wc * w) * Jc.transpose() * Jc;
            g += (2.0 * wc * w * e_c0) * Jc.transpose();
            H += (2.0 * wl * w) * Jl.transpose() * Jl;
            g += (2.0 * wl * w * e_l0) * Jl.transpose();

            // Heading
            if (wh > 0.0) {
                double path_theta = std::atan2(ref.sin_theta, ref.cos_theta);
                double h_err0 = normalize_angle(X[k](2) - path_theta);
                H += (2.0 * wh * w) * Jh.transpose() * Jh;
                g += (2.0 * wh * w * h_err0) * Jh.transpose();
            }

            // Boundary penalty (soft)
            if (k < (int)boundaries.size()) {
                const auto& bd = boundaries[k];
                Eigen::MatrixXd Jpos(2, n_dec);
                Jpos.setZero();
                for (int j = 0; j < k; j++)
                    Jpos.block<2, NU>(0, NU * j) = M_current[j].block<2, NU>(0, 0);

                double lv = bd.nx * X[k](0) + bd.ny * X[k](1) - bd.b_left;
                if (lv > 0) {
                    Eigen::RowVectorXd Jb = bd.nx * Jpos.row(0) + bd.ny * Jpos.row(1);
                    H += (2.0 * config.boundary_weight) * Jb.transpose() * Jb;
                    g += (2.0 * config.boundary_weight * lv) * Jb.transpose();
                }
                double rv = -bd.nx * X[k](0) - bd.ny * X[k](1) - bd.b_right;
                if (rv > 0) {
                    Eigen::RowVectorXd Jb = -bd.nx * Jpos.row(0) - bd.ny * Jpos.row(1);
                    H += (2.0 * config.boundary_weight) * Jb.transpose() * Jb;
                    g += (2.0 * config.boundary_weight * rv) * Jb.transpose();
                }
            }

            // Obstacle halfspace constraints
            if (k < (int)obs_hs.size()) {
                for (const auto& hs : obs_hs[k]) {
                    if (!hs.active) continue;
                    Eigen::RowVectorXd Cn = Eigen::RowVectorXd::Zero(n_dec);
                    for (int j = 0; j < k; j++)
                        Cn.segment<NU>(NU * j) = hs.nx * M_current[j].row(0)
                                                + hs.ny * M_current[j].row(1);
                    if (ci < n_obs) {
                        C_obs.row(ci) = Cn;
                        d_obs(ci) = hs.b - (hs.nx * X[k](0) + hs.ny * X[k](1));
                        ci++;
                    }
                }
            }
        }

        // ==== Control-dependent costs (velocity tracking, steering tracking, smoothness) ====
        for (int k = 0; k < N; k++) {
            PathRef ref = get_path_ref(k, path_refs, X[k]);

            // Velocity tracking: wv * (v[k] - v_ref)²
            // v[k] = U[k](0), so this is diagonal on Δu[k](0)
            double v_base = lerp_weight(0.20, config.reference_velocity, sp);
            double v_ref = v_base * std::exp(curv_decay * std::abs(ref.curvature));
            v_ref = std::clamp(v_ref, config.min_velocity, config.max_velocity);
            double v_err = U[k](0) - v_ref;
            H(NU * k, NU * k) += 2.0 * wv;
            g(NU * k) += 2.0 * wv * v_err;

            // Steering tracking: ws * (δ[k] - 0)² — reference tracks δ_ref=0
            // Reference MPCC.py uses u_ref = [v_ref, 0] — NO feedforward.
            // The solver is purely error-driven: contouring/lag/heading costs
            // determine steering. This weight just regularizes toward zero.
            H(NU * k + 1, NU * k + 1) += 2.0 * config.steering_weight;
            g(NU * k + 1) += 2.0 * config.steering_weight * U[k](1);

            // Progress reward: -gamma * v[k] * dt (matching reference MPCC.py line 90)
            // In reference, V[k] is arc-length progress speed (decision variable).
            // Here, v[k] = U[k](0) is vehicle speed. Since v[k] = U0[k] + Δu[k](0),
            // the linear cost -gamma*v*dt contributes -gamma*dt to gradient on Δu[k](0).
            double wp = lerp_weight(config.startup_progress_weight, config.progress_weight, sp);
            g(NU * k) -= wp * config.dt;

            // Control smoothness: (u[k] - u[k-1])² matching reference R_u
            // For k=0: smoothness against u_prev_ (matching reference R_u_prev)
            // For k>0: smoothness against u[k-1] (matching reference R_u)
            Vec2 u_prev_k = (k == 0) ? u_prev_ : U[k - 1];
            Vec2 du = U[k] - u_prev_k;

            // Speed smoothness (R_u[0] = acceleration_weight)
            H(NU * k, NU * k) += 2.0 * config.acceleration_weight;
            g(NU * k) += 2.0 * config.acceleration_weight * du(0);
            // Steering smoothness (R_u[1] = steering_rate_weight)
            H(NU * k + 1, NU * k + 1) += 2.0 * sr;
            g(NU * k + 1) += 2.0 * sr * du(1);

            // Cross terms for k>0: (u[k]-u[k-1])² adds coupling
            if (k > 0) {
                int ik = NU * k, ikm = NU * (k - 1);
                // Speed coupling
                H(ikm, ikm) += 2.0 * config.acceleration_weight;
                H(ik, ikm)  -= 2.0 * config.acceleration_weight;
                H(ikm, ik)  -= 2.0 * config.acceleration_weight;
                g(ikm) -= 2.0 * config.acceleration_weight * du(0);
                // Steering coupling
                H(ikm + 1, ikm + 1) += 2.0 * sr;
                H(ik + 1, ikm + 1)  -= 2.0 * sr;
                H(ikm + 1, ik + 1)  -= 2.0 * sr;
                g(ikm + 1) -= 2.0 * sr * du(1);
            }
        }

        // Regularize
        H.diagonal().array() += 1e-6;

        // Box constraints: v ∈ [min_v, max_v], δ ∈ [-max_steer, max_steer]
        Eigen::VectorXd lb(n_dec), ub(n_dec);
        for (int k = 0; k < N; k++) {
            lb(NU * k) = config.min_velocity - U[k](0);
            ub(NU * k) = config.max_velocity - U[k](0);
            lb(NU * k + 1) = -config.max_steering - U[k](1);
            ub(NU * k + 1) = config.max_steering - U[k](1);
        }

        Eigen::MatrixXd C = C_obs.topRows(ci);
        Eigen::VectorXd d = d_obs.head(ci);

        return solve_admm_qp(H, g, C, d, lb, ub);
    }

    Eigen::VectorXd solve_admm_qp(
        const Eigen::MatrixXd& H,
        const Eigen::VectorXd& g_vec,
        const Eigen::MatrixXd& C,
        const Eigen::VectorXd& d,
        const Eigen::VectorXd& lb,
        const Eigen::VectorXd& ub)
    {
        int n = (int)H.rows();
        int m = (int)C.rows();
        int m_total = m + n;

        if (m == 0) {
            Eigen::LLT<Eigen::MatrixXd> llt(H);
            if (llt.info() != Eigen::Success) {
                Eigen::MatrixXd Hr = H + 1e-6 * Eigen::MatrixXd::Identity(n, n);
                llt.compute(Hr);
            }
            Eigen::VectorXd x = llt.solve(-g_vec);
            // Apply box constraints
            for (int i = 0; i < n; i++)
                x(i) = std::clamp(x(i), lb(i), ub(i));
            return x;
        }

        Eigen::MatrixXd A_aug(m_total, n);
        A_aug.topRows(m) = C;
        A_aug.bottomRows(n) = Eigen::MatrixXd::Identity(n, n);

        double rho = 1.0;
        const int max_iter = config.max_qp_iterations * 10;
        const double abs_tol = config.qp_tolerance;
        const double rel_tol = 1e-3;

        Eigen::VectorXd x = Eigen::VectorXd::Zero(n);
        Eigen::VectorXd z = Eigen::VectorXd::Zero(m_total);
        Eigen::VectorXd lambda = Eigen::VectorXd::Zero(m_total);

        auto factorize = [&](double rv) -> Eigen::LLT<Eigen::MatrixXd> {
            Eigen::MatrixXd KKT = H + rv * A_aug.transpose() * A_aug;
            KKT.diagonal().array() += 1e-8;
            return Eigen::LLT<Eigen::MatrixXd>(KKT);
        };

        auto llt = factorize(rho);

        for (int iter = 0; iter < max_iter; iter++) {
            Eigen::VectorXd z_prev = z;

            Eigen::VectorXd rhs = -g_vec + rho * A_aug.transpose() * (z - lambda);
            x = llt.solve(rhs);

            Eigen::VectorXd Ax = A_aug * x;
            // Project: ineq >= d, box [lb, ub]
            z = Ax + lambda;
            for (int i = 0; i < m; i++) z(i) = std::max(z(i), d(i));
            for (int i = 0; i < n; i++) z(m + i) = std::clamp(z(m + i), lb(i), ub(i));

            Eigen::VectorXd r = Ax - z;
            lambda += r;

            double pri = r.norm();
            double dua = (rho * A_aug.transpose() * (z - z_prev)).norm();
            double eps_p = abs_tol * std::sqrt(m_total) + rel_tol * std::max(Ax.norm(), z.norm());
            double eps_d = abs_tol * std::sqrt(n) + rel_tol * (rho * A_aug.transpose() * lambda).norm();

            if (pri < eps_p && dua < eps_d) break;

            if (iter > 0 && iter % 10 == 0) {
                double ratio = pri / (dua + 1e-10);
                if (ratio > 10.0) {
                    rho = std::min(rho * 2.0, 1e6);
                    llt = factorize(rho);
                    lambda *= 0.5;
                } else if (ratio < 0.1) {
                    rho = std::max(rho / 2.0, 1e-6);
                    llt = factorize(rho);
                    lambda *= 2.0;
                }
            }
        }
        return x;
    }

    double compute_total_cost(
        const std::vector<Vec3>& X,
        const std::vector<Vec2>& U,
        const std::vector<PathRef>& path_refs,
        const std::vector<std::vector<ObstacleHalfspace>>& obs_hs,
        const std::vector<BoundaryConstraint>& boundaries)
    {
        int N = config.horizon;
        double total = 0.0;
        double sp = startup_progress();
        double wc = lerp_weight(config.startup_contour_weight, config.contour_weight, sp);
        double wl = lerp_weight(config.startup_lag_weight, config.lag_weight, sp);
        double wv = lerp_weight(config.startup_velocity_weight, config.velocity_weight, sp);
        double wh = lerp_weight(config.startup_heading_weight, config.heading_weight, sp);
        double sr = lerp_weight(config.startup_steering_rate_weight, config.steering_rate_weight, sp);
        double curv_decay = lerp_weight(config.startup_curvature_decay, -0.4, sp);

        for (int k = 0; k <= N; k++) {
            PathRef ref = get_path_ref(k, path_refs, X[k]);
            double w = (k == N) ? 2.0 : 1.0;
            double dx = X[k](0) - ref.x, dy = X[k](1) - ref.y;
            double e_c = -ref.sin_theta * dx + ref.cos_theta * dy;
            double e_l =  ref.cos_theta * dx + ref.sin_theta * dy;
            total += w * wc * e_c * e_c;
            total += w * wl * e_l * e_l;

            if (wh > 0.0) {
                double h_err = normalize_angle(X[k](2) - std::atan2(ref.sin_theta, ref.cos_theta));
                total += w * wh * h_err * h_err;
            }

            if (k < (int)obs_hs.size()) {
                for (const auto& hs : obs_hs[k]) {
                    if (!hs.active) continue;
                    double viol = hs.b - (hs.nx * X[k](0) + hs.ny * X[k](1));
                    if (viol > 0) total += hs.weight * viol * viol;
                }
            }
            if (k < (int)boundaries.size()) {
                const auto& bd = boundaries[k];
                double lv = bd.nx * X[k](0) + bd.ny * X[k](1) - bd.b_left;
                if (lv > 0) total += config.boundary_weight * lv * lv;
                double rv = -bd.nx * X[k](0) - bd.ny * X[k](1) - bd.b_right;
                if (rv > 0) total += config.boundary_weight * rv * rv;
            }
        }

        for (int k = 0; k < N; k++) {
            PathRef ref = get_path_ref(k, path_refs, X[k]);
            double v_base = lerp_weight(0.20, config.reference_velocity, sp);
            double v_ref = v_base * std::exp(curv_decay * std::abs(ref.curvature));
            v_ref = std::clamp(v_ref, config.min_velocity, config.max_velocity);
            total += wv * (U[k](0) - v_ref) * (U[k](0) - v_ref);

            // Steering tracking toward zero (no feedforward) — matches reference u_ref=[v,0]
            total += config.steering_weight * U[k](1) * U[k](1);

            // Progress reward: -gamma * v[k] * dt
            double wp = lerp_weight(config.startup_progress_weight, config.progress_weight, sp);
            total -= wp * U[k](0) * config.dt;

            Vec2 u_prev_k = (k == 0) ? u_prev_ : U[k - 1];
            Vec2 du = U[k] - u_prev_k;
            total += config.acceleration_weight * du(0) * du(0);
            total += sr * du(1) * du(1);
        }
        return total;
    }
};

}  // namespace mpcc


// ============================================================================
// C API for Python ctypes binding
// ============================================================================

extern "C" {

struct MPCCParams {
    int horizon;
    double dt, wheelbase;
    double max_velocity, min_velocity, max_steering;
    double max_acceleration, max_steering_rate;
    double reference_velocity;
    double contour_weight, lag_weight, velocity_weight;
    double steering_weight, acceleration_weight, steering_rate_weight;
    double jerk_weight;
    double robot_radius, safety_margin, obstacle_weight;
    double boundary_weight;
    int max_sqp_iterations, max_qp_iterations;
    double qp_tolerance;
};

struct MPCCPathPoint {
    double x, y, cos_theta, sin_theta, curvature;
};

struct MPCCObstacle {
    double x, y, radius;
    double vx, vy;
};

struct MPCCBoundary {
    double nx, ny, b_left, b_right;
};

struct MPCCResultC {
    double v_cmd, delta_cmd, omega_cmd;
    double solve_time_us;
    int success;
    double predicted_x[50];
    double predicted_y[50];
    double predicted_theta[50];
    int predicted_len;
};

void* mpcc_create(const MPCCParams* params);
void mpcc_destroy(void* solver);
void mpcc_reset(void* solver);

int mpcc_solve(
    void* solver,
    const double* state,
    const MPCCPathPoint* path,
    int n_path,
    const MPCCObstacle* obstacles,
    int n_obstacles,
    const MPCCBoundary* boundaries,
    int n_boundaries,
    double current_progress,
    double path_total_length,
    MPCCResultC* result);

}  // extern "C"

#endif  // MPCC_SOLVER_H
