/**
 * MPCC Solver - Model Predictive Contouring Control for QCar2
 *
 * High-performance C++ implementation using Eigen.
 * Uses proper iLQR (iterative Linear Quadratic Regulator) with:
 * - Full Hessian for x-y coupling (not diagonal approximation)
 * - Feedback gains (K matrix) in forward rollout
 * - Cost-based line search for robust convergence
 * - Bicycle model with slip angle (matching reference repo)
 *
 * Key features:
 * - Ackermann dynamics with slip angle beta
 * - RK4 for simulation, Euler for Jacobian linearization
 * - Contouring + lag cost with proper x-y Hessian coupling
 * - Linearized obstacle avoidance (soft penalty)
 * - Road boundary soft constraints
 * - Warm-starting from previous solution
 * - ~150us solve time for N=25 horizon
 */

#ifndef MPCC_SOLVER_H
#define MPCC_SOLVER_H

#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace mpcc {

struct Config {
    int horizon = 20;
    double dt = 0.1;
    double wheelbase = 0.256;
    double max_velocity = 0.40;
    double min_velocity = 0.0;
    double max_steering = 0.45;
    double max_acceleration = 0.6;
    double max_steering_rate = 0.6;
    double reference_velocity = 0.35;

    // Cost weights — CONTOUR > LAG to prevent lane violations.
    // Previous ratio (8:15) caused vehicle to prioritize forward progress
    // over staying centered in lane, resulting in corner cutting.
    double contour_weight = 25.0;
    double lag_weight = 5.0;
    double velocity_weight = 2.0;
    double steering_weight = 3.0;
    double acceleration_weight = 1.5;
    double steering_rate_weight = 4.0;
    double jerk_weight = 0.5;

    // Startup ramp: for the first startup_ramp_duration_s seconds,
    // use lower reference velocity to let the controller settle.
    double startup_ramp_duration_s = 3.0;
    double startup_elapsed_s = 0.0;  // Set by the caller each solve

    // Obstacle
    double robot_radius = 0.13;
    double safety_margin = 0.10;
    double obstacle_weight = 200.0;

    // Boundary
    double boundary_weight = 30.0;

    // Road half-width for boundary generation
    double boundary_default_width = 0.22;

    // Solver
    int max_sqp_iterations = 3;
    int max_qp_iterations = 10;
    double qp_tolerance = 1e-5;
};

struct PathRef {
    double x, y, cos_theta, sin_theta, curvature;
};

struct Obstacle {
    double x, y, radius;
};

struct BoundaryConstraint {
    double nx, ny;  // Normal vector
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

// Type aliases for clarity
using VecX = Eigen::Matrix<double, 5, 1>;
using VecU = Eigen::Matrix<double, 2, 1>;
using MatXX = Eigen::Matrix<double, 5, 5>;
using MatXU = Eigen::Matrix<double, 5, 2>;
using MatUX = Eigen::Matrix<double, 2, 5>;
using MatUU = Eigen::Matrix<double, 2, 2>;

/**
 * Ackermann/bicycle dynamics for QCar2.
 * State: [x, y, theta, v, delta] (5D)
 * Control: [a, delta_dot] (2D)
 *
 * Uses bicycle model with slip angle beta for accuracy:
 *   beta = atan(tan(delta) / 2)
 *   dx/dt = v * cos(theta + beta)
 *   dy/dt = v * sin(theta + beta)
 *   dtheta/dt = v / L * sin(beta)
 *   dv/dt = a
 *   ddelta/dt = delta_dot
 */
class AckermannModel {
public:
    static constexpr int NX = 5;
    static constexpr int NU = 2;

    double L;  // wheelbase

    AckermannModel(double wheelbase = 0.256) : L(wheelbase) {}

    // Continuous dynamics: dx/dt = f(x, u) with slip angle
    VecX dynamics(const VecX& x, const VecU& u) const
    {
        VecX xdot;
        double v = x(3);
        double delta = x(4);
        double theta = x(2);

        // Slip angle: beta = atan(tan(delta) / 2)
        // This accounts for CG being at L/2 from rear axle
        double beta = std::atan(std::tan(delta) / 2.0);

        xdot(0) = v * std::cos(theta + beta);
        xdot(1) = v * std::sin(theta + beta);
        xdot(2) = v / L * std::sin(beta);
        xdot(3) = u(0);  // acceleration
        xdot(4) = u(1);  // steering rate
        return xdot;
    }

    // RK4 step
    VecX rk4_step(const VecX& x, const VecU& u, double dt) const
    {
        auto k1 = dynamics(x, u);
        auto k2 = dynamics(x + 0.5 * dt * k1, u);
        auto k3 = dynamics(x + 0.5 * dt * k2, u);
        auto k4 = dynamics(x + dt * k3, u);
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }

    // Linearize: x_{k+1} = A*x_k + B*u_k + c
    void linearize(
        const VecX& x_ref,
        const VecU& u_ref,
        double dt,
        MatXX& A,
        MatXU& B,
        VecX& c) const
    {
        double v = x_ref(3);
        double delta = x_ref(4);
        double theta = x_ref(2);

        // Slip angle and its derivatives
        double tan_del = std::tan(delta);
        double beta = std::atan(tan_del / 2.0);
        double cos_del = std::cos(delta);
        double sec2_del = 1.0 / (cos_del * cos_del);
        // d(beta)/d(delta) = (sec^2(delta) / 2) / (1 + tan^2(delta)/4)
        double dbeta_ddelta = (sec2_del / 2.0) / (1.0 + tan_del * tan_del / 4.0);

        double cos_tb = std::cos(theta + beta);
        double sin_tb = std::sin(theta + beta);
        double sin_beta = std::sin(beta);
        double cos_beta = std::cos(beta);

        // Jacobian df/dx (continuous)
        MatXX Ac = MatXX::Zero();
        // dx/dtheta = -v * sin(theta + beta)
        Ac(0, 2) = -v * sin_tb;
        // dx/dv = cos(theta + beta)
        Ac(0, 3) = cos_tb;
        // dx/ddelta = -v * sin(theta + beta) * dbeta_ddelta
        Ac(0, 4) = -v * sin_tb * dbeta_ddelta;

        // dy/dtheta = v * cos(theta + beta)
        Ac(1, 2) = v * cos_tb;
        // dy/dv = sin(theta + beta)
        Ac(1, 3) = sin_tb;
        // dy/ddelta = v * cos(theta + beta) * dbeta_ddelta
        Ac(1, 4) = v * cos_tb * dbeta_ddelta;

        // dtheta/dv = sin(beta) / L
        Ac(2, 3) = sin_beta / L;
        // dtheta/ddelta = v / L * cos(beta) * dbeta_ddelta
        Ac(2, 4) = v / L * cos_beta * dbeta_ddelta;

        // Jacobian df/du (continuous)
        MatXU Bc = MatXU::Zero();
        Bc(3, 0) = 1.0;  // dv/da = 1
        Bc(4, 1) = 1.0;  // ddelta/ddelta_dot = 1

        // Euler discretization
        A = MatXX::Identity() + dt * Ac;
        B = dt * Bc;

        // Affine term: c = x_ref + dt*f(x_ref, u_ref) - A*x_ref - B*u_ref
        auto f_ref = dynamics(x_ref, u_ref);
        c = x_ref + dt * f_ref - A * x_ref - B * u_ref;
    }
};

/**
 * MPCC Solver using proper iLQR with full Hessian.
 *
 * Key improvements over previous version:
 * 1. Full NX x NX Hessian (not diagonal) for proper x-y coupling
 * 2. Feedback gains K in forward rollout
 * 3. Cost-based line search
 * 4. Bicycle model with slip angle
 */
class Solver {
public:
    static constexpr int NX = AckermannModel::NX;
    static constexpr int NU = AckermannModel::NU;

    Config config;
    AckermannModel model;

    // Warm-start trajectory
    std::vector<VecX> X_warm;
    std::vector<VecU> U_warm;
    bool has_warmstart = false;

    Solver() : model(0.256) {}

    void init(const Config& cfg) {
        config = cfg;
        model = AckermannModel(cfg.wheelbase);
        has_warmstart = false;
        X_warm.resize(cfg.horizon + 1);
        U_warm.resize(cfg.horizon);
    }

    void reset() {
        has_warmstart = false;
    }

    Result solve(
        const VecX& x0,
        const std::vector<PathRef>& path_refs,
        double current_progress,
        double path_total_length,
        const std::vector<Obstacle>& obstacles,
        const std::vector<BoundaryConstraint>& boundaries)
    {
        auto t_start = std::chrono::high_resolution_clock::now();

        int N = config.horizon;
        Result result;
        result.success = false;

        // Initialize trajectory
        std::vector<VecX> X(N + 1);
        std::vector<VecU> U(N);

        if (has_warmstart) {
            // Shift warm-start forward by one step
            for (int k = 0; k < N; k++) {
                X[k] = (k + 1 < (int)X_warm.size()) ? X_warm[k + 1] : X_warm.back();
                U[k] = (k + 1 < (int)U_warm.size()) ? U_warm[k + 1] : U_warm.back();
            }
            X[N] = X_warm.back();
            X[0] = x0;
        } else {
            // Cold start: straight-line prediction
            double v0 = std::max(x0(3), 0.1);
            for (int k = 0; k <= N; k++) {
                X[k] = x0;
                X[k](0) = x0(0) + k * config.dt * v0 * std::cos(x0(2));
                X[k](1) = x0(1) + k * config.dt * v0 * std::sin(x0(2));
                X[k](3) = v0;
            }
            for (int k = 0; k < N; k++) {
                U[k] = VecU::Zero();
            }
        }

        // SQP iterations: re-linearize and solve iLQR
        for (int sqp = 0; sqp < config.max_sqp_iterations; sqp++) {
            // Linearize dynamics around current trajectory
            std::vector<MatXX> As(N);
            std::vector<MatXU> Bs(N);
            std::vector<VecX> cs(N);

            for (int k = 0; k < N; k++) {
                model.linearize(X[k], U[k], config.dt, As[k], Bs[k], cs[k]);
            }

            // Run iLQR backward-forward pass
            bool converged = solve_ilqr(x0, X, U, As, Bs, cs, path_refs,
                                         obstacles, boundaries);

            // Re-simulate forward with nonlinear (RK4) dynamics
            X[0] = x0;
            for (int k = 0; k < N; k++) {
                U[k](0) = clamp(U[k](0), -config.max_acceleration, config.max_acceleration);
                U[k](1) = clamp(U[k](1), -config.max_steering_rate, config.max_steering_rate);
                X[k + 1] = model.rk4_step(X[k], U[k], config.dt);
                X[k + 1](3) = clamp(X[k + 1](3), config.min_velocity, config.max_velocity);
                X[k + 1](4) = clamp(X[k + 1](4), -config.max_steering, config.max_steering);
            }

            if (converged) break;
        }

        // Store warm-start
        X_warm = X;
        U_warm = U;
        has_warmstart = true;

        // Extract result (use step 1, not step 0, for one-step delay)
        result.v_cmd = clamp(X[1](3), 0.0, config.max_velocity);
        result.delta_cmd = clamp(X[1](4), -config.max_steering, config.max_steering);

        // Compute angular velocity for Twist message
        if (std::abs(result.v_cmd) > 0.001) {
            result.omega_cmd = result.v_cmd * std::tan(result.delta_cmd) / config.wheelbase;
        } else {
            result.omega_cmd = 0.0;
        }

        // Store predicted trajectory
        result.predicted_x.resize(N + 1);
        result.predicted_y.resize(N + 1);
        result.predicted_theta.resize(N + 1);
        for (int k = 0; k <= N; k++) {
            result.predicted_x[k] = X[k](0);
            result.predicted_y[k] = X[k](1);
            result.predicted_theta[k] = X[k](2);
        }

        result.cost = compute_total_cost(X, U, path_refs, obstacles, boundaries);
        result.success = true;

        auto t_end = std::chrono::high_resolution_clock::now();
        result.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();

        return result;
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

    /**
     * Compute stage cost, gradient, and FULL Hessian for a state.
     * Returns the scalar cost value.
     *
     * Uses full NX x NX Hessian (not diagonal) to properly couple
     * x and y in the contouring/lag cost.
     */
    double compute_state_cost_full(
        const VecX& xk,
        const PathRef& ref,
        const std::vector<Obstacle>& obstacles,
        const std::vector<BoundaryConstraint>& boundaries,
        int k,
        VecX& grad_x,
        MatXX& hess_x)
    {
        grad_x.setZero();
        hess_x.setZero();
        double cost = 0.0;

        double dx = xk(0) - ref.x;
        double dy = xk(1) - ref.y;
        double e_c = -ref.sin_theta * dx + ref.cos_theta * dy;  // contouring (lateral)
        double e_l =  ref.cos_theta * dx + ref.sin_theta * dy;  // lag (longitudinal)

        cost += config.contour_weight * e_c * e_c;
        cost += config.lag_weight * e_l * e_l;

        // Curvature-adaptive velocity reference
        // Stronger decay (-1.2) for tight turns on 1:10 scale track
        double v_ref;
        if (config.startup_elapsed_s < config.startup_ramp_duration_s) {
            v_ref = 0.15 * std::exp(-3.0 * std::abs(ref.curvature));
        } else {
            v_ref = config.reference_velocity * std::exp(-1.2 * std::abs(ref.curvature));
        }
        v_ref = std::clamp(v_ref, 0.08, config.max_velocity);

        // --- Contouring + lag gradient ---
        // d(e_c)/dx = -sin_theta, d(e_c)/dy = cos_theta
        // d(e_l)/dx = cos_theta,  d(e_l)/dy = sin_theta
        grad_x(0) = 2.0 * config.contour_weight * e_c * (-ref.sin_theta)
                   + 2.0 * config.lag_weight * e_l * ref.cos_theta;
        grad_x(1) = 2.0 * config.contour_weight * e_c * ref.cos_theta
                   + 2.0 * config.lag_weight * e_l * ref.sin_theta;

        // --- Contouring + lag FULL Hessian for (x, y) block ---
        // H = 2*w_c * [d(e_c)/dx]^T [d(e_c)/dx] + 2*w_l * [d(e_l)/dx]^T [d(e_l)/dx]
        //   = 2*w_c * [-sin; cos] [-sin, cos] + 2*w_l * [cos; sin] [cos, sin]
        double s = ref.sin_theta;
        double c = ref.cos_theta;
        double wc = config.contour_weight;
        double wl = config.lag_weight;

        // Full 2x2 block for states 0,1 (x,y):
        hess_x(0, 0) = 2.0 * (wc * s * s + wl * c * c);
        hess_x(0, 1) = 2.0 * (-wc * s * c + wl * c * s);  // = 2*(wl - wc)*s*c
        hess_x(1, 0) = hess_x(0, 1);  // symmetric
        hess_x(1, 1) = 2.0 * (wc * c * c + wl * s * s);

        // --- Velocity tracking ---
        double v_err = xk(3) - v_ref;
        cost += config.velocity_weight * v_err * v_err;
        grad_x(3) = 2.0 * config.velocity_weight * v_err;
        hess_x(3, 3) = 2.0 * config.velocity_weight;

        // --- Steering penalty ---
        cost += config.steering_weight * xk(4) * xk(4);
        grad_x(4) = 2.0 * config.steering_weight * xk(4);
        hess_x(4, 4) = 2.0 * config.steering_weight;

        // --- Obstacle penalty (smooth barrier) ---
        for (const auto& obs : obstacles) {
            double odx = xk(0) - obs.x;
            double ody = xk(1) - obs.y;
            double dist_sq = odx * odx + ody * ody;
            double dist = std::sqrt(dist_sq);
            double safe_r = obs.radius + config.robot_radius + config.safety_margin;

            if (dist < safe_r + 0.5 && dist > 1e-4) {
                double violation = safe_r - dist;
                if (violation > 0) {
                    cost += config.obstacle_weight * violation * violation;
                    double factor = 2.0 * config.obstacle_weight * violation;
                    double nx = odx / dist;
                    double ny = ody / dist;
                    grad_x(0) -= factor * nx;
                    grad_x(1) -= factor * ny;
                    // Hessian: add w * (I/dist - n*n^T/dist) + w * n*n^T
                    // Simplified: just add diagonal for numerical stability
                    hess_x(0, 0) += 2.0 * config.obstacle_weight;
                    hess_x(1, 1) += 2.0 * config.obstacle_weight;
                }
            }
        }

        // --- Boundary penalty (soft) ---
        if (k < (int)boundaries.size()) {
            const auto& bd = boundaries[k];
            double left_val = bd.nx * xk(0) + bd.ny * xk(1) - bd.b_left;
            double right_val = -bd.nx * xk(0) - bd.ny * xk(1) - bd.b_right;

            if (left_val > 0) {
                cost += config.boundary_weight * left_val * left_val;
                grad_x(0) += 2.0 * config.boundary_weight * left_val * bd.nx;
                grad_x(1) += 2.0 * config.boundary_weight * left_val * bd.ny;
                hess_x(0, 0) += 2.0 * config.boundary_weight * bd.nx * bd.nx;
                hess_x(0, 1) += 2.0 * config.boundary_weight * bd.nx * bd.ny;
                hess_x(1, 0) += 2.0 * config.boundary_weight * bd.ny * bd.nx;
                hess_x(1, 1) += 2.0 * config.boundary_weight * bd.ny * bd.ny;
            }
            if (right_val > 0) {
                cost += config.boundary_weight * right_val * right_val;
                grad_x(0) -= 2.0 * config.boundary_weight * right_val * bd.nx;
                grad_x(1) -= 2.0 * config.boundary_weight * right_val * bd.ny;
                hess_x(0, 0) += 2.0 * config.boundary_weight * bd.nx * bd.nx;
                hess_x(0, 1) += 2.0 * config.boundary_weight * bd.nx * bd.ny;
                hess_x(1, 0) += 2.0 * config.boundary_weight * bd.ny * bd.nx;
                hess_x(1, 1) += 2.0 * config.boundary_weight * bd.ny * bd.ny;
            }
        }

        return cost;
    }

    /**
     * Compute total trajectory cost (for line search evaluation).
     */
    double compute_total_cost(
        const std::vector<VecX>& X,
        const std::vector<VecU>& U,
        const std::vector<PathRef>& path_refs,
        const std::vector<Obstacle>& obstacles,
        const std::vector<BoundaryConstraint>& boundaries)
    {
        int N = config.horizon;
        double total = 0.0;

        for (int k = 0; k <= N; k++) {
            PathRef ref = get_path_ref(k, path_refs, X[k]);
            VecX g; MatXX H;
            double weight = (k == N) ? 2.0 : 1.0;
            total += weight * compute_state_cost_full(X[k], ref, obstacles, boundaries, k, g, H);
        }

        for (int k = 0; k < N; k++) {
            total += config.acceleration_weight * U[k](0) * U[k](0);
            total += config.steering_rate_weight * U[k](1) * U[k](1);
            if (k > 0) {
                VecU du = U[k] - U[k-1];
                total += config.jerk_weight * du.squaredNorm();
            }
        }

        return total;
    }

    PathRef get_path_ref(int k, const std::vector<PathRef>& path_refs, const VecX& xk) {
        if (k < (int)path_refs.size()) {
            return path_refs[k];
        } else if (!path_refs.empty()) {
            return path_refs.back();
        } else {
            PathRef ref;
            ref.x = xk(0); ref.y = xk(1);
            ref.cos_theta = std::cos(xk(2));
            ref.sin_theta = std::sin(xk(2));
            ref.curvature = 0.0;
            return ref;
        }
    }

    /**
     * Proper iLQR with full Hessian, feedback gains, and line search.
     *
     * This is the standard iLQR algorithm:
     * 1. Backward pass: compute feedback gains K_k and feedforward k_k
     * 2. Forward pass: roll out new trajectory with line search
     *
     * Returns true if converged.
     */
    bool solve_ilqr(
        const VecX& x0,
        std::vector<VecX>& X,
        std::vector<VecU>& U,
        const std::vector<MatXX>& As,
        const std::vector<MatXU>& Bs,
        const std::vector<VecX>& cs,
        const std::vector<PathRef>& path_refs,
        const std::vector<Obstacle>& obstacles,
        const std::vector<BoundaryConstraint>& boundaries)
    {
        int N = config.horizon;
        double mu = 1e-3;  // Regularization parameter

        for (int iter = 0; iter < config.max_qp_iterations; iter++) {

            // === BACKWARD PASS ===
            // Compute value function at terminal state
            PathRef ref_N = get_path_ref(N, path_refs, X[N]);
            VecX lx_N; MatXX lxx_N;
            compute_state_cost_full(X[N], ref_N, obstacles, boundaries, N, lx_N, lxx_N);
            // Terminal cost weighted 2x
            VecX Vx = 2.0 * lx_N;
            MatXX Vxx = 2.0 * lxx_N;

            // Storage for feedback gains and feedforward terms
            std::vector<MatUX> Ks(N);    // Feedback gains
            std::vector<VecU> ks(N);     // Feedforward terms
            double expected_reduction = 0.0;

            bool backward_ok = true;
            for (int k = N - 1; k >= 0; k--) {
                PathRef ref = get_path_ref(k, path_refs, X[k]);

                // Stage cost gradient and Hessian (full)
                VecX lx; MatXX lxx;
                compute_state_cost_full(X[k], ref, obstacles, boundaries, k, lx, lxx);

                // Control cost gradient and Hessian
                VecU lu;
                MatUU luu = MatUU::Zero();
                lu(0) = 2.0 * config.acceleration_weight * U[k](0);
                lu(1) = 2.0 * config.steering_rate_weight * U[k](1);
                luu(0, 0) = 2.0 * config.acceleration_weight;
                luu(1, 1) = 2.0 * config.steering_rate_weight;

                // Jerk penalty
                if (k > 0) {
                    lu(0) += 2.0 * config.jerk_weight * (U[k](0) - U[k-1](0));
                    lu(1) += 2.0 * config.jerk_weight * (U[k](1) - U[k-1](1));
                }
                luu(0, 0) += 2.0 * config.jerk_weight;
                luu(1, 1) += 2.0 * config.jerk_weight;

                // Cross-term lxu is zero for our cost function (state and control costs are separable)
                // MatXU lxu = MatXU::Zero();

                // Q-function derivatives (standard iLQR formulas)
                VecX Qx = lx + As[k].transpose() * Vx;
                VecU Qu = lu + Bs[k].transpose() * Vx;
                MatXX Qxx = lxx + As[k].transpose() * Vxx * As[k];
                MatUU Quu = luu + Bs[k].transpose() * Vxx * Bs[k];
                MatUX Qux = Bs[k].transpose() * Vxx * As[k];

                // Regularize Quu to ensure positive definiteness
                Quu(0, 0) += mu;
                Quu(1, 1) += mu;

                // Check positive definiteness via Cholesky
                Eigen::LLT<MatUU> llt(Quu);
                if (llt.info() != Eigen::Success) {
                    // Increase regularization and retry
                    mu *= 10.0;
                    backward_ok = false;
                    break;
                }

                // Feedback gain: K = -Quu^{-1} * Qux
                // Feedforward:   k = -Quu^{-1} * Qu
                MatUU Quu_inv = llt.solve(MatUU::Identity());
                ks[k] = -Quu_inv * Qu;
                Ks[k] = -Quu_inv * Qux;

                // Clamp feedforward to prevent huge steps
                ks[k](0) = clamp(ks[k](0), -1.0, 1.0);
                ks[k](1) = clamp(ks[k](1), -1.0, 1.0);

                // Expected cost reduction for line search
                expected_reduction += ks[k].transpose() * Qu;

                // Update value function (standard iLQR)
                Vx = Qx + Ks[k].transpose() * Quu * ks[k]
                   + Ks[k].transpose() * Qu + Qux.transpose() * ks[k];
                Vxx = Qxx + Ks[k].transpose() * Quu * Ks[k]
                    + Ks[k].transpose() * Qux + Qux.transpose() * Ks[k];
                // Ensure symmetry
                Vxx = 0.5 * (Vxx + Vxx.transpose());
            }

            if (!backward_ok) {
                // Backward pass failed - try next iteration with higher regularization
                continue;
            }

            // Decrease regularization on successful backward pass
            mu = std::max(mu * 0.5, 1e-6);

            // === FORWARD PASS with line search ===
            double current_cost = compute_total_cost(X, U, path_refs, obstacles, boundaries);

            bool improved = false;
            double alpha = 1.0;

            for (int ls = 0; ls < 6; ls++) {
                std::vector<VecX> X_new(N + 1);
                std::vector<VecU> U_new(N);

                X_new[0] = x0;
                for (int k = 0; k < N; k++) {
                    // u_new = u + alpha * k + K * (x_new - x_ref)
                    VecX dx = X_new[k] - X[k];
                    U_new[k] = U[k] + alpha * ks[k] + Ks[k] * dx;

                    // Clamp controls
                    U_new[k](0) = clamp(U_new[k](0),
                        -config.max_acceleration, config.max_acceleration);
                    U_new[k](1) = clamp(U_new[k](1),
                        -config.max_steering_rate, config.max_steering_rate);

                    // Forward simulate using linearized dynamics
                    X_new[k + 1] = As[k] * X_new[k] + Bs[k] * U_new[k] + cs[k];

                    // Clamp states
                    X_new[k + 1](3) = clamp(X_new[k + 1](3),
                        config.min_velocity, config.max_velocity);
                    X_new[k + 1](4) = clamp(X_new[k + 1](4),
                        -config.max_steering, config.max_steering);
                }

                double new_cost = compute_total_cost(X_new, U_new, path_refs, obstacles, boundaries);

                if (new_cost < current_cost) {
                    X = X_new;
                    U = U_new;
                    improved = true;
                    break;
                }

                alpha *= 0.5;  // Backtracking
            }

            if (!improved) {
                // No improvement found - accept current trajectory
                // Try with smaller step next outer SQP iteration
                break;
            }

            // Check convergence
            double max_du = 0.0;
            for (int k = 0; k < N; k++) {
                max_du = std::max(max_du, ks[k].squaredNorm());
            }
            if (max_du < config.qp_tolerance) {
                return true;  // Converged
            }
        }

        return false;
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
    const double* state,          // [x, y, theta, v, delta]
    const MPCCPathPoint* path,    // array of path references
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
