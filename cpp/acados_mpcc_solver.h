/**
 * acados MPCC Solver — Production nonlinear MPC via acados.
 *
 * Uses acados-generated C code for the MPCC OCP:
 * - State (nx=4): [X, Y, psi, theta_A] — position, heading, arc-length progress
 * - Controls (nu=3): [V, delta, V_theta] — speed, steering, progress speed
 * - theta_A is a proper decision variable (matches PolyCtrl 2025 reference exactly)
 * - qpOASES QP solver with exact Hessian
 *
 * Provides the mpcc::AcadosSolver class used by all MPCC components.
 */

#ifndef ACADOS_MPCC_SOLVER_H
#define ACADOS_MPCC_SOLVER_H

#include "mpcc_types.h"          // Config, Result, PathRef, Obstacle, etc.
#include "cubic_spline_path.h"   // CubicSplinePath for spline evaluation

#include <vector>
#include <cmath>
#include <chrono>
#include <memory>

// acados C API (generated code)
extern "C" {
#include "acados_ocp/c_generated_code/acados_solver_mpcc_qcar2.h"
#include "acados/utils/math.h"
}

namespace mpcc {

// Parameter indices — must match generate_mpcc_solver.py
namespace acados_param {
    constexpr int IDX_XREF = 0;
    constexpr int IDX_YREF = 1;
    constexpr int IDX_DXREF = 2;
    constexpr int IDX_DYREF = 3;
    constexpr int IDX_QC = 4;
    constexpr int IDX_QL = 5;
    constexpr int IDX_GAMMA = 6;
    constexpr int IDX_RV_ACCEL = 7;
    constexpr int IDX_RV_STEER = 8;
    constexpr int IDX_RV_THETA = 9;
    constexpr int IDX_RREF_VEL = 10;
    constexpr int IDX_RREF_STEER = 11;
    constexpr int IDX_VREF = 12;
    constexpr int IDX_DELTAREF = 13;
    constexpr int IDX_VPREV = 14;
    constexpr int IDX_DELTAPREV = 15;
    constexpr int IDX_VTHETAPREV = 16;
    constexpr int IDX_OBSX = 17;
    constexpr int IDX_OBSY = 18;
    constexpr int IDX_OBSR = 19;
    constexpr int IDX_WLEFT = 20;   // road half-width to left boundary
    constexpr int IDX_WRIGHT = 21;  // road half-width to right boundary
    constexpr int N_PARAMS = 22;
}

/**
 * AcadosSolver — acados-based MPCC with theta_A as a state variable.
 *
 * Provides init/solve/reset API for the MPCC controller.
 * Key differences:
 * - theta_A (arc-length progress) is a state, V_theta is a control
 * - No PathLookup callback needed (spline evaluated at solver's theta_A)
 * - spline_path must be set for spline reference evaluation
 */
class AcadosSolver {
public:
    static constexpr int NX = 3;  // External API: [x, y, theta] (matches Solver)
    static constexpr int NU = 2;  // External API: [v, delta] (matches Solver)

    Config config;
    KinematicModel model;
    AckermannModel ackermann_model;  // For simulation tests

    // Spline path for evaluating references at theta_A positions
    acc::CubicSplinePath* spline_path = nullptr;

    // PathLookup — kept for API compatibility but NOT used by acados solver
    // (theta_A as state variable replaces the need for external re-projection)
    PathLookup path_lookup;

    // Previous control for first-step smoothness
    Vec2 u_prev_ = Vec2(0.0, 0.0);
    double vtheta_prev_ = 0.0;

    // Current arc-length progress
    double current_progress_ = 0.0;

    // Warm-start trajectory (for API compatibility)
    std::vector<Vec3> X_warm;
    std::vector<Vec2> U_warm;
    bool has_warmstart = false;

    AcadosSolver() : model(0.256), ackermann_model(0.256), capsule_(nullptr), warmstart_shift_count_(0) {}

    ~AcadosSolver() {
        destroy();
    }

    void init(const Config& cfg) {
        config = cfg;
        model = KinematicModel(cfg.wheelbase);
        ackermann_model = AckermannModel(cfg.wheelbase);

        // Create acados solver capsule
        destroy();  // Clean up any previous instance
        capsule_ = mpcc_qcar2_acados_create_capsule();
        int status = mpcc_qcar2_acados_create(capsule_);
        if (status != 0) {
            throw std::runtime_error("acados solver creation failed with status " + std::to_string(status));
        }

        nlp_config_ = mpcc_qcar2_acados_get_nlp_config(capsule_);
        nlp_dims_ = mpcc_qcar2_acados_get_nlp_dims(capsule_);
        nlp_in_ = mpcc_qcar2_acados_get_nlp_in(capsule_);
        nlp_out_ = mpcc_qcar2_acados_get_nlp_out(capsule_);
        nlp_solver_ = mpcc_qcar2_acados_get_nlp_solver(capsule_);
        nlp_opts_ = mpcc_qcar2_acados_get_nlp_opts(capsule_);

        has_warmstart = false;
        u_prev_ = Vec2(0.2, 0.0);
        vtheta_prev_ = 0.0;

        X_warm.resize(cfg.horizon + 1);
        U_warm.resize(cfg.horizon);
    }

    void reset() {
        has_warmstart = false;
        u_prev_ = Vec2(0.0, 0.0);  // Vehicle is stopped at leg transitions
        vtheta_prev_ = 0.0;
        warmstart_shift_count_ = 0;

        if (capsule_) {
            mpcc_qcar2_acados_reset(capsule_, 1);
            // Reset QP memory for clean warm-start
            ocp_nlp_solver_reset_qp_memory(nlp_solver_, nlp_in_, nlp_out_);
        }
    }

    /**
     * Solve MPCC for one control cycle.
     *
     * Internally maps to 4D acados state [X, Y, psi, theta_A] and
     * 3D acados controls [V, delta, V_theta].
     */
    Result solve(
        const Vec3& x0,
        const std::vector<PathRef>& path_refs,
        double current_progress,
        double path_total_length,
        const std::vector<Obstacle>& obstacles,
        const std::vector<BoundaryConstraint>& /*boundaries*/,
        double measured_v = -1.0,
        double measured_delta = -999.0)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        int N = config.horizon;
        current_progress_ = current_progress;
        Result result;
        result.success = false;

        if (!capsule_) {
            result.v_cmd = 0.0;
            result.delta_cmd = 0.0;
            result.omega_cmd = 0.0;
            return result;
        }

        // Update u_prev with measured values (matching reference MPC_node.py:553)
        if (measured_v >= 0.0) {
            u_prev_(0) = measured_v;
        }
        if (measured_delta > -900.0) {
            u_prev_(1) = measured_delta;
        }

        // Compute startup ramp weights
        double sp = startup_progress();
        double wc = lerp_weight(config.startup_contour_weight, config.contour_weight, sp);
        double wl = lerp_weight(config.startup_lag_weight, config.lag_weight, sp);
        double wv = lerp_weight(config.startup_velocity_weight, config.velocity_weight, sp);
        double sr = lerp_weight(config.startup_steering_rate_weight, config.steering_rate_weight, sp);
        double wp = lerp_weight(config.startup_progress_weight, config.progress_weight, sp);
        double curv_decay = lerp_weight(config.startup_curvature_decay, -0.4, sp);

        // Initial state: [X, Y, psi, theta_A]
        double x0_4d[4] = {x0(0), x0(1), x0(2), current_progress};

        // Set initial state constraint
        // API: ocp_nlp_constraints_model_set(config, dims, in, out, stage, field, value)
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", (void*)x0_4d);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", (void*)x0_4d);

        // Initialize trajectory (matching mpc_planner warm-start logic)
        if (!has_warmstart) {
            // Cold start: straight-line extrapolation from current state.
            // mpc_planner uses braking trajectory for failure recovery, but for
            // normal cold start a straight-line guess converges well with SQP.
            double v0 = std::max(measured_v >= 0.0 ? measured_v : u_prev_(0), 0.15);
            double d0 = measured_delta > -900.0 ? measured_delta : u_prev_(1);
            double vth0 = std::max(v0, 0.2);

            for (int k = 0; k <= N; k++) {
                double theta_k = current_progress + k * vth0 * config.dt;
                theta_k = std::min(theta_k, path_total_length - 0.001);
                double xk[4] = {
                    x0(0) + k * v0 * config.dt * std::cos(x0(2)),
                    x0(1) + k * v0 * config.dt * std::sin(x0(2)),
                    x0(2),
                    theta_k
                };
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", (void*)xk);
                if (k < N) {
                    double uk[3] = {v0, d0, vth0};
                    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", (void*)uk);
                }
            }
        } else {
            // Warm-start: shift previous trajectory forward by 1 step
            // (matching mpc_planner initializeWarmstart with shift_forward=true)
            // Pattern: [x0_new, x2_prev, x3_prev, ..., x_{N-1}_prev, x_{N-1}_prev]
            //
            // Read all previous data first to avoid overwrite-before-read issues.
            warmstart_shift_count_++;
            double prev_x[11][4];  // N+1 max
            double prev_u[10][3];  // N max
            for (int k = 0; k <= N; k++)
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", (void*)prev_x[k]);
            for (int k = 0; k < N; k++)
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "u", (void*)prev_u[k]);

            // k=0: current measured state
            double xk0[4] = {x0(0), x0(1), x0(2), current_progress};
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, 0, "x", (void*)xk0);

            // k=1..N-2: shift from k+1
            for (int k = 1; k <= N - 2; k++) {
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", (void*)prev_x[k+1]);
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", (void*)prev_u[std::min(k+1, N-1)]);
            }
            // k=N-1 and k=N: extrapolate with prev_x[N] (terminal)
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N - 1, "x", (void*)prev_x[N]);
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N - 1, "u", (void*)prev_u[N-1]);
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, N, "x", (void*)prev_x[N]);
        }

        // Update theta_A upper bound to path length
        for (int k = 0; k <= N; k++) {
            double ubx[4] = {50.0, 50.0, 1e4, path_total_length};
            ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_,
                                          k, "ubx", (void*)ubx);
            if (k > 0) {
                double lbx[4] = {-50.0, -50.0, -1e4, 0.0};
                ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_,
                                              k, "lbx", (void*)lbx);
            }
        }

        // Set per-stage parameters
        // Find closest obstacle (pick the nearest one for the single-obstacle constraint)
        Obstacle closest_obs;
        // Default: inactive obstacle with zero radius, placed 5m behind the vehicle.
        // Matching mpc_planner pattern: inactive obstacles have zero-sized shape,
        // constraint is trivially satisfied. Keeping it nearby (not 1000m) avoids
        // ill-conditioned Jacobians.
        closest_obs.x = x0(0) - 5.0 * std::cos(x0(2));
        closest_obs.y = x0(1) - 5.0 * std::sin(x0(2));
        closest_obs.radius = 0.0;  // zero radius = inactive
        double min_obs_dist = 1e9;
        for (const auto& obs : obstacles) {
            double d = std::hypot(obs.x - x0(0), obs.y - x0(1));
            if (d < min_obs_dist) {
                min_obs_dist = d;
                closest_obs = obs;
            }
        }

        // Diagnostics: pre-allocate per-stage ref storage
        double v_ref_k0 = 0.0;
        std::vector<double> diag_ref_x, diag_ref_y, diag_ref_v, diag_ref_curv;
        if (config.per_stage_logging) {
            diag_ref_x.resize(N + 1);
            diag_ref_y.resize(N + 1);
            diag_ref_v.resize(N + 1);
            diag_ref_curv.resize(N + 1);
        }

        for (int k = 0; k <= N; k++) {
            double p_k[acados_param::N_PARAMS];

            // Get path reference for this stage.
            // Matching mpc_planner pattern: evaluate spline at current_progress + k offset,
            // NOT at predicted theta_A from the trajectory guess. Parameters are fixed for
            // the entire SQP solve; using theta_A from warm-start produces stale references
            // that degrade tracking (0.5m+ CTE). The solver's theta_A decision variable
            // still optimizes contouring/lag cost — but the reference *positions* come from
            // the externally-computed closest point, stepped forward by curvature-adaptive speed.
            double ref_x, ref_y, ref_dx, ref_dy, ref_curv;
            if (spline_path && spline_path->is_built() && k < (int)path_refs.size()) {
                // Use caller-provided path_refs (computed from current_progress with
                // curvature-adaptive stepping in get_spline_path_refs). Also use spline
                // for smooth tangent/curvature evaluation at the same arc-length position.
                ref_x = path_refs[k].x;
                ref_y = path_refs[k].y;
                ref_dx = path_refs[k].cos_theta;
                ref_dy = path_refs[k].sin_theta;
                ref_curv = path_refs[k].curvature;
            } else if (k < (int)path_refs.size()) {
                ref_x = path_refs[k].x;
                ref_y = path_refs[k].y;
                ref_dx = path_refs[k].cos_theta;
                ref_dy = path_refs[k].sin_theta;
                ref_curv = path_refs[k].curvature;
            } else {
                ref_x = x0(0);
                ref_y = x0(1);
                ref_dx = std::cos(x0(2));
                ref_dy = std::sin(x0(2));
                ref_curv = 0.0;
            }

            // Spline references
            p_k[acados_param::IDX_XREF] = ref_x;
            p_k[acados_param::IDX_YREF] = ref_y;
            p_k[acados_param::IDX_DXREF] = ref_dx;
            p_k[acados_param::IDX_DYREF] = ref_dy;

            // Weights
            p_k[acados_param::IDX_QC] = wc;
            p_k[acados_param::IDX_QL] = wl;
            p_k[acados_param::IDX_GAMMA] = wp;
            p_k[acados_param::IDX_RV_ACCEL] = config.acceleration_weight;
            p_k[acados_param::IDX_RV_STEER] = sr;
            p_k[acados_param::IDX_RV_THETA] = 0.1;  // Smoothness for V_theta
            p_k[acados_param::IDX_RREF_VEL] = wv;  // Velocity tracking at ALL stages (mpc_planner applies at all k)
            p_k[acados_param::IDX_RREF_STEER] = config.steering_weight;

            // Velocity reference (curvature-adaptive)
            double v_base = lerp_weight(0.20, config.reference_velocity, sp);
            double v_ref = v_base * std::exp(curv_decay * std::abs(ref_curv));
            v_ref = std::clamp(v_ref, config.min_velocity, config.max_velocity);
            p_k[acados_param::IDX_VREF] = v_ref;
            p_k[acados_param::IDX_DELTAREF] = 0.0;  // Reference tracks δ=0

            // Track diagnostics data
            if (k == 0) v_ref_k0 = v_ref;
            if (config.per_stage_logging) {
                diag_ref_x[k] = ref_x;
                diag_ref_y[k] = ref_y;
                diag_ref_v[k] = v_ref;
                diag_ref_curv[k] = ref_curv;
            }

            // Previous controls (for smoothness at k=0)
            p_k[acados_param::IDX_VPREV] = u_prev_(0);
            p_k[acados_param::IDX_DELTAPREV] = u_prev_(1);
            p_k[acados_param::IDX_VTHETAPREV] = vtheta_prev_;

            // Obstacle (with velocity prediction)
            double obs_x_k = closest_obs.x + k * config.dt * closest_obs.vx;
            double obs_y_k = closest_obs.y + k * config.dt * closest_obs.vy;
            p_k[acados_param::IDX_OBSX] = obs_x_k;
            p_k[acados_param::IDX_OBSY] = obs_y_k;
            p_k[acados_param::IDX_OBSR] = closest_obs.radius + config.safety_margin;

            // Road boundary widths (half-widths from path centerline)
            // e_c > 0 = vehicle left of path, e_c < 0 = vehicle right of path.
            // width_right: ALWAYS enforced (vehicle must never cross right lane edge).
            // width_left: relaxed when obstacle nearby (vehicle can enter oncoming lane).
            // When boundary_weight == 0, use large widths (disabled).
            double w_right = (config.boundary_weight > 0.0) ?
                config.boundary_default_width : 5.0;
            double w_left = w_right;  // Same as right by default

            if (min_obs_dist < 2.0 && config.boundary_weight > 0.0) {
                // Obstacle nearby: only widen LEFT boundary for avoidance into oncoming lane.
                // Right boundary stays tight — vehicle must NOT go outside right lane edge.
                w_left = std::max(w_left, 0.50);  // Allow up to 0.50m into oncoming lane
            }
            p_k[acados_param::IDX_WLEFT] = w_left;
            p_k[acados_param::IDX_WRIGHT] = w_right;

            mpcc_qcar2_acados_update_params(capsule_, k, p_k, acados_param::N_PARAMS);
        }

        // SQP solve (up to 10 iterations with HPIPM QP solver).
        int acados_status = mpcc_qcar2_acados_solve(capsule_);

        // Extract result
        double u0[3];
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", (void*)u0);

        result.v_cmd = std::clamp(u0[0], config.min_velocity, config.max_velocity);
        result.delta_cmd = std::clamp(u0[1], -config.max_steering, config.max_steering);
        vtheta_prev_ = u0[2];  // Store V_theta for next step's smoothness

        if (std::abs(result.v_cmd) > 0.001) {
            result.omega_cmd = result.v_cmd * std::tan(result.delta_cmd) / config.wheelbase;
        } else {
            result.omega_cmd = 0.0;
        }

        // Extract predicted trajectory
        result.predicted_x.resize(N + 1);
        result.predicted_y.resize(N + 1);
        result.predicted_theta.resize(N + 1);
        for (int k = 0; k <= N; k++) {
            double xk[4];
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", (void*)xk);
            result.predicted_x[k] = xk[0];
            result.predicted_y[k] = xk[1];
            result.predicted_theta[k] = xk[2];
        }

        // Store warm-start data and u_prev
        has_warmstart = true;
        u_prev_ = Vec2(result.v_cmd, result.delta_cmd);

        // Also store in X_warm/U_warm for API compatibility
        for (int k = 0; k <= N; k++) {
            X_warm[k] = Vec3(result.predicted_x[k], result.predicted_y[k], result.predicted_theta[k]);
            if (k < N) {
                double uk[3];
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "u", (void*)uk);
                U_warm[k] = Vec2(uk[0], uk[1]);
            }
        }

        // Accept status 0 (converged), 2 (max iter reached but result usable),
        // 3 (QP MINSTEP — effectively converged), or 4 (NLP MINSTEP — SQP step small)
        result.success = (acados_status == 0 || acados_status == 2 ||
                          acados_status == 3 || acados_status == 4);

        // Store acados status in diagnostics before any reset
        result.diag.acados_status = acados_status;

        // On failure: reset solver state (matching mpc_planner completeOneIteration)
        if (!result.success) {
            mpcc_qcar2_acados_reset(capsule_, 1);
            ocp_nlp_solver_reset_qp_memory(nlp_solver_, nlp_in_, nlp_out_);
            has_warmstart = false;  // Force cold start on next call
        }

        // Get solve time and cost from acados
        double t_solve = 0.0;
        ocp_nlp_get(nlp_solver_, "time_tot", (void*)&t_solve);
        // Evaluate and extract cost value
        ocp_nlp_eval_cost(nlp_solver_, nlp_in_, nlp_out_);
        double cost_val = 0.0;
        ocp_nlp_get(nlp_solver_, "cost_value", (void*)&cost_val);
        result.cost = cost_val;

        // Extract diagnostics when enabled
        if (config.diagnostics_enabled) {
            extract_diagnostics(result, N, sp, wc, wl, wv, sr, wp, v_ref_k0,
                                closest_obs, min_obs_dist,
                                diag_ref_x, diag_ref_y, diag_ref_v, diag_ref_curv);
        }

        // Auto-advance startup timer (for standalone tests that don't set it externally).
        // In deployment, the controller node sets config.startup_elapsed_s = ROS elapsed time
        // BEFORE each call, so this auto-increment gets overwritten — no double-counting.
        config.startup_elapsed_s += config.dt;

        auto t_end = std::chrono::high_resolution_clock::now();
        result.solve_time_us = std::chrono::duration_cast<std::chrono::microseconds>(
            t_end - t_start).count();

        return result;
    }

    // Overload: accept 5D state for backward compatibility
    Result solve(
        const VecX& x0_5d,
        const std::vector<PathRef>& path_refs,
        double current_progress,
        double path_total_length,
        const std::vector<Obstacle>& obstacles,
        const std::vector<BoundaryConstraint>& boundaries)
    {
        Vec3 x0_3d(x0_5d(0), x0_5d(1), x0_5d(2));
        return solve(x0_3d, path_refs, current_progress, path_total_length,
                     obstacles, boundaries, x0_5d(3), x0_5d(4));
    }

private:
    mpcc_qcar2_solver_capsule* capsule_ = nullptr;
    ocp_nlp_config* nlp_config_ = nullptr;
    ocp_nlp_dims* nlp_dims_ = nullptr;
    ocp_nlp_in* nlp_in_ = nullptr;
    ocp_nlp_out* nlp_out_ = nullptr;
    ocp_nlp_solver* nlp_solver_ = nullptr;
    void* nlp_opts_ = nullptr;
    int warmstart_shift_count_ = 0;

    void extract_diagnostics(Result& result, int N, double sp,
                              double wc, double wl, double wv, double sr, double wp,
                              double v_ref_k0, const Obstacle& closest_obs,
                              double min_obs_dist,
                              const std::vector<double>& diag_ref_x,
                              const std::vector<double>& diag_ref_y,
                              const std::vector<double>& diag_ref_v,
                              const std::vector<double>& diag_ref_curv) {
        auto& d = result.diag;

        // SQP iteration count
        int sqp_iter = 0;
        ocp_nlp_get(nlp_solver_, "sqp_iter", &sqp_iter);
        d.sqp_iter = sqp_iter;

        // KKT norm (extracted from stage 0 — overall convergence indicator)
        // Note: acados stores kkt_norm_inf at solver level, not stage level
        double kkt = 0.0;
        ocp_nlp_get(nlp_solver_, "res_stat", &kkt);
        d.kkt_norm_inf = kkt;

        // QP status
        int qp_stat = 0;
        ocp_nlp_get(nlp_solver_, "qp_status", &qp_stat);
        d.qp_status = qp_stat;

        // Residuals
        double res_eq = 0, res_ineq = 0, res_comp = 0, res_stat = 0;
        ocp_nlp_get(nlp_solver_, "res_eq", &res_eq);
        ocp_nlp_get(nlp_solver_, "res_ineq", &res_ineq);
        ocp_nlp_get(nlp_solver_, "res_comp", &res_comp);
        ocp_nlp_get(nlp_solver_, "res_stat", &res_stat);
        d.res_eq = res_eq;
        d.res_ineq = res_ineq;
        d.res_comp = res_comp;
        d.res_stat = res_stat;

        // Timing
        double t_tot = 0, t_qp = 0;
        ocp_nlp_get(nlp_solver_, "time_tot", &t_tot);
        ocp_nlp_get(nlp_solver_, "time_qp_sol", &t_qp);
        d.acados_time_tot_ms = t_tot * 1000.0;
        d.acados_time_qp_ms = t_qp * 1000.0;

        // Startup progress and effective weights
        d.startup_progress = sp;
        d.eff_contour_w = wc;
        d.eff_lag_w = wl;
        d.eff_vel_w = wv;
        d.eff_sr_w = sr;
        d.eff_progress_w = wp;
        d.eff_v_ref_k0 = v_ref_k0;

        // Warmstart
        d.warmstart_used = has_warmstart;
        d.warmstart_shift_count = warmstart_shift_count_;

        // Obstacle data
        d.obs_x = closest_obs.x;
        d.obs_y = closest_obs.y;
        d.obs_r = closest_obs.radius;
        d.obs_dist = min_obs_dist;

        // Per-stage trajectory
        if (config.per_stage_logging) {
            d.stage_x.resize(N + 1);
            d.stage_y.resize(N + 1);
            d.stage_psi.resize(N + 1);
            d.stage_theta_a.resize(N + 1);
            d.stage_v.resize(N);
            d.stage_delta.resize(N);
            d.stage_v_theta.resize(N);

            for (int k = 0; k <= N; k++) {
                double xk[4];
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", (void*)xk);
                d.stage_x[k] = xk[0];
                d.stage_y[k] = xk[1];
                d.stage_psi[k] = xk[2];
                d.stage_theta_a[k] = xk[3];
                if (k < N) {
                    double uk[3];
                    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "u", (void*)uk);
                    d.stage_v[k] = uk[0];
                    d.stage_delta[k] = uk[1];
                    d.stage_v_theta[k] = uk[2];
                }
            }

            // Per-stage references
            d.ref_x = diag_ref_x;
            d.ref_y = diag_ref_y;
            d.ref_v = diag_ref_v;
            d.ref_curv = diag_ref_curv;
        }
    }

    void destroy() {
        if (capsule_) {
            mpcc_qcar2_acados_free(capsule_);
            mpcc_qcar2_acados_free_capsule(capsule_);
            capsule_ = nullptr;
        }
    }

    static double clamp(double v, double lo, double hi) {
        return std::max(lo, std::min(hi, v));
    }

    double startup_progress() const {
        if (config.startup_ramp_duration_s <= 0.0) return 1.0;
        return std::clamp(config.startup_elapsed_s / config.startup_ramp_duration_s, 0.0, 1.0);
    }

    static double lerp_weight(double startup_val, double normal_val, double progress) {
        return startup_val + progress * (normal_val - startup_val);
    }
};

}  // namespace mpcc

#endif  // ACADOS_MPCC_SOLVER_H
