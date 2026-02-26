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
    constexpr int N_PARAMS = 20;
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

    AcadosSolver() : model(0.256), ackermann_model(0.256), capsule_(nullptr) {}

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
        u_prev_ = Vec2(0.2, 0.0);
        vtheta_prev_ = 0.0;

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

        // Initialize trajectory if no warm-start
        if (!has_warmstart) {
            double v0 = std::max(measured_v >= 0.0 ? measured_v : u_prev_(0), 0.15);
            double d0 = measured_delta > -900.0 ? measured_delta : u_prev_(1);
            double vth0 = std::max(v0, 0.2);  // Initial progress speed

            for (int k = 0; k <= N; k++) {
                double theta_k = current_progress + k * vth0 * config.dt;
                theta_k = std::min(theta_k, path_total_length - 0.001);
                double xk[4] = {
                    x0(0) + k * v0 * config.dt * std::cos(x0(2)),
                    x0(1) + k * v0 * config.dt * std::sin(x0(2)),
                    x0(2),
                    theta_k
                };
                // API: ocp_nlp_out_set(config, dims, out, in, stage, field, value)
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", (void*)xk);

                if (k < N) {
                    double uk[3] = {v0, d0, vth0};
                    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", (void*)uk);
                }
            }
        } else {
            // Warm-start: shift trajectory by one step
            for (int k = 0; k < N; k++) {
                int src = std::min(k + 1, N);
                double xk[4];
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, src, "x", (void*)xk);
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", (void*)xk);

                if (k < N - 1) {
                    double uk[3];
                    int usrc = std::min(k + 1, N - 1);
                    ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, usrc, "u", (void*)uk);
                    ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", (void*)uk);
                }
            }
            // Fix initial state
            double xk0[4] = {x0(0), x0(1), x0(2), current_progress};
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, 0, "x", (void*)xk0);
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
        closest_obs.x = 1000.0;
        closest_obs.y = 1000.0;
        closest_obs.radius = 0.1;
        double min_obs_dist = 1e9;
        for (const auto& obs : obstacles) {
            double d = std::hypot(obs.x - x0(0), obs.y - x0(1));
            if (d < min_obs_dist) {
                min_obs_dist = d;
                closest_obs = obs;
            }
        }

        for (int k = 0; k <= N; k++) {
            double p_k[acados_param::N_PARAMS];

            // Get path reference for this stage
            // If spline_path is available, evaluate at predicted theta_A
            double ref_x, ref_y, ref_dx, ref_dy, ref_curv;
            if (spline_path && spline_path->is_built()) {
                // Get predicted theta_A from current trajectory
                double xk[4];
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, k, "x", (void*)xk);
                double s = std::clamp(xk[3], 0.0, spline_path->total_length() - 0.001);

                double sx, sy;
                spline_path->get_position(s, sx, sy);
                double tangent = spline_path->get_tangent(s);
                ref_x = sx;
                ref_y = sy;
                ref_dx = std::cos(tangent);
                ref_dy = std::sin(tangent);
                ref_curv = spline_path->get_curvature(s);
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
            p_k[acados_param::IDX_RREF_VEL] = (k == 0) ? wv : 0.0;  // R_ref at k=0 only
            p_k[acados_param::IDX_RREF_STEER] = config.steering_weight;

            // Velocity reference (curvature-adaptive)
            double v_base = lerp_weight(0.20, config.reference_velocity, sp);
            double v_ref = v_base * std::exp(curv_decay * std::abs(ref_curv));
            v_ref = std::clamp(v_ref, config.min_velocity, config.max_velocity);
            p_k[acados_param::IDX_VREF] = v_ref;
            p_k[acados_param::IDX_DELTAREF] = 0.0;  // Reference tracks δ=0

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

            mpcc_qcar2_acados_update_params(capsule_, k, p_k, acados_param::N_PARAMS);
        }

        // Precompute (condensing + expansion) — MUST be called after parameter updates
        ocp_nlp_precompute(nlp_solver_, nlp_in_, nlp_out_);

        // Solve (SQP with up to 20 iterations internally)
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

        // Get solve time and cost from acados
        double t_solve = 0.0;
        ocp_nlp_get(nlp_solver_, "time_tot", (void*)&t_solve);
        // Evaluate and extract cost value
        ocp_nlp_eval_cost(nlp_solver_, nlp_in_, nlp_out_);
        double cost_val = 0.0;
        ocp_nlp_get(nlp_solver_, "cost_value", (void*)&cost_val);
        result.cost = cost_val;

        // Auto-advance startup timer
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
