/**
 * Diagnostic: Why does the solver command max steering at startup on a straight path?
 * Simulates the first 20 steps with two plants:
 *   (A) Rate-limited AckermannModel (like test_deployment.cpp)
 *   (B) Direct steering (like real deployment — servo reaches command instantly)
 *
 * Build:
 *   cd /home/stephen/quanser-acc/cpp/test_build
 *   g++ -std=c++17 -O2 -I.. -I/usr/include/eigen3 \
 *       -o diagnose_steering diagnose_steering.cpp ../road_graph.cpp
 */

#include <cmath>
#include <cstdio>
#include <vector>
#include "coordinate_transform.h"
#include "cubic_spline_path.h"
#include "road_graph.h"
#include "mpcc_solver_interface.h"

struct PDSpeedController {
    double actual_speed = 0.0;
    double kp = 20.0, kd = 0.1;
    double prev_error = 0.0;
    double step(double desired_speed, double dt_outer) {
        int inner_steps = 7;
        double inner_dt = dt_outer / inner_steps;
        for (int i = 0; i < inner_steps; i++) {
            double error = desired_speed - actual_speed;
            double d_error = (error - prev_error) / inner_dt;
            double pwm = kp * error * 0.0047 / 12.0 + kd * d_error * 0.0047 / 12.0;
            pwm = std::clamp(pwm, -0.3, 0.3);
            actual_speed += pwm * (12.0 / 0.0047) * inner_dt * 0.3;
            actual_speed = std::max(actual_speed, 0.0);
            prev_error = error;
        }
        return actual_speed;
    }
};

int main() {
    // Build the same hub->pickup path used in deployment
    acc::RoadGraph rg;
    auto path_opt = rg.plan_path_for_mission_leg("hub_to_pickup", acc::HUB_X, acc::HUB_Y);
    auto& path_xy = *path_opt;
    auto& path_wx = path_xy.first;
    auto& path_wy = path_xy.second;

    acc::CubicSplinePath spline;
    spline.build(path_wx, path_wy, true);
    double total_len = spline.total_length();

    // Common config (matching deployment)
    mpcc::Config cfg;
    cfg.horizon = 10;
    cfg.dt = 0.1;
    cfg.wheelbase = 0.256;
    cfg.max_velocity = 0.55;
    cfg.min_velocity = 0.0;
    cfg.max_steering = 0.45;  // hardware servo limit
    cfg.max_acceleration = 1.5;
    cfg.max_steering_rate = 1.5;
    cfg.reference_velocity = 0.45;
    cfg.contour_weight = 15.0;      // Matches deployed
    cfg.lag_weight = 10.0;
    cfg.velocity_weight = 15.0;
    cfg.steering_weight = 0.05;
    cfg.acceleration_weight = 0.01;
    cfg.steering_rate_weight = 1.5;  // Matches deployed
    cfg.heading_weight = 0.0;        // Reference-matched (was 2.0, caused swerving)
    cfg.progress_weight = 1.0;
    cfg.jerk_weight = 0.0;
    cfg.boundary_weight = 0.0;
    cfg.max_sqp_iterations = 5;
    cfg.max_qp_iterations = 20;
    cfg.startup_ramp_duration_s = 3.0;  // Matches deployed
    cfg.startup_elapsed_s = 0.0;
    cfg.startup_progress_weight = 1.0;

    auto make_lookup = [&spline, total_len](
        double px, double py, double s_min, double* s_out) -> mpcc::PathRef {
        double s = spline.find_closest_progress_from(px, py, s_min);
        s = std::clamp(s, 0.0, total_len - 0.001);
        if (s_out) *s_out = s;
        mpcc::PathRef ref;
        double ct, st;
        spline.get_path_reference(s, ref.x, ref.y, ct, st);
        ref.cos_theta = ct; ref.sin_theta = st;
        ref.curvature = spline.get_curvature(s);
        return ref;
    };

    double init_x, init_y;
    spline.get_position(0.0, init_x, init_y);
    double init_theta = spline.get_tangent(0.0);
    printf("Path start: (%.4f, %.4f) theta=%.4f rad (%.1f deg)\n",
           init_x, init_y, init_theta, init_theta * 180/M_PI);

    // ====== Scenario A: Rate-limited plant (like test_deployment.cpp) ======
    printf("\n=== Scenario A: Rate-limited AckermannModel plant ===\n");
    printf("step | v_cmd  delta_cmd | plant_v plant_delta | x       y       theta   | CTE     heading_err\n");
    printf("-----|------------------|---------------------|-------------------------|--------------------\n");

    mpcc::ActiveSolver solverA;
    solverA.init(cfg);
    solverA.path_lookup.lookup = make_lookup;

    mpcc::AckermannModel plantA(cfg.wheelbase);
    mpcc::VecX stateA;
    stateA << init_x, init_y, init_theta, 0.0, 0.0;
    PDSpeedController pdA;
    double progressA = 0.0;
    int cfailA = 0;

    for (int step = 0; step < 30; step++) {
        std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
        double s = progressA;
        double lv = std::max(stateA(3), cfg.reference_velocity * 0.5);
        for (int k = 0; k <= cfg.horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);
            double rx, ry, ct, st;
            spline.get_path_reference(s, rx, ry, ct, st);
            refs[k].x = rx; refs[k].y = ry;
            refs[k].cos_theta = ct; refs[k].sin_theta = st;
            refs[k].curvature = spline.get_curvature(s);
            s += std::max(0.10, lv * std::exp(-0.4 * std::abs(refs[k].curvature))) * cfg.dt;
        }

        auto result = solverA.solve(stateA, refs, progressA, total_len, {}, {});
        if (!result.success) { if (++cfailA >= 5) { printf("  SOLVER FAILED at step %d\n", step); break; } continue; }
        cfailA = 0;

        double v_cmd = std::clamp(result.v_cmd, cfg.min_velocity, cfg.max_velocity);
        double delta_cmd = std::clamp(result.delta_cmd, -cfg.max_steering, cfg.max_steering);

        double cp = spline.find_closest_progress(stateA(0), stateA(1));
        double rpx, rpy; spline.get_position(cp, rpx, rpy);
        double cte = std::hypot(stateA(0) - rpx, stateA(1) - rpy);
        double path_theta = spline.get_tangent(cp);
        double herr = stateA(2) - path_theta;
        while (herr > M_PI) herr -= 2*M_PI;
        while (herr < -M_PI) herr += 2*M_PI;

        printf("%4d | %6.3f %8.4f | %7.3f %11.4f | %7.4f %7.4f %7.4f | %7.4f %7.4f\n",
               step, v_cmd, delta_cmd, stateA(3), stateA(4),
               stateA(0), stateA(1), stateA(2), cte, herr);
        if (cte > 1.0) break;

        double actual_v = pdA.step(v_cmd, cfg.dt);
        mpcc::VecU u;
        u(0) = (actual_v - stateA(3)) / cfg.dt;
        u(1) = (delta_cmd - stateA(4)) / cfg.dt;
        u(0) = std::clamp(u(0), -cfg.max_acceleration, cfg.max_acceleration);
        u(1) = std::clamp(u(1), -cfg.max_steering_rate, cfg.max_steering_rate);
        stateA = plantA.rk4_step(stateA, u, cfg.dt);
        stateA(3) = std::clamp(stateA(3), cfg.min_velocity, cfg.max_velocity);
        stateA(4) = std::clamp(stateA(4), -cfg.max_steering, cfg.max_steering);

        double np = spline.find_closest_progress(stateA(0), stateA(1));
        if (np > progressA) progressA = np;
    }

    // ====== Scenario B: Direct steering plant (like real deployment) ======
    printf("\n=== Scenario B: Direct steering plant (no rate limit) ===\n");
    printf("step | v_cmd  delta_cmd | plant_v plant_delta | x       y       theta   | CTE     heading_err\n");
    printf("-----|------------------|---------------------|-------------------------|--------------------\n");

    mpcc::ActiveSolver solverB;
    solverB.init(cfg);
    solverB.path_lookup.lookup = make_lookup;

    mpcc::KinematicModel plantB(cfg.wheelbase);
    mpcc::Vec3 stateB;
    stateB << init_x, init_y, init_theta;
    double actual_v_B = 0.0;
    double actual_delta_B = 0.0;
    PDSpeedController pdB;
    double progressB = 0.0;
    int cfailB = 0;

    for (int step = 0; step < 30; step++) {
        std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
        double s = progressB;
        double lv = std::max(actual_v_B, cfg.reference_velocity * 0.5);
        for (int k = 0; k <= cfg.horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);
            double rx, ry, ct, st;
            spline.get_path_reference(s, rx, ry, ct, st);
            refs[k].x = rx; refs[k].y = ry;
            refs[k].cos_theta = ct; refs[k].sin_theta = st;
            refs[k].curvature = spline.get_curvature(s);
            s += std::max(0.10, lv * std::exp(-0.4 * std::abs(refs[k].curvature))) * cfg.dt;
        }

        mpcc::VecX x0_5d;
        x0_5d << stateB(0), stateB(1), stateB(2), actual_v_B, actual_delta_B;
        auto result = solverB.solve(x0_5d, refs, progressB, total_len, {}, {});
        if (!result.success) { if (++cfailB >= 5) { printf("  SOLVER FAILED at step %d\n", step); break; } continue; }
        cfailB = 0;

        double v_cmd = std::clamp(result.v_cmd, cfg.min_velocity, cfg.max_velocity);
        double delta_cmd = std::clamp(result.delta_cmd, -cfg.max_steering, cfg.max_steering);

        double cp = spline.find_closest_progress(stateB(0), stateB(1));
        double rpx, rpy; spline.get_position(cp, rpx, rpy);
        double cte = std::hypot(stateB(0) - rpx, stateB(1) - rpy);
        double path_theta = spline.get_tangent(cp);
        double herr = stateB(2) - path_theta;
        while (herr > M_PI) herr -= 2*M_PI;
        while (herr < -M_PI) herr += 2*M_PI;

        printf("%4d | %6.3f %8.4f | %7.3f %11.4f | %7.4f %7.4f %7.4f | %7.4f %7.4f\n",
               step, v_cmd, delta_cmd, actual_v_B, actual_delta_B,
               stateB(0), stateB(1), stateB(2), cte, herr);
        if (cte > 1.0) break;

        actual_v_B = pdB.step(v_cmd, cfg.dt);
        actual_delta_B = delta_cmd;  // Direct! No rate limit.

        mpcc::Vec2 u_direct(actual_v_B, actual_delta_B);
        stateB = plantB.rk4_step(stateB, u_direct, cfg.dt);

        double np = spline.find_closest_progress(stateB(0), stateB(1));
        if (np > progressB) progressB = np;
    }

    printf("\nKey: Rate-limited plant dampens solver's steering commands.\n");
    printf("Direct plant applies full command immediately — if solver is aggressive, oscillation follows.\n");

    return 0;
}
