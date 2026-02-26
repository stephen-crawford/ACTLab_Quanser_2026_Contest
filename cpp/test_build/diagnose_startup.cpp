/**
 * Diagnostic: Reproduce EXACT deployment startup scenario.
 *
 * The deployment log (mpcc_20260226_105501.csv) shows:
 *   - Vehicle at (0.016, 0.004) heading 0.014 rad, v=0.034
 *   - delta_cmd = 0.5236 (MAX) for 27 consecutive steps
 *   - CTE ≈ -0.005, heading_err ≈ -0.024
 *   - progress_pct = 0.0 (no progress for 2.5s)
 *
 * This test reproduces the EXACT state and checks if the solver
 * also produces max steering. If it does, the bug is in the solver;
 * if not, the bug is in the deployment pipeline.
 *
 * Build:
 *   cd /home/stephen/quanser-acc/cpp/test_build
 *   g++ -std=c++17 -O2 -I.. -I/usr/include/eigen3 \
 *       -o diagnose_startup diagnose_startup.cpp ../road_graph.cpp
 */

#include <cmath>
#include <cstdio>
#include <vector>
#include "coordinate_transform.h"
#include "cubic_spline_path.h"
#include "road_graph.h"
#include "mpcc_solver_interface.h"

int main() {
    printf("=== Diagnose Startup Max Steering ===\n\n");

    // Build path exactly as deployment does (road graph returns QLabs coords)
    acc::RoadGraph rg;
    auto path_opt = rg.plan_path_for_mission_leg("hub_to_pickup", acc::HUB_X, acc::HUB_Y);
    auto& path_xy = *path_opt;
    auto& qlabs_wx = path_xy.first;
    auto& qlabs_wy = path_xy.second;

    printf("Path: %zu waypoints\n", qlabs_wx.size());
    printf("Path start (QLabs): (%.6f, %.6f)\n", qlabs_wx[0], qlabs_wy[0]);

    // Transform to map frame (CRITICAL — mission_manager does this at line 858)
    acc::TransformParams tp;
    std::vector<double> path_wx, path_wy;
    acc::qlabs_path_to_map(qlabs_wx, qlabs_wy, tp, path_wx, path_wy);

    printf("Path start (map):   (%.6f, %.6f)\n", path_wx[0], path_wy[0]);
    printf("Path[1] (map):      (%.6f, %.6f)\n", path_wx[1], path_wy[1]);
    printf("Path[100] (map):    (%.6f, %.6f)\n", path_wx[100], path_wy[100]);

    // Build cubic spline (exactly as controller does)
    acc::CubicSplinePath spline;
    spline.build(path_wx, path_wy, true);
    double total_len = spline.total_length();
    printf("Spline total length: %.4fm\n\n", total_len);

    // Path tangent at start
    double tangent0 = spline.get_tangent(0.0);
    double curv0 = spline.get_curvature(0.0);
    printf("Path tangent at s=0: %.6f rad (%.2f deg)\n", tangent0, tangent0 * 180 / M_PI);
    printf("Path curvature at s=0: %.6f\n\n", curv0);

    // Deployment config (exact match to controller node)
    mpcc::Config cfg;
    cfg.horizon = 10;
    cfg.dt = 0.1;
    cfg.wheelbase = 0.256;
    cfg.max_velocity = 1.2;
    cfg.min_velocity = 0.0;
    cfg.max_steering = 0.45;  // hardware servo limit
    cfg.max_acceleration = 1.5;
    cfg.max_steering_rate = 1.5;
    cfg.reference_velocity = 0.45;
    cfg.contour_weight = 20.0;
    cfg.lag_weight = 10.0;
    cfg.velocity_weight = 15.0;
    cfg.steering_weight = 0.05;
    cfg.acceleration_weight = 0.01;
    cfg.steering_rate_weight = 1.0;
    cfg.heading_weight = 3.0;
    cfg.jerk_weight = 0.0;
    cfg.boundary_weight = 0.0;
    cfg.progress_weight = 1.0;
    cfg.max_sqp_iterations = 5;
    cfg.max_qp_iterations = 20;
    cfg.startup_ramp_duration_s = 0.0;  // disabled
    cfg.startup_elapsed_s = 0.0;
    cfg.startup_progress_weight = 5.0;

    // ====== Test 1: Solver with NO path lookup (pre-computed refs) ======
    printf("=== Test 1: Pre-computed path refs (no adaptive re-projection) ===\n");
    {
        mpcc::ActiveSolver solver;
        solver.init(cfg);
        // NO path_lookup — just like diagnose_steering.cpp which showed δ≈0

        double state_x = 0.016, state_y = 0.004, state_theta = 0.014;
        double state_v = 0.034, state_delta = 0.0;
        double progress = 0.0;

        // Build path refs exactly as controller does
        double lookahead_v = std::max(state_v, cfg.reference_velocity * 0.5);
        std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
        double s = progress;
        for (int k = 0; k <= cfg.horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);
            double rx, ry, ct, st;
            spline.get_path_reference(s, rx, ry, ct, st);
            refs[k].x = rx; refs[k].y = ry;
            refs[k].cos_theta = ct; refs[k].sin_theta = st;
            refs[k].curvature = spline.get_curvature(s);
            double step = std::max(0.10, lookahead_v * std::exp(-0.4 * std::abs(refs[k].curvature))) * cfg.dt;
            s += step;
        }

        printf("Path refs:\n");
        for (int k = 0; k <= cfg.horizon; k++) {
            double pt = std::atan2(refs[k].sin_theta, refs[k].cos_theta);
            printf("  k=%d: (%.6f, %.6f) theta=%.4f curv=%.4f\n",
                   k, refs[k].x, refs[k].y, pt, refs[k].curvature);
        }

        mpcc::VecX x0;
        x0 << state_x, state_y, state_theta, state_v, state_delta;
        auto result = solver.solve(x0, refs, progress, total_len, {}, {});

        printf("\nResult: v_cmd=%.4f, delta_cmd=%.4f (%.1f deg), success=%d, solve=%.0fus\n",
               result.v_cmd, result.delta_cmd, result.delta_cmd * 180 / M_PI,
               result.success, result.solve_time_us);
        printf("Expected delta_cmd ≈ 0.0, got %.4f\n\n", result.delta_cmd);
    }

    // ====== Test 2: Solver WITH path lookup (adaptive re-projection) ======
    printf("=== Test 2: With adaptive path re-projection (as deployed) ===\n");
    {
        mpcc::ActiveSolver solver;
        solver.init(cfg);

        // Set up path lookup exactly as controller does
        solver.path_lookup.lookup = [&spline, total_len](
            double px, double py, double s_min, double* s_out) -> mpcc::PathRef
        {
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

        double state_x = 0.016, state_y = 0.004, state_theta = 0.014;
        double state_v = 0.034, state_delta = 0.0;
        double progress = 0.0;

        double lookahead_v = std::max(state_v, cfg.reference_velocity * 0.5);
        std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
        double s = progress;
        for (int k = 0; k <= cfg.horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);
            double rx, ry, ct, st;
            spline.get_path_reference(s, rx, ry, ct, st);
            refs[k].x = rx; refs[k].y = ry;
            refs[k].cos_theta = ct; refs[k].sin_theta = st;
            refs[k].curvature = spline.get_curvature(s);
            double step = std::max(0.10, lookahead_v * std::exp(-0.4 * std::abs(refs[k].curvature))) * cfg.dt;
            s += step;
        }

        mpcc::VecX x0;
        x0 << state_x, state_y, state_theta, state_v, state_delta;
        auto result = solver.solve(x0, refs, progress, total_len, {}, {});

        printf("Result: v_cmd=%.4f, delta_cmd=%.4f (%.1f deg), success=%d, solve=%.0fus\n",
               result.v_cmd, result.delta_cmd, result.delta_cmd * 180 / M_PI,
               result.success, result.solve_time_us);
        printf("Expected delta_cmd ≈ 0.0, got %.4f\n\n", result.delta_cmd);
    }

    // ====== Test 3: Multiple steps with state_delta feedback (deployment loop) ======
    printf("=== Test 3: Multi-step simulation with state_delta = delta_cmd (deployment bug?) ===\n");
    printf("step | v_cmd  delta_cmd | state_v  state_delta | x       y       theta   | CTE     h_err\n");
    printf("-----|------------------|----------------------|-------------------------|-------------\n");
    {
        mpcc::ActiveSolver solver;
        solver.init(cfg);

        // Set up path lookup
        solver.path_lookup.lookup = [&spline, total_len](
            double px, double py, double s_min, double* s_out) -> mpcc::PathRef
        {
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

        // Initial state from deployment log
        double state_x = 0.016, state_y = 0.004, state_theta = 0.014;
        double state_v = 0.034, state_delta = 0.0;
        double progress = 0.0;

        // Simulate without plant model — just test what the solver does
        // when state_delta is set to the commanded value (as deployment does)
        for (int step = 0; step < 15; step++) {
            double lookahead_v = std::max(state_v, cfg.reference_velocity * 0.5);
            std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
            double s = progress;
            for (int k = 0; k <= cfg.horizon; k++) {
                s = std::clamp(s, 0.0, total_len - 0.001);
                double rx, ry, ct, st;
                spline.get_path_reference(s, rx, ry, ct, st);
                refs[k].x = rx; refs[k].y = ry;
                refs[k].cos_theta = ct; refs[k].sin_theta = st;
                refs[k].curvature = spline.get_curvature(s);
                double step_size = std::max(0.10, lookahead_v * std::exp(-0.4 * std::abs(refs[k].curvature))) * cfg.dt;
                s += step_size;
            }

            mpcc::VecX x0;
            x0 << state_x, state_y, state_theta, state_v, state_delta;
            auto result = solver.solve(x0, refs, progress, total_len, {}, {});

            // Compute CTE and heading error
            double cp = spline.find_closest_progress(state_x, state_y);
            double rpx, rpy; spline.get_position(cp, rpx, rpy);
            double cte = std::hypot(state_x - rpx, state_y - rpy);
            double path_theta = spline.get_tangent(cp);
            double herr = state_theta - path_theta;
            while (herr > M_PI) herr -= 2*M_PI;
            while (herr < -M_PI) herr += 2*M_PI;

            printf("%4d | %6.3f %8.4f | %7.3f %12.4f | %7.4f %7.4f %7.4f | %7.4f %7.4f\n",
                   step, result.v_cmd, result.delta_cmd, state_v, state_delta,
                   state_x, state_y, state_theta, cte, herr);

            if (!result.success) { printf("  SOLVER FAILED!\n"); break; }

            // Deployment feedback: state_delta = delta_cmd (NOT measured from plant!)
            // This is what mpcc_controller_node.cpp line 1113 does
            state_delta = result.delta_cmd;

            // Don't update position (vehicle barely moves at v≈0.034)
            // Just let solver re-run with same position but new state_delta
        }
    }

    // ====== Test 4: Same but with state_delta always 0 (measured, not commanded) ======
    printf("\n=== Test 4: Multi-step with state_delta = 0 (measured from stationary vehicle) ===\n");
    printf("step | v_cmd  delta_cmd | state_v  state_delta | x       y       theta   | CTE     h_err\n");
    printf("-----|------------------|----------------------|-------------------------|-------------\n");
    {
        mpcc::ActiveSolver solver;
        solver.init(cfg);

        solver.path_lookup.lookup = [&spline, total_len](
            double px, double py, double s_min, double* s_out) -> mpcc::PathRef
        {
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

        double state_x = 0.016, state_y = 0.004, state_theta = 0.014;
        double state_v = 0.034, state_delta = 0.0;
        double progress = 0.0;

        for (int step = 0; step < 15; step++) {
            double lookahead_v = std::max(state_v, cfg.reference_velocity * 0.5);
            std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
            double s = progress;
            for (int k = 0; k <= cfg.horizon; k++) {
                s = std::clamp(s, 0.0, total_len - 0.001);
                double rx, ry, ct, st;
                spline.get_path_reference(s, rx, ry, ct, st);
                refs[k].x = rx; refs[k].y = ry;
                refs[k].cos_theta = ct; refs[k].sin_theta = st;
                refs[k].curvature = spline.get_curvature(s);
                double step_size = std::max(0.10, lookahead_v * std::exp(-0.4 * std::abs(refs[k].curvature))) * cfg.dt;
                s += step_size;
            }

            mpcc::VecX x0;
            // Keep state_delta = 0 (as if measured from actual vehicle)
            x0 << state_x, state_y, state_theta, state_v, 0.0;
            auto result = solver.solve(x0, refs, progress, total_len, {}, {});

            double cp = spline.find_closest_progress(state_x, state_y);
            double rpx, rpy; spline.get_position(cp, rpx, rpy);
            double cte = std::hypot(state_x - rpx, state_y - rpy);
            double path_theta = spline.get_tangent(cp);
            double herr = state_theta - path_theta;
            while (herr > M_PI) herr -= 2*M_PI;
            while (herr < -M_PI) herr += 2*M_PI;

            printf("%4d | %6.3f %8.4f | %7.3f %12.4f | %7.4f %7.4f %7.4f | %7.4f %7.4f\n",
                   step, result.v_cmd, result.delta_cmd, state_v, 0.0,
                   state_x, state_y, state_theta, cte, herr);
        }
    }

    // ====== Test 5: Check max_steering = 0.45 (hardware limit) ======
    printf("\n=== Test 5: Same test with max_steering=0.45 (hardware limit) ===\n");
    {
        mpcc::Config cfg45 = cfg;
        cfg45.max_steering = 0.45;  // True hardware limit

        mpcc::ActiveSolver solver;
        solver.init(cfg45);

        solver.path_lookup.lookup = [&spline, total_len](
            double px, double py, double s_min, double* s_out) -> mpcc::PathRef
        {
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

        double state_x = 0.016, state_y = 0.004, state_theta = 0.014;
        double state_v = 0.034, state_delta = 0.0;
        double progress = 0.0;

        mpcc::VecX x0;
        x0 << state_x, state_y, state_theta, state_v, state_delta;
        auto result = solver.solve(x0, {}, progress, total_len, {}, {});
        // Note: empty refs should trigger get_path_ref fallback

        // Redo with proper refs
        double lookahead_v = std::max(state_v, cfg45.reference_velocity * 0.5);
        std::vector<mpcc::PathRef> refs(cfg45.horizon + 1);
        double s = progress;
        for (int k = 0; k <= cfg45.horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);
            double rx, ry, ct, st;
            spline.get_path_reference(s, rx, ry, ct, st);
            refs[k].x = rx; refs[k].y = ry;
            refs[k].cos_theta = ct; refs[k].sin_theta = st;
            refs[k].curvature = spline.get_curvature(s);
            double step_size = std::max(0.10, lookahead_v * std::exp(-0.4 * std::abs(refs[k].curvature))) * cfg45.dt;
            s += step_size;
        }

        result = solver.solve(x0, refs, progress, total_len, {}, {});
        printf("With max_steering=0.45: v_cmd=%.4f, delta_cmd=%.4f (%.1f deg)\n",
               result.v_cmd, result.delta_cmd, result.delta_cmd * 180 / M_PI);
    }

    printf("\n=== Summary ===\n");
    printf("If Tests 1 & 2 produce delta≈0 but Test 3 shows escalating delta,\n");
    printf("then the bug is state_delta = delta_cmd feedback (line 1113).\n");
    printf("The solver warm-starts from its own commands, and if those commands\n");
    printf("don't match the actual vehicle state, oscillation follows.\n");

    return 0;
}
