/**
 * Test: Map-frame startup simulation — matches deployment pipeline exactly.
 *
 * This test reproduces the FULL deployment flow:
 * 1. Road graph generates path in QLabs frame
 * 2. Transform to map frame (qlabs_path_to_map)
 * 3. Hermite blend aligns path start to vehicle heading (0° in map frame)
 * 4. CubicSplinePath with Gaussian smoothing
 * 5. MPCC solver with adaptive path re-projection
 * 6. Vehicle starts at (0, 0, 0°) — Cartographer initialization
 *
 * Previously, tests started the vehicle at the path start position AND heading,
 * which masked the 35° heading mismatch between Cartographer's initial heading
 * and the road graph path direction in map frame.
 *
 * Build:
 *   cd /home/stephen/quanser-acc/cpp/test_build
 *   g++ -std=c++17 -O2 -I.. -I/usr/include/eigen3 \
 *       -o test_mapframe_startup test_mapframe_startup.cpp ../road_graph.cpp
 */

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "coordinate_transform.h"
#include "cubic_spline_path.h"
#include "road_graph.h"
#include "mpcc_solver.h"

// Hermite path blending — MUST match mission_manager_node.cpp
static void align_path_to_vehicle_heading(
    std::vector<double>& mx, std::vector<double>& my,
    double veh_x, double veh_y, double veh_yaw)
{
    if (mx.size() < 2) return;

    double path_dx = mx[1] - mx[0];
    double path_dy = my[1] - my[0];
    double path_yaw = std::atan2(path_dy, path_dx);
    double heading_err = acc::normalize_angle(path_yaw - veh_yaw);
    if (std::abs(heading_err) < 5.0 * M_PI / 180.0) return;

    // Extended blend to cover solver horizon
    double base_blend = 0.50;
    double err_scale = std::abs(heading_err) / (M_PI / 6.0);
    double blend_dist = base_blend + 1.0 * std::min(err_scale, 2.0);
    blend_dist = std::min(blend_dist, 2.5);

    double cum = 0.0;
    int rejoin_idx = 1;
    for (size_t i = 1; i < mx.size(); i++) {
        cum += std::hypot(mx[i] - mx[i-1], my[i] - my[i-1]);
        if (cum >= blend_dist) { rejoin_idx = static_cast<int>(i); break; }
    }
    if (rejoin_idx <= 0) return;

    double rx = mx[rejoin_idx], ry = my[rejoin_idx];
    double rtx, rty;
    if (rejoin_idx + 1 < static_cast<int>(mx.size())) {
        rtx = mx[rejoin_idx + 1] - mx[rejoin_idx];
        rty = my[rejoin_idx + 1] - my[rejoin_idx];
    } else {
        rtx = mx[rejoin_idx] - mx[rejoin_idx - 1];
        rty = my[rejoin_idx] - my[rejoin_idx - 1];
    }
    double rlen = std::hypot(rtx, rty);
    if (rlen > 1e-6) { rtx /= rlen; rty /= rlen; }

    double chord = std::hypot(rx - veh_x, ry - veh_y);
    double tang_scale = chord;

    double p0x = veh_x, p0y = veh_y;
    double m0x = std::cos(veh_yaw) * tang_scale;
    double m0y = std::sin(veh_yaw) * tang_scale;
    double p1x = rx, p1y = ry;
    double m1x = rtx * tang_scale, m1y = rty * tang_scale;

    double ds = 0.001;
    int n_pts = std::max(static_cast<int>(chord / ds), 10);
    std::vector<double> new_x, new_y;
    new_x.reserve(n_pts + mx.size());
    new_y.reserve(n_pts + mx.size());

    for (int i = 0; i <= n_pts; i++) {
        double t = static_cast<double>(i) / n_pts;
        double t2 = t * t, t3 = t2 * t;
        double h00 = 2*t3 - 3*t2 + 1;
        double h10 = t3 - 2*t2 + t;
        double h01 = -2*t3 + 3*t2;
        double h11 = t3 - t2;
        new_x.push_back(h00*p0x + h10*m0x + h01*p1x + h11*m1x);
        new_y.push_back(h00*p0y + h10*m0y + h01*p1y + h11*m1y);
    }

    for (size_t i = rejoin_idx + 1; i < mx.size(); i++) {
        new_x.push_back(mx[i]);
        new_y.push_back(my[i]);
    }

    printf("    Hermite blend: heading_err=%.1fdeg, blend_dist=%.2fm, chord=%.2fm, rejoin_idx=%d\n",
           heading_err * 180/M_PI, blend_dist, chord, rejoin_idx);

    mx = std::move(new_x);
    my = std::move(new_y);
}

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
    printf("=== Map-Frame Startup Test ===\n\n");

    // Step 1: Generate path in QLabs frame
    acc::RoadGraph rg;
    auto path_opt = rg.plan_path_for_mission_leg("hub_to_pickup", acc::HUB_X, acc::HUB_Y);
    auto& [qx, qy] = *path_opt;
    printf("Path: %zu QLabs waypoints\n", qx.size());

    // Step 2: Transform to map frame
    acc::TransformParams tp;
    std::vector<double> mx, my;
    acc::qlabs_path_to_map(qx, qy, tp, mx, my);
    double path_tangent_raw = std::atan2(my[1] - my[0], mx[1] - mx[0]);
    printf("Path start (map): (%.6f, %.6f) tangent=%.1f deg\n",
           mx[0], my[0], path_tangent_raw * 180/M_PI);

    // Step 3: Vehicle starts at (0, 0, 0°) — Cartographer initialization
    double veh_x = 0.0, veh_y = 0.0, veh_yaw = 0.0;
    printf("Vehicle start: (%.3f, %.3f) heading=%.1f deg\n", veh_x, veh_y, veh_yaw * 180/M_PI);
    printf("Heading mismatch: %.1f deg\n\n", (path_tangent_raw - veh_yaw) * 180/M_PI);

    // Step 4: Hermite blend
    printf("--- Applying Hermite blend ---\n");
    align_path_to_vehicle_heading(mx, my, veh_x, veh_y, veh_yaw);
    double blended_tangent = std::atan2(my[1] - my[0], mx[1] - mx[0]);
    printf("After blend: %zu waypoints, tangent=%.1f deg (was %.1f deg)\n\n",
           mx.size(), blended_tangent * 180/M_PI, path_tangent_raw * 180/M_PI);

    // Step 5: Build CubicSplinePath with Gaussian smoothing
    acc::CubicSplinePath spline;
    spline.build(mx, my, true);
    double total_len = spline.total_length();
    double spline_tangent = spline.get_tangent(0.0);
    printf("Spline tangent at s=0: %.1f deg (blended), total_len=%.2fm\n\n",
           spline_tangent * 180/M_PI, total_len);

    // Step 6: Config matching deployment
    mpcc::Config cfg;
    cfg.horizon = 10;
    cfg.dt = 0.1;
    cfg.wheelbase = 0.256;
    cfg.max_velocity = 1.2;
    cfg.min_velocity = 0.0;
    cfg.max_steering = 0.45;
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
    cfg.startup_ramp_duration_s = 0.0;

    // Step 7: Full deployment simulation
    printf("=== Deployment-Realistic Simulation ===\n");
    printf("step | v_cmd  delta_cmd | plant_v plant_delta | x       y       theta   | CTE     h_err   | progress\n");
    printf("-----|------------------|---------------------|-------------------------|-----------------|--------\n");

    mpcc::Solver solver;
    solver.init(cfg);

    // Adaptive path re-projection (as deployed)
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

    // Plant model: direct steering (servo reaches command instantly)
    mpcc::KinematicModel plant(cfg.wheelbase);
    mpcc::Vec3 state;
    state << veh_x, veh_y, veh_yaw;
    double actual_v = 0.0;
    double actual_delta = 0.0;
    PDSpeedController pd;
    double progress = 0.0;

    double max_cte = 0.0, sum_cte = 0.0;
    int steer_sat_count = 0;
    int n_steps = 0;

    // Trace data for CSV
    std::vector<double> tr_t, tr_x, tr_y, tr_theta, tr_v, tr_vcmd;
    std::vector<double> tr_delta, tr_cte, tr_herr, tr_prog;

    for (int step = 0; step < 100; step++) {
        // Build path refs (as controller does)
        double lookahead_v = std::max(actual_v, cfg.reference_velocity * 0.5);
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

        // Solve (5D state as controller does)
        mpcc::VecX x0;
        x0 << state(0), state(1), state(2), actual_v, actual_delta;
        auto result = solver.solve(x0, refs, progress, total_len, {}, {});
        if (!result.success) { printf("  SOLVER FAILED at step %d\n", step); break; }

        double v_cmd = std::clamp(result.v_cmd, cfg.min_velocity, cfg.max_velocity);
        double delta_cmd = std::clamp(result.delta_cmd, -cfg.max_steering, cfg.max_steering);

        // Compute CTE and heading error
        double cp = spline.find_closest_progress(state(0), state(1));
        double rpx, rpy; spline.get_position(cp, rpx, rpy);
        double cte = std::hypot(state(0) - rpx, state(1) - rpy);
        double path_theta = spline.get_tangent(cp);
        double herr = state(2) - path_theta;
        while (herr > M_PI) herr -= 2*M_PI;
        while (herr < -M_PI) herr += 2*M_PI;

        double progress_pct = 100.0 * progress / total_len;

        if (step < 20 || step % 10 == 0) {
            printf("%4d | %6.3f %8.4f | %7.3f %11.4f | %7.4f %7.4f %7.4f | %7.4f %7.4f | %5.1f%%\n",
                   step, v_cmd, delta_cmd, actual_v, actual_delta,
                   state(0), state(1), state(2), cte, herr, progress_pct);
        }

        // Record trace
        tr_t.push_back(step * cfg.dt);
        tr_x.push_back(state(0));
        tr_y.push_back(state(1));
        tr_theta.push_back(state(2));
        tr_v.push_back(actual_v);
        tr_vcmd.push_back(v_cmd);
        tr_delta.push_back(delta_cmd);
        tr_cte.push_back(cte);
        tr_herr.push_back(herr);
        tr_prog.push_back(progress_pct);

        max_cte = std::max(max_cte, cte);
        sum_cte += cte;
        if (std::abs(delta_cmd) > cfg.max_steering - 0.01) steer_sat_count++;
        n_steps++;

        // Plant: direct steering, PD speed control
        actual_v = pd.step(v_cmd, cfg.dt);
        actual_delta = delta_cmd;  // Direct! No rate limit.

        mpcc::Vec2 u_direct(actual_v, actual_delta);
        state = plant.rk4_step(state, u_direct, cfg.dt);

        // Update progress (monotonic)
        double np = spline.find_closest_progress(state(0), state(1));
        if (np > progress) progress = np;

        // Check goal
        double remaining = total_len - progress;
        if (remaining < 0.15) {
            printf("  Goal reached at step %d (%.1f%%)\n", step, 100.0 * progress / total_len);
            break;
        }
    }

    // Write CSV for plotting
    {
        std::ofstream f("mapframe_startup.csv");
        f << "elapsed_s,x,y,theta,v_meas,v_cmd,delta_cmd,cross_track_err,heading_err,progress_pct\n";
        for (size_t i = 0; i < tr_t.size(); i++) {
            f << std::fixed << std::setprecision(4)
              << tr_t[i] << "," << tr_x[i] << "," << tr_y[i] << ","
              << tr_theta[i] << "," << tr_v[i] << "," << tr_vcmd[i] << ","
              << tr_delta[i] << "," << tr_cte[i] << "," << tr_herr[i] << ","
              << std::setprecision(1) << tr_prog[i] << "\n";
        }
        // Write reference path (blended, in map frame)
        f << "\n# Reference path (map frame)\n";
        f << "ref_x,ref_y\n";
        for (size_t i = 0; i < mx.size(); i += std::max(size_t(1), mx.size()/500)) {
            f << std::fixed << std::setprecision(6) << mx[i] << "," << my[i] << "\n";
        }
        // Write original path (pre-blend, in map frame)
        // (not available here since mx was overwritten by blend — write qlabs path in qlabs frame)
        f.close();
        printf("  CSV written: mapframe_startup.csv\n");
    }

    double avg_cte = sum_cte / std::max(n_steps, 1);
    double steer_sat_pct = 100.0 * steer_sat_count / std::max(n_steps, 1);

    printf("\n=== Results ===\n");
    printf("Max CTE:     %.4fm\n", max_cte);
    printf("Avg CTE:     %.4fm\n", avg_cte);
    printf("Steer sat:   %.0f%% (%d/%d steps)\n", steer_sat_pct, steer_sat_count, n_steps);
    printf("Duration:    %d steps (%.1fs)\n", n_steps, n_steps * cfg.dt);

    // PASS/FAIL criteria
    bool pass = true;
    if (max_cte > 0.25) {
        printf("FAIL: max_cte %.4f > 0.25m\n", max_cte);
        pass = false;
    }
    if (steer_sat_pct > 10.0) {
        printf("FAIL: steering saturation %.0f%% > 10%%\n", steer_sat_pct);
        pass = false;
    }

    printf("\n%s\n", pass ? "=== PASS ===" : "=== FAIL ===");
    return pass ? 0 : 1;
}
