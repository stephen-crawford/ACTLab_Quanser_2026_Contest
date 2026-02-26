/**
 * Full-mission closed-loop simulation.
 *
 * Simulates the ENTIRE 3-leg mission (Hub→Pickup→Dropoff→Hub) continuously
 * with carry-over state between legs, realistic PD speed controller lag,
 * and outputs comprehensive CSV data for visualization.
 *
 * This test matches the deployment pipeline exactly:
 * 1. Generates path from road_graph for each leg
 * 2. Transforms QLabs→map frame
 * 3. Builds CubicSplinePath with Gaussian smoothing
 * 4. Runs MPCC solver with adaptive path re-projection
 * 5. Uses PD speed controller (simulating qcar2_hardware)
 * 6. Carries vehicle state between legs (heading, position offsets)
 * 7. Outputs map-frame AND QLabs-frame coordinates for plotting
 *
 * Build:
 *   cd /home/stephen/quanser-acc/cpp/test_build
 *   g++ -std=c++17 -O2 -I.. -I/usr/include/eigen3 \
 *       -o test_full_mission_sim test_full_mission_sim.cpp ../road_graph.cpp
 *
 * Run:
 *   ./test_full_mission_sim
 *
 * Generates:
 *   full_mission_sim.csv - trace data matching MPCC CSV format
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "mpcc_solver_interface.h"
#include "cubic_spline_path.h"
#include "road_graph.h"
#include "coordinate_transform.h"

// PD Speed Controller simulation (matches qcar2_hardware.cpp)
struct PDSpeedController {
    double kp = 20.0;
    double kd = 0.1;
    double dt_inner = 0.015;
    double pwm_max = 0.3;
    double km = 0.0047;
    double battery_voltage = 7.2;

    double motor_speed_cmd = 0.0;
    double prior_speed_error = 0.0;
    double actual_speed = 0.0;

    double step(double desired_speed, double dt_outer) {
        int n_inner = static_cast<int>(dt_outer / dt_inner);
        for (int i = 0; i < n_inner; i++) {
            double speed_error = desired_speed - actual_speed;
            motor_speed_cmd += (speed_error * kp +
                               (speed_error - prior_speed_error) / dt_inner * kd)
                               * km / battery_voltage;
            motor_speed_cmd = std::clamp(motor_speed_cmd, -pwm_max, pwm_max);
            if (motor_speed_cmd < 0.01 && motor_speed_cmd >= 0 && desired_speed > 0)
                motor_speed_cmd = 0.01 + motor_speed_cmd;
            prior_speed_error = speed_error;
            double target_speed = motor_speed_cmd * (0.65 / 0.3);
            double tau = 0.05;
            actual_speed += (target_speed - actual_speed) * (dt_inner / tau);
            actual_speed = std::max(0.0, actual_speed);
        }
        return actual_speed;
    }

    void reset() {
        motor_speed_cmd = 0.0;
        prior_speed_error = 0.0;
        actual_speed = 0.0;
    }
};

// CSV output structure
struct TracePoint {
    double elapsed_s;
    double x_map, y_map;   // map frame
    double x_ql, y_ql;     // QLabs frame
    double theta;
    double v_meas, v_cmd, delta_cmd;
    double cte;
    double heading_err;
    double progress_pct;
    double curvature;
    std::string leg_name;
};

// Store blended path for visualization
struct BlendedPath {
    std::string name;
    std::vector<double> x_ql, y_ql;  // QLabs frame
};

// Configure solver matching controller node
mpcc::Config make_deployment_config() {
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
    cfg.progress_weight = 1.0;
    cfg.jerk_weight = 0.0;
    cfg.boundary_weight = 0.0;  // Disabled (ref has 0; boundary fights contour on curves)
    cfg.boundary_default_width = 0.22;
    cfg.max_sqp_iterations = 5;
    cfg.max_qp_iterations = 20;
    cfg.qp_tolerance = 1e-5;
    // Startup ramp disabled (matching fix)
    cfg.startup_ramp_duration_s = 0.0;
    cfg.startup_elapsed_s = 0.0;
    cfg.startup_progress_weight = 5.0;
    return cfg;
}

// Run one mission leg, returning trace and final state
struct LegResult {
    std::vector<TracePoint> trace;
    mpcc::VecX final_state;  // 5D [x, y, theta, v, delta]
    BlendedPath blended_path;  // actual path being tracked (after Hermite blend)
    double max_cte = 0.0;
    double avg_cte = 0.0;
    double max_heading_err = 0.0;
    int steering_saturated_steps = 0;
    bool completed = false;
};

// Hermite path blending — matches mission_manager_node.cpp::align_path_to_vehicle_heading
void align_path_to_vehicle_heading(
    std::vector<double>& mx, std::vector<double>& my,
    double veh_x, double veh_y, double veh_yaw)
{
    if (mx.size() < 2) return;

    double path_dx = mx[1] - mx[0];
    double path_dy = my[1] - my[0];
    double path_yaw = std::atan2(path_dy, path_dx);
    double heading_err = acc::normalize_angle(path_yaw - veh_yaw);
    if (std::abs(heading_err) < 5.0 * M_PI / 180.0) return;

    // Scale blend distance with heading error — larger errors need longer arcs
    // to avoid extreme curvature in the Hermite transition
    double base_blend = 0.50;
    double err_scale = std::abs(heading_err) / (M_PI / 6.0);  // 1.0 at 30°
    double blend_dist = base_blend + 1.0 * std::min(err_scale, 2.0);
    blend_dist = std::min(blend_dist, 2.5);  // cover full solver horizon

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

    std::printf("    Hermite blend: heading_err=%.1fdeg, blend_dist=%.2fm, chord=%.2fm\n",
        heading_err * 180.0 / M_PI, blend_dist, chord);

    mx = std::move(new_x);
    my = std::move(new_y);
}

LegResult run_leg(
    const std::vector<double>& path_x_qlabs,
    const std::vector<double>& path_y_qlabs,
    const mpcc::VecX& initial_state,
    double elapsed_start,
    const std::string& leg_name,
    int max_steps,
    PDSpeedController& pd_ctrl)
{
    LegResult result;
    acc::TransformParams tp;

    // Transform QLabs → map frame
    std::vector<double> map_x, map_y;
    acc::qlabs_path_to_map(path_x_qlabs, path_y_qlabs, tp, map_x, map_y);

    // Align path start with vehicle heading (matches mission_manager behavior)
    align_path_to_vehicle_heading(map_x, map_y,
        initial_state(0), initial_state(1), initial_state(2));

    // Build CubicSplinePath (with Gaussian smoothing)
    acc::CubicSplinePath spline;
    spline.build(map_x, map_y, true);
    double total_len = spline.total_length();

    // Store blended path in QLabs frame for visualization
    result.blended_path.name = leg_name;
    int n_vis = static_cast<int>(total_len / 0.005) + 1;
    result.blended_path.x_ql.resize(n_vis);
    result.blended_path.y_ql.resize(n_vis);
    for (int i = 0; i < n_vis; i++) {
        double s = total_len * i / (n_vis - 1);
        s = std::min(s, total_len - 0.001);
        double px, py;
        spline.get_position(s, px, py);
        acc::map_to_qlabs_2d(px, py, tp, result.blended_path.x_ql[i],
                             result.blended_path.y_ql[i]);
    }

    // Initialize solver
    mpcc::ActiveSolver solver;
    auto cfg = make_deployment_config();
    solver.init(cfg);

    // Set up adaptive path re-projection
    solver.path_lookup.lookup = [&spline, total_len](
        double px, double py, double s_min, double* s_out) -> mpcc::PathRef
    {
        double s = spline.find_closest_progress_from(px, py, s_min);
        s = std::clamp(s, 0.0, total_len - 0.001);
        if (s_out) *s_out = s;
        mpcc::PathRef ref;
        double ct, st;
        spline.get_path_reference(s, ref.x, ref.y, ct, st);
        ref.cos_theta = ct;
        ref.sin_theta = st;
        ref.curvature = spline.get_curvature(s);
        return ref;
    };

    // Initialize state from input (carry-over from previous leg)
    mpcc::AckermannModel plant(cfg.wheelbase);
    mpcc::VecX state = initial_state;

    // Find initial progress on this path
    double progress = spline.find_closest_progress(state(0), state(1));
    double cte_sum = 0.0;
    int step_count = 0;

    for (int step = 0; step < max_steps; step++) {
        if (progress >= total_len - 0.1) {
            result.completed = true;
            break;
        }

        // Generate path references with curvature-adaptive spacing
        // (matches mpcc_controller_node.cpp::get_spline_path_refs exactly)
        std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
        double s = progress;
        double lookahead_v = std::max(state(3), cfg.reference_velocity * 0.5);
        for (int k = 0; k <= cfg.horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);
            double rx, ry, ct, st;
            spline.get_path_reference(s, rx, ry, ct, st);
            refs[k].x = rx;  refs[k].y = ry;
            refs[k].cos_theta = ct;  refs[k].sin_theta = st;
            refs[k].curvature = spline.get_curvature(s);
            // Curvature-adaptive spacing (matching controller)
            double curv = std::abs(refs[k].curvature);
            double step_speed = lookahead_v * std::exp(-0.4 * curv);
            step_speed = std::max(step_speed, 0.10);
            s += step_speed * cfg.dt;
        }

        // Solve
        auto sol = solver.solve(state, refs, progress, total_len, {}, {});
        if (!sol.success) {
            std::printf("  [%s] Solver failed at step %d, progress=%.1f%%\n",
                leg_name.c_str(), step, 100.0 * progress / total_len);
            break;
        }

        double v_cmd = sol.v_cmd;
        double delta_cmd = sol.delta_cmd;

        // Decelerate near goal
        double remaining = total_len - progress;
        if (remaining < 0.5) {
            double decel_factor = remaining / 0.5;
            v_cmd *= decel_factor;
            if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
        }

        v_cmd = std::clamp(v_cmd, cfg.min_velocity, cfg.max_velocity);
        delta_cmd = std::clamp(delta_cmd, -cfg.max_steering, cfg.max_steering);

        // Apply speed through PD controller
        double actual_v = pd_ctrl.step(v_cmd, cfg.dt);

        // Apply controls to plant
        mpcc::VecU u;
        u(0) = (actual_v - state(3)) / cfg.dt;
        u(1) = (delta_cmd - state(4)) / cfg.dt;
        u(0) = std::clamp(u(0), -cfg.max_acceleration, cfg.max_acceleration);
        u(1) = std::clamp(u(1), -cfg.max_steering_rate, cfg.max_steering_rate);

        state = plant.rk4_step(state, u, cfg.dt);
        state(3) = std::clamp(state(3), cfg.min_velocity, cfg.max_velocity);
        state(4) = std::clamp(state(4), -cfg.max_steering, cfg.max_steering);

        // Update progress (monotonic)
        double new_progress = spline.find_closest_progress(state(0), state(1));
        if (new_progress > progress) progress = new_progress;

        // Compute CTE
        double cp = spline.find_closest_progress(state(0), state(1));
        double rx, ry;
        spline.get_position(cp, rx, ry);
        double cte = std::hypot(state(0) - rx, state(1) - ry);

        // Compute heading error
        double path_tangent = spline.get_tangent(cp);
        double heading_err = state(2) - path_tangent;
        while (heading_err > M_PI) heading_err -= 2*M_PI;
        while (heading_err < -M_PI) heading_err += 2*M_PI;

        if (cte > result.max_cte) {
            result.max_cte = cte;
        }
        result.max_heading_err = std::max(result.max_heading_err, std::abs(heading_err));
        cte_sum += cte;
        step_count++;

        // Track steering saturation
        if (std::abs(std::abs(delta_cmd) - cfg.max_steering) < 0.01)
            result.steering_saturated_steps++;

        // Convert to QLabs frame for output
        double x_ql, y_ql;
        acc::map_to_qlabs_2d(state(0), state(1), tp, x_ql, y_ql);

        // Store trace point
        TracePoint tp_out;
        tp_out.elapsed_s = elapsed_start + step * cfg.dt;
        tp_out.x_map = state(0);
        tp_out.y_map = state(1);
        tp_out.x_ql = x_ql;
        tp_out.y_ql = y_ql;
        tp_out.theta = state(2);
        tp_out.v_meas = state(3);
        tp_out.v_cmd = v_cmd;
        tp_out.delta_cmd = delta_cmd;
        tp_out.cte = cte;
        tp_out.heading_err = heading_err;
        tp_out.progress_pct = 100.0 * progress / total_len;
        tp_out.curvature = spline.get_curvature(cp);
        tp_out.leg_name = leg_name;
        result.trace.push_back(tp_out);
    }

    result.avg_cte = (step_count > 0) ? cte_sum / step_count : 0.0;
    result.final_state = state;

    if (progress >= total_len - 0.1) result.completed = true;

    return result;
}

// Traffic event: describes what happens at a given progress percentage
struct TrafficEvent {
    double trigger_progress_pct; // progress % at which event triggers
    enum Type { STOP_SIGN, RED_LIGHT, OBSTACLE_CREEP, GREEN_LIGHT } type;
    double duration_s;           // stop/creep duration (0 = instantaneous)
};

// Run one mission leg with traffic events (stops, obstacles, speed limits)
LegResult run_leg_with_events(
    const std::vector<double>& path_x_qlabs,
    const std::vector<double>& path_y_qlabs,
    const mpcc::VecX& initial_state,
    double elapsed_start,
    const std::string& leg_name,
    int max_steps,
    PDSpeedController& pd_ctrl,
    const std::vector<TrafficEvent>& events)
{
    LegResult result;
    acc::TransformParams tp;

    std::vector<double> map_x, map_y;
    acc::qlabs_path_to_map(path_x_qlabs, path_y_qlabs, tp, map_x, map_y);
    align_path_to_vehicle_heading(map_x, map_y,
        initial_state(0), initial_state(1), initial_state(2));

    acc::CubicSplinePath spline;
    spline.build(map_x, map_y, true);
    double total_len = spline.total_length();

    // Store blended path for visualization
    result.blended_path.name = leg_name;
    int n_vis = static_cast<int>(total_len / 0.005) + 1;
    result.blended_path.x_ql.resize(n_vis);
    result.blended_path.y_ql.resize(n_vis);
    for (int i = 0; i < n_vis; i++) {
        double s = total_len * i / (n_vis - 1);
        s = std::min(s, total_len - 0.001);
        double px, py;
        spline.get_position(s, px, py);
        acc::map_to_qlabs_2d(px, py, tp, result.blended_path.x_ql[i],
                             result.blended_path.y_ql[i]);
    }

    mpcc::ActiveSolver solver;
    auto cfg = make_deployment_config();
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
        ref.cos_theta = ct;
        ref.sin_theta = st;
        ref.curvature = spline.get_curvature(s);
        return ref;
    };

    mpcc::AckermannModel plant(cfg.wheelbase);
    mpcc::VecX state = initial_state;
    double progress = spline.find_closest_progress(state(0), state(1));
    double cte_sum = 0.0;
    int step_count = 0;

    // Event state
    size_t next_event = 0;
    double stop_until = -1.0;       // elapsed time when stop ends
    bool creep_mode = false;
    double creep_until = -1.0;
    int total_stops = 0;

    for (int step = 0; step < max_steps; step++) {
        double elapsed = elapsed_start + step * cfg.dt;

        if (progress >= total_len - 0.1) {
            result.completed = true;
            break;
        }

        double progress_pct = 100.0 * progress / total_len;

        // Check for traffic events
        while (next_event < events.size() &&
               progress_pct >= events[next_event].trigger_progress_pct) {
            auto& ev = events[next_event];
            if (ev.type == TrafficEvent::STOP_SIGN || ev.type == TrafficEvent::RED_LIGHT) {
                stop_until = elapsed + ev.duration_s;
                total_stops++;
            } else if (ev.type == TrafficEvent::OBSTACLE_CREEP) {
                creep_mode = true;
                creep_until = elapsed + ev.duration_s;
            } else if (ev.type == TrafficEvent::GREEN_LIGHT) {
                stop_until = -1.0;  // clear any pending stop
            }
            next_event++;
        }

        // If stopped (traffic stop / red light)
        if (elapsed < stop_until) {
            // Vehicle fully stopped — matching publish_stop() behavior
            pd_ctrl.step(0.0, cfg.dt);
            state(3) = std::max(0.0, state(3) - 2.0 * cfg.dt); // decelerate
            state(3) = std::max(0.0, state(3));

            TracePoint tp_out;
            tp_out.elapsed_s = elapsed;
            tp_out.x_map = state(0); tp_out.y_map = state(1);
            acc::map_to_qlabs_2d(state(0), state(1), tp, tp_out.x_ql, tp_out.y_ql);
            tp_out.theta = state(2);
            tp_out.v_meas = state(3); tp_out.v_cmd = 0.0;
            tp_out.delta_cmd = state(4);
            double cp2 = spline.find_closest_progress(state(0), state(1));
            double rx2, ry2; spline.get_position(cp2, rx2, ry2);
            tp_out.cte = std::hypot(state(0)-rx2, state(1)-ry2);
            double pt2 = spline.get_tangent(cp2);
            tp_out.heading_err = acc::normalize_angle(state(2) - pt2);
            tp_out.progress_pct = progress_pct;
            tp_out.leg_name = leg_name;
            result.trace.push_back(tp_out);
            result.max_cte = std::max(result.max_cte, tp_out.cte);
            cte_sum += tp_out.cte; step_count++;
            continue;
        }

        // Check if creep mode expired
        if (creep_mode && elapsed >= creep_until) {
            creep_mode = false;
        }

        // Generate path references with curvature-adaptive spacing
        // (matches mpcc_controller_node.cpp::get_spline_path_refs exactly)
        std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
        double s = progress;
        double lookahead_v = std::max(state(3), cfg.reference_velocity * 0.5);
        for (int k = 0; k <= cfg.horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);
            double rx, ry, ct, st;
            spline.get_path_reference(s, rx, ry, ct, st);
            refs[k].x = rx;  refs[k].y = ry;
            refs[k].cos_theta = ct;  refs[k].sin_theta = st;
            refs[k].curvature = spline.get_curvature(s);
            double curv = std::abs(refs[k].curvature);
            double step_speed = lookahead_v * std::exp(-0.4 * curv);
            step_speed = std::max(step_speed, 0.10);
            s += step_speed * cfg.dt;
        }

        auto sol = solver.solve(state, refs, progress, total_len, {}, {});
        if (!sol.success) break;

        double v_cmd = sol.v_cmd;
        double delta_cmd = sol.delta_cmd;

        // Decelerate near goal
        double remaining = total_len - progress;
        if (remaining < 0.5) {
            double decel_factor = remaining / 0.5;
            v_cmd *= decel_factor;
            if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
        }

        // Apply obstacle creep speed limit (matches controller line 1085)
        if (creep_mode) {
            v_cmd = std::min(v_cmd, 0.20);
        }

        v_cmd = std::clamp(v_cmd, cfg.min_velocity, cfg.max_velocity);
        delta_cmd = std::clamp(delta_cmd, -cfg.max_steering, cfg.max_steering);

        double actual_v = pd_ctrl.step(v_cmd, cfg.dt);

        mpcc::VecU u;
        u(0) = (actual_v - state(3)) / cfg.dt;
        u(1) = (delta_cmd - state(4)) / cfg.dt;
        u(0) = std::clamp(u(0), -cfg.max_acceleration, cfg.max_acceleration);
        u(1) = std::clamp(u(1), -cfg.max_steering_rate, cfg.max_steering_rate);

        state = plant.rk4_step(state, u, cfg.dt);
        state(3) = std::clamp(state(3), cfg.min_velocity, cfg.max_velocity);
        state(4) = std::clamp(state(4), -cfg.max_steering, cfg.max_steering);

        double new_progress = spline.find_closest_progress(state(0), state(1));
        if (new_progress > progress) progress = new_progress;

        double cp = spline.find_closest_progress(state(0), state(1));
        double rx, ry;
        spline.get_position(cp, rx, ry);
        double cte = std::hypot(state(0) - rx, state(1) - ry);
        double path_tangent = spline.get_tangent(cp);
        double heading_err = acc::normalize_angle(state(2) - path_tangent);

        result.max_cte = std::max(result.max_cte, cte);
        result.max_heading_err = std::max(result.max_heading_err, std::abs(heading_err));
        cte_sum += cte; step_count++;

        if (std::abs(std::abs(delta_cmd) - cfg.max_steering) < 0.01)
            result.steering_saturated_steps++;

        double x_ql, y_ql;
        acc::map_to_qlabs_2d(state(0), state(1), tp, x_ql, y_ql);

        TracePoint tp_out;
        tp_out.elapsed_s = elapsed;
        tp_out.x_map = state(0); tp_out.y_map = state(1);
        tp_out.x_ql = x_ql; tp_out.y_ql = y_ql;
        tp_out.theta = state(2);
        tp_out.v_meas = state(3); tp_out.v_cmd = v_cmd;
        tp_out.delta_cmd = delta_cmd;
        tp_out.cte = cte; tp_out.heading_err = heading_err;
        tp_out.progress_pct = 100.0 * progress / total_len;
        tp_out.curvature = spline.get_curvature(cp);
        tp_out.leg_name = leg_name;
        result.trace.push_back(tp_out);
    }

    result.avg_cte = (step_count > 0) ? cte_sum / step_count : 0.0;
    result.final_state = state;
    if (progress >= total_len - 0.1) result.completed = true;

    std::printf("    [%s] %d traffic stops, creep=%s, max_cte=%.3f\n",
        leg_name.c_str(), total_stops, creep_mode ? "active" : "off", result.max_cte);

    return result;
}

int main() {
    std::printf("=== Full Mission Closed-Loop Simulation ===\n\n");

    // Generate paths for all 3 legs
    acc::RoadGraph road_graph(0.001);

    struct LegDef {
        std::string name;
        double start_x, start_y;
        int max_steps;
    };
    std::vector<LegDef> legs = {
        {"hub_to_pickup", acc::HUB_X, acc::HUB_Y, 600},
        {"pickup_to_dropoff", acc::PICKUP_X, acc::PICKUP_Y, 600},
        {"dropoff_to_hub", acc::DROPOFF_X, acc::DROPOFF_Y, 800},
    };

    // Initial state: at hub, aligned with path, from stop
    acc::TransformParams tp;
    double hub_map_x, hub_map_y;
    acc::qlabs_to_map_2d(acc::HUB_X, acc::HUB_Y, tp, hub_map_x, hub_map_y);

    // Get initial heading from first leg's path
    auto route0 = road_graph.plan_path_for_mission_leg("hub_to_pickup",
        acc::HUB_X, acc::HUB_Y);
    if (!route0) {
        std::printf("ERROR: Failed to generate initial path\n");
        return 1;
    }

    std::vector<double> init_mx, init_my;
    acc::qlabs_path_to_map(route0->first, route0->second, tp, init_mx, init_my);
    acc::CubicSplinePath init_spline;
    init_spline.build(init_mx, init_my, true);
    double init_theta = init_spline.get_tangent(0.0);

    mpcc::VecX state;
    state << hub_map_x, hub_map_y, init_theta, 0.0, 0.0;

    PDSpeedController pd_ctrl;
    double elapsed = 0.0;
    std::vector<TracePoint> all_trace;
    std::vector<BlendedPath> blended_paths;
    double overall_max_cte = 0.0;

    for (auto& leg : legs) {
        std::printf("Running %s...\n", leg.name.c_str());

        auto route = road_graph.plan_path_for_mission_leg(leg.name,
            leg.start_x, leg.start_y);
        if (!route) {
            std::printf("  ERROR: Failed to plan %s\n", leg.name.c_str());
            return 1;
        }

        auto result = run_leg(route->first, route->second,
            state, elapsed, leg.name, leg.max_steps, pd_ctrl);

        std::printf("  %s: max_cte=%.3f avg_cte=%.3f steps=%zu prog=%.0f%% "
                    "heading_err_max=%.1fdeg steer_sat=%d completed=%s\n",
            leg.name.c_str(), result.max_cte, result.avg_cte,
            result.trace.size(),
            result.trace.empty() ? 0.0 : result.trace.back().progress_pct,
            result.max_heading_err * 180.0 / M_PI,
            result.steering_saturated_steps,
            result.completed ? "yes" : "NO");

        overall_max_cte = std::max(overall_max_cte, result.max_cte);
        blended_paths.push_back(result.blended_path);

        // Carry state to next leg
        state = result.final_state;
        if (!result.trace.empty())
            elapsed = result.trace.back().elapsed_s + 0.1;

        // Simulate stop at waypoint (1 second)
        pd_ctrl.actual_speed = 0.0;
        state(3) = 0.0;  // velocity = 0
        elapsed += 1.0;

        // Append trace
        all_trace.insert(all_trace.end(),
            result.trace.begin(), result.trace.end());
    }

    // Write blended paths for visualization
    for (auto& bp : blended_paths) {
        std::string bp_path = "blended_" + bp.name + ".csv";
        std::ofstream f(bp_path);
        f << "x_qlabs,y_qlabs\n";
        for (size_t i = 0; i < bp.x_ql.size(); i++) {
            f << std::fixed << std::setprecision(6) << bp.x_ql[i] << "," << bp.y_ql[i] << "\n";
        }
        std::printf("Blended path: %s (%zu points)\n", bp_path.c_str(), bp.x_ql.size());
    }

    // Write CSV matching MPCC log format (for generate_report.py compatibility)
    std::string csv_path = "full_mission_sim.csv";
    {
        std::ofstream f(csv_path);
        f << "elapsed_s,x,y,theta,v_meas,v_cmd,delta_cmd,progress_pct,"
             "solve_time_us,cross_track_err,heading_err,n_obstacles,"
             "motion_enabled,traffic_stop\n";
        for (auto& tp : all_trace) {
            f << std::fixed;
            f << std::setprecision(3) << tp.elapsed_s << ","
              << std::setprecision(4) << tp.x_map << "," << tp.y_map << ","
              << std::setprecision(4) << tp.theta << ","
              << std::setprecision(4) << tp.v_meas << "," << tp.v_cmd << ","
              << tp.delta_cmd << ","
              << std::setprecision(1) << tp.progress_pct << ","
              << "50,"  // solve_time_us (mock)
              << std::setprecision(4) << tp.cte << "," << tp.heading_err << ","
              << "0,1,0\n";
        }
    }
    std::printf("\nCSV written: %s (%zu data points)\n", csv_path.c_str(), all_trace.size());

    // Also write QLabs-frame trace for direct plotting
    std::string ql_csv_path = "full_mission_sim_qlabs.csv";
    {
        std::ofstream f(ql_csv_path);
        f << "elapsed_s,x_qlabs,y_qlabs,cte,v,delta,heading_err,leg\n";
        for (auto& tp : all_trace) {
            f << std::fixed << std::setprecision(3) << tp.elapsed_s << ","
              << std::setprecision(4) << tp.x_ql << "," << tp.y_ql << ","
              << tp.cte << "," << tp.v_meas << "," << tp.delta_cmd << ","
              << tp.heading_err << "," << tp.leg_name << "\n";
        }
    }

    // Summary
    double total_time = all_trace.empty() ? 0.0 :
        all_trace.back().elapsed_s - all_trace.front().elapsed_s;
    double total_cte_sum = 0.0;
    for (auto& tp : all_trace) total_cte_sum += tp.cte;
    double total_avg_cte = all_trace.empty() ? 0.0 : total_cte_sum / all_trace.size();
    double total_avg_speed = 0.0;
    for (auto& tp : all_trace) total_avg_speed += tp.v_meas;
    total_avg_speed = all_trace.empty() ? 0.0 : total_avg_speed / all_trace.size();

    std::printf("\n=== Mission Summary ===\n");
    std::printf("Duration: %.1fs\n", total_time);
    std::printf("Max CTE:  %.3fm\n", overall_max_cte);
    std::printf("Avg CTE:  %.3fm\n", total_avg_cte);
    std::printf("Avg speed: %.2f m/s\n", total_avg_speed);
    std::printf("Points:   %zu\n", all_trace.size());

    if (overall_max_cte > 0.30) {
        std::printf("\nFAIL: Max CTE %.3fm exceeds 0.30m threshold\n", overall_max_cte);
        return 1;
    }
    std::printf("\nPASS: Max CTE %.3fm within 0.30m threshold\n", overall_max_cte);

    // =====================================================================
    // Test 2: Full mission with traffic events (stop signs, red lights, obstacles)
    // =====================================================================
    std::printf("\n\n=== Full Mission with Traffic Events ===\n\n");

    // Realistic traffic events for the SDCS scenario:
    // Hub→Pickup: stop sign at ~30%, red light at ~60%
    // Pickup→Dropoff: obstacle (cone) at ~40%
    // Dropoff→Hub: stop sign at ~20%, red light at ~50%, obstacle at ~75%
    struct LegEvents {
        std::string name;
        double start_x, start_y;
        int max_steps;
        std::vector<TrafficEvent> events;
    };
    std::vector<LegEvents> traffic_legs = {
        {"hub_to_pickup", acc::HUB_X, acc::HUB_Y, 800, {
            {30.0, TrafficEvent::STOP_SIGN, 3.0},     // 3s stop at stop sign
            {60.0, TrafficEvent::RED_LIGHT, 5.0},      // 5s wait at red light
        }},
        {"pickup_to_dropoff", acc::PICKUP_X, acc::PICKUP_Y, 800, {
            {40.0, TrafficEvent::OBSTACLE_CREEP, 3.0},  // 3s creep past cone
        }},
        {"dropoff_to_hub", acc::DROPOFF_X, acc::DROPOFF_Y, 1200, {
            {20.0, TrafficEvent::STOP_SIGN, 3.0},      // stop sign
            {50.0, TrafficEvent::RED_LIGHT, 8.0},       // long red light
            {75.0, TrafficEvent::OBSTACLE_CREEP, 4.0},  // cone avoidance
        }},
    };

    // Reset to hub
    state << hub_map_x, hub_map_y, init_theta, 0.0, 0.0;
    pd_ctrl.reset();
    elapsed = 0.0;
    std::vector<TracePoint> traffic_trace;
    double traffic_max_cte = 0.0;

    for (auto& leg : traffic_legs) {
        std::printf("Running %s (with traffic)...\n", leg.name.c_str());

        auto route = road_graph.plan_path_for_mission_leg(leg.name,
            leg.start_x, leg.start_y);
        if (!route) {
            std::printf("  ERROR: Failed to plan %s\n", leg.name.c_str());
            return 1;
        }

        auto result = run_leg_with_events(route->first, route->second,
            state, elapsed, leg.name, leg.max_steps, pd_ctrl, leg.events);

        std::printf("  %s: max_cte=%.3f avg_cte=%.3f steps=%zu prog=%.0f%% "
                    "heading_err_max=%.1fdeg steer_sat=%d completed=%s\n",
            leg.name.c_str(), result.max_cte, result.avg_cte,
            result.trace.size(),
            result.trace.empty() ? 0.0 : result.trace.back().progress_pct,
            result.max_heading_err * 180.0 / M_PI,
            result.steering_saturated_steps,
            result.completed ? "yes" : "NO");

        traffic_max_cte = std::max(traffic_max_cte, result.max_cte);

        state = result.final_state;
        if (!result.trace.empty())
            elapsed = result.trace.back().elapsed_s + 0.1;
        pd_ctrl.actual_speed = 0.0;
        state(3) = 0.0;
        elapsed += 1.0;

        traffic_trace.insert(traffic_trace.end(),
            result.trace.begin(), result.trace.end());
    }

    // Write traffic CSV
    {
        std::ofstream f("full_mission_traffic.csv");
        f << "elapsed_s,x,y,theta,v_meas,v_cmd,delta_cmd,progress_pct,"
             "solve_time_us,cross_track_err,heading_err,n_obstacles,"
             "motion_enabled,traffic_stop\n";
        for (auto& tp : traffic_trace) {
            f << std::fixed
              << std::setprecision(3) << tp.elapsed_s << ","
              << std::setprecision(4) << tp.x_map << "," << tp.y_map << ","
              << tp.theta << "," << tp.v_meas << "," << tp.v_cmd << ","
              << tp.delta_cmd << ","
              << std::setprecision(1) << tp.progress_pct << ","
              << "50," << std::setprecision(4) << tp.cte << "," << tp.heading_err << ","
              << "0,1,0\n";
        }
    }

    double traffic_time = traffic_trace.empty() ? 0.0 :
        traffic_trace.back().elapsed_s - traffic_trace.front().elapsed_s;
    double traffic_cte_sum = 0.0;
    for (auto& tp : traffic_trace) traffic_cte_sum += tp.cte;
    double traffic_avg_cte = traffic_trace.empty() ? 0.0 : traffic_cte_sum / traffic_trace.size();

    std::printf("\n=== Traffic Mission Summary ===\n");
    std::printf("Duration: %.1fs\n", traffic_time);
    std::printf("Max CTE:  %.3fm\n", traffic_max_cte);
    std::printf("Avg CTE:  %.3fm\n", traffic_avg_cte);
    std::printf("Points:   %zu\n", traffic_trace.size());

    if (traffic_max_cte > 0.30) {
        std::printf("\nFAIL: Traffic test Max CTE %.3fm exceeds 0.30m\n", traffic_max_cte);
        return 1;
    }
    std::printf("\nPASS: Traffic test Max CTE %.3fm within 0.30m\n", traffic_max_cte);

    // =====================================================================
    // Test 3: Worst-case scenario (frequent stops + high speed resume)
    // =====================================================================
    std::printf("\n\n=== Stress Test: Frequent Stops ===\n\n");

    state << hub_map_x, hub_map_y, init_theta, 0.0, 0.0;
    pd_ctrl.reset();
    elapsed = 0.0;

    // Hub→Pickup with stops every 10% progress
    std::vector<TrafficEvent> stress_events;
    for (double p = 10.0; p <= 90.0; p += 10.0) {
        stress_events.push_back({p, TrafficEvent::STOP_SIGN, 2.0});
    }

    auto stress_route = road_graph.plan_path_for_mission_leg("hub_to_pickup",
        acc::HUB_X, acc::HUB_Y);
    auto stress_result = run_leg_with_events(stress_route->first, stress_route->second,
        state, elapsed, "stress_hub_to_pickup", 1500, pd_ctrl, stress_events);

    std::printf("  Stress test: max_cte=%.3f avg_cte=%.3f steps=%zu prog=%.0f%% completed=%s\n",
        stress_result.max_cte, stress_result.avg_cte,
        stress_result.trace.size(),
        stress_result.trace.empty() ? 0.0 : stress_result.trace.back().progress_pct,
        stress_result.completed ? "yes" : "NO");

    if (stress_result.max_cte > 0.30) {
        std::printf("FAIL: Stress test Max CTE %.3fm exceeds 0.30m\n", stress_result.max_cte);
        return 1;
    }
    if (!stress_result.completed) {
        std::printf("FAIL: Stress test did not complete\n");
        return 1;
    }
    std::printf("PASS: Stress test Max CTE %.3fm within 0.30m\n", stress_result.max_cte);

    // =====================================================================
    // Test 4: Sim-to-Real Gap Analysis
    // Run with boundary_weight=8.0 and velocity estimation lag to prove
    // these cause the CTE degradation seen in deployment (0.067→0.203m)
    // =====================================================================
    std::printf("\n\n=== Sim-to-Real Gap Analysis ===\n");

    // --- Test 4a: With boundary_weight=8.0 only ---
    {
        std::printf("\n[Gap Test 4a] boundary_weight=8.0 (deployment had this, ref has 0)\n");
        state << hub_map_x, hub_map_y, init_theta, 0.0, 0.0;
        pd_ctrl.reset();
        elapsed = 0.0;
        double gap_max_cte = 0.0;

        // Temporarily modify make_deployment_config to use boundary_weight=8.0
        // We do this by running each leg with a modified config
        for (auto& leg : legs) {
            auto route = road_graph.plan_path_for_mission_leg(leg.name,
                leg.start_x, leg.start_y);
            if (!route) continue;

            // Run with boundaries using a modified run_leg approach
            // We construct a custom run that overrides boundary_weight
            acc::TransformParams tp_gap;
            std::vector<double> gmx, gmy;
            acc::qlabs_path_to_map(route->first, route->second, tp_gap, gmx, gmy);
            align_path_to_vehicle_heading(gmx, gmy,
                state(0), state(1), state(2));

            acc::CubicSplinePath gspline;
            gspline.build(gmx, gmy, true);
            double glen = gspline.total_length();

            mpcc::ActiveSolver gsolver;
            auto gcfg = make_deployment_config();
            gcfg.boundary_weight = 8.0;  // <-- DEPLOY HAD THIS
            gsolver.init(gcfg);

            gsolver.path_lookup.lookup = [&gspline, glen](
                double px, double py, double s_min, double* s_out) -> mpcc::PathRef
            {
                double s = gspline.find_closest_progress_from(px, py, s_min);
                s = std::clamp(s, 0.0, glen - 0.001);
                if (s_out) *s_out = s;
                mpcc::PathRef ref;
                double ct, st;
                gspline.get_path_reference(s, ref.x, ref.y, ct, st);
                ref.cos_theta = ct; ref.sin_theta = st;
                ref.curvature = gspline.get_curvature(s);
                return ref;
            };

            mpcc::AckermannModel gplant(gcfg.wheelbase);
            double gprogress = gspline.find_closest_progress(state(0), state(1));

            // Generate path-tangent-based boundary constraints (like controller node fallback)
            double half_width = gcfg.boundary_default_width;

            for (int step = 0; step < leg.max_steps; step++) {
                if (gprogress >= glen - 0.1) break;

                std::vector<mpcc::PathRef> grefs(gcfg.horizon + 1);
                double gs = gprogress;
                double gv = std::max(state(3), gcfg.reference_velocity * 0.5);
                for (int k = 0; k <= gcfg.horizon; k++) {
                    gs = std::clamp(gs, 0.0, glen - 0.001);
                    double rx, ry, ct, st;
                    gspline.get_path_reference(gs, rx, ry, ct, st);
                    grefs[k] = {rx, ry, ct, st, gspline.get_curvature(gs)};
                    double gcurv = std::abs(gspline.get_curvature(gs));
                    gs += std::max(gv * std::exp(-0.4 * gcurv), 0.10) * gcfg.dt;
                }

                // Build boundary constraints (matching controller node lines 821-837)
                std::vector<mpcc::BoundaryConstraint> gbounds(gcfg.horizon + 1);
                gs = gprogress;
                for (int k = 0; k <= gcfg.horizon; k++) {
                    gs = std::clamp(gs, 0.0, glen - 0.001);
                    double cx, cy, ct, st;
                    gspline.get_path_reference(gs, cx, cy, ct, st);
                    double ta = std::atan2(st, ct);
                    double nx = -std::sin(ta);
                    double ny =  std::cos(ta);
                    double center_proj = nx * cx + ny * cy;
                    gbounds[k].nx = nx;
                    gbounds[k].ny = ny;
                    gbounds[k].b_left = center_proj + half_width;
                    gbounds[k].b_right = -(center_proj - half_width);
                    double curv = gspline.get_curvature(gs);
                    gs += std::max(gv * std::exp(-0.4 * std::abs(curv)), 0.10) * gcfg.dt;
                }

                auto sol = gsolver.solve(state, grefs, gprogress, glen, {}, gbounds);
                if (!sol.success) break;

                double v_cmd = sol.v_cmd;
                double delta_cmd = sol.delta_cmd;
                double remaining = glen - gprogress;
                if (remaining < 0.5) {
                    double df = remaining / 0.5;
                    v_cmd *= df;
                    if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
                }
                v_cmd = std::clamp(v_cmd, gcfg.min_velocity, gcfg.max_velocity);
                delta_cmd = std::clamp(delta_cmd, -gcfg.max_steering, gcfg.max_steering);

                double actual_v = pd_ctrl.step(v_cmd, gcfg.dt);
                mpcc::VecU gu;
                gu(0) = (actual_v - state(3)) / gcfg.dt;
                gu(1) = (delta_cmd - state(4)) / gcfg.dt;
                gu(0) = std::clamp(gu(0), -gcfg.max_acceleration, gcfg.max_acceleration);
                gu(1) = std::clamp(gu(1), -gcfg.max_steering_rate, gcfg.max_steering_rate);

                state = gplant.rk4_step(state, gu, gcfg.dt);
                state(3) = std::clamp(state(3), gcfg.min_velocity, gcfg.max_velocity);
                state(4) = std::clamp(state(4), -gcfg.max_steering, gcfg.max_steering);

                double np = gspline.find_closest_progress(state(0), state(1));
                if (np > gprogress) gprogress = np;

                double cp = gspline.find_closest_progress(state(0), state(1));
                double rx, ry;
                gspline.get_position(cp, rx, ry);
                double cte = std::hypot(state(0) - rx, state(1) - ry);
                gap_max_cte = std::max(gap_max_cte, cte);
            }

            pd_ctrl.actual_speed = 0.0;
            state(3) = 0.0;
            elapsed += 1.0;
        }
        std::printf("  boundary_weight=8.0: max_cte=%.3fm (vs 0.067m without boundaries)\n",
            gap_max_cte);
        if (gap_max_cte > overall_max_cte + 0.01) {
            std::printf("  CONFIRMED: Boundary constraints degrade CTE by +%.3fm\n",
                gap_max_cte - overall_max_cte);
        }
    }

    // --- Test 4b: With velocity estimation lag only ---
    {
        std::printf("\n[Gap Test 4b] Velocity estimation lag (EKF ~100ms delay)\n");
        state << hub_map_x, hub_map_y, init_theta, 0.0, 0.0;
        pd_ctrl.reset();
        elapsed = 0.0;
        double gap_max_cte = 0.0;

        // Simple velocity lag model: solver sees velocity from 1 step ago
        double lagged_v = 0.0;

        for (auto& leg : legs) {
            auto route = road_graph.plan_path_for_mission_leg(leg.name,
                leg.start_x, leg.start_y);
            if (!route) continue;

            acc::TransformParams tp_gap;
            std::vector<double> gmx, gmy;
            acc::qlabs_path_to_map(route->first, route->second, tp_gap, gmx, gmy);
            align_path_to_vehicle_heading(gmx, gmy,
                state(0), state(1), state(2));

            acc::CubicSplinePath gspline;
            gspline.build(gmx, gmy, true);
            double glen = gspline.total_length();

            mpcc::ActiveSolver gsolver;
            auto gcfg = make_deployment_config();
            gsolver.init(gcfg);

            gsolver.path_lookup.lookup = [&gspline, glen](
                double px, double py, double s_min, double* s_out) -> mpcc::PathRef
            {
                double s = gspline.find_closest_progress_from(px, py, s_min);
                s = std::clamp(s, 0.0, glen - 0.001);
                if (s_out) *s_out = s;
                mpcc::PathRef ref;
                double ct, st;
                gspline.get_path_reference(s, ref.x, ref.y, ct, st);
                ref.cos_theta = ct; ref.sin_theta = st;
                ref.curvature = gspline.get_curvature(s);
                return ref;
            };

            mpcc::AckermannModel gplant(gcfg.wheelbase);
            double gprogress = gspline.find_closest_progress(state(0), state(1));

            for (int step = 0; step < leg.max_steps; step++) {
                if (gprogress >= glen - 0.1) break;

                std::vector<mpcc::PathRef> grefs(gcfg.horizon + 1);
                double gs = gprogress;
                double gv = std::max(state(3), gcfg.reference_velocity * 0.5);
                for (int k = 0; k <= gcfg.horizon; k++) {
                    gs = std::clamp(gs, 0.0, glen - 0.001);
                    double rx, ry, ct, st;
                    gspline.get_path_reference(gs, rx, ry, ct, st);
                    grefs[k] = {rx, ry, ct, st, gspline.get_curvature(gs)};
                    double gcurv = std::abs(gspline.get_curvature(gs));
                    gs += std::max(gv * std::exp(-0.4 * gcurv), 0.10) * gcfg.dt;
                }

                // Feed LAGGED velocity to solver (simulates EKF ~100ms delay)
                mpcc::VecX solver_state = state;
                solver_state(3) = lagged_v;  // <-- delayed velocity

                auto sol = gsolver.solve(solver_state, grefs, gprogress, glen, {}, {});
                if (!sol.success) break;

                double v_cmd = sol.v_cmd;
                double delta_cmd = sol.delta_cmd;
                double remaining = glen - gprogress;
                if (remaining < 0.5) {
                    double df = remaining / 0.5;
                    v_cmd *= df;
                    if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
                }
                v_cmd = std::clamp(v_cmd, gcfg.min_velocity, gcfg.max_velocity);
                delta_cmd = std::clamp(delta_cmd, -gcfg.max_steering, gcfg.max_steering);

                double actual_v = pd_ctrl.step(v_cmd, gcfg.dt);
                mpcc::VecU gu;
                gu(0) = (actual_v - state(3)) / gcfg.dt;
                gu(1) = (delta_cmd - state(4)) / gcfg.dt;
                gu(0) = std::clamp(gu(0), -gcfg.max_acceleration, gcfg.max_acceleration);
                gu(1) = std::clamp(gu(1), -gcfg.max_steering_rate, gcfg.max_steering_rate);

                // Update lagged velocity (1 step behind)
                lagged_v = state(3);

                state = gplant.rk4_step(state, gu, gcfg.dt);
                state(3) = std::clamp(state(3), gcfg.min_velocity, gcfg.max_velocity);
                state(4) = std::clamp(state(4), -gcfg.max_steering, gcfg.max_steering);

                double np = gspline.find_closest_progress(state(0), state(1));
                if (np > gprogress) gprogress = np;

                double cp = gspline.find_closest_progress(state(0), state(1));
                double rx, ry;
                gspline.get_position(cp, rx, ry);
                double cte = std::hypot(state(0) - rx, state(1) - ry);
                gap_max_cte = std::max(gap_max_cte, cte);
            }

            lagged_v = 0.0;
            pd_ctrl.actual_speed = 0.0;
            state(3) = 0.0;
            elapsed += 1.0;
        }
        std::printf("  velocity_lag=100ms: max_cte=%.3fm (vs 0.067m without lag)\n",
            gap_max_cte);
        if (gap_max_cte > overall_max_cte + 0.01) {
            std::printf("  CONFIRMED: Velocity lag degrades CTE by +%.3fm\n",
                gap_max_cte - overall_max_cte);
        }
    }

    // --- Test 4c: With BOTH boundary + velocity lag (worst case) ---
    {
        std::printf("\n[Gap Test 4c] boundary_weight=8.0 + velocity_lag=100ms (full deployment model)\n");
        state << hub_map_x, hub_map_y, init_theta, 0.0, 0.0;
        pd_ctrl.reset();
        elapsed = 0.0;
        double gap_max_cte = 0.0;
        double lagged_v = 0.0;

        for (auto& leg : legs) {
            auto route = road_graph.plan_path_for_mission_leg(leg.name,
                leg.start_x, leg.start_y);
            if (!route) continue;

            acc::TransformParams tp_gap;
            std::vector<double> gmx, gmy;
            acc::qlabs_path_to_map(route->first, route->second, tp_gap, gmx, gmy);
            align_path_to_vehicle_heading(gmx, gmy,
                state(0), state(1), state(2));

            acc::CubicSplinePath gspline;
            gspline.build(gmx, gmy, true);
            double glen = gspline.total_length();

            mpcc::ActiveSolver gsolver;
            auto gcfg = make_deployment_config();
            gcfg.boundary_weight = 8.0;  // <-- DEPLOY HAD THIS
            gsolver.init(gcfg);

            gsolver.path_lookup.lookup = [&gspline, glen](
                double px, double py, double s_min, double* s_out) -> mpcc::PathRef
            {
                double s = gspline.find_closest_progress_from(px, py, s_min);
                s = std::clamp(s, 0.0, glen - 0.001);
                if (s_out) *s_out = s;
                mpcc::PathRef ref;
                double ct, st;
                gspline.get_path_reference(s, ref.x, ref.y, ct, st);
                ref.cos_theta = ct; ref.sin_theta = st;
                ref.curvature = gspline.get_curvature(s);
                return ref;
            };

            mpcc::AckermannModel gplant(gcfg.wheelbase);
            double gprogress = gspline.find_closest_progress(state(0), state(1));
            double half_width = gcfg.boundary_default_width;

            for (int step = 0; step < leg.max_steps; step++) {
                if (gprogress >= glen - 0.1) break;

                std::vector<mpcc::PathRef> grefs(gcfg.horizon + 1);
                double gs = gprogress;
                double gv = std::max(state(3), gcfg.reference_velocity * 0.5);
                for (int k = 0; k <= gcfg.horizon; k++) {
                    gs = std::clamp(gs, 0.0, glen - 0.001);
                    double rx, ry, ct, st;
                    gspline.get_path_reference(gs, rx, ry, ct, st);
                    grefs[k] = {rx, ry, ct, st, gspline.get_curvature(gs)};
                    double gcurv = std::abs(gspline.get_curvature(gs));
                    gs += std::max(gv * std::exp(-0.4 * gcurv), 0.10) * gcfg.dt;
                }

                std::vector<mpcc::BoundaryConstraint> gbounds(gcfg.horizon + 1);
                gs = gprogress;
                for (int k = 0; k <= gcfg.horizon; k++) {
                    gs = std::clamp(gs, 0.0, glen - 0.001);
                    double cx, cy, ct, st;
                    gspline.get_path_reference(gs, cx, cy, ct, st);
                    double ta = std::atan2(st, ct);
                    double nx = -std::sin(ta);
                    double ny =  std::cos(ta);
                    double center_proj = nx * cx + ny * cy;
                    gbounds[k].nx = nx;
                    gbounds[k].ny = ny;
                    gbounds[k].b_left = center_proj + half_width;
                    gbounds[k].b_right = -(center_proj - half_width);
                    double curv = gspline.get_curvature(gs);
                    gs += std::max(gv * std::exp(-0.4 * std::abs(curv)), 0.10) * gcfg.dt;
                }

                mpcc::VecX solver_state = state;
                solver_state(3) = lagged_v;

                auto sol = gsolver.solve(solver_state, grefs, gprogress, glen, {}, gbounds);
                if (!sol.success) break;

                double v_cmd = sol.v_cmd;
                double delta_cmd = sol.delta_cmd;
                double remaining = glen - gprogress;
                if (remaining < 0.5) {
                    double df = remaining / 0.5;
                    v_cmd *= df;
                    if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
                }
                v_cmd = std::clamp(v_cmd, gcfg.min_velocity, gcfg.max_velocity);
                delta_cmd = std::clamp(delta_cmd, -gcfg.max_steering, gcfg.max_steering);

                double actual_v = pd_ctrl.step(v_cmd, gcfg.dt);
                mpcc::VecU gu;
                gu(0) = (actual_v - state(3)) / gcfg.dt;
                gu(1) = (delta_cmd - state(4)) / gcfg.dt;
                gu(0) = std::clamp(gu(0), -gcfg.max_acceleration, gcfg.max_acceleration);
                gu(1) = std::clamp(gu(1), -gcfg.max_steering_rate, gcfg.max_steering_rate);

                lagged_v = state(3);

                state = gplant.rk4_step(state, gu, gcfg.dt);
                state(3) = std::clamp(state(3), gcfg.min_velocity, gcfg.max_velocity);
                state(4) = std::clamp(state(4), -gcfg.max_steering, gcfg.max_steering);

                double np = gspline.find_closest_progress(state(0), state(1));
                if (np > gprogress) gprogress = np;

                double cp = gspline.find_closest_progress(state(0), state(1));
                double rx, ry;
                gspline.get_position(cp, rx, ry);
                double cte = std::hypot(state(0) - rx, state(1) - ry);
                gap_max_cte = std::max(gap_max_cte, cte);
            }

            lagged_v = 0.0;
            pd_ctrl.actual_speed = 0.0;
            state(3) = 0.0;
            elapsed += 1.0;
        }
        std::printf("  boundary+lag combined: max_cte=%.3fm (deployment saw 0.203m)\n",
            gap_max_cte);
        if (gap_max_cte > overall_max_cte + 0.01) {
            std::printf("  CONFIRMED: Combined gap degrades CTE by +%.3fm\n",
                gap_max_cte - overall_max_cte);
        }
    }

    // --- Test 4d: Position/heading noise (Cartographer SLAM jitter) ---
    {
        std::printf("\n[Gap Test 4d] Position noise σ=10mm + heading noise σ=2° (SLAM jitter)\n");
        state << hub_map_x, hub_map_y, init_theta, 0.0, 0.0;
        pd_ctrl.reset();
        elapsed = 0.0;
        double gap_max_cte = 0.0;
        double gap_cte_sum = 0.0;
        int gap_steps = 0;

        // Simple LCG random for reproducibility
        uint32_t rng = 42;
        auto next_rng = [&rng]() -> double {
            rng = rng * 1103515245 + 12345;
            return ((rng >> 16) & 0x7fff) / 32768.0;
        };
        auto gauss_rng = [&next_rng]() -> double {
            // Box-Muller
            double u1 = next_rng() * 0.9998 + 0.0001;
            double u2 = next_rng();
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        };

        for (auto& leg : legs) {
            auto route = road_graph.plan_path_for_mission_leg(leg.name,
                leg.start_x, leg.start_y);
            if (!route) continue;

            acc::TransformParams tp_gap;
            std::vector<double> gmx, gmy;
            acc::qlabs_path_to_map(route->first, route->second, tp_gap, gmx, gmy);
            align_path_to_vehicle_heading(gmx, gmy,
                state(0), state(1), state(2));

            acc::CubicSplinePath gspline;
            gspline.build(gmx, gmy, true);
            double glen = gspline.total_length();

            mpcc::ActiveSolver gsolver;
            auto gcfg = make_deployment_config();
            gsolver.init(gcfg);

            gsolver.path_lookup.lookup = [&gspline, glen](
                double px, double py, double s_min, double* s_out) -> mpcc::PathRef
            {
                double s = gspline.find_closest_progress_from(px, py, s_min);
                s = std::clamp(s, 0.0, glen - 0.001);
                if (s_out) *s_out = s;
                mpcc::PathRef ref;
                double ct, st;
                gspline.get_path_reference(s, ref.x, ref.y, ct, st);
                ref.cos_theta = ct; ref.sin_theta = st;
                ref.curvature = gspline.get_curvature(s);
                return ref;
            };

            mpcc::AckermannModel gplant(gcfg.wheelbase);
            double gprogress = gspline.find_closest_progress(state(0), state(1));

            for (int step = 0; step < leg.max_steps; step++) {
                if (gprogress >= glen - 0.1) break;

                // Add SLAM noise to state fed to solver
                mpcc::VecX noisy_state = state;
                noisy_state(0) += gauss_rng() * 0.010;  // 10mm position noise
                noisy_state(1) += gauss_rng() * 0.010;
                noisy_state(2) += gauss_rng() * (2.0 * M_PI / 180.0);  // 2° heading noise

                std::vector<mpcc::PathRef> grefs(gcfg.horizon + 1);
                double gs = gprogress;
                double gv = std::max(noisy_state(3), gcfg.reference_velocity * 0.5);
                for (int k = 0; k <= gcfg.horizon; k++) {
                    gs = std::clamp(gs, 0.0, glen - 0.001);
                    double rx, ry, ct, st;
                    gspline.get_path_reference(gs, rx, ry, ct, st);
                    grefs[k] = {rx, ry, ct, st, gspline.get_curvature(gs)};
                    double gcurv = std::abs(gspline.get_curvature(gs));
                    gs += std::max(gv * std::exp(-0.4 * gcurv), 0.10) * gcfg.dt;
                }

                auto sol = gsolver.solve(noisy_state, grefs, gprogress, glen, {}, {});
                if (!sol.success) break;

                double v_cmd = sol.v_cmd;
                double delta_cmd = sol.delta_cmd;
                double remaining = glen - gprogress;
                if (remaining < 0.5) {
                    double df = remaining / 0.5;
                    v_cmd *= df;
                    if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
                }
                v_cmd = std::clamp(v_cmd, gcfg.min_velocity, gcfg.max_velocity);
                delta_cmd = std::clamp(delta_cmd, -gcfg.max_steering, gcfg.max_steering);

                double actual_v = pd_ctrl.step(v_cmd, gcfg.dt);
                mpcc::VecU gu;
                gu(0) = (actual_v - state(3)) / gcfg.dt;
                gu(1) = (delta_cmd - state(4)) / gcfg.dt;
                gu(0) = std::clamp(gu(0), -gcfg.max_acceleration, gcfg.max_acceleration);
                gu(1) = std::clamp(gu(1), -gcfg.max_steering_rate, gcfg.max_steering_rate);

                state = gplant.rk4_step(state, gu, gcfg.dt);
                state(3) = std::clamp(state(3), gcfg.min_velocity, gcfg.max_velocity);
                state(4) = std::clamp(state(4), -gcfg.max_steering, gcfg.max_steering);

                double np = gspline.find_closest_progress(state(0), state(1));
                if (np > gprogress) gprogress = np;

                double cp = gspline.find_closest_progress(state(0), state(1));
                double rx, ry;
                gspline.get_position(cp, rx, ry);
                double cte = std::hypot(state(0) - rx, state(1) - ry);
                gap_max_cte = std::max(gap_max_cte, cte);
                gap_cte_sum += cte;
                gap_steps++;
            }

            pd_ctrl.actual_speed = 0.0;
            state(3) = 0.0;
            elapsed += 1.0;
        }
        double gap_avg = gap_steps > 0 ? gap_cte_sum / gap_steps : 0.0;
        std::printf("  noise σ_pos=10mm σ_θ=2°: max_cte=%.3fm avg_cte=%.3fm\n",
            gap_max_cte, gap_avg);
        if (gap_max_cte > overall_max_cte + 0.01) {
            std::printf("  CONFIRMED: SLAM noise degrades CTE by +%.3fm\n",
                gap_max_cte - overall_max_cte);
        }
    }

    // --- Test 4e: All factors combined (boundary + velocity lag + SLAM noise) ---
    {
        std::printf("\n[Gap Test 4e] ALL: boundary=8.0 + vel_lag + noise σ=10mm/2°\n");
        state << hub_map_x, hub_map_y, init_theta, 0.0, 0.0;
        pd_ctrl.reset();
        elapsed = 0.0;
        double gap_max_cte = 0.0;
        double gap_cte_sum = 0.0;
        int gap_steps = 0;
        double lagged_v = 0.0;

        uint32_t rng = 42;
        auto next_rng = [&rng]() -> double {
            rng = rng * 1103515245 + 12345;
            return ((rng >> 16) & 0x7fff) / 32768.0;
        };
        auto gauss_rng = [&next_rng]() -> double {
            double u1 = next_rng() * 0.9998 + 0.0001;
            double u2 = next_rng();
            return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * M_PI * u2);
        };

        for (auto& leg : legs) {
            auto route = road_graph.plan_path_for_mission_leg(leg.name,
                leg.start_x, leg.start_y);
            if (!route) continue;

            acc::TransformParams tp_gap;
            std::vector<double> gmx, gmy;
            acc::qlabs_path_to_map(route->first, route->second, tp_gap, gmx, gmy);
            align_path_to_vehicle_heading(gmx, gmy,
                state(0), state(1), state(2));

            acc::CubicSplinePath gspline;
            gspline.build(gmx, gmy, true);
            double glen = gspline.total_length();

            mpcc::ActiveSolver gsolver;
            auto gcfg = make_deployment_config();
            gcfg.boundary_weight = 8.0;
            gsolver.init(gcfg);

            gsolver.path_lookup.lookup = [&gspline, glen](
                double px, double py, double s_min, double* s_out) -> mpcc::PathRef
            {
                double s = gspline.find_closest_progress_from(px, py, s_min);
                s = std::clamp(s, 0.0, glen - 0.001);
                if (s_out) *s_out = s;
                mpcc::PathRef ref;
                double ct, st;
                gspline.get_path_reference(s, ref.x, ref.y, ct, st);
                ref.cos_theta = ct; ref.sin_theta = st;
                ref.curvature = gspline.get_curvature(s);
                return ref;
            };

            mpcc::AckermannModel gplant(gcfg.wheelbase);
            double gprogress = gspline.find_closest_progress(state(0), state(1));
            double half_width = gcfg.boundary_default_width;

            for (int step = 0; step < leg.max_steps; step++) {
                if (gprogress >= glen - 0.1) break;

                mpcc::VecX noisy_state = state;
                noisy_state(0) += gauss_rng() * 0.010;
                noisy_state(1) += gauss_rng() * 0.010;
                noisy_state(2) += gauss_rng() * (2.0 * M_PI / 180.0);
                noisy_state(3) = lagged_v;

                std::vector<mpcc::PathRef> grefs(gcfg.horizon + 1);
                double gs = gprogress;
                double gv = std::max(noisy_state(3), gcfg.reference_velocity * 0.5);
                for (int k = 0; k <= gcfg.horizon; k++) {
                    gs = std::clamp(gs, 0.0, glen - 0.001);
                    double rx, ry, ct, st;
                    gspline.get_path_reference(gs, rx, ry, ct, st);
                    grefs[k] = {rx, ry, ct, st, gspline.get_curvature(gs)};
                    double gcurv = std::abs(gspline.get_curvature(gs));
                    gs += std::max(gv * std::exp(-0.4 * gcurv), 0.10) * gcfg.dt;
                }

                std::vector<mpcc::BoundaryConstraint> gbounds(gcfg.horizon + 1);
                gs = gprogress;
                for (int k = 0; k <= gcfg.horizon; k++) {
                    gs = std::clamp(gs, 0.0, glen - 0.001);
                    double cx, cy, ct, st;
                    gspline.get_path_reference(gs, cx, cy, ct, st);
                    double ta = std::atan2(st, ct);
                    double nx = -std::sin(ta);
                    double ny =  std::cos(ta);
                    double center_proj = nx * cx + ny * cy;
                    gbounds[k].nx = nx;
                    gbounds[k].ny = ny;
                    gbounds[k].b_left = center_proj + half_width;
                    gbounds[k].b_right = -(center_proj - half_width);
                    double curv = gspline.get_curvature(gs);
                    gs += std::max(gv * std::exp(-0.4 * std::abs(curv)), 0.10) * gcfg.dt;
                }

                auto sol = gsolver.solve(noisy_state, grefs, gprogress, glen, {}, gbounds);
                if (!sol.success) break;

                double v_cmd = sol.v_cmd;
                double delta_cmd = sol.delta_cmd;
                double remaining = glen - gprogress;
                if (remaining < 0.5) {
                    double df = remaining / 0.5;
                    v_cmd *= df;
                    if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
                }
                v_cmd = std::clamp(v_cmd, gcfg.min_velocity, gcfg.max_velocity);
                delta_cmd = std::clamp(delta_cmd, -gcfg.max_steering, gcfg.max_steering);

                double actual_v = pd_ctrl.step(v_cmd, gcfg.dt);
                mpcc::VecU gu;
                gu(0) = (actual_v - state(3)) / gcfg.dt;
                gu(1) = (delta_cmd - state(4)) / gcfg.dt;
                gu(0) = std::clamp(gu(0), -gcfg.max_acceleration, gcfg.max_acceleration);
                gu(1) = std::clamp(gu(1), -gcfg.max_steering_rate, gcfg.max_steering_rate);

                lagged_v = state(3);

                state = gplant.rk4_step(state, gu, gcfg.dt);
                state(3) = std::clamp(state(3), gcfg.min_velocity, gcfg.max_velocity);
                state(4) = std::clamp(state(4), -gcfg.max_steering, gcfg.max_steering);

                double np = gspline.find_closest_progress(state(0), state(1));
                if (np > gprogress) gprogress = np;

                double cp = gspline.find_closest_progress(state(0), state(1));
                double rx, ry;
                gspline.get_position(cp, rx, ry);
                double cte = std::hypot(state(0) - rx, state(1) - ry);
                gap_max_cte = std::max(gap_max_cte, cte);
                gap_cte_sum += cte;
                gap_steps++;
            }

            lagged_v = 0.0;
            pd_ctrl.actual_speed = 0.0;
            state(3) = 0.0;
            elapsed += 1.0;
        }
        double gap_avg = gap_steps > 0 ? gap_cte_sum / gap_steps : 0.0;
        std::printf("  ALL combined: max_cte=%.3fm avg_cte=%.3fm (deployment saw max=0.203 avg=0.090)\n",
            gap_max_cte, gap_avg);
        if (gap_max_cte > overall_max_cte + 0.01) {
            std::printf("  CONFIRMED: Combined factors degrade CTE by +%.3fm\n",
                gap_max_cte - overall_max_cte);
        }
    }

    std::printf("\n=== Gap analysis complete ===\n");
    std::printf("Fix applied: boundary_weight=0, use_state_estimator=false\n");

    std::printf("\n=== ALL TESTS PASSED ===\n");
    return 0;
}
