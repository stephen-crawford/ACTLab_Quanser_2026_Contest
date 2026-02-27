/**
 * Diagnostic: Reproduce and fix real-deployment swerving.
 *
 * Root cause analysis (Feb 26, 2026):
 * In simulation without servo delay, ALL parameter configs show excellent CTE
 * (~0.02m). The swerving in QLabs must come from deployment-specific factors:
 *
 * 1. heading_weight=2.0 (NOT in reference) — creates oscillation on curves by
 *    fighting the contour cost. The heading cost penalizes future heading deviation
 *    from path tangent, but on curves the optimal steering naturally creates heading
 *    error that the heading cost then tries to correct → oscillation.
 *
 * 2. Servo delay (~50ms) — real QCar2 servo has mechanical delay. The solver assumes
 *    instant steering but actual execution is delayed, creating systematic overshoot.
 *
 * 3. TF latency — Cartographer publishes at ~10Hz with processing delay. The solver
 *    uses stale position data, creating ~0.05m position errors at 0.45 m/s.
 *
 * This test models ALL THREE factors and sweeps heading_weight to find optimal value.
 *
 * Build:
 *   cd /home/stephen/quanser-acc/cpp/test_build
 *   g++ -std=c++17 -O2 -I.. -I/usr/include/eigen3 \
 *       -o diagnose_swerving diagnose_swerving.cpp ../road_graph.cpp
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <string>
#include <chrono>
#include "coordinate_transform.h"
#include "cubic_spline_path.h"
#include "road_graph.h"
#include "mpcc_solver_interface.h"

struct PDSpeedController {
    double kp = 20.0, kd = 0.1, dt_inner = 0.015;
    double pwm_max = 0.3, km = 0.0047, battery_voltage = 7.2;
    double motor_speed_cmd = 0.0, prior_speed_error = 0.0, actual_speed = 0.0;

    double step(double desired_speed, double dt_outer) {
        int n = static_cast<int>(dt_outer / dt_inner);
        for (int i = 0; i < n; i++) {
            double e = desired_speed - actual_speed;
            motor_speed_cmd += (e * kp + (e - prior_speed_error) / dt_inner * kd)
                               * km / battery_voltage;
            motor_speed_cmd = std::clamp(motor_speed_cmd, -pwm_max, pwm_max);
            if (motor_speed_cmd < 0.01 && motor_speed_cmd >= 0 && desired_speed > 0)
                motor_speed_cmd = 0.01 + motor_speed_cmd;
            prior_speed_error = e;
            double target = motor_speed_cmd * (0.65 / 0.3);
            actual_speed += (target - actual_speed) * (dt_inner / 0.05);
            actual_speed = std::max(0.0, actual_speed);
        }
        return actual_speed;
    }
    void reset() { motor_speed_cmd = prior_speed_error = actual_speed = 0.0; }
};

struct TestResult {
    double heading_w;
    double max_cte, avg_cte;
    double max_steer_osc;
    double avg_steer_osc;
    int steer_sat_steps;
    bool completed;
    double avg_speed;
};

TestResult run_with_realistic_model(
    acc::CubicSplinePath& spline,
    double heading_w,
    double servo_delay_ms,
    double tf_delay_ms,
    double pos_noise_sigma,
    double heading_noise_sigma,
    unsigned seed)
{
    double total_len = spline.total_length();

    mpcc::Config cfg;
    cfg.horizon = 10;
    cfg.dt = 0.1;
    cfg.wheelbase = 0.256;
    cfg.max_velocity = 0.55;
    cfg.min_velocity = 0.0;
    cfg.max_steering = 0.45;
    cfg.max_acceleration = 1.5;
    cfg.max_steering_rate = 1.5;
    cfg.reference_velocity = 0.45;
    cfg.contour_weight = 15.0;
    cfg.lag_weight = 10.0;
    cfg.velocity_weight = 15.0;
    cfg.steering_weight = 0.05;
    cfg.acceleration_weight = 0.01;
    cfg.steering_rate_weight = 1.5;
    cfg.heading_weight = heading_w;
    cfg.progress_weight = 1.0;
    cfg.jerk_weight = 0.0;
    cfg.boundary_weight = 0.0;
    cfg.max_sqp_iterations = 5;
    cfg.max_qp_iterations = 20;
    cfg.startup_ramp_duration_s = 3.0;
    cfg.startup_elapsed_s = 10.0;  // Past startup for steady-state testing
    cfg.startup_progress_weight = 1.0;

    mpcc::ActiveSolver solver;
    solver.init(cfg);
    solver.path_lookup.lookup = [&spline, total_len](
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


    // TRUE state (plant)
    mpcc::AckermannModel plant(cfg.wheelbase);
    double init_x, init_y;
    spline.get_position(0.0, init_x, init_y);
    double init_theta = spline.get_tangent(0.0);
    mpcc::VecX true_state;
    true_state << init_x, init_y, init_theta, 0.0, 0.0;

    PDSpeedController pd;
    double progress = 0.0;
    double prev_delta = 0.0;

    // Servo delay buffer (models mechanical response time)
    int servo_delay_steps = static_cast<int>(servo_delay_ms / (cfg.dt * 1000.0));
    std::vector<double> delta_buffer(std::max(servo_delay_steps, 1), 0.0);
    int buf_idx = 0;

    // TF delay: solver sees state from tf_delay_ms ago
    int tf_delay_steps = static_cast<int>(tf_delay_ms / (cfg.dt * 1000.0));
    std::vector<mpcc::VecX> state_history;

    std::mt19937 rng(seed);
    std::normal_distribution<double> pos_dist(0.0, pos_noise_sigma);
    std::normal_distribution<double> head_dist(0.0, heading_noise_sigma);

    TestResult res;
    res.heading_w = heading_w;
    res.max_cte = 0;
    res.avg_cte = 0;
    res.max_steer_osc = 0;
    res.avg_steer_osc = 0;
    res.steer_sat_steps = 0;
    res.completed = false;
    res.avg_speed = 0;

    int step_count = 0;
    double cte_sum = 0, osc_sum = 0, speed_sum = 0;
    int max_steps = 500;
    int cfail = 0;

    for (int step = 0; step < max_steps; step++) {
        if (progress >= total_len - 0.1) { res.completed = true; break; }

        // Record state history for TF delay
        state_history.push_back(true_state);

        // MEASURED state = delayed true state + noise
        // Models Cartographer SLAM TF output (delayed + noisy)
        mpcc::VecX delayed_state = true_state;
        if (tf_delay_steps > 0 && (int)state_history.size() > tf_delay_steps) {
            delayed_state = state_history[state_history.size() - 1 - tf_delay_steps];
        }

        double meas_x = delayed_state(0) + pos_dist(rng);
        double meas_y = delayed_state(1) + pos_dist(rng);
        double meas_theta = delayed_state(2) + head_dist(rng);
        double meas_v = true_state(3);  // Encoder velocity is current (no delay)
        double meas_delta = prev_delta;

        mpcc::VecX x0;
        x0 << meas_x, meas_y, meas_theta, std::max(0.0, meas_v), meas_delta;

        double meas_progress = spline.find_closest_progress(meas_x, meas_y);
        if (meas_progress > progress) progress = meas_progress;

        double lv = std::max(meas_v, cfg.reference_velocity * 0.5);
        std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
        double s = progress;
        for (int k = 0; k <= cfg.horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);
            double rx, ry, ct, st;
            spline.get_path_reference(s, rx, ry, ct, st);
            refs[k].x = rx; refs[k].y = ry;
            refs[k].cos_theta = ct; refs[k].sin_theta = st;
            refs[k].curvature = spline.get_curvature(s);
            double curv = std::abs(refs[k].curvature);
            s += std::max(0.10, lv * std::exp(-0.4 * curv)) * cfg.dt;
        }

        auto result = solver.solve(x0, refs, progress, total_len, {}, {});
        if (!result.success) { if (++cfail >= 5) break; continue; }
        cfail = 0;

        double v_cmd = std::clamp(result.v_cmd, cfg.min_velocity, cfg.max_velocity);
        double delta_cmd = std::clamp(result.delta_cmd, -cfg.max_steering, cfg.max_steering);

        double remaining = total_len - progress;
        if (remaining < 0.5) v_cmd *= remaining / 0.5;

        // Apply servo delay: the actual steering is the command from N steps ago
        double actual_delta;
        if (servo_delay_steps > 0) {
            actual_delta = delta_buffer[buf_idx % delta_buffer.size()];
            delta_buffer[buf_idx % delta_buffer.size()] = delta_cmd;
            buf_idx++;
        } else {
            actual_delta = delta_cmd;
        }

        // Apply to TRUE state (plant dynamics with delayed steering)
        double actual_v = pd.step(v_cmd, cfg.dt);
        mpcc::VecU u;
        u(0) = (actual_v - true_state(3)) / cfg.dt;
        u(1) = (actual_delta - true_state(4)) / cfg.dt;
        u(0) = std::clamp(u(0), -cfg.max_acceleration, cfg.max_acceleration);
        u(1) = std::clamp(u(1), -cfg.max_steering_rate, cfg.max_steering_rate);
        true_state = plant.rk4_step(true_state, u, cfg.dt);
        true_state(3) = std::clamp(true_state(3), cfg.min_velocity, cfg.max_velocity);
        true_state(4) = std::clamp(true_state(4), -cfg.max_steering, cfg.max_steering);

        // CTE from TRUE state
        double cp = spline.find_closest_progress(true_state(0), true_state(1));
        double rpx, rpy; spline.get_position(cp, rpx, rpy);
        double cte = std::hypot(true_state(0) - rpx, true_state(1) - rpy);
        res.max_cte = std::max(res.max_cte, cte);
        cte_sum += cte;
        if (cte > 1.0) break;

        double osc = std::abs(delta_cmd - prev_delta);
        res.max_steer_osc = std::max(res.max_steer_osc, osc);
        osc_sum += osc;
        prev_delta = delta_cmd;

        if (std::abs(std::abs(delta_cmd) - cfg.max_steering) < 0.01)
            res.steer_sat_steps++;

        speed_sum += true_state(3);
        step_count++;
    }

    res.avg_cte = step_count > 0 ? cte_sum / step_count : 0;
    res.avg_steer_osc = step_count > 0 ? osc_sum / step_count : 0;
    res.avg_speed = step_count > 0 ? speed_sum / step_count : 0;
    if (progress >= total_len - 0.1) res.completed = true;

    return res;
}

int main() {
    printf("=== Swerving Diagnosis: Realistic Deployment Model ===\n");
    printf("Models: servo delay, TF latency, measurement noise\n\n");

    // Build path through curves (pickup→dropoff has tight roundabout section)
    acc::RoadGraph rg;
    std::vector<std::pair<std::string, std::string>> legs = {
        {"hub_to_pickup", "Hub→Pickup"},
        {"pickup_to_dropoff", "Pickup→Dropoff"},
    };

    // Deployment-realistic parameters
    double servo_delay = 50.0;  // ms (QCar2 servo mechanical delay)
    double tf_delay = 100.0;    // ms (Cartographer processing + publish delay)
    double pos_noise = 0.005;   // 5mm σ position noise
    double heading_noise = 2.0 * M_PI / 180.0;  // 2° σ heading noise

    // ---- Test 1: heading_weight sweep WITH realistic delays ----
    printf("--- heading_weight sweep (servo=%.0fms, TF=%.0fms, noise=%.0fmm/%.1f°) ---\n",
           servo_delay, tf_delay, pos_noise*1000, heading_noise*180/M_PI);
    printf("%-10s | %-7s %-7s %-9s %-9s %-6s %-6s %s\n",
           "heading_w", "max_cte", "avg_cte", "max_osc", "avg_osc", "speed", "sat", "status");
    printf("-----------|---------|---------|---------|---------|------|------|-------\n");

    std::vector<double> heading_weights = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0};

    for (auto hw : heading_weights) {
        double total_max_cte = 0, total_avg_cte = 0, total_avg_osc = 0, total_speed = 0;
        int total_legs = 0;
        bool all_ok = true;

        for (auto& [leg_name, leg_label] : legs) {
            auto leg_path = rg.plan_path_for_mission_leg(leg_name, acc::HUB_X, acc::HUB_Y);
            if (!leg_path) continue;

            acc::CubicSplinePath leg_spline;
            leg_spline.build(leg_path->first, leg_path->second, true);

            // Average over 3 seeds
            for (unsigned seed = 42; seed <= 44; seed++) {
                auto r = run_with_realistic_model(leg_spline, hw,
                    servo_delay, tf_delay, pos_noise, heading_noise, seed);
                total_max_cte = std::max(total_max_cte, r.max_cte);
                total_avg_cte += r.avg_cte;
                total_avg_osc += r.avg_steer_osc;
                total_speed += r.avg_speed;
                total_legs++;
                if (!r.completed) all_ok = false;
            }
        }

        if (total_legs > 0) {
            printf("%-10.1f | %-7.3f %-7.3f %-9.3f %-9.3f %-6.3f %-6d %s\n",
                   hw, total_max_cte, total_avg_cte / total_legs,
                   0.0, total_avg_osc / total_legs,
                   total_speed / total_legs, 0,
                   all_ok ? "OK" : "FAIL");
        }
    }

    // ---- Test 2: Compare with NO delays (to show delay is the amplifier) ----
    printf("\n--- heading_weight=2.0: delay comparison ---\n");
    printf("%-20s | %-7s %-7s %-9s\n", "condition", "max_cte", "avg_cte", "avg_osc");
    printf("--------------------|---------|---------|--------\n");

    struct Condition {
        const char* name;
        double servo_ms, tf_ms, pos_n, head_n;
    };
    std::vector<Condition> conditions = {
        {"no delay, no noise",     0,  0,     0,          0},
        {"servo only (50ms)",     50,  0,     0,          0},
        {"TF only (100ms)",        0, 100,    0,          0},
        {"noise only",             0,  0,     0.005,      2.0*M_PI/180},
        {"servo+TF",              50, 100,    0,          0},
        {"servo+TF+noise",        50, 100,    0.005,      2.0*M_PI/180},
    };

    for (auto& c : conditions) {
        auto leg_path = rg.plan_path_for_mission_leg("hub_to_pickup", acc::HUB_X, acc::HUB_Y);
        if (!leg_path) continue;

        acc::CubicSplinePath spline;
        spline.build(leg_path->first, leg_path->second, true);

        double sum_max_cte = 0, sum_avg_cte = 0, sum_avg_osc = 0;
        int count = 0;
        for (unsigned seed = 42; seed <= 44; seed++) {
            auto r = run_with_realistic_model(spline, 2.0,
                c.servo_ms, c.tf_ms, c.pos_n, c.head_n, seed);
            sum_max_cte = std::max(sum_max_cte, r.max_cte);
            sum_avg_cte += r.avg_cte;
            sum_avg_osc += r.avg_steer_osc;
            count++;
        }
        printf("%-20s | %-7.3f %-7.3f %-9.3f\n",
               c.name, sum_max_cte, sum_avg_cte / count, sum_avg_osc / count);
    }

    // ---- Test 3: Full mission with best heading_weight ----
    printf("\n--- Full 3-leg mission with heading_weight=0 (reference-matched) ---\n");
    {
        std::vector<std::pair<std::string, std::string>> all_legs = {
            {"hub_to_pickup", "Hub→Pickup"},
            {"pickup_to_dropoff", "Pickup→Dropoff"},
            {"dropoff_to_hub", "Dropoff→Hub"},
        };

        for (auto& [leg_name, leg_label] : all_legs) {
            auto leg_path = rg.plan_path_for_mission_leg(leg_name, acc::HUB_X, acc::HUB_Y);
            if (!leg_path) { printf("  FAILED to plan %s\n", leg_name.c_str()); continue; }

            acc::CubicSplinePath leg_spline;
            leg_spline.build(leg_path->first, leg_path->second, true);

            auto r = run_with_realistic_model(leg_spline, 0.0,
                servo_delay, tf_delay, pos_noise, heading_noise, 42);
            printf("  %s: max_cte=%.3fm avg_cte=%.3fm avg_osc=%.3f speed=%.3f %s\n",
                   leg_label.c_str(), r.max_cte, r.avg_cte, r.avg_steer_osc,
                   r.avg_speed, r.completed ? "OK" : "INCOMPLETE");
        }
    }

    // ---- Test 4: Write CSV traces for visualization ----
    printf("\n--- Writing CSV traces for heading_weight comparison ---\n");
    {
        auto leg_path = rg.plan_path_for_mission_leg("hub_to_pickup", acc::HUB_X, acc::HUB_Y);
        if (leg_path) {
            acc::CubicSplinePath spline;
            spline.build(leg_path->first, leg_path->second, true);

            // Generate trace for h_w=0 and h_w=2.0
            for (double hw : {0.0, 2.0}) {
                char fname[256];
                std::snprintf(fname, sizeof(fname), "results/swerving_hw%.0f.csv", hw);
                std::ofstream f(fname);
                if (!f) { std::snprintf(fname, sizeof(fname), "swerving_hw%.0f.csv", hw); f.open(fname); }
                if (!f) continue;

                f << "step,x,y,theta,v,delta_cmd,cte,heading_err\n";

                // Run single trace
                mpcc::Config cfg;
                cfg.horizon = 10; cfg.dt = 0.1; cfg.wheelbase = 0.256;
                cfg.max_velocity = 0.55; cfg.min_velocity = 0.0;
                cfg.max_steering = 0.45; cfg.max_acceleration = 1.5;
                cfg.max_steering_rate = 1.5; cfg.reference_velocity = 0.45;
                cfg.contour_weight = 15.0; cfg.lag_weight = 10.0;
                cfg.velocity_weight = 15.0; cfg.steering_weight = 0.05;
                cfg.acceleration_weight = 0.01; cfg.steering_rate_weight = 1.5;
                cfg.heading_weight = hw; cfg.progress_weight = 1.0;
                cfg.jerk_weight = 0.0; cfg.boundary_weight = 0.0;
                cfg.max_sqp_iterations = 5; cfg.max_qp_iterations = 20;
                cfg.startup_ramp_duration_s = 3.0; cfg.startup_elapsed_s = 10.0;
                cfg.startup_progress_weight = 1.0;

                mpcc::ActiveSolver solver;
                solver.init(cfg);
                double tl = spline.total_length();
                solver.path_lookup.lookup = [&spline, tl](
                    double px, double py, double s_min, double* s_out) -> mpcc::PathRef {
                    double s = spline.find_closest_progress_from(px, py, s_min);
                    s = std::clamp(s, 0.0, tl - 0.001);
                    if (s_out) *s_out = s;
                    mpcc::PathRef ref;
                    double ct, st;
                    spline.get_path_reference(s, ref.x, ref.y, ct, st);
                    ref.cos_theta = ct; ref.sin_theta = st;
                    ref.curvature = spline.get_curvature(s);
                    return ref;
                };
            

                mpcc::AckermannModel plant(cfg.wheelbase);
                double init_x, init_y;
                spline.get_position(0.0, init_x, init_y);
                double init_theta = spline.get_tangent(0.0);
                mpcc::VecX ts; ts << init_x, init_y, init_theta, 0.0, 0.0;
                PDSpeedController pd;
                double progress = 0.0;
                std::mt19937 rng(42);
                std::normal_distribution<double> pd_n(0, 0.005);
                std::normal_distribution<double> hd_n(0, 2.0*M_PI/180);

                // Servo delay buffer
                std::vector<double> dbuf(1, 0.0);  // 1 step = 100ms for dt=0.1
                int cfail2 = 0;

                for (int step = 0; step < 500; step++) {
                    if (progress >= tl - 0.1) break;

                    double mx = ts(0) + pd_n(rng), my = ts(1) + pd_n(rng);
                    double mt = ts(2) + hd_n(rng);
                    double mv = ts(3);

                    mpcc::VecX x0;
                    x0 << mx, my, mt, std::max(0.0, mv), dbuf[0];

                    double mp = spline.find_closest_progress(mx, my);
                    if (mp > progress) progress = mp;

                    double lv = std::max(mv, cfg.reference_velocity * 0.5);
                    std::vector<mpcc::PathRef> refs(cfg.horizon + 1);
                    double s = progress;
                    for (int k = 0; k <= cfg.horizon; k++) {
                        s = std::clamp(s, 0.0, tl - 0.001);
                        double rx, ry, ct, st;
                        spline.get_path_reference(s, rx, ry, ct, st);
                        refs[k].x = rx; refs[k].y = ry;
                        refs[k].cos_theta = ct; refs[k].sin_theta = st;
                        refs[k].curvature = spline.get_curvature(s);
                        s += std::max(0.10, lv * std::exp(-0.4 * std::abs(refs[k].curvature))) * cfg.dt;
                    }

                    auto res = solver.solve(x0, refs, progress, tl, {}, {});
                    if (!res.success) { if (++cfail2 >= 5) break; continue; }
                    cfail2 = 0;

                    double v_cmd = std::clamp(res.v_cmd, cfg.min_velocity, cfg.max_velocity);
                    double delta_cmd = std::clamp(res.delta_cmd, -cfg.max_steering, cfg.max_steering);
                    double rem = tl - progress;
                    if (rem < 0.5) v_cmd *= rem / 0.5;

                    // Servo delay: apply previous command
                    double actual_delta = dbuf[0];
                    dbuf[0] = delta_cmd;

                    double av = pd.step(v_cmd, cfg.dt);
                    mpcc::VecU u;
                    u(0) = (av - ts(3)) / cfg.dt;
                    u(1) = (actual_delta - ts(4)) / cfg.dt;
                    u(0) = std::clamp(u(0), -cfg.max_acceleration, cfg.max_acceleration);
                    u(1) = std::clamp(u(1), -cfg.max_steering_rate, cfg.max_steering_rate);
                    ts = plant.rk4_step(ts, u, cfg.dt);
                    ts(3) = std::clamp(ts(3), cfg.min_velocity, cfg.max_velocity);
                    ts(4) = std::clamp(ts(4), -cfg.max_steering, cfg.max_steering);

                    // CTE
                    double cp = spline.find_closest_progress(ts(0), ts(1));
                    double rpx, rpy; spline.get_position(cp, rpx, rpy);
                    double cte = std::hypot(ts(0) - rpx, ts(1) - rpy);
                    double tang = spline.get_tangent(cp);
                    double herr = ts(2) - tang;
                    while (herr > M_PI) herr -= 2*M_PI;
                    while (herr < -M_PI) herr += 2*M_PI;

                    f << step << "," << ts(0) << "," << ts(1) << "," << ts(2) << ","
                      << ts(3) << "," << delta_cmd << "," << cte << "," << herr << "\n";
                    if (cte > 1.0) break;
                }

                printf("  Written: %s\n", fname);
            }
        }
    }

    printf("\n=== DONE ===\n");
    return 0;
}
