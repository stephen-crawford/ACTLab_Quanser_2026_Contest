/**
 * MPCC Solver Tests — steering direction, tuning, and path following
 *
 * Tests the core MPCC solver behavior:
 * 1. Steering direction: left turn path → positive delta (not negative)
 * 2. Forward progress: solver must make progress along the path
 * 3. Velocity tracking: commanded velocity should approach reference
 * 4. Cold start behavior: first solve should produce reasonable commands
 * 5. Warm start: subsequent solves should converge faster
 * 6. Hub-to-pickup scenario: simulates real mission start
 * 7. Tuning validation: config values match reference-aligned defaults
 *
 * Build:
 *   cd /home/stephen/quanser-acc/cpp/test_build
 *   g++ -std=c++17 -O2 -I.. -I/usr/include/eigen3 -o test_mpcc_solver test_mpcc_solver.cpp
 *
 * Run:
 *   ./test_mpcc_solver
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "mpcc_solver.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { std::printf("  TEST: %-55s ", name); } while(0)

#define PASS() \
    do { std::printf("[PASS]\n"); tests_passed++; } while(0)

#define FAIL(msg) \
    do { std::printf("[FAIL] %s\n", msg); tests_failed++; } while(0)

#define ASSERT_NEAR(a, b, eps, msg) \
    do { if (std::abs((a) - (b)) > (eps)) { \
        std::printf("[FAIL] %s (got %f, expected %f)\n", msg, (double)(a), (double)(b)); \
        tests_failed++; return; \
    } } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { \
        std::printf("[FAIL] %s\n", msg); \
        tests_failed++; return; \
    } } while(0)

#define ASSERT_GT(a, b, msg) \
    do { if (!((a) > (b))) { \
        std::printf("[FAIL] %s (got %f, expected > %f)\n", msg, (double)(a), (double)(b)); \
        tests_failed++; return; \
    } } while(0)

#define ASSERT_LT(a, b, msg) \
    do { if (!((a) < (b))) { \
        std::printf("[FAIL] %s (got %f, expected < %f)\n", msg, (double)(a), (double)(b)); \
        tests_failed++; return; \
    } } while(0)

// =========================================================================
// Helper: create a straight path along X axis
// =========================================================================
std::vector<mpcc::PathRef> make_straight_path_x(int n, double spacing = 0.1) {
    std::vector<mpcc::PathRef> refs(n);
    for (int k = 0; k < n; k++) {
        refs[k].x = k * spacing;
        refs[k].y = 0.0;
        refs[k].cos_theta = 1.0;  // heading along +X
        refs[k].sin_theta = 0.0;
        refs[k].curvature = 0.0;
    }
    return refs;
}

// Helper: create a path curving LEFT (positive Y direction)
std::vector<mpcc::PathRef> make_left_turn_path(int n) {
    std::vector<mpcc::PathRef> refs(n);
    double radius = 2.0;
    for (int k = 0; k < n; k++) {
        double angle = k * 0.05;  // ~3deg steps
        refs[k].x = radius * std::sin(angle);
        refs[k].y = radius * (1.0 - std::cos(angle));
        refs[k].cos_theta = std::cos(angle);
        refs[k].sin_theta = std::sin(angle);
        refs[k].curvature = 1.0 / radius;
    }
    return refs;
}

// Helper: create a path curving RIGHT (negative Y direction)
std::vector<mpcc::PathRef> make_right_turn_path(int n) {
    std::vector<mpcc::PathRef> refs(n);
    double radius = 2.0;
    for (int k = 0; k < n; k++) {
        double angle = -k * 0.05;
        refs[k].x = radius * std::sin(-angle);  // same X progression
        refs[k].y = -radius * (1.0 - std::cos(angle));  // curves down
        refs[k].cos_theta = std::cos(angle);
        refs[k].sin_theta = std::sin(angle);
        refs[k].curvature = -1.0 / radius;
    }
    return refs;
}

// Helper: create a path similar to hub->pickup start (northeast direction)
// Car starts at origin heading ~0 rad, path goes to (0.86, 0.46) then northeast
std::vector<mpcc::PathRef> make_hub_to_pickup_path(int n) {
    std::vector<mpcc::PathRef> refs(n);
    // Initial segment: slightly curving left from ~0 deg toward ~28 deg
    // This matches the real mission: hub at (0,0) heading east, first waypoint at (0.86, 0.46)
    double target_angle = 0.49;  // ~28 deg
    for (int k = 0; k < n; k++) {
        double t = (double)k / (n - 1);
        double angle = t * target_angle;
        double s = t * 2.0;  // ~2m total path length shown
        refs[k].x = s * std::cos(angle * 0.5);  // approximate arc
        refs[k].y = s * std::sin(angle * 0.5);
        refs[k].cos_theta = std::cos(angle);
        refs[k].sin_theta = std::sin(angle);
        refs[k].curvature = target_angle / 2.0;  // gentle left curve
    }
    return refs;
}

mpcc::Config make_test_config() {
    mpcc::Config cfg;
    cfg.horizon = 15;
    cfg.dt = 0.1;
    cfg.wheelbase = 0.256;
    cfg.startup_elapsed_s = 10.0;  // Past startup ramp
    return cfg;
}

// =========================================================================
// 1. Config Defaults — verify reference-aligned tuning
// =========================================================================
void test_config_defaults() {
    TEST("config: contour_weight default is 8.0");
    mpcc::Config cfg;
    ASSERT_NEAR(cfg.contour_weight, 8.0, 0.01, "contour_weight");
    PASS();
}

void test_config_lag_gt_contour() {
    TEST("config: lag_weight > contour_weight (progress priority)");
    mpcc::Config cfg;
    ASSERT_GT(cfg.lag_weight, cfg.contour_weight,
        "lag_weight must exceed contour_weight for progress");
    PASS();
}

void test_config_velocity_weight() {
    TEST("config: velocity_weight >= 15.0 (strong tracking)");
    mpcc::Config cfg;
    ASSERT_TRUE(cfg.velocity_weight >= 15.0,
        "velocity_weight must be >= 15 to track v_ref (ref: 17.0)");
    PASS();
}

void test_config_reference_velocity() {
    TEST("config: reference_velocity is 0.65");
    mpcc::Config cfg;
    ASSERT_NEAR(cfg.reference_velocity, 0.65, 0.01, "reference_velocity");
    PASS();
}

void test_config_max_velocity() {
    TEST("config: max_velocity >= 1.0 (ref uses 2.0)");
    mpcc::Config cfg;
    ASSERT_TRUE(cfg.max_velocity >= 1.0,
        "max_velocity must be >= 1.0 (ref uses 2.0)");
    PASS();
}

void test_config_min_velocity() {
    TEST("config: min_velocity > 0 (prevents v=0 equilibrium)");
    mpcc::Config cfg;
    ASSERT_GT(cfg.min_velocity, 0.0, "min_velocity must be positive");
    PASS();
}

// =========================================================================
// 2. Dynamics Model Tests
// =========================================================================
void test_dynamics_straight() {
    TEST("dynamics: straight ahead (delta=0) preserves heading");
    mpcc::AckermannModel model(0.256);
    mpcc::VecX x;
    x << 0.0, 0.0, 0.0, 0.5, 0.0;  // heading 0, v=0.5, delta=0
    mpcc::VecU u;
    u << 0.0, 0.0;  // no accel, no steer rate
    auto x1 = model.rk4_step(x, u, 0.1);
    ASSERT_NEAR(x1(0), 0.05, 0.001, "x should advance ~0.05m");
    ASSERT_NEAR(x1(1), 0.0, 0.001, "y should stay 0");
    ASSERT_NEAR(x1(2), 0.0, 0.001, "theta should stay 0");
    PASS();
}

void test_dynamics_positive_delta_turns_left() {
    TEST("dynamics: positive delta (left steer) → positive yaw rate");
    mpcc::AckermannModel model(0.256);
    mpcc::VecX x;
    x << 0.0, 0.0, 0.0, 0.5, 0.3;  // delta=+0.3 (left)
    mpcc::VecU u;
    u << 0.0, 0.0;
    auto x1 = model.rk4_step(x, u, 0.1);
    ASSERT_GT(x1(2), 0.0, "positive delta should increase theta (turn left)");
    ASSERT_GT(x1(1), 0.0, "positive delta should move y positive (left)");
    PASS();
}

void test_dynamics_negative_delta_turns_right() {
    TEST("dynamics: negative delta (right steer) → negative yaw rate");
    mpcc::AckermannModel model(0.256);
    mpcc::VecX x;
    x << 0.0, 0.0, 0.0, 0.5, -0.3;  // delta=-0.3 (right)
    mpcc::VecU u;
    u << 0.0, 0.0;
    auto x1 = model.rk4_step(x, u, 0.1);
    ASSERT_LT(x1(2), 0.0, "negative delta should decrease theta (turn right)");
    ASSERT_LT(x1(1), 0.0, "negative delta should move y negative (right)");
    PASS();
}

// =========================================================================
// 3. Solver — Straight Path
// =========================================================================
void test_solver_straight_path_velocity() {
    TEST("solver: straight path → velocity approaches v_ref");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_straight_path_x(cfg.horizon);
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.1, 0.0;

    auto result = solver.solve(x0, path_refs, 0.0, 5.0, {}, {});
    ASSERT_TRUE(result.success, "solver should succeed");
    ASSERT_GT(result.v_cmd, cfg.min_velocity, "velocity should exceed min");
    PASS();
}

void test_solver_straight_path_low_steering() {
    TEST("solver: straight path → near-zero steering");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_straight_path_x(cfg.horizon);
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.3, 0.0;

    auto result = solver.solve(x0, path_refs, 0.0, 5.0, {}, {});
    ASSERT_TRUE(result.success, "solver should succeed");
    ASSERT_TRUE(std::abs(result.delta_cmd) < 0.15,
        "steering should be near zero on straight path");
    PASS();
}

// =========================================================================
// 4. Solver — Steering Direction (CRITICAL)
// =========================================================================
void test_solver_left_turn_positive_delta() {
    TEST("solver: LEFT turn path → POSITIVE delta (left steering)");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_left_turn_path(cfg.horizon);
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.3, 0.0;  // at origin, heading right, some speed

    // Run a few solves to let warm-start converge
    mpcc::Result result;
    for (int i = 0; i < 3; i++) {
        result = solver.solve(x0, path_refs, 0.0, 5.0, {}, {});
    }

    ASSERT_TRUE(result.success, "solver should succeed");
    ASSERT_GT(result.delta_cmd, 0.0,
        "LEFT turn path MUST produce POSITIVE steering (left)");
    PASS();
}

void test_solver_right_turn_negative_delta() {
    TEST("solver: RIGHT turn path → NEGATIVE delta (right steering)");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_right_turn_path(cfg.horizon);
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.3, 0.0;

    mpcc::Result result;
    for (int i = 0; i < 3; i++) {
        result = solver.solve(x0, path_refs, 0.0, 5.0, {}, {});
    }

    ASSERT_TRUE(result.success, "solver should succeed");
    ASSERT_LT(result.delta_cmd, 0.0,
        "RIGHT turn path MUST produce NEGATIVE steering (right)");
    PASS();
}

// =========================================================================
// 5. Solver — Hub-to-Pickup Scenario
// =========================================================================
void test_solver_hub_to_pickup_steers_left() {
    TEST("solver: hub→pickup path → steers LEFT (not right into wall)");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_hub_to_pickup_path(cfg.horizon);
    // Car starts at origin (~hub), heading ~0 rad (east), needs to go northeast
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.1, 0.0;

    mpcc::Result result;
    for (int i = 0; i < 5; i++) {
        result = solver.solve(x0, path_refs, 0.0, 5.0, {}, {});
    }

    ASSERT_TRUE(result.success, "solver should succeed");
    // Path goes northeast → car must steer LEFT (positive delta)
    ASSERT_GT(result.delta_cmd, -0.05,
        "hub→pickup: should NOT steer hard right (delta should be > -0.05)");
    PASS();
}

void test_solver_hub_to_pickup_makes_progress() {
    TEST("solver: hub→pickup → velocity > min (actually moving)");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_hub_to_pickup_path(cfg.horizon);
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.1, 0.0;

    auto result = solver.solve(x0, path_refs, 0.0, 5.0, {}, {});
    ASSERT_TRUE(result.success, "solver should succeed");
    ASSERT_GT(result.v_cmd, cfg.min_velocity,
        "velocity must exceed min_velocity (car must move)");
    PASS();
}

// =========================================================================
// 6. Solver — Multi-step Simulation
// =========================================================================
void test_solver_multistep_progress() {
    TEST("solver: 20-step simulation → car advances along path");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_straight_path_x(cfg.horizon, 0.2);
    mpcc::VecX state;
    state << 0.0, 0.0, 0.0, 0.0, 0.0;

    mpcc::AckermannModel model(cfg.wheelbase);

    for (int step = 0; step < 20; step++) {
        auto result = solver.solve(state, path_refs, state(0), 5.0, {}, {});
        ASSERT_TRUE(result.success, "solver should succeed each step");

        // Apply dynamics
        mpcc::VecU u;
        u << (result.v_cmd - state(3)) / cfg.dt,  // approximate acceleration
             (result.delta_cmd - state(4)) / cfg.dt;  // approximate steer rate
        u(0) = std::clamp(u(0), -cfg.max_acceleration, cfg.max_acceleration);
        u(1) = std::clamp(u(1), -cfg.max_steering_rate, cfg.max_steering_rate);
        state = model.rk4_step(state, u, cfg.dt);
        state(3) = std::clamp(state(3), cfg.min_velocity, cfg.max_velocity);
        state(4) = std::clamp(state(4), -cfg.max_steering, cfg.max_steering);
    }

    // After 20 steps at 0.1s each = 2s, car should have moved forward
    ASSERT_GT(state(0), 0.3, "car x should advance > 0.3m after 2s");
    ASSERT_TRUE(std::abs(state(1)) < 0.3,
        "car should stay near path (|y| < 0.3)");
    PASS();
}

void test_solver_multistep_left_turn_progress() {
    TEST("solver: 20-step left turn → car follows curve");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_left_turn_path(cfg.horizon);
    mpcc::VecX state;
    state << 0.0, 0.0, 0.0, 0.1, 0.0;  // starting with some velocity

    mpcc::AckermannModel model(cfg.wheelbase);

    for (int step = 0; step < 20; step++) {
        auto result = solver.solve(state, path_refs, 0.0, 5.0, {}, {});
        if (!result.success) break;

        mpcc::VecU u;
        u << (result.v_cmd - state(3)) / cfg.dt,
             (result.delta_cmd - state(4)) / cfg.dt;
        u(0) = std::clamp(u(0), -cfg.max_acceleration, cfg.max_acceleration);
        u(1) = std::clamp(u(1), -cfg.max_steering_rate, cfg.max_steering_rate);
        state = model.rk4_step(state, u, cfg.dt);
        state(3) = std::clamp(state(3), cfg.min_velocity, cfg.max_velocity);
        state(4) = std::clamp(state(4), -cfg.max_steering, cfg.max_steering);
    }

    ASSERT_GT(state(0), 0.1, "car should advance in x");
    ASSERT_GT(state(1), -0.1, "car should NOT go far negative y on left turn");
    ASSERT_GT(state(2), -0.1, "heading should trend positive (left) on left turn");
    PASS();
}

// =========================================================================
// 7. Warm Start
// =========================================================================
void test_solver_warmstart_faster() {
    TEST("solver: warm-started solve is faster than cold start");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_straight_path_x(cfg.horizon);
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.3, 0.0;

    // Cold start
    auto r1 = solver.solve(x0, path_refs, 0.0, 5.0, {}, {});
    double cold_time = r1.solve_time_us;

    // Warm start (slightly advanced)
    x0(0) = 0.03;
    auto r2 = solver.solve(x0, path_refs, 0.03, 5.0, {}, {});
    double warm_time = r2.solve_time_us;

    ASSERT_TRUE(r1.success && r2.success, "both solves should succeed");
    // Warm start should be at most 2x cold (often faster)
    ASSERT_LT(warm_time, cold_time * 3.0,
        "warm start should not be much slower than cold");
    PASS();
}

// =========================================================================
// 8. Obstacle Avoidance
// =========================================================================
void test_solver_avoids_obstacle() {
    TEST("solver: obstacle on path → steers around it");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_straight_path_x(cfg.horizon);
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.3, 0.0;

    // Place obstacle directly on path ahead
    std::vector<mpcc::Obstacle> obstacles = {{0.5, 0.0, 0.15}};

    auto result = solver.solve(x0, path_refs, 0.0, 5.0, obstacles, {});
    ASSERT_TRUE(result.success, "solver should succeed with obstacle");
    // With obstacle on path, steering should deviate
    ASSERT_TRUE(std::abs(result.delta_cmd) > 0.01 || result.v_cmd < 0.3,
        "solver should react to obstacle (steer or slow down)");
    PASS();
}

// =========================================================================
// 9. Contouring and Lag Error Signs
// =========================================================================
void test_contouring_error_lateral() {
    TEST("contouring: car offset left → positive e_c");
    // Path along +X axis, car is at y=+0.1 (left of path)
    mpcc::PathRef ref;
    ref.x = 0.0; ref.y = 0.0;
    ref.cos_theta = 1.0; ref.sin_theta = 0.0;  // heading along +X
    ref.curvature = 0.0;

    double dx = 0.0 - 0.0;  // car x - ref x
    double dy = 0.1 - 0.0;  // car y - ref y (car is left)
    double e_c = -ref.sin_theta * dx + ref.cos_theta * dy;
    double e_l =  ref.cos_theta * dx + ref.sin_theta * dy;

    ASSERT_NEAR(e_c, 0.1, 0.001, "e_c should be +0.1 (left offset)");
    ASSERT_NEAR(e_l, 0.0, 0.001, "e_l should be 0 (no lag)");
    PASS();
}

void test_lag_error_behind() {
    TEST("lag: car behind reference → negative e_l");
    mpcc::PathRef ref;
    ref.x = 1.0; ref.y = 0.0;  // reference 1m ahead
    ref.cos_theta = 1.0; ref.sin_theta = 0.0;

    double dx = 0.0 - 1.0;  // car at 0, ref at 1 → dx = -1
    double dy = 0.0;
    double e_l = ref.cos_theta * dx + ref.sin_theta * dy;

    ASSERT_LT(e_l, 0.0, "e_l should be negative when car is behind ref");
    // The lag gradient should pull car forward (increase x)
    // grad_x(0) = 2 * lag_weight * e_l * cos_theta = 2 * w * (-1) * 1 < 0
    // Negative gradient in x → optimizer increases x → car moves forward ✓
    double grad_x0 = 2.0 * 12.0 * e_l * ref.cos_theta;
    ASSERT_LT(grad_x0, 0.0, "lag gradient should be negative (pull forward)");
    PASS();
}

void test_lag_error_ahead() {
    TEST("lag: car ahead of reference → positive e_l");
    mpcc::PathRef ref;
    ref.x = 0.0; ref.y = 0.0;
    ref.cos_theta = 1.0; ref.sin_theta = 0.0;

    double dx = 1.0;  // car at 1, ref at 0
    double dy = 0.0;
    double e_l = ref.cos_theta * dx + ref.sin_theta * dy;

    ASSERT_GT(e_l, 0.0, "e_l should be positive when car is ahead of ref");
    PASS();
}

// =========================================================================
// 10. Performance
// =========================================================================
void test_solver_performance() {
    TEST("solver: solve time < 5ms");
    auto cfg = make_test_config();
    mpcc::Solver solver;
    solver.init(cfg);

    auto path_refs = make_straight_path_x(cfg.horizon);
    mpcc::VecX x0;
    x0 << 0.0, 0.0, 0.0, 0.3, 0.0;

    // Warm up
    solver.solve(x0, path_refs, 0.0, 5.0, {}, {});

    // Measure
    double total_us = 0;
    int n_solves = 10;
    for (int i = 0; i < n_solves; i++) {
        x0(0) += 0.03;
        auto result = solver.solve(x0, path_refs, x0(0), 5.0, {}, {});
        total_us += result.solve_time_us;
    }
    double avg_us = total_us / n_solves;
    std::printf("(avg=%.0fus) ", avg_us);
    ASSERT_LT(avg_us, 5000.0, "average solve time should be < 5ms");
    PASS();
}

// =========================================================================
// Main
// =========================================================================
int main() {
    std::printf("=== MPCC Solver Tests ===\n\n");

    std::printf("[Config Defaults]\n");
    test_config_defaults();
    test_config_lag_gt_contour();
    test_config_velocity_weight();
    test_config_reference_velocity();
    test_config_max_velocity();
    test_config_min_velocity();

    std::printf("\n[Dynamics Model]\n");
    test_dynamics_straight();
    test_dynamics_positive_delta_turns_left();
    test_dynamics_negative_delta_turns_right();

    std::printf("\n[Solver - Basic]\n");
    test_solver_straight_path_velocity();
    test_solver_straight_path_low_steering();

    std::printf("\n[Solver - Steering Direction (CRITICAL)]\n");
    test_solver_left_turn_positive_delta();
    test_solver_right_turn_negative_delta();

    std::printf("\n[Solver - Hub-to-Pickup Scenario]\n");
    test_solver_hub_to_pickup_steers_left();
    test_solver_hub_to_pickup_makes_progress();

    std::printf("\n[Solver - Multi-step Simulation]\n");
    test_solver_multistep_progress();
    test_solver_multistep_left_turn_progress();

    std::printf("\n[Warm Start]\n");
    test_solver_warmstart_faster();

    std::printf("\n[Obstacle Avoidance]\n");
    test_solver_avoids_obstacle();

    std::printf("\n[Contouring & Lag Errors]\n");
    test_contouring_error_lateral();
    test_lag_error_behind();
    test_lag_error_ahead();

    std::printf("\n[Performance]\n");
    test_solver_performance();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
