/**
 * MPCC Solver implementation - C API for Python ctypes.
 */

#include "mpcc_solver.h"
#include <iostream>

extern "C" {

void* mpcc_create(const MPCCParams* params) {
    auto* solver = new mpcc::Solver();
    mpcc::Config cfg;

    cfg.horizon = params->horizon;
    cfg.dt = params->dt;
    cfg.wheelbase = params->wheelbase;
    cfg.max_velocity = params->max_velocity;
    cfg.min_velocity = params->min_velocity;
    cfg.max_steering = params->max_steering;
    cfg.max_acceleration = params->max_acceleration;
    cfg.max_steering_rate = params->max_steering_rate;
    cfg.reference_velocity = params->reference_velocity;
    cfg.contour_weight = params->contour_weight;
    cfg.lag_weight = params->lag_weight;
    cfg.velocity_weight = params->velocity_weight;
    cfg.steering_weight = params->steering_weight;
    cfg.acceleration_weight = params->acceleration_weight;
    cfg.steering_rate_weight = params->steering_rate_weight;
    cfg.jerk_weight = params->jerk_weight;
    cfg.robot_radius = params->robot_radius;
    cfg.safety_margin = params->safety_margin;
    cfg.obstacle_weight = params->obstacle_weight;
    cfg.boundary_weight = params->boundary_weight;
    cfg.max_sqp_iterations = params->max_sqp_iterations;
    cfg.max_qp_iterations = params->max_qp_iterations;
    cfg.qp_tolerance = params->qp_tolerance;

    solver->init(cfg);
    return solver;
}

void mpcc_destroy(void* solver) {
    delete static_cast<mpcc::Solver*>(solver);
}

void mpcc_reset(void* solver) {
    static_cast<mpcc::Solver*>(solver)->reset();
}

int mpcc_solve(
    void* solver_ptr,
    const double* state,
    const MPCCPathPoint* path,
    int n_path,
    const MPCCObstacle* obstacles,
    int n_obstacles,
    const MPCCBoundary* boundaries,
    int n_boundaries,
    double current_progress,
    double path_total_length,
    MPCCResultC* result)
{
    auto* solver = static_cast<mpcc::Solver*>(solver_ptr);

    // Convert state
    Eigen::Matrix<double, 5, 1> x0;
    x0 << state[0], state[1], state[2], state[3], state[4];

    // Convert path references
    std::vector<mpcc::PathRef> path_refs(n_path);
    for (int i = 0; i < n_path; i++) {
        path_refs[i].x = path[i].x;
        path_refs[i].y = path[i].y;
        path_refs[i].cos_theta = path[i].cos_theta;
        path_refs[i].sin_theta = path[i].sin_theta;
        path_refs[i].curvature = path[i].curvature;
    }

    // Convert obstacles
    std::vector<mpcc::Obstacle> obs_vec(n_obstacles);
    for (int i = 0; i < n_obstacles; i++) {
        obs_vec[i].x = obstacles[i].x;
        obs_vec[i].y = obstacles[i].y;
        obs_vec[i].radius = obstacles[i].radius;
    }

    // Convert boundaries
    std::vector<mpcc::BoundaryConstraint> bd_vec(n_boundaries);
    for (int i = 0; i < n_boundaries; i++) {
        bd_vec[i].nx = boundaries[i].nx;
        bd_vec[i].ny = boundaries[i].ny;
        bd_vec[i].b_left = boundaries[i].b_left;
        bd_vec[i].b_right = boundaries[i].b_right;
    }

    // Solve
    auto res = solver->solve(x0, path_refs, current_progress,
                             path_total_length, obs_vec, bd_vec);

    // Fill result
    result->v_cmd = res.v_cmd;
    result->delta_cmd = res.delta_cmd;
    result->omega_cmd = res.omega_cmd;
    result->solve_time_us = res.solve_time_us;
    result->success = res.success ? 1 : 0;

    int n = std::min((int)res.predicted_x.size(), 50);
    result->predicted_len = n;
    for (int i = 0; i < n; i++) {
        result->predicted_x[i] = res.predicted_x[i];
        result->predicted_y[i] = res.predicted_y[i];
        result->predicted_theta[i] = res.predicted_theta[i];
    }

    return res.success ? 0 : -1;
}

}  // extern "C"
