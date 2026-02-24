/**
 * Coordinate transform utilities for QLabs <-> Cartographer map frame.
 *
 * Ported from mission_manager.py (qlabs_to_map_frame, map_to_qlabs_frame)
 * and road_graph.py (qlabs_path_to_map_path).
 *
 * The transform consists of:
 *   1. Translate by the car's spawn position in QLabs
 *   2. Rotate by the calibrated transform angle (0.7177 rad)
 */

#pragma once

#include <cmath>
#include <vector>

namespace acc {

struct TransformParams {
    double origin_x = -1.205;
    double origin_y = -0.83;
    double origin_heading_rad = 0.7177;
};

/// Normalize angle to [-pi, pi]
inline double normalize_angle(double angle) {
    while (angle > M_PI)  angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

/// Transform a single point from QLabs world frame to Cartographer map frame.
inline void qlabs_to_map(double qlabs_x, double qlabs_y, double qlabs_yaw,
                         const TransformParams& tp,
                         double& map_x, double& map_y, double& map_yaw)
{
    double dx = qlabs_x - tp.origin_x;
    double dy = qlabs_y - tp.origin_y;
    double cos_t = std::cos(tp.origin_heading_rad);
    double sin_t = std::sin(tp.origin_heading_rad);

    map_x =  cos_t * dx + sin_t * dy;
    map_y = -sin_t * dx + cos_t * dy;
    map_yaw = normalize_angle(qlabs_yaw + tp.origin_heading_rad);
}

/// Transform a single point from Cartographer map frame to QLabs world frame.
inline void map_to_qlabs(double map_x, double map_y, double map_yaw,
                         const TransformParams& tp,
                         double& qlabs_x, double& qlabs_y, double& qlabs_yaw)
{
    double cos_t = std::cos(tp.origin_heading_rad);
    double sin_t = std::sin(tp.origin_heading_rad);

    double rx = map_x * cos_t - map_y * sin_t;
    double ry = map_x * sin_t + map_y * cos_t;

    qlabs_x = rx + tp.origin_x;
    qlabs_y = ry + tp.origin_y;
    qlabs_yaw = normalize_angle(map_yaw - tp.origin_heading_rad);
}

/// Transform a 2D point (no heading) from QLabs to map frame.
inline void qlabs_to_map_2d(double qlabs_x, double qlabs_y,
                            const TransformParams& tp,
                            double& map_x, double& map_y)
{
    double dx = qlabs_x - tp.origin_x;
    double dy = qlabs_y - tp.origin_y;
    double cos_t = std::cos(tp.origin_heading_rad);
    double sin_t = std::sin(tp.origin_heading_rad);
    map_x =  cos_t * dx + sin_t * dy;
    map_y = -sin_t * dx + cos_t * dy;
}

/// Transform a 2D point (no heading) from map to QLabs frame.
inline void map_to_qlabs_2d(double map_x, double map_y,
                            const TransformParams& tp,
                            double& qlabs_x, double& qlabs_y)
{
    double cos_t = std::cos(tp.origin_heading_rad);
    double sin_t = std::sin(tp.origin_heading_rad);
    qlabs_x = map_x * cos_t - map_y * sin_t + tp.origin_x;
    qlabs_y = map_x * sin_t + map_y * cos_t + tp.origin_y;
}

/// Batch transform Nx2 waypoints from QLabs to map frame.
/// Input/output: vectors of x, y coordinates (same size).
inline void qlabs_path_to_map(const std::vector<double>& qlabs_x,
                              const std::vector<double>& qlabs_y,
                              const TransformParams& tp,
                              std::vector<double>& map_x,
                              std::vector<double>& map_y)
{
    size_t n = qlabs_x.size();
    map_x.resize(n);
    map_y.resize(n);
    double cos_t = std::cos(tp.origin_heading_rad);
    double sin_t = std::sin(tp.origin_heading_rad);
    for (size_t i = 0; i < n; ++i) {
        double dx = qlabs_x[i] - tp.origin_x;
        double dy = qlabs_y[i] - tp.origin_y;
        map_x[i] =  cos_t * dx + sin_t * dy;
        map_y[i] = -sin_t * dx + cos_t * dy;
    }
}

}  // namespace acc
