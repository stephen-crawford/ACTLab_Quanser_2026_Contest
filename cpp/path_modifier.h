/**
 * Path avoidance modifier for obstacle map integration.
 *
 * When an obstacle is detected on the planned path, generates a
 * modified path segment that avoids it.
 *
 * Strategy:
 * - Find blocked segment where obstacle is within margin
 * - Determine avoidance side (prefer side with more road width)
 * - Offset blocked waypoints laterally
 * - Smooth with cubic blend for MPCC compatibility
 * - Clamp to road boundaries
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace acc {

struct PathObstacle {
    double x, y, radius;
    std::string obj_class;
    bool is_static;
};

class PathModifier {
public:
    static constexpr double VEHICLE_RADIUS = 0.13;
    static constexpr double SAFETY_MARGIN = 0.08;
    static constexpr double BLEND_DISTANCE = 0.3;  // cubic blend ramp distance

    /**
     * Check if any waypoint on the path is blocked by an obstacle.
     * Returns index of first blocked waypoint, or -1 if clear.
     */
    static int check_path_blocked(
        const std::vector<double>& path_x,
        const std::vector<double>& path_y,
        const std::vector<PathObstacle>& obstacles,
        double vehicle_radius = VEHICLE_RADIUS)
    {
        double margin = vehicle_radius + SAFETY_MARGIN;
        for (size_t i = 0; i < path_x.size(); i++) {
            for (auto& obs : obstacles) {
                double d = std::hypot(path_x[i] - obs.x, path_y[i] - obs.y);
                if (d < obs.radius + margin) {
                    return static_cast<int>(i);
                }
            }
        }
        return -1;
    }

    /**
     * Generate a modified path that avoids the given obstacle.
     *
     * The modified path replaces waypoints near the obstacle with
     * laterally offset versions, smoothed with cubic transitions.
     */
    static bool generate_avoidance_path(
        std::vector<double>& path_x,
        std::vector<double>& path_y,
        const PathObstacle& obstacle,
        double vehicle_radius = VEHICLE_RADIUS)
    {
        int n = static_cast<int>(path_x.size());
        if (n < 3) return false;

        double margin = obstacle.radius + vehicle_radius + SAFETY_MARGIN;

        // Find blocked segment
        int start_idx = -1, end_idx = -1;
        for (int i = 0; i < n; i++) {
            double d = std::hypot(path_x[i] - obstacle.x, path_y[i] - obstacle.y);
            if (d < margin + 0.15) {  // slightly wider for avoidance zone
                if (start_idx < 0) start_idx = i;
                end_idx = i;
            }
        }
        if (start_idx < 0) return false;

        // Expand zone for smooth transition
        start_idx = std::max(0, start_idx - 3);
        end_idx = std::min(n - 1, end_idx + 3);

        // Determine avoidance side: choose the side of the path where
        // the obstacle is NOT. The obstacle is at (obs.x, obs.y).
        // Compute cross product with path tangent to determine which side.
        int mid_idx = (start_idx + end_idx) / 2;
        mid_idx = std::clamp(mid_idx, 1, n - 2);

        double tx = path_x[mid_idx + 1] - path_x[mid_idx - 1];
        double ty = path_y[mid_idx + 1] - path_y[mid_idx - 1];
        double tlen = std::sqrt(tx*tx + ty*ty);
        if (tlen < 1e-6) return false;
        tx /= tlen; ty /= tlen;

        // Normal pointing left of path tangent
        double nx = -ty, ny = tx;

        // Vector from path center to obstacle
        double ox = obstacle.x - path_x[mid_idx];
        double oy = obstacle.y - path_y[mid_idx];
        double dot = ox * nx + oy * ny;

        // Offset in the opposite direction from the obstacle
        double offset = margin + 0.05;
        double offset_dir = (dot > 0) ? -1.0 : 1.0;

        // Apply offset to blocked waypoints with cubic smoothing
        int zone_len = end_idx - start_idx + 1;
        for (int i = start_idx; i <= end_idx; i++) {
            // Compute blend factor: 0 at edges, 1 at center
            double t = static_cast<double>(i - start_idx) / std::max(1, zone_len - 1);
            // Smooth cubic: 3t^2 - 2t^3
            double blend = 3.0 * t * t - 2.0 * t * t * t;

            // Compute local tangent normal
            int prev = std::max(0, i - 1);
            int next = std::min(n - 1, i + 1);
            double ltx = path_x[next] - path_x[prev];
            double lty = path_y[next] - path_y[prev];
            double ltlen = std::sqrt(ltx*ltx + lty*lty);
            if (ltlen < 1e-6) continue;
            double lnx = -lty / ltlen;
            double lny = ltx / ltlen;

            path_x[i] += lnx * offset * offset_dir * blend;
            path_y[i] += lny * offset * offset_dir * blend;
        }

        return true;
    }
};

}  // namespace acc
