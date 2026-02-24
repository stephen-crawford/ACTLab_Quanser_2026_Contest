/**
 * Road boundary constraint generation for MPCC controller.
 *
 * Ported from road_boundaries.py: RoadSegment, TrafficControl,
 * ObstacleZone, RoadBoundarySpline.
 *
 * All internal storage is in MAP FRAME after transformation from QLabs.
 */

#pragma once

#include "coordinate_transform.h"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace acc {

// -------------------------------------------------------------------------
// Road segment
// -------------------------------------------------------------------------

struct CenterlinePoint {
    double x, y;           // map frame
    double width_left;
    double width_right;
};

struct RoadSegment {
    std::string name;
    std::string type;  // "spline" or "circular"

    // Spline segment data
    std::vector<CenterlinePoint> centerline;

    // Circular segment data
    double center_x = 0, center_y = 0;
    double radius = 0;
    double width = 0.24;

    bool contains_point(double x, double y, double margin = 0.5) const;

    /// Returns (nearest_x, nearest_y, tangent_angle, width_left, width_right)
    struct NearestResult {
        double x, y, tangent, w_left, w_right;
    };
    NearestResult get_nearest_point_and_tangent(double x, double y) const;
};

// -------------------------------------------------------------------------
// Traffic control
// -------------------------------------------------------------------------

struct TrafficControl {
    std::string type;   // "stop_sign", "traffic_light"
    std::string name;
    double x = 0, y = 0;  // map frame
    double stop_line_distance = 0.2;
};

// -------------------------------------------------------------------------
// Obstacle zone
// -------------------------------------------------------------------------

struct ObstacleZone {
    std::string name;
    std::string type;  // "circle" or "rectangle"
    double center_x = 0, center_y = 0;  // map frame
    double radius = 0;       // for circle
    double width = 0, height = 0;  // for rectangle
    double max_velocity = 0.4;

    bool contains_point(double x, double y) const;
};

// -------------------------------------------------------------------------
// Boundary constraint result
// -------------------------------------------------------------------------

struct BoundaryResult {
    double nx, ny;      // normal vector (pointing left)
    double b_left;
    double b_right;
};

// -------------------------------------------------------------------------
// RoadBoundarySpline
// -------------------------------------------------------------------------

class RoadBoundarySpline {
public:
    explicit RoadBoundarySpline(const std::string& config_path = "");

    /// Load configuration from YAML file.
    void load_config(const std::string& config_path);

    /// Find the road segment containing (x, y) in map frame.
    const RoadSegment* get_active_segment(double x, double y) const;

    /// Get linearized boundary constraints at position (map frame).
    BoundaryResult get_boundary_constraints(double x, double y, double theta) const;

    /// Get boundary constraints from path geometry (simpler, fixed-width).
    BoundaryResult get_boundary_constraints_from_path(
        double path_x, double path_y, double path_theta,
        double default_width = 0.25) const;

    /// Get velocity limit from obstacle zones (map frame).
    double get_velocity_limit(double x, double y) const;

    /// Get nearby traffic controls (map frame).
    struct NearbyControl {
        const TrafficControl* control;
        double distance;
    };
    std::vector<NearbyControl> get_nearby_traffic_controls(
        double x, double y, double theta, double max_distance = 1.5) const;

    // Accessors
    const std::vector<RoadSegment>& segments() const { return segments_; }
    const std::vector<TrafficControl>& traffic_controls() const { return traffic_controls_; }
    const std::vector<ObstacleZone>& obstacle_zones() const { return obstacle_zones_; }

private:
    double road_width_ = 0.30;
    double vehicle_half_width_ = 0.08;
    double safety_margin_ = 0.02;

    TransformParams tp_;

    std::vector<RoadSegment> segments_;
    std::vector<TrafficControl> traffic_controls_;
    std::vector<ObstacleZone> obstacle_zones_;
};

}  // namespace acc
