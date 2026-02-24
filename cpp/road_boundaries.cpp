/**
 * Road boundary constraint generation implementation.
 * Ported from road_boundaries.py.
 */

#include "road_boundaries.h"
#include "yaml_config.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace acc {

// -------------------------------------------------------------------------
// RoadSegment
// -------------------------------------------------------------------------

bool RoadSegment::contains_point(double x, double y, double margin) const {
    if (type == "circular") {
        double dist = std::hypot(x - center_x, y - center_y);
        return dist <= radius + width + margin;
    }
    // Spline segment
    for (auto& pt : centerline) {
        double dist = std::hypot(x - pt.x, y - pt.y);
        double max_w = std::max(pt.width_left, pt.width_right);
        if (dist <= max_w + margin) return true;
    }
    return false;
}

RoadSegment::NearestResult
RoadSegment::get_nearest_point_and_tangent(double x, double y) const
{
    NearestResult r;
    if (type == "circular") {
        double dx = x - center_x;
        double dy = y - center_y;
        double dist = std::hypot(dx, dy);
        if (dist < 0.01) {
            r = {center_x, center_y, 0.0, width, width};
            return r;
        }
        r.x = center_x + radius * dx / dist;
        r.y = center_y + radius * dy / dist;
        r.tangent = std::atan2(-dx, dy);
        r.w_left = width;
        r.w_right = width;
        return r;
    }

    // Spline segment - find nearest segment
    if (centerline.size() < 2) {
        if (centerline.size() == 1) {
            r = {centerline[0].x, centerline[0].y, 0.0,
                 centerline[0].width_left, centerline[0].width_right};
            return r;
        }
        r = {x, y, 0.0, 0.24, 0.24};
        return r;
    }

    double min_dist = 1e18;
    r = {centerline[0].x, centerline[0].y, 0.0,
         centerline[0].width_left, centerline[0].width_right};
    int best_idx = 0;
    double best_t = 0.0;

    for (size_t i = 0; i + 1 < centerline.size(); ++i) {
        double p1x = centerline[i].x, p1y = centerline[i].y;
        double p2x = centerline[i+1].x, p2y = centerline[i+1].y;
        double vx = p2x - p1x, vy = p2y - p1y;
        double ux = x - p1x, uy = y - p1y;
        double seg_sq = vx * vx + vy * vy;

        double t = 0.0;
        double proj_x = p1x, proj_y = p1y;
        if (seg_sq > 1e-10) {
            t = std::clamp((ux * vx + uy * vy) / seg_sq, 0.0, 1.0);
            proj_x = p1x + t * vx;
            proj_y = p1y + t * vy;
        }

        double dist = std::hypot(x - proj_x, y - proj_y);
        if (dist < min_dist) {
            min_dist = dist;
            r.x = proj_x;
            r.y = proj_y;
            r.tangent = std::atan2(vy, vx);
            best_idx = static_cast<int>(i);
            best_t = t;
        }
    }

    // Interpolate widths
    int next_idx = std::min(best_idx + 1,
                            static_cast<int>(centerline.size()) - 1);
    r.w_left = (1.0 - best_t) * centerline[best_idx].width_left +
               best_t * centerline[next_idx].width_left;
    r.w_right = (1.0 - best_t) * centerline[best_idx].width_right +
                best_t * centerline[next_idx].width_right;
    return r;
}

// -------------------------------------------------------------------------
// ObstacleZone
// -------------------------------------------------------------------------

bool ObstacleZone::contains_point(double x, double y) const {
    if (type == "circle") {
        return std::hypot(x - center_x, y - center_y) <= radius;
    }
    if (type == "rectangle") {
        return std::abs(x - center_x) <= width / 2.0 &&
               std::abs(y - center_y) <= height / 2.0;
    }
    return false;
}

// -------------------------------------------------------------------------
// RoadBoundarySpline
// -------------------------------------------------------------------------

RoadBoundarySpline::RoadBoundarySpline(const std::string& config_path) {
    if (!config_path.empty()) {
        load_config(config_path);
    }
}

void RoadBoundarySpline::load_config(const std::string& config_path) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const std::exception& e) {
        std::cerr << "[RoadBoundarySpline] Failed to load: " << config_path
                  << " - " << e.what() << std::endl;
        return;
    }

    road_width_ = yaml_double(config, "road_width", 0.30);
    vehicle_half_width_ = yaml_double(config, "vehicle_half_width", 0.08);
    safety_margin_ = yaml_double(config, "safety_margin", 0.02);

    if (config["transform"]) {
        auto tf = config["transform"];
        tp_.origin_x = yaml_double(tf, "origin_x", -1.205);
        tp_.origin_y = yaml_double(tf, "origin_y", -0.83);
        if (tf["origin_heading_rad"]) {
            tp_.origin_heading_rad = tf["origin_heading_rad"].as<double>();
        } else {
            double deg = yaml_double(tf, "origin_heading_deg", -44.7);
            tp_.origin_heading_rad = -deg * M_PI / 180.0;
        }
    }

    // Load road segments
    segments_.clear();
    if (config["road_segments"]) {
        for (const auto& seg_node : config["road_segments"]) {
            RoadSegment seg;
            seg.name = yaml_str(seg_node, "name", "unnamed");
            seg.type = yaml_str(seg_node, "type", "spline");

            if (seg.type == "circular") {
                double cx = seg_node["center"]["x"].as<double>();
                double cy = seg_node["center"]["y"].as<double>();
                qlabs_to_map_2d(cx, cy, tp_, seg.center_x, seg.center_y);
                seg.radius = yaml_double(seg_node, "radius", 0.5);
                seg.width = yaml_double(seg_node, "width", 0.24);
            } else {
                // Spline segment
                if (seg_node["centerline"]) {
                    for (const auto& pt_node : seg_node["centerline"]) {
                        CenterlinePoint cp;
                        double qx = pt_node["x"].as<double>();
                        double qy = pt_node["y"].as<double>();
                        qlabs_to_map_2d(qx, qy, tp_, cp.x, cp.y);
                        cp.width_left = yaml_double(pt_node, "width_left", 0.24);
                        cp.width_right = yaml_double(pt_node, "width_right", 0.24);
                        seg.centerline.push_back(cp);
                    }
                }
            }
            segments_.push_back(std::move(seg));
        }
    }

    // Load traffic controls
    traffic_controls_.clear();
    if (config["traffic_controls"]) {
        for (const auto& tc_node : config["traffic_controls"]) {
            TrafficControl tc;
            tc.type = yaml_str(tc_node, "type", "stop_sign");
            tc.name = yaml_str(tc_node, "name", "unnamed");
            double qx = tc_node["position"]["x"].as<double>();
            double qy = tc_node["position"]["y"].as<double>();
            qlabs_to_map_2d(qx, qy, tp_, tc.x, tc.y);
            tc.stop_line_distance = yaml_double(tc_node, "stop_line_distance", 0.2);
            traffic_controls_.push_back(tc);
        }
    }

    // Load obstacle zones
    obstacle_zones_.clear();
    if (config["obstacle_zones"]) {
        for (const auto& oz_node : config["obstacle_zones"]) {
            ObstacleZone oz;
            oz.name = yaml_str(oz_node, "name", "unnamed");
            oz.type = yaml_str(oz_node, "type", "circle");
            double qx = oz_node["center"]["x"].as<double>();
            double qy = oz_node["center"]["y"].as<double>();
            qlabs_to_map_2d(qx, qy, tp_, oz.center_x, oz.center_y);
            oz.max_velocity = yaml_double(oz_node, "max_velocity", 0.4);
            if (oz.type == "circle") {
                oz.radius = yaml_double(oz_node, "radius", 0.5);
            } else if (oz.type == "rectangle") {
                oz.width = yaml_double(oz_node, "width", 1.0);
                oz.height = yaml_double(oz_node, "height", 1.0);
            }
            obstacle_zones_.push_back(oz);
        }
    }

    std::cout << "[RoadBoundarySpline] Loaded " << segments_.size() << " segments, "
              << traffic_controls_.size() << " traffic controls, "
              << obstacle_zones_.size() << " obstacle zones" << std::endl;
}

const RoadSegment* RoadBoundarySpline::get_active_segment(double x, double y) const {
    for (auto& seg : segments_) {
        if (seg.contains_point(x, y)) return &seg;
    }
    return nullptr;
}

BoundaryResult RoadBoundarySpline::get_boundary_constraints(
    double x, double y, double theta) const
{
    BoundaryResult br;
    const RoadSegment* seg = get_active_segment(x, y);

    if (!seg) {
        // No segment - use path tangent (theta)
        double dx_n = std::cos(theta);
        double dy_n = std::sin(theta);
        br.nx = dy_n;
        br.ny = -dx_n;
        double eff_w = road_width_ * 2.0 - vehicle_half_width_;
        double proj = br.nx * x + br.ny * y;
        br.b_left = proj + eff_w;
        br.b_right = -proj + eff_w;
        return br;
    }

    // Get YAML segment widths (accurate per-segment road width data)
    auto nr = seg->get_nearest_point_and_tangent(x, y);

    // Use the PATH tangent (theta) for the boundary normal direction, NOT the
    // coarse YAML centerline tangent. The YAML centerline has only 4-8 control
    // points per segment, so its tangent angle diverges from the smooth cubic
    // spline path on curves. Using the YAML tangent creates conflicting
    // constraints: the path says "curve left" but the boundary says "stay in
    // a corridor aligned to a different angle". This causes the solver to
    // fight itself and the vehicle fails to turn properly.
    double dx_n = std::cos(theta);
    double dy_n = std::sin(theta);
    br.nx = dy_n;
    br.ny = -dx_n;

    double eff_left = std::max(nr.w_left - vehicle_half_width_ - safety_margin_, 0.10);
    double eff_right = std::max(nr.w_right - vehicle_half_width_ - safety_margin_, 0.10);

    // Project the nearest YAML centerline point onto the path-tangent normal
    // to get boundary offsets relative to the actual path direction
    double center_proj = br.nx * nr.x + br.ny * nr.y;
    br.b_left = center_proj + eff_left;
    br.b_right = -center_proj + eff_right;

    return br;
}

BoundaryResult RoadBoundarySpline::get_boundary_constraints_from_path(
    double path_x, double path_y, double path_theta,
    double default_width) const
{
    BoundaryResult br;
    double dx_n = std::cos(path_theta);
    double dy_n = std::sin(path_theta);
    br.nx = dy_n;
    br.ny = -dx_n;

    double eff_w = std::max(default_width - vehicle_half_width_ - safety_margin_, 0.08);
    double proj = br.nx * path_x + br.ny * path_y;
    br.b_left = proj + eff_w;
    br.b_right = -proj + eff_w;
    return br;
}

double RoadBoundarySpline::get_velocity_limit(double x, double y) const {
    double min_v = 0.6;
    for (auto& zone : obstacle_zones_) {
        if (zone.contains_point(x, y)) {
            min_v = std::min(min_v, zone.max_velocity);
        }
    }
    return min_v;
}

std::vector<RoadBoundarySpline::NearbyControl>
RoadBoundarySpline::get_nearby_traffic_controls(
    double x, double y, double theta, double max_distance) const
{
    std::vector<NearbyControl> results;
    for (auto& tc : traffic_controls_) {
        double dx = tc.x - x, dy = tc.y - y;
        double dist = std::hypot(dx, dy);
        if (dist > max_distance) continue;

        // Check if approaching (within 90 deg of heading)
        double angle_to = std::atan2(dy, dx);
        double diff = std::abs(normalize_angle(angle_to - theta));
        if (diff > M_PI / 2.0) continue;

        results.push_back({&tc, dist});
    }
    std::sort(results.begin(), results.end(),
              [](auto& a, auto& b) { return a.distance < b.distance; });
    return results;
}

}  // namespace acc
