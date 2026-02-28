/**
 * Road network path planner for SDCS competition track.
 *
 * Ported from road_graph.py: SDCSRoadMap, SCSPath, A*, RoadGraph.
 * Uses Straight-Curve-Straight path generation with A* graph search.
 *
 * All coordinates are in QLabs world frame (meters).
 */

#pragma once

#include <cmath>
#include <memory>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace acc {

// -------------------------------------------------------------------------
// SCSPath - Straight-Curve-Straight path generation
// -------------------------------------------------------------------------

/// Result of SCSPath computation
struct SCSResult {
    std::vector<double> x, y;  // waypoint coordinates
    double length = 0.0;
    bool valid = false;
};

/// Calculate SCS path between two poses with at most one turn.
/// startPose/endPose: {x, y, theta}
SCSResult SCSPath(const double start[3], const double end[3],
                  double radius, double stepSize = 0.01);

// -------------------------------------------------------------------------
// Road map graph structures
// -------------------------------------------------------------------------

struct RoadMapNode {
    double pose[3];  // x, y, theta
    int id;
    std::vector<int> out_edges;  // indices into RoadMap::edges
    std::vector<int> in_edges;
};

struct RoadMapEdge {
    int from_node, to_node;
    std::vector<double> wp_x, wp_y;  // waypoints
    double length = 0.0;
};

class RoadMap {
public:
    void add_node(double x, double y, double theta);
    void add_edge(int from_idx, int to_idx, double radius);

    /// A* shortest path from startNode to goalNode.
    /// Returns path as (x[], y[]) or nullopt.
    std::optional<std::pair<std::vector<double>, std::vector<double>>>
    find_shortest_path(int start_idx, int goal_idx) const;

    /// Generate path through a sequence of node indices.
    std::optional<std::pair<std::vector<double>, std::vector<double>>>
    generate_path(const std::vector<int>& node_sequence) const;

    int num_nodes() const { return static_cast<int>(nodes_.size()); }

protected:
    std::vector<RoadMapNode> nodes_;
    std::vector<RoadMapEdge> edges_;
};

// -------------------------------------------------------------------------
// SDCSRoadMap - the competition track road network
// -------------------------------------------------------------------------

class SDCSRoadMap : public RoadMap {
public:
    SDCSRoadMap(bool left_hand_traffic = false, bool use_small_map = false);
};

// -------------------------------------------------------------------------
// RoadGraph - high-level interface for mission_manager
// -------------------------------------------------------------------------

/// Mission locations (QLabs world coordinates)
constexpr double HUB_X = -1.205, HUB_Y = -0.83;
constexpr double PICKUP_X = 0.125, PICKUP_Y = 4.395;
constexpr double DROPOFF_X = -0.905, DROPOFF_Y = 0.800;

/// Resample a 2D path to uniform spacing using natural cubic spline interpolation.
/// Reference 2025 uses scipy splprep/splev; this is a C++ equivalent.
void resample_path(std::vector<double>& x, std::vector<double>& y,
                   double spacing = 0.001);

class RoadGraph {
public:
    explicit RoadGraph(double ds = 0.001);

    /// Get pre-computed route waypoints. Returns (x[], y[]) or nullopt.
    std::optional<std::pair<std::vector<double>, std::vector<double>>>
    get_route(const std::string& route_name) const;

    /// Get route names.
    std::vector<std::string> get_route_names() const;

    /// Determine which route to use based on mission leg labels.
    std::string get_route_for_leg(const std::string& start_label,
                                  const std::string& goal_label) const;

    /// Get path for a specific mission leg, starting from current position.
    /// Returns (x[], y[]) in QLabs frame, or nullopt.
    std::optional<std::pair<std::vector<double>, std::vector<double>>>
    plan_path_for_mission_leg(const std::string& route_name,
                              double cur_x, double cur_y,
                              double cur_yaw = std::numeric_limits<double>::quiet_NaN()) const;

    /// Find the best pre-computed route from start to goal (QLabs coords).
    /// Returns (x[], y[]) or nullopt.
    std::optional<std::pair<std::vector<double>, std::vector<double>>>
    plan_path(double start_x, double start_y,
              double goal_x, double goal_y) const;

    /// Plan a path from current position to goal using the best matching route.
    /// Returns (x[], y[]) or nullopt.
    std::optional<std::pair<std::vector<double>, std::vector<double>>>
    plan_path_from_pose(double cur_x, double cur_y,
                        double goal_x, double goal_y) const;

    /// Get the full loop path (all waypoints, QLabs frame).
    const std::vector<double>& loop_x() const { return loop_x_; }
    const std::vector<double>& loop_y() const { return loop_y_; }

    /// Get waypoint indices for key locations on the loop.
    int pickup_index() const { return pickup_idx_; }
    int dropoff_index() const { return dropoff_idx_; }
    int hub_index() const { return hub_idx_; }

private:
    double ds_;
    SDCSRoadMap roadmap_;

    // Full loop path (reference 2025 style: single [24,20,9,10] path)
    std::vector<double> loop_x_, loop_y_;
    int pickup_idx_ = -1;   // waypoint index closest to pickup
    int dropoff_idx_ = -1;  // waypoint index closest to dropoff
    int hub_idx_ = -1;      // waypoint index closest to hub (end of loop)

    struct Route {
        std::vector<double> x, y;
    };
    std::unordered_map<std::string, Route> routes_;

    static int find_closest_idx(const std::vector<double>& wx,
                                const std::vector<double>& wy,
                                double px, double py);

    /// Find the first local minimum within threshold of (px,py).
    /// Avoids picking a later pass when the path visits the same area twice.
    static int find_first_local_min(const std::vector<double>& wx,
                                    const std::vector<double>& wy,
                                    double px, double py,
                                    double threshold = 0.5,
                                    int start_from = 0);
};

}  // namespace acc
