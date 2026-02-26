/**
 * Diagnose path tangent mismatches at mission waypoint transitions.
 *
 * This checks whether the vehicle arrives at each waypoint with a heading
 * that matches the start tangent of the next leg's path. Large mismatches
 * here explain deployment swerving — the solver has to correct a large
 * heading error at the start of each leg.
 */

#include <cstdio>
#include <cmath>
#include <vector>

#include "cubic_spline_path.h"
#include "road_graph.h"
#include "coordinate_transform.h"

int main() {
    std::printf("=== Path Tangent Analysis at Mission Waypoints ===\n\n");

    acc::RoadGraph road_graph(0.001);
    acc::TransformParams tp;

    struct LegInfo {
        std::string name;
        double start_x, start_y;
    };

    std::vector<LegInfo> legs = {
        {"hub_to_pickup", acc::HUB_X, acc::HUB_Y},
        {"pickup_to_dropoff", acc::PICKUP_X, acc::PICKUP_Y},
        {"dropoff_to_hub", acc::DROPOFF_X, acc::DROPOFF_Y},
    };

    struct LegAnalysis {
        std::string name;
        double start_tangent_map;  // start tangent in map frame (rad)
        double end_tangent_map;    // end tangent in map frame (rad)
        double start_tangent_ql;   // start tangent in QLabs frame (rad)
        double end_tangent_ql;     // end tangent in QLabs frame (rad)
        double start_x, start_y;  // map frame start
        double end_x, end_y;      // map frame end
        double length;
    };

    std::vector<LegAnalysis> analyses;

    for (auto& leg : legs) {
        auto route = road_graph.plan_path_for_mission_leg(leg.name,
            leg.start_x, leg.start_y);
        if (!route) {
            std::printf("ERROR: Failed to plan %s\n", leg.name.c_str());
            return 1;
        }

        // Transform to map frame
        std::vector<double> mx, my;
        acc::qlabs_path_to_map(route->first, route->second, tp, mx, my);

        // Build spline
        acc::CubicSplinePath spline;
        spline.build(mx, my, true);
        double total_len = spline.total_length();

        LegAnalysis la;
        la.name = leg.name;
        la.start_tangent_map = spline.get_tangent(0.0);
        la.end_tangent_map = spline.get_tangent(total_len - 0.001);
        la.start_tangent_ql = la.start_tangent_map - tp.origin_heading_rad;
        la.end_tangent_ql = la.end_tangent_map - tp.origin_heading_rad;
        spline.get_position(0.0, la.start_x, la.start_y);
        spline.get_position(total_len - 0.001, la.end_x, la.end_y);
        la.length = total_len;

        analyses.push_back(la);

        std::printf("%s:\n", leg.name.c_str());
        std::printf("  Length: %.3f m\n", total_len);
        std::printf("  Start: (%.4f, %.4f) tangent=%.1f deg (map)\n",
            la.start_x, la.start_y, la.start_tangent_map * 180.0 / M_PI);
        std::printf("  End:   (%.4f, %.4f) tangent=%.1f deg (map)\n",
            la.end_x, la.end_y, la.end_tangent_map * 180.0 / M_PI);
        std::printf("\n");
    }

    // Check transitions
    std::printf("=== Transition Analysis ===\n\n");

    for (size_t i = 0; i < analyses.size(); i++) {
        size_t next = (i + 1) % analyses.size();
        auto& cur = analyses[i];
        auto& nxt = analyses[next];

        double pos_gap = std::hypot(cur.end_x - nxt.start_x,
                                     cur.end_y - nxt.start_y);
        double heading_gap = cur.end_tangent_map - nxt.start_tangent_map;
        while (heading_gap > M_PI) heading_gap -= 2*M_PI;
        while (heading_gap < -M_PI) heading_gap += 2*M_PI;

        std::printf("%s → %s:\n", cur.name.c_str(), nxt.name.c_str());
        std::printf("  Position gap: %.3f m\n", pos_gap);
        std::printf("  Heading gap:  %.1f deg\n",
            heading_gap * 180.0 / M_PI);
        std::printf("  (End tangent: %.1f°, Next start tangent: %.1f°)\n",
            cur.end_tangent_map * 180.0 / M_PI,
            nxt.start_tangent_map * 180.0 / M_PI);

        if (std::abs(heading_gap) > 10.0 * M_PI / 180.0) {
            std::printf("  *** WARNING: Heading gap > 10 deg! ***\n");
        }
        std::printf("\n");
    }

    // Also check: what heading does the vehicle have when it arrives at each waypoint?
    // The loop path passes through each waypoint at a particular angle.
    // If the vehicle follows the path perfectly, it arrives with the END tangent of the current leg.
    std::printf("=== Expected Vehicle Heading at Waypoints ===\n\n");
    std::printf("At Pickup (end of hub→pickup):\n");
    std::printf("  Vehicle heading (map): %.1f deg\n",
        analyses[0].end_tangent_map * 180.0 / M_PI);
    std::printf("  Next path start (map): %.1f deg\n",
        analyses[1].start_tangent_map * 180.0 / M_PI);
    double pickup_gap = analyses[0].end_tangent_map - analyses[1].start_tangent_map;
    while (pickup_gap > M_PI) pickup_gap -= 2*M_PI;
    while (pickup_gap < -M_PI) pickup_gap += 2*M_PI;
    std::printf("  Heading mismatch: %.1f deg\n\n", pickup_gap * 180.0 / M_PI);

    std::printf("At Dropoff (end of pickup→dropoff):\n");
    std::printf("  Vehicle heading (map): %.1f deg\n",
        analyses[1].end_tangent_map * 180.0 / M_PI);
    std::printf("  Next path start (map): %.1f deg\n",
        analyses[2].start_tangent_map * 180.0 / M_PI);
    double dropoff_gap = analyses[1].end_tangent_map - analyses[2].start_tangent_map;
    while (dropoff_gap > M_PI) dropoff_gap -= 2*M_PI;
    while (dropoff_gap < -M_PI) dropoff_gap += 2*M_PI;
    std::printf("  Heading mismatch: %.1f deg\n\n", dropoff_gap * 180.0 / M_PI);

    std::printf("At Hub (end of dropoff→hub):\n");
    std::printf("  Vehicle heading (map): %.1f deg\n",
        analyses[2].end_tangent_map * 180.0 / M_PI);
    std::printf("  Next path start (map): %.1f deg\n",
        analyses[0].start_tangent_map * 180.0 / M_PI);
    double hub_gap = analyses[2].end_tangent_map - analyses[0].start_tangent_map;
    while (hub_gap > M_PI) hub_gap -= 2*M_PI;
    while (hub_gap < -M_PI) hub_gap += 2*M_PI;
    std::printf("  Heading mismatch: %.1f deg\n", hub_gap * 180.0 / M_PI);

    return 0;
}
