/**
 * Standalone compilation and logic test for all ported C++ code.
 *
 * Tests:
 * 1. coordinate_transform.h - QLabs <-> map transforms
 * 2. cubic_spline_path.h    - Spline building and querying
 * 3. road_graph.h/.cpp      - SCSPath, A*, SDCSRoadMap, RoadGraph
 * 4. road_boundaries.h/.cpp - Boundary constraints, obstacle zones
 * 5. Integration: types and signatures match between components
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

// Our headers
#include "coordinate_transform.h"
#include "cubic_spline_path.h"
#include "yaml_config.h"
#include "road_graph.h"
#include "road_boundaries.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { std::printf("  TEST: %-50s ", name); } while(0)

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
    do { if (!(cond)) { FAIL(msg); return; } } while(0)

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
// 1. Coordinate Transform Tests
// =========================================================================
void test_coordinate_transform_roundtrip() {
    TEST("coord: roundtrip qlabs->map->qlabs");
    acc::TransformParams tp;

    double qx = 0.5, qy = 1.0, qtheta = 0.3;
    double mx, my, mtheta;
    acc::qlabs_to_map(qx, qy, qtheta, tp, mx, my, mtheta);

    double qx2, qy2, qtheta2;
    acc::map_to_qlabs(mx, my, mtheta, tp, qx2, qy2, qtheta2);

    ASSERT_NEAR(qx, qx2, 1e-10, "x roundtrip");
    ASSERT_NEAR(qy, qy2, 1e-10, "y roundtrip");
    ASSERT_NEAR(qtheta, qtheta2, 1e-10, "theta roundtrip");
    PASS();
}

void test_coordinate_transform_origin() {
    TEST("coord: origin maps to (0,0)");
    acc::TransformParams tp;

    double mx, my, mtheta;
    acc::qlabs_to_map(tp.origin_x, tp.origin_y, 0.0, tp, mx, my, mtheta);

    ASSERT_NEAR(mx, 0.0, 1e-10, "origin x should be 0");
    ASSERT_NEAR(my, 0.0, 1e-10, "origin y should be 0");
    PASS();
}

void test_coordinate_transform_2d() {
    TEST("coord: 2D variant matches 3D");
    acc::TransformParams tp;

    double qx = 0.125, qy = 4.395;
    double mx1, my1, mtheta1;
    acc::qlabs_to_map(qx, qy, 0.0, tp, mx1, my1, mtheta1);

    double mx2, my2;
    acc::qlabs_to_map_2d(qx, qy, tp, mx2, my2);

    ASSERT_NEAR(mx1, mx2, 1e-10, "2D x mismatch");
    ASSERT_NEAR(my1, my2, 1e-10, "2D y mismatch");
    PASS();
}

void test_coordinate_transform_batch() {
    TEST("coord: batch path transform");
    acc::TransformParams tp;

    std::vector<double> qx = {0.0, 1.0, 2.0};
    std::vector<double> qy = {0.0, 1.0, 0.5};
    std::vector<double> mx, my;
    acc::qlabs_path_to_map(qx, qy, tp, mx, my);

    ASSERT_TRUE(mx.size() == 3, "batch size mismatch");

    // Verify each point matches individual transform
    for (size_t i = 0; i < qx.size(); i++) {
        double mx2, my2;
        acc::qlabs_to_map_2d(qx[i], qy[i], tp, mx2, my2);
        ASSERT_NEAR(mx[i], mx2, 1e-10, "batch x");
        ASSERT_NEAR(my[i], my2, 1e-10, "batch y");
    }
    PASS();
}

void test_normalize_angle() {
    TEST("coord: normalize_angle");
    ASSERT_NEAR(acc::normalize_angle(0.0), 0.0, 1e-10, "0");
    ASSERT_NEAR(acc::normalize_angle(M_PI), M_PI, 1e-10, "PI");
    ASSERT_NEAR(acc::normalize_angle(-M_PI), -M_PI, 1e-10, "-PI");
    ASSERT_NEAR(acc::normalize_angle(3*M_PI), M_PI, 1e-10, "3PI");
    ASSERT_NEAR(acc::normalize_angle(-3*M_PI), -M_PI, 1e-10, "-3PI");
    PASS();
}

// =========================================================================
// 2. Cubic Spline Path Tests
// =========================================================================
void test_spline_build_line() {
    TEST("spline: build straight line");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    ASSERT_NEAR(sp.total_length(), 4.0, 0.1, "total length ~4");
    PASS();
}

void test_spline_position_interpolation() {
    TEST("spline: position interpolation on line");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    double px, py;
    sp.get_position(2.0, px, py);
    ASSERT_NEAR(px, 2.0, 0.1, "x at s=2");
    ASSERT_NEAR(py, 0.0, 0.1, "y at s=2");
    PASS();
}

void test_spline_tangent() {
    TEST("spline: tangent on straight line");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    double t = sp.get_tangent(2.0);
    ASSERT_NEAR(t, 0.0, 0.1, "tangent ~0 (horizontal)");
    PASS();
}

void test_spline_curvature_line() {
    TEST("spline: curvature on straight line ~0");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    double k = sp.get_curvature(2.0);
    ASSERT_NEAR(k, 0.0, 0.05, "curvature ~0 for line");
    PASS();
}

void test_spline_circle() {
    TEST("spline: curvature on quarter circle ~1/R");
    acc::CubicSplinePath sp;
    double R = 1.0;
    int N = 20;
    std::vector<double> x, y;
    for (int i = 0; i <= N; i++) {
        double theta = M_PI / 2.0 * i / N;
        x.push_back(R * std::cos(theta));
        y.push_back(R * std::sin(theta));
    }
    sp.build(x, y);

    double k = sp.get_curvature(sp.total_length() / 2.0);
    ASSERT_NEAR(std::abs(k), 1.0 / R, 0.15, "curvature ~1/R");
    PASS();
}

void test_spline_closest_progress() {
    TEST("spline: find_closest_progress");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    double s = sp.find_closest_progress(2.0, 0.5);
    ASSERT_NEAR(s, 2.0, 0.2, "closest progress to (2,0.5) ~2");
    PASS();
}

void test_spline_contouring_errors() {
    TEST("spline: contouring/lag errors");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    double e_c, e_l;
    sp.compute_contouring_errors(2.0, 0.5, 2.0, e_c, e_l);
    ASSERT_NEAR(std::abs(e_c), 0.5, 0.15, "contouring error ~0.5");
    ASSERT_NEAR(std::abs(e_l), 0.0, 0.15, "lag error ~0");
    PASS();
}

void test_spline_path_reference() {
    TEST("spline: get_path_reference");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    double ref_x, ref_y, cos_t, sin_t;
    sp.get_path_reference(2.0, ref_x, ref_y, cos_t, sin_t);
    ASSERT_NEAR(ref_x, 2.0, 0.1, "ref x ~2");
    ASSERT_NEAR(ref_y, 0.0, 0.1, "ref y ~0");
    ASSERT_NEAR(cos_t, 1.0, 0.1, "cos_theta ~1 (horizontal)");
    ASSERT_NEAR(sin_t, 0.0, 0.1, "sin_theta ~0");
    PASS();
}

void test_spline_build_from_pairs() {
    TEST("spline: build from pairs");
    acc::CubicSplinePath sp;
    std::vector<std::pair<double,double>> pts = {
        {0,0}, {1,1}, {2,0}, {3,1}, {4,0}
    };
    sp.build(pts);
    ASSERT_TRUE(sp.total_length() > 4.0, "zigzag path > 4");

    double px, py;
    sp.get_position(0.0, px, py);
    ASSERT_NEAR(px, 0.0, 0.01, "start x = 0");
    ASSERT_NEAR(py, 0.0, 0.01, "start y = 0");
    PASS();
}

// =========================================================================
// 3. Road Graph Tests
// =========================================================================
void test_scs_path_straight() {
    TEST("road_graph: SCSPath straight line");
    double start[3] = {0, 0, 0};
    double end[3] = {2, 0, 0};
    auto result = acc::SCSPath(start, end, 0.3);
    ASSERT_TRUE(result.valid, "straight SCS should succeed");
    ASSERT_TRUE(result.x.size() >= 2, "should have points");
    PASS();
}

void test_scs_path_turn() {
    TEST("road_graph: SCSPath with turn");
    double start[3] = {0, 0, 0};
    double end[3] = {1, 1, M_PI/2};
    auto result = acc::SCSPath(start, end, 0.3);
    ASSERT_TRUE(result.valid, "turning SCS should succeed");
    ASSERT_TRUE(result.x.size() >= 5, "should have multiple points");
    PASS();
}

void test_scs_path_zero_radius() {
    TEST("road_graph: SCSPath zero radius (straight)");
    double start[3] = {0, 0, 0};
    double end[3] = {1, 0, 0};
    auto result = acc::SCSPath(start, end, 0.0);
    ASSERT_TRUE(result.valid, "zero radius should succeed");
    ASSERT_TRUE(result.length > 0.9 && result.length < 1.1, "length ~1");
    PASS();
}

void test_sdcs_roadmap_construction() {
    TEST("road_graph: SDCSRoadMap has 25 nodes");
    acc::SDCSRoadMap sdcs;
    ASSERT_TRUE(sdcs.num_nodes() == 25, "should have 25 nodes");
    PASS();
}

void test_sdcs_roadmap_pathfinding() {
    TEST("road_graph: SDCSRoadMap path from node 0 to 20");
    acc::SDCSRoadMap sdcs;
    auto path = sdcs.find_shortest_path(0, 20);
    ASSERT_TRUE(path.has_value(), "should find a path from 0 to 20");
    auto& [px, py] = *path;
    ASSERT_TRUE(!px.empty(), "path x should not be empty");
    ASSERT_TRUE(px.size() == py.size(), "x and y sizes should match");
    PASS();
}

void test_roadgraph_construction() {
    TEST("road_graph: RoadGraph default construction");
    acc::RoadGraph rg(0.01);
    auto names = rg.get_route_names();
    ASSERT_TRUE(names.size() == 3, "should have 3 routes");
    PASS();
}

void test_roadgraph_route_names() {
    TEST("road_graph: RoadGraph route leg lookup");
    acc::RoadGraph rg(0.01);
    ASSERT_TRUE(rg.get_route_for_leg("hub", "pickup") == "hub_to_pickup",
                "hub->pickup");
    ASSERT_TRUE(rg.get_route_for_leg("pickup", "dropoff") == "pickup_to_dropoff",
                "pickup->dropoff");
    ASSERT_TRUE(rg.get_route_for_leg("dropoff", "hub") == "dropoff_to_hub",
                "dropoff->hub");
    PASS();
}

void test_roadgraph_plan_path() {
    TEST("road_graph: RoadGraph plan hub_to_pickup path");
    acc::RoadGraph rg(0.01);
    auto path = rg.plan_path_for_mission_leg("hub_to_pickup", -1.205, -0.83);
    ASSERT_TRUE(path.has_value(), "should produce waypoints");
    auto& [px, py] = *path;
    ASSERT_TRUE(px.size() > 5, "path should have many waypoints");
    PASS();
}

void test_roadgraph_get_route() {
    TEST("road_graph: RoadGraph get_route");
    acc::RoadGraph rg(0.01);
    auto r1 = rg.get_route("hub_to_pickup");
    auto r2 = rg.get_route("pickup_to_dropoff");
    auto r3 = rg.get_route("dropoff_to_hub");
    auto r4 = rg.get_route("nonexistent");

    ASSERT_TRUE(r1.has_value(), "hub_to_pickup exists");
    ASSERT_TRUE(r2.has_value(), "pickup_to_dropoff exists");
    ASSERT_TRUE(r3.has_value(), "dropoff_to_hub exists");
    ASSERT_TRUE(!r4.has_value(), "nonexistent should be nullopt");
    PASS();
}

void test_roadmap_generate_path() {
    TEST("road_graph: generate_path through nodes");
    acc::SDCSRoadMap sdcs;
    // Simple: 0 -> 2 -> 4
    auto path = sdcs.generate_path({0, 2, 4});
    ASSERT_TRUE(path.has_value(), "should generate path");
    auto& [px, py] = *path;
    ASSERT_TRUE(px.size() > 3, "should have interpolated points");
    PASS();
}

// =========================================================================
// 4. Road Boundaries Tests
// =========================================================================
void test_road_segment_circular_contains() {
    TEST("boundary: circular segment contains_point");
    acc::RoadSegment seg;
    seg.type = "circular";
    seg.center_x = 1.0;
    seg.center_y = 1.0;
    seg.radius = 0.5;
    seg.width = 0.24;

    ASSERT_TRUE(seg.contains_point(1.0, 1.0), "center inside");
    ASSERT_TRUE(seg.contains_point(1.5, 1.0), "on radius inside");
    ASSERT_TRUE(!seg.contains_point(5.0, 5.0), "far point outside");
    PASS();
}

void test_road_segment_circular_nearest() {
    TEST("boundary: circular segment nearest point");
    acc::RoadSegment seg;
    seg.type = "circular";
    seg.center_x = 0.0;
    seg.center_y = 0.0;
    seg.radius = 1.0;
    seg.width = 0.24;

    auto nr = seg.get_nearest_point_and_tangent(2.0, 0.0);
    ASSERT_NEAR(nr.x, 1.0, 0.01, "nearest x = 1");
    ASSERT_NEAR(nr.y, 0.0, 0.01, "nearest y = 0");
    PASS();
}

void test_road_segment_spline_contains() {
    TEST("boundary: spline segment contains_point");
    acc::RoadSegment seg;
    seg.type = "spline";
    seg.centerline = {
        {0.0, 0.0, 0.3, 0.3},
        {1.0, 0.0, 0.3, 0.3},
        {2.0, 0.0, 0.3, 0.3},
    };

    ASSERT_TRUE(seg.contains_point(1.0, 0.0), "on centerline");
    ASSERT_TRUE(seg.contains_point(1.0, 0.2), "within width");
    ASSERT_TRUE(!seg.contains_point(1.0, 5.0), "far away");
    PASS();
}

void test_road_segment_spline_nearest() {
    TEST("boundary: spline segment nearest + tangent");
    acc::RoadSegment seg;
    seg.type = "spline";
    seg.centerline = {
        {0.0, 0.0, 0.3, 0.3},
        {1.0, 0.0, 0.3, 0.3},
        {2.0, 0.0, 0.3, 0.3},
    };

    auto nr = seg.get_nearest_point_and_tangent(1.0, 0.5);
    ASSERT_NEAR(nr.x, 1.0, 0.01, "nearest x = 1");
    ASSERT_NEAR(nr.y, 0.0, 0.01, "nearest y = 0");
    ASSERT_NEAR(nr.tangent, 0.0, 0.01, "tangent = 0 (horizontal)");
    PASS();
}

void test_obstacle_zone_circle() {
    TEST("boundary: circle obstacle zone");
    acc::ObstacleZone oz;
    oz.type = "circle";
    oz.center_x = 1.0;
    oz.center_y = 1.0;
    oz.radius = 0.5;

    ASSERT_TRUE(oz.contains_point(1.0, 1.0), "center inside");
    ASSERT_TRUE(oz.contains_point(1.3, 1.0), "near center inside");
    ASSERT_TRUE(!oz.contains_point(2.0, 2.0), "outside");
    PASS();
}

void test_obstacle_zone_rectangle() {
    TEST("boundary: rectangle obstacle zone");
    acc::ObstacleZone oz;
    oz.type = "rectangle";
    oz.center_x = 0.0;
    oz.center_y = 0.0;
    oz.width = 2.0;
    oz.height = 1.0;

    ASSERT_TRUE(oz.contains_point(0.0, 0.0), "center inside");
    ASSERT_TRUE(oz.contains_point(0.9, 0.4), "corner-ish inside");
    ASSERT_TRUE(!oz.contains_point(1.5, 0.0), "outside width");
    ASSERT_TRUE(!oz.contains_point(0.0, 0.8), "outside height");
    PASS();
}

void test_road_boundary_spline_no_config() {
    TEST("boundary: RoadBoundarySpline default (no config)");
    acc::RoadBoundarySpline rbs;
    auto br = rbs.get_boundary_constraints(0.0, 0.0, 0.0);
    ASSERT_TRUE(std::isfinite(br.nx), "nx finite");
    ASSERT_TRUE(std::isfinite(br.ny), "ny finite");
    ASSERT_TRUE(std::isfinite(br.b_left), "b_left finite");
    ASSERT_TRUE(std::isfinite(br.b_right), "b_right finite");
    PASS();
}

void test_road_boundary_path_constraints() {
    TEST("boundary: path-based boundary constraints");
    acc::RoadBoundarySpline rbs;
    auto br = rbs.get_boundary_constraints_from_path(1.0, 0.0, 0.0, 0.25);
    ASSERT_TRUE(std::isfinite(br.nx), "nx finite");
    ASSERT_TRUE(std::isfinite(br.b_left), "b_left finite");
    // Normal perpendicular to heading=0: nx=sin(0)=0, ny=-cos(0)=-1
    ASSERT_NEAR(br.nx, 0.0, 0.01, "nx ~0 for heading=0");
    PASS();
}

void test_road_boundary_velocity_limit() {
    TEST("boundary: velocity limit with no zones");
    acc::RoadBoundarySpline rbs;
    double v = rbs.get_velocity_limit(0.0, 0.0);
    ASSERT_NEAR(v, 0.6, 0.01, "default velocity limit = 0.6");
    PASS();
}

void test_road_boundary_traffic_controls() {
    TEST("boundary: nearby traffic controls (empty)");
    acc::RoadBoundarySpline rbs;
    auto tc = rbs.get_nearby_traffic_controls(0.0, 0.0, 0.0);
    ASSERT_TRUE(tc.empty(), "no traffic controls by default");
    PASS();
}

void test_boundary_normal_unit_vector() {
    TEST("boundary: normal is unit vector at all headings");
    acc::RoadBoundarySpline rbs;
    for (double theta = -M_PI; theta <= M_PI; theta += M_PI / 4) {
        auto br = rbs.get_boundary_constraints_from_path(0.0, 0.0, theta, 0.25);
        double n_len = std::hypot(br.nx, br.ny);
        ASSERT_NEAR(n_len, 1.0, 0.01, "normal should be unit vector");
        ASSERT_TRUE(br.b_left > 0, "b_left positive");
        ASSERT_TRUE(br.b_right > 0, "b_right positive");
    }
    PASS();
}

// =========================================================================
// 5. Integration Tests
// =========================================================================
void test_integration_spline_with_roadgraph() {
    TEST("integration: CubicSplinePath from RoadGraph waypoints");
    acc::RoadGraph rg(0.01);

    auto path = rg.plan_path_for_mission_leg("hub_to_pickup", -1.205, -0.83);
    ASSERT_TRUE(path.has_value(), "path should exist");
    auto& [wx, wy] = *path;
    ASSERT_TRUE(wx.size() >= 4, "need >= 4 points for spline");

    acc::CubicSplinePath sp;
    sp.build(wx, wy);

    ASSERT_TRUE(sp.total_length() > 0.5, "path should have positive length");

    // Query all along the path
    for (double s = 0; s < sp.total_length(); s += sp.total_length() / 10.0) {
        double px, py;
        sp.get_position(s, px, py);
        ASSERT_TRUE(std::isfinite(px) && std::isfinite(py), "position finite");
        double k = sp.get_curvature(s);
        ASSERT_TRUE(std::isfinite(k), "curvature finite");
    }
    PASS();
}

void test_integration_all_mission_legs() {
    TEST("integration: all 3 mission legs produce valid splines");
    acc::RoadGraph rg(0.01);

    const char* legs[] = {"hub_to_pickup", "pickup_to_dropoff", "dropoff_to_hub"};
    for (auto leg : legs) {
        auto path = rg.plan_path_for_mission_leg(leg, -1.205, -0.83);
        ASSERT_TRUE(path.has_value(), "path should not be empty");
        auto& [wx, wy] = *path;

        if (wx.size() >= 4) {
            acc::CubicSplinePath sp;
            sp.build(wx, wy);
            ASSERT_TRUE(sp.total_length() > 0.1, "spline should have length");
        }
    }
    PASS();
}

void test_integration_transform_with_roadgraph() {
    TEST("integration: RoadGraph paths transform correctly");
    acc::TransformParams tp;

    // Hub in QLabs: (-1.205, -0.83) -> should map to (0, 0)
    double mx, my;
    acc::qlabs_to_map_2d(-1.205, -0.83, tp, mx, my);
    ASSERT_NEAR(mx, 0.0, 0.01, "hub maps to x=0");
    ASSERT_NEAR(my, 0.0, 0.01, "hub maps to y=0");

    // RoadGraph paths are in QLabs frame - transform to map
    acc::RoadGraph rg(0.01);
    auto path = rg.get_route("hub_to_pickup");
    ASSERT_TRUE(path.has_value(), "route exists");
    auto& [qx, qy] = *path;

    std::vector<double> map_x, map_y;
    acc::qlabs_path_to_map(qx, qy, tp, map_x, map_y);
    ASSERT_TRUE(map_x.size() == qx.size(), "transformed size matches");
    ASSERT_TRUE(std::isfinite(map_x.front()) && std::isfinite(map_y.front()),
                "first point finite");
    PASS();
}

void test_integration_boundary_constraints_on_path() {
    TEST("integration: boundary constraints along transformed path");
    acc::TransformParams tp;
    acc::RoadGraph rg(0.01);
    acc::RoadBoundarySpline rbs;  // no YAML config, uses heading fallback

    auto route = rg.get_route("hub_to_pickup");
    ASSERT_TRUE(route.has_value(), "route exists");
    auto& [qx, qy] = *route;

    std::vector<double> mx, my;
    acc::qlabs_path_to_map(qx, qy, tp, mx, my);

    // Build spline from transformed path
    acc::CubicSplinePath sp;
    sp.build(mx, my);

    // Get boundary constraints at several points along the path
    for (double s = 0; s < sp.total_length(); s += sp.total_length() / 5.0) {
        double px, py;
        sp.get_position(s, px, py);
        double theta = sp.get_tangent(s);
        auto br = rbs.get_boundary_constraints(px, py, theta);
        ASSERT_TRUE(std::isfinite(br.nx) && std::isfinite(br.ny), "normal finite");
        ASSERT_TRUE(std::isfinite(br.b_left) && std::isfinite(br.b_right), "bounds finite");
    }
    PASS();
}

void test_integration_spline_errors_along_path() {
    TEST("integration: contouring errors near path are small");
    acc::RoadGraph rg(0.01);
    auto path = rg.plan_path_for_mission_leg("dropoff_to_hub", -0.905, 0.800);
    ASSERT_TRUE(path.has_value(), "path exists");
    auto& [wx, wy] = *path;

    acc::CubicSplinePath sp;
    sp.build(wx, wy);

    // Points on the path should have small contouring errors
    for (double s = 0.1; s < sp.total_length() - 0.1; s += sp.total_length() / 10.0) {
        double px, py;
        sp.get_position(s, px, py);
        double e_c, e_l;
        sp.compute_contouring_errors(px, py, s, e_c, e_l);
        ASSERT_TRUE(std::abs(e_c) < 0.01, "contouring error should be ~0 on path");
        ASSERT_TRUE(std::abs(e_l) < 0.01, "lag error should be ~0 on path");
    }
    PASS();
}

// =========================================================================
// 6. Path Direction Tests (regression for southward arc bug)
// =========================================================================
void test_hub_to_pickup_path_heads_northeast() {
    TEST("path direction: hub_to_pickup first 10 waypoints go NE");
    acc::RoadGraph rg(0.01);
    auto path = rg.plan_path_for_mission_leg("hub_to_pickup", -1.205, -0.83);
    ASSERT_TRUE(path.has_value(), "path should exist");
    auto& [wx, wy] = *path;

    // Transform to map frame
    acc::TransformParams tp;
    std::vector<double> mx, my;
    acc::qlabs_path_to_map(wx, wy, tp, mx, my);

    // First 20 waypoints should NOT go far south (negative Y in map)
    // Before fix: Y went to -0.79. After fix: should stay > -0.15
    for (size_t i = 0; i < std::min(mx.size(), size_t(20)); i++) {
        ASSERT_TRUE(my[i] > -0.15,
            "early waypoints should not go south (was the 270deg arc bug)");
    }

    // Path should head northeast (positive X and positive Y overall)
    if (mx.size() > 50) {
        double dx = mx[50] - mx[0];
        double dy = my[50] - my[0];
        ASSERT_TRUE(dx > 0.0, "path should advance in +X direction");
        ASSERT_TRUE(dy > -0.2, "path should not go far south");
    }
    PASS();
}

void test_hub_to_pickup_initial_tangent() {
    TEST("path direction: initial tangent angle is north of east");
    acc::RoadGraph rg(0.01);
    auto path = rg.plan_path_for_mission_leg("hub_to_pickup", -1.205, -0.83);
    ASSERT_TRUE(path.has_value(), "path should exist");
    auto& [wx, wy] = *path;

    acc::TransformParams tp;
    std::vector<double> mx, my;
    acc::qlabs_path_to_map(wx, wy, tp, mx, my);

    ASSERT_TRUE(mx.size() >= 10, "need enough points");
    // Tangent from waypoint 5 to waypoint 10
    double dx = mx[10] - mx[5];
    double dy = my[10] - my[5];
    double angle = std::atan2(dy, dx) * 180.0 / M_PI;
    // Should be between -45 and +90 degrees (east to northeast, NOT south)
    ASSERT_TRUE(angle > -45.0 && angle < 90.0,
        "initial path tangent should be east-to-northeast, not south");
    PASS();
}

// =========================================================================
// 7. Path Recalculation Tests
// =========================================================================
void test_plan_path_hub_to_pickup() {
    TEST("road_graph: plan_path finds route from hub to pickup");
    acc::RoadGraph rg(0.01);
    auto path = rg.plan_path(acc::HUB_X, acc::HUB_Y, acc::PICKUP_X, acc::PICKUP_Y);
    ASSERT_TRUE(path.has_value(), "plan_path should find a route");
    auto& [px, py] = *path;
    ASSERT_TRUE(px.size() > 10, "path should have many waypoints");
    // First point should be near hub
    double start_dist = std::hypot(px.front() - acc::HUB_X, py.front() - acc::HUB_Y);
    ASSERT_TRUE(start_dist < 1.0, "path start should be near hub");
    PASS();
}

void test_plan_path_from_pose_mid_route() {
    TEST("road_graph: plan_path_from_pose slices from current position");
    acc::RoadGraph rg(0.01);
    // Get the full hub_to_pickup route to find a mid-route position
    auto full = rg.get_route("hub_to_pickup");
    ASSERT_TRUE(full.has_value(), "route should exist");
    auto& [fx, fy] = *full;
    int mid = static_cast<int>(fx.size()) / 2;
    double mid_x = fx[mid], mid_y = fy[mid];

    // Plan from mid-route to pickup
    auto path = rg.plan_path_from_pose(mid_x, mid_y, acc::PICKUP_X, acc::PICKUP_Y);
    ASSERT_TRUE(path.has_value(), "plan_path_from_pose should succeed");
    auto& [px, py] = *path;
    // Path should start near the mid-route position
    double start_dist = std::hypot(px.front() - mid_x, py.front() - mid_y);
    ASSERT_TRUE(start_dist < 0.5, "path should start near current position");
    // Path should be shorter than full route
    ASSERT_TRUE(px.size() < fx.size(), "sliced path should be shorter than full route");
    PASS();
}

void test_plan_path_from_pose_direction_check() {
    TEST("road_graph: plan_path_from_pose prepends with direction check");
    acc::RoadGraph rg(0.01);
    // Use a position slightly offset from the route
    auto full = rg.get_route("hub_to_pickup");
    ASSERT_TRUE(full.has_value(), "route should exist");
    auto& [fx, fy] = *full;
    int mid = static_cast<int>(fx.size()) / 3;
    // Offset slightly perpendicular to route direction
    double mid_x = fx[mid] + 0.05, mid_y = fy[mid] + 0.05;

    auto path = rg.plan_path_from_pose(mid_x, mid_y, acc::PICKUP_X, acc::PICKUP_Y);
    ASSERT_TRUE(path.has_value(), "plan_path_from_pose should succeed");
    auto& [px, py] = *path;
    ASSERT_TRUE(px.size() >= 5, "path should have at least 5 points");
    PASS();
}

void test_plan_path_from_pose_min_segment() {
    TEST("road_graph: plan_path_from_pose minimum segment length");
    acc::RoadGraph rg(0.01);
    // Position near the end of the route
    auto full = rg.get_route("hub_to_pickup");
    ASSERT_TRUE(full.has_value(), "route should exist");
    auto& [fx, fy] = *full;
    int near_end = static_cast<int>(fx.size()) - 3;
    double end_x = fx[near_end], end_y = fy[near_end];

    auto path = rg.plan_path_from_pose(end_x, end_y, acc::PICKUP_X, acc::PICKUP_Y);
    ASSERT_TRUE(path.has_value(), "should still produce a path");
    auto& [px, py] = *path;
    // When close to end, it pads to at least 20 points (start_idx = max(0, n-20))
    ASSERT_TRUE(px.size() >= 5, "path near end should have at least 5 points");
    PASS();
}

void test_plan_path_goal_before_start() {
    TEST("road_graph: plan_path returns full route when goal <= start");
    acc::RoadGraph rg(0.01);
    // Swap: goal is at hub (start of route), start is at pickup (end of route)
    auto path = rg.plan_path(acc::PICKUP_X, acc::PICKUP_Y, acc::HUB_X, acc::HUB_Y);
    ASSERT_TRUE(path.has_value(), "plan_path should succeed");
    auto& [px, py] = *path;
    // When goal_idx <= start_idx, returns full route
    ASSERT_TRUE(px.size() > 100, "should return full route when goal before start");
    PASS();
}

// =========================================================================
// 8. Contouring Error Math Tests
// =========================================================================
void test_contouring_error_right_offset() {
    TEST("spline: contouring error sign for right-offset point");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    double e_c, e_l;
    // Point at y=-0.3 (right of horizontal path)
    sp.compute_contouring_errors(2.0, -0.3, 2.0, e_c, e_l);
    // For path heading=0: e_c = -sin(0)*dx + cos(0)*dy = dy = -0.3
    ASSERT_LT(e_c, 0.0, "e_c should be negative for right-offset point");
    ASSERT_NEAR(std::abs(e_c), 0.3, 0.1, "e_c magnitude should be ~0.3");
    PASS();
}

void test_lag_error_ahead_of_progress() {
    TEST("spline: lag error for point ahead of progress");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    double e_c, e_l;
    // Point at x=2.5, progress s=2.0 → ref at (2,0)
    sp.compute_contouring_errors(2.5, 0.0, 2.0, e_c, e_l);
    // For path heading=0: e_l = cos(0)*(x-ref_x) + sin(0)*(y-ref_y) = x-ref_x = 0.5
    ASSERT_GT(e_l, 0.0, "e_l should be positive when point ahead of progress");
    ASSERT_NEAR(e_l, 0.5, 0.15, "e_l magnitude should be ~0.5");
    PASS();
}

void test_contouring_lag_45deg_path() {
    TEST("spline: contouring+lag errors on 45-degree path");
    acc::CubicSplinePath sp;
    // Path at 45 degrees
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 1, 2, 3, 4};
    sp.build(x, y);

    double mid_s = sp.total_length() / 2.0;
    double ref_x, ref_y, cos_t, sin_t;
    sp.get_path_reference(mid_s, ref_x, ref_y, cos_t, sin_t);

    // Tangent should be ~45 deg
    double tangent_angle = std::atan2(sin_t, cos_t);
    ASSERT_NEAR(tangent_angle, M_PI / 4.0, 0.15,
        "tangent angle should be ~45deg on diagonal path");

    // Point on path should have near-zero errors
    double e_c, e_l;
    sp.compute_contouring_errors(ref_x, ref_y, mid_s, e_c, e_l);
    ASSERT_TRUE(std::abs(e_c) < 0.05, "on-path contouring error should be ~0");
    ASSERT_TRUE(std::abs(e_l) < 0.05, "on-path lag error should be ~0");
    PASS();
}

// =========================================================================
// 9. Spline Rebuild & Closest Progress Tests
// =========================================================================
void test_spline_rebuild_different_waypoints() {
    TEST("spline: rebuild with different waypoints");
    acc::CubicSplinePath sp;
    std::vector<double> x1 = {0, 1, 2, 3, 4};
    std::vector<double> y1 = {0, 0, 0, 0, 0};
    sp.build(x1, y1);
    double len1 = sp.total_length();

    // Rebuild with longer path
    std::vector<double> x2 = {0, 2, 4, 6, 8};
    std::vector<double> y2 = {0, 0, 0, 0, 0};
    sp.build(x2, y2);
    double len2 = sp.total_length();

    ASSERT_GT(len2, len1 * 1.5, "rebuilt path should be longer");

    // Query the rebuilt path
    double px, py;
    sp.get_position(len2 / 2.0, px, py);
    ASSERT_NEAR(px, 4.0, 0.5, "midpoint x should be ~4.0 on rebuilt path");
    ASSERT_NEAR(py, 0.0, 0.1, "midpoint y should be ~0.0");
    PASS();
}

void test_spline_closest_progress_offsets() {
    TEST("spline: find_closest_progress at multiple offsets");
    acc::CubicSplinePath sp;
    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 0, 0, 0, 0};
    sp.build(x, y);

    // Points along the path at different positions
    double s0 = sp.find_closest_progress(0.0, 0.3);
    double s1 = sp.find_closest_progress(1.0, -0.2);
    double s2 = sp.find_closest_progress(3.0, 0.1);

    ASSERT_NEAR(s0, 0.0, 0.3, "closest progress for (0,0.3) should be ~0");
    ASSERT_NEAR(s1, 1.0, 0.3, "closest progress for (1,-0.2) should be ~1");
    ASSERT_NEAR(s2, 3.0, 0.3, "closest progress for (3,0.1) should be ~3");
    // They should be monotonically increasing
    ASSERT_TRUE(s0 < s1, "s0 < s1");
    ASSERT_TRUE(s1 < s2, "s1 < s2");
    PASS();
}

// =========================================================================
// Main
// =========================================================================
int main() {
    std::printf("=== C++ Port Compilation & Logic Tests ===\n\n");

    std::printf("[Coordinate Transform]\n");
    test_coordinate_transform_roundtrip();
    test_coordinate_transform_origin();
    test_coordinate_transform_2d();
    test_coordinate_transform_batch();
    test_normalize_angle();

    std::printf("\n[Cubic Spline Path]\n");
    test_spline_build_line();
    test_spline_position_interpolation();
    test_spline_tangent();
    test_spline_curvature_line();
    test_spline_circle();
    test_spline_closest_progress();
    test_spline_contouring_errors();
    test_spline_path_reference();
    test_spline_build_from_pairs();

    std::printf("\n[Road Graph]\n");
    test_scs_path_straight();
    test_scs_path_turn();
    test_scs_path_zero_radius();
    test_sdcs_roadmap_construction();
    test_sdcs_roadmap_pathfinding();
    test_roadgraph_construction();
    test_roadgraph_route_names();
    test_roadgraph_plan_path();
    test_roadgraph_get_route();
    test_roadmap_generate_path();

    std::printf("\n[Road Boundaries]\n");
    test_road_segment_circular_contains();
    test_road_segment_circular_nearest();
    test_road_segment_spline_contains();
    test_road_segment_spline_nearest();
    test_obstacle_zone_circle();
    test_obstacle_zone_rectangle();
    test_road_boundary_spline_no_config();
    test_road_boundary_path_constraints();
    test_road_boundary_velocity_limit();
    test_road_boundary_traffic_controls();
    test_boundary_normal_unit_vector();

    std::printf("\n[Integration]\n");
    test_integration_spline_with_roadgraph();
    test_integration_all_mission_legs();
    test_integration_transform_with_roadgraph();
    test_integration_boundary_constraints_on_path();
    test_integration_spline_errors_along_path();

    std::printf("\n[Path Direction]\n");
    test_hub_to_pickup_path_heads_northeast();
    test_hub_to_pickup_initial_tangent();

    std::printf("\n[Path Recalculation]\n");
    test_plan_path_hub_to_pickup();
    test_plan_path_from_pose_mid_route();
    test_plan_path_from_pose_direction_check();
    test_plan_path_from_pose_min_segment();
    test_plan_path_goal_before_start();

    std::printf("\n[Contouring Error Math]\n");
    test_contouring_error_right_offset();
    test_lag_error_ahead_of_progress();
    test_contouring_lag_45deg_path();

    std::printf("\n[Spline Rebuild & Closest Progress]\n");
    test_spline_rebuild_different_waypoints();
    test_spline_closest_progress_offsets();

    std::printf("\n=== Results: %d passed, %d failed ===\n",
                tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
