/**
 * Road network path planner implementation.
 * Ported from road_graph.py.
 */

#include "road_graph.h"
#include <algorithm>
#include <cstring>
#include <functional>
#include <set>

namespace acc {

// -------------------------------------------------------------------------
// Math utilities
// -------------------------------------------------------------------------
static constexpr double TWO_PI = 2.0 * M_PI;

static double wrap_to_2pi(double th) {
    th = std::fmod(std::fmod(th, TWO_PI) + TWO_PI, TWO_PI);
    return th;
}

static double wrap_to_pi(double th) {
    th = std::fmod(th, TWO_PI);
    th = std::fmod(th + TWO_PI, TWO_PI);
    if (th > M_PI) th -= TWO_PI;
    return th;
}

static double signed_angle_between(double v1x, double v1y, double v2x, double v2y) {
    return wrap_to_pi(std::atan2(v2y, v2x) - std::atan2(v1y, v1x));
}

// -------------------------------------------------------------------------
// SCSPath
// -------------------------------------------------------------------------

SCSResult SCSPath(const double start[3], const double end[3],
                  double radius, double stepSize)
{
    SCSResult result;
    double p1x = start[0], p1y = start[1], th1 = start[2];
    double p2x = end[0],   p2y = end[1],   th2 = end[2];

    if (radius < 1e-6) {
        // Straight line
        double dx = p2x - p1x, dy = p2y - p1y;
        double dist = std::hypot(dx, dy);
        if (dist < 1e-6) {
            result.valid = true;
            result.length = 0.0;
            return result;
        }
        int n_pts = std::max(static_cast<int>(dist / stepSize), 2);
        result.x.reserve(n_pts);
        result.y.reserve(n_pts);
        for (int i = 1; i < n_pts; ++i) {
            double t = static_cast<double>(i) / n_pts;
            result.x.push_back(p1x + t * dx);
            result.y.push_back(p1y + t * dy);
        }
        result.length = dist;
        result.valid = true;
        return result;
    }

    double t1x = std::cos(th1), t1y = std::sin(th1);
    double t2x = std::cos(th2), t2y = std::sin(th2);

    // Direction of turn
    double dp_x = p2x - p1x, dp_y = p2y - p1y;
    double sa = signed_angle_between(t1x, t1y, dp_x, dp_y);
    int direction = (sa > 0) ? 1 : -1;

    double n1x = radius * (-t1y) * direction;
    double n1y = radius * ( t1x) * direction;
    double n2x = radius * (-t2y) * direction;
    double n2y = radius * ( t2x) * direction;

    double cx, cy;  // circle center
    double tol = 0.01;

    double angle_diff = wrap_to_pi(th2 - th1);

    if (std::abs(angle_diff) < tol) {
        // Nearly parallel headings
        double vx = p2x - p1x, vy = p2y - p1y;
        double v_norm = std::hypot(vx, vy);
        if (v_norm < 1e-9) {
            result.valid = true;
            result.length = 0.0;
            return result;
        }
        double vux = vx / v_norm, vuy = vy / v_norm;
        double dot = t1x * vux + t1y * vuy;
        if (1.0 - std::abs(dot) < tol) {
            cx = p2x + n1x;
            cy = p2y + n1y;
        } else {
            return result;  // invalid
        }
    } else if (std::abs(wrap_to_pi(th2 - th1 + M_PI)) < tol) {
        // Anti-parallel headings
        double vx = (p2x + 2 * n2x) - p1x;
        double vy = (p2y + 2 * n2y) - p1y;
        double v_norm = std::hypot(vx, vy);
        if (v_norm < 1e-9) {
            result.valid = true;
            result.length = 0.0;
            return result;
        }
        double vux = vx / v_norm, vuy = vy / v_norm;
        double dot = t1x * vux + t1y * vuy;
        if (1.0 - std::abs(dot) < tol) {
            double s = t1x * vx + t1y * vy;
            if (s < tol) {
                cx = p1x + n1x;
                cy = p1y + n1y;
            } else {
                cx = p2x + n2x;
                cy = p2y + n2y;
            }
        } else {
            return result;  // invalid
        }
    } else {
        // General case: solve 2x2 linear system
        double d1x = p1x + n1x, d1y = p1y + n1y;
        double d2x = p2x + n2x, d2y = p2y + n2y;
        // A = [t1x, -t2x; t1y, -t2y], b = [d2x-d1x; d2y-d1y]
        double det = t1x * (-t2y) - (-t2x) * t1y;
        if (std::abs(det) < 1e-10) return result;

        double bx = d2x - d1x, by = d2y - d1y;
        double alpha = ((-t2y) * bx - (-t2x) * by) / det;
        double beta  = (t1x * by - t1y * bx) / det;

        if (alpha >= -tol && beta <= tol) {
            cx = d1x + alpha * t1x;
            cy = d1y + alpha * t1y;
        } else {
            return result;  // invalid
        }
    }

    // Tangent points on circle
    double b1x = cx - n1x, b1y = cy - n1y;
    double b2x = cx - n2x, b2y = cy - n2y;

    // Discretize line p1 -> b1
    double line1_len = std::hypot(b1x - p1x, b1y - p1y);
    if (line1_len > stepSize) {
        double ds = stepSize / line1_len;
        for (double s = ds; s < 1.0; s += ds) {
            result.x.push_back(p1x + s * (b1x - p1x));
            result.y.push_back(p1y + s * (b1y - p1y));
        }
    }

    // Discretize arc b1 -> b2
    double av1x = b1x - cx, av1y = b1y - cy;
    double av2x = b2x - cx, av2y = b2y - cy;
    double ang_dist = wrap_to_2pi(direction *
        wrap_to_pi(std::atan2(av2y, av2x) - std::atan2(av1y, av1x)));
    double arc_length = std::abs(ang_dist * radius);
    if (arc_length > stepSize) {
        double start_angle = std::atan2(av1y, av1x);
        double dth = stepSize / radius;
        for (double s = dth; s < ang_dist; s += dth) {
            double th = start_angle + s * direction;
            result.x.push_back(cx + std::cos(th) * radius);
            result.y.push_back(cy + std::sin(th) * radius);
        }
    }

    // Discretize line b2 -> p2
    double line2_len = std::hypot(b2x - p2x, b2y - p2y);
    if (line2_len > stepSize) {
        double ds = stepSize / line2_len;
        for (double s = ds; s < 1.0; s += ds) {
            result.x.push_back(b2x + s * (p2x - b2x));
            result.y.push_back(b2y + s * (p2y - b2y));
        }
    }

    result.length = line1_len + arc_length + line2_len;
    result.valid = true;
    return result;
}

// -------------------------------------------------------------------------
// RoadMap
// -------------------------------------------------------------------------

void RoadMap::add_node(double x, double y, double theta) {
    RoadMapNode node;
    node.pose[0] = x;
    node.pose[1] = y;
    node.pose[2] = theta;
    node.id = static_cast<int>(nodes_.size());
    nodes_.push_back(node);
}

void RoadMap::add_edge(int from_idx, int to_idx, double radius) {
    int edge_idx = static_cast<int>(edges_.size());
    RoadMapEdge edge;
    edge.from_node = from_idx;
    edge.to_node = to_idx;

    auto scs = SCSPath(nodes_[from_idx].pose, nodes_[to_idx].pose, radius);
    edge.wp_x = std::move(scs.x);
    edge.wp_y = std::move(scs.y);
    edge.length = scs.length;

    edges_.push_back(std::move(edge));
    nodes_[from_idx].out_edges.push_back(edge_idx);
    nodes_[to_idx].in_edges.push_back(edge_idx);
}

std::optional<std::pair<std::vector<double>, std::vector<double>>>
RoadMap::find_shortest_path(int start_idx, int goal_idx) const
{
    if (start_idx == goal_idx) return std::nullopt;

    int N = static_cast<int>(nodes_.size());
    std::vector<double> g_score(N, 1e18);
    g_score[start_idx] = 0.0;

    // came_from: node -> (prev_node, edge_idx)
    std::vector<std::pair<int, int>> came_from(N, {-1, -1});

    // Priority queue: (f_score, node_idx)
    using PQItem = std::pair<double, int>;
    std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> open_set;

    double gx = nodes_[goal_idx].pose[0], gy = nodes_[goal_idx].pose[1];
    double h0 = std::hypot(gx - nodes_[start_idx].pose[0],
                           gy - nodes_[start_idx].pose[1]);
    open_set.push({h0, start_idx});

    std::set<int> closed;

    while (!open_set.empty()) {
        auto [f, current] = open_set.top();
        open_set.pop();

        if (current == goal_idx) {
            // Reconstruct path
            std::vector<double> px, py;
            // Add goal node
            px.push_back(nodes_[goal_idx].pose[0]);
            py.push_back(nodes_[goal_idx].pose[1]);

            int node = goal_idx;
            while (came_from[node].first >= 0) {
                int prev = came_from[node].first;
                int eidx = came_from[node].second;
                const auto& e = edges_[eidx];

                // Prepend: node pose + edge waypoints (reversed)
                std::vector<double> seg_x, seg_y;
                seg_x.push_back(nodes_[prev].pose[0]);
                seg_y.push_back(nodes_[prev].pose[1]);
                seg_x.insert(seg_x.end(), e.wp_x.begin(), e.wp_x.end());
                seg_y.insert(seg_y.end(), e.wp_y.begin(), e.wp_y.end());
                seg_x.insert(seg_x.end(), px.begin(), px.end());
                seg_y.insert(seg_y.end(), py.begin(), py.end());
                px = std::move(seg_x);
                py = std::move(seg_y);

                node = prev;
                if (came_from[node].first < 0) break;
            }
            return std::make_pair(std::move(px), std::move(py));
        }

        if (closed.count(current)) continue;
        closed.insert(current);

        for (int eidx : nodes_[current].out_edges) {
            const auto& edge = edges_[eidx];
            int neighbor = edge.to_node;
            if (closed.count(neighbor)) continue;
            if (edge.length <= 0) continue;  // invalid edge

            double tent_g = g_score[current] + edge.length;
            if (tent_g < g_score[neighbor]) {
                came_from[neighbor] = {current, eidx};
                g_score[neighbor] = tent_g;
                double h = std::hypot(gx - nodes_[neighbor].pose[0],
                                      gy - nodes_[neighbor].pose[1]);
                open_set.push({tent_g + h, neighbor});
            }
        }
    }
    return std::nullopt;
}

std::optional<std::pair<std::vector<double>, std::vector<double>>>
RoadMap::generate_path(const std::vector<int>& node_sequence) const
{
    std::vector<double> px, py;
    for (size_t i = 0; i + 1 < node_sequence.size(); ++i) {
        auto seg = find_shortest_path(node_sequence[i], node_sequence[i + 1]);
        if (!seg) return std::nullopt;
        // Append all but last point (to avoid duplicates)
        auto& sx = seg->first;
        auto& sy = seg->second;
        for (size_t j = 0; j + 1 < sx.size(); ++j) {
            px.push_back(sx[j]);
            py.push_back(sy[j]);
        }
    }
    // Add final node
    int last = node_sequence.back();
    px.push_back(nodes_[last].pose[0]);
    py.push_back(nodes_[last].pose[1]);
    return std::make_pair(std::move(px), std::move(py));
}

// -------------------------------------------------------------------------
// SDCSRoadMap
// -------------------------------------------------------------------------

SDCSRoadMap::SDCSRoadMap(bool /*left_hand_traffic*/, bool use_small_map)
{
    double scale = 0.002035;
    double xOffset = 1134;
    double yOffset = 2363;

    double innerR  = 305.5 * scale;    // 0.622
    double outerR  = 438.0 * scale;    // 0.891
    double circleR = 333.0 * scale;    // 0.678
    double onewayR = 350.0 * scale;    // 0.712
    double kinkR   = 375.0 * scale;    // 0.763

    double pi = M_PI;
    double hpi = M_PI / 2.0;

    // Right-hand traffic node poses (pixel coords -> QLabs)
    struct NP { double px, py, th; };
    std::vector<NP> nodePoses = {
        {1134, 2299, -hpi},        // 0
        {1266, 2323,  hpi},        // 1
        {1688, 2896,  0},          // 2
        {1688, 2763,  pi},         // 3
        {2242, 2323,  hpi},        // 4
        {2109, 2323, -hpi},        // 5
        {1632, 1822,  pi},         // 6
        {1741, 1955,  0},          // 7
        { 766, 1822,  pi},         // 8
        { 766, 1955,  0},          // 9
        { 504, 2589, -42*pi/180},  // 10
    };

    if (!use_small_map) {
        std::vector<NP> extra = {
            {1134, 1300, -hpi},              // 11
            {1134, 1454, -hpi},              // 12
            {1266, 1454,  hpi},              // 13
            {2242,  905,  hpi},              // 14
            {2109, 1454, -hpi},              // 15
            {1580,  540, -80.6*pi/180},      // 16
            {1854.4, 814.5, -9.4*pi/180},   // 17
            {1440,  856, -138*pi/180},       // 18
            {1523,  958,  42*pi/180},        // 19
            {1134,  153,  pi},               // 20
            {1134,  286,  0},                // 21
            { 159,  905, -hpi},              // 22
            { 291,  905,  hpi},              // 23
        };
        nodePoses.insert(nodePoses.end(), extra.begin(), extra.end());
    }

    // Scale and add nodes
    for (auto& np : nodePoses) {
        double x = scale * (np.px - xOffset);
        double y = scale * (yOffset - np.py);
        add_node(x, y, np.th);
    }

    // Edge configs: [from, to, radius]
    struct EC { int from, to; double r; };
    std::vector<EC> edgeConfigs = {
        {0, 2, outerR},  {1, 7, innerR}, {1, 8, outerR},
        {2, 4, outerR},  {3, 1, innerR}, {4, 6, outerR},
        {5, 3, innerR},  {6, 0, outerR}, {6, 8, 0},
        {7, 5, innerR},  {8, 10, onewayR}, {9, 0, innerR},
        {9, 7, 0},       {10, 1, innerR}, {10, 2, innerR},
    };

    if (!use_small_map) {
        std::vector<EC> extra = {
            {1, 13, 0},      {4, 14, 0},       {6, 13, innerR},
            {7, 14, outerR}, {8, 23, innerR},  {9, 13, outerR},
            {11, 12, 0},     {12, 0, 0},       {12, 7, outerR},
            {12, 8, innerR}, {13, 19, innerR}, {14, 16, circleR},
            {14, 20, circleR},{15, 5, outerR}, {15, 6, innerR},
            {16, 17, circleR},{16, 18, innerR},{17, 15, innerR},
            {17, 16, circleR},{17, 20, circleR},{18, 11, kinkR},
            {19, 17, innerR},{20, 22, outerR}, {21, 16, innerR},
            {22, 9, outerR}, {22, 10, outerR}, {23, 21, innerR},
        };
        edgeConfigs.insert(edgeConfigs.end(), extra.begin(), extra.end());
    }

    for (auto& ec : edgeConfigs) {
        add_edge(ec.from, ec.to, ec.r);
    }

    // Spawn node (node 24): vehicle starting position at hub
    double spawn_heading = std::fmod(-44.7 * pi / 180.0, 2.0 * pi);
    if (spawn_heading < 0) spawn_heading += 2.0 * pi;
    add_node(-1.205, -0.83, spawn_heading);  // node 24
    add_edge(24, 2, 0.0);
    add_edge(10, 24, 0.0);
    add_edge(24, 1, 0.0);  // Straight line (was 0.866 — the arc honored spawn heading 315deg,
                            // creating a 270deg southward sweep before heading northeast)
}

// -------------------------------------------------------------------------
// RoadGraph
// -------------------------------------------------------------------------

int RoadGraph::find_closest_idx(const std::vector<double>& wx,
                                const std::vector<double>& wy,
                                double px, double py)
{
    double best = 1e18;
    int idx = 0;
    for (size_t i = 0; i < wx.size(); ++i) {
        double d = (wx[i] - px) * (wx[i] - px) + (wy[i] - py) * (wy[i] - py);
        if (d < best) { best = d; idx = static_cast<int>(i); }
    }
    return idx;
}

void RoadGraph::interpolate_gap(double x1, double y1, double x2, double y2,
                                double ds,
                                std::vector<double>& out_x,
                                std::vector<double>& out_y)
{
    double dist = std::hypot(x2 - x1, y2 - y1);
    if (dist < ds * 1.5) return;
    int n = std::max(static_cast<int>(dist / ds), 2);
    for (int i = 1; i < n; ++i) {
        double t = static_cast<double>(i) / n;
        out_x.push_back(x1 + t * (x2 - x1));
        out_y.push_back(y1 + t * (y2 - y1));
    }
}

void RoadGraph::attach_endpoints(const std::string& route_name,
                                 std::vector<double>& rx,
                                 std::vector<double>& ry)
{
    if (rx.empty()) return;

    auto prepend = [&](double px, double py) {
        if (std::hypot(rx.front() - px, ry.front() - py) > 0.02) {
            std::vector<double> gx, gy;
            interpolate_gap(px, py, rx.front(), ry.front(), ds_, gx, gy);
            std::vector<double> nx, ny;
            nx.push_back(px); ny.push_back(py);
            nx.insert(nx.end(), gx.begin(), gx.end());
            ny.insert(ny.end(), gy.begin(), gy.end());
            nx.insert(nx.end(), rx.begin(), rx.end());
            ny.insert(ny.end(), ry.begin(), ry.end());
            rx = std::move(nx); ry = std::move(ny);
        }
    };

    auto append = [&](double px, double py) {
        if (std::hypot(rx.back() - px, ry.back() - py) > 0.02) {
            std::vector<double> gx, gy;
            interpolate_gap(rx.back(), ry.back(), px, py, ds_, gx, gy);
            rx.insert(rx.end(), gx.begin(), gx.end());
            ry.insert(ry.end(), gy.begin(), gy.end());
            rx.push_back(px); ry.push_back(py);
        }
    };

    if (route_name == "hub_to_pickup") {
        append(PICKUP_X, PICKUP_Y);
    } else if (route_name == "pickup_to_dropoff") {
        prepend(PICKUP_X, PICKUP_Y);
        append(DROPOFF_X, DROPOFF_Y);
    } else if (route_name == "dropoff_to_hub") {
        prepend(DROPOFF_X, DROPOFF_Y);
    }
}

RoadGraph::RoadGraph(double ds) : ds_(ds) {
    struct RouteSeq {
        std::string name;
        std::vector<int> nodes;
    };

    std::vector<RouteSeq> route_defs = {
        {"hub_to_pickup",      {24, 1, 13, 19, 17, 20}},
        {"pickup_to_dropoff",  {21, 16, 18, 11, 12, 8}},
        {"dropoff_to_hub",     {8, 10, 24}},
    };

    for (auto& rd : route_defs) {
        auto path = roadmap_.generate_path(rd.nodes);
        if (path) {
            auto& [px, py] = *path;
            attach_endpoints(rd.name, px, py);
            routes_[rd.name] = {std::move(px), std::move(py)};
        }
    }
}

std::optional<std::pair<std::vector<double>, std::vector<double>>>
RoadGraph::get_route(const std::string& route_name) const
{
    auto it = routes_.find(route_name);
    if (it == routes_.end()) return std::nullopt;
    return std::make_pair(it->second.x, it->second.y);
}

std::vector<std::string> RoadGraph::get_route_names() const {
    std::vector<std::string> names;
    for (auto& [k, v] : routes_)
        names.push_back(k);
    return names;
}

std::string RoadGraph::get_route_for_leg(const std::string& start_label,
                                         const std::string& goal_label) const
{
    auto contains = [](const std::string& s, const std::string& sub) {
        return s.find(sub) != std::string::npos;
    };

    if (contains(goal_label, "pickup") && (contains(start_label, "hub") || start_label.empty()))
        return "hub_to_pickup";
    if (contains(goal_label, "dropoff") && contains(start_label, "pickup"))
        return "pickup_to_dropoff";
    if (contains(goal_label, "hub") && contains(start_label, "dropoff"))
        return "dropoff_to_hub";
    return "";
}

std::optional<std::pair<std::vector<double>, std::vector<double>>>
RoadGraph::plan_path_for_mission_leg(const std::string& route_name,
                                     double cur_x, double cur_y) const
{
    auto it = routes_.find(route_name);
    if (it == routes_.end()) return std::nullopt;

    const auto& route = it->second;
    int start_idx = find_closest_idx(route.x, route.y, cur_x, cur_y);

    int n = static_cast<int>(route.x.size());
    int segment_start = start_idx;
    int segment_len = n - segment_start;

    if (segment_len < 5) {
        segment_start = std::max(0, n - 20);
        segment_len = n - segment_start;
    }

    std::vector<double> rx(route.x.begin() + segment_start,
                           route.x.begin() + segment_start + segment_len);
    std::vector<double> ry(route.y.begin() + segment_start,
                           route.y.begin() + segment_start + segment_len);

    // Prepend current position if not already close
    if (std::hypot(cur_x - rx.front(), cur_y - ry.front()) > 0.02) {
        rx.insert(rx.begin(), cur_x);
        ry.insert(ry.begin(), cur_y);
    }

    return std::make_pair(std::move(rx), std::move(ry));
}

}  // namespace acc
