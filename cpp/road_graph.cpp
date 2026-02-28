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

// Shift a centerline path to right-lane centerline.
// For right-hand traffic, "right" is -left-normal, where left-normal = (-ty, tx).
static void offset_path_to_right_lane_center(std::vector<double>& x,
                                             std::vector<double>& y,
                                             double offset_m)
{
    if (x.size() < 2 || y.size() != x.size() || std::abs(offset_m) < 1e-9) return;
    std::vector<double> ox(x.size()), oy(y.size());
    for (size_t i = 0; i < x.size(); ++i) {
        size_t ip = (i == 0) ? 0 : i - 1;
        size_t in = std::min(i + 1, x.size() - 1);
        double tx = x[in] - x[ip];
        double ty = y[in] - y[ip];
        double tlen = std::hypot(tx, ty);
        if (tlen < 1e-9) {
            ox[i] = x[i];
            oy[i] = y[i];
            continue;
        }
        tx /= tlen;
        ty /= tlen;
        // left normal = (-ty, tx), right normal = (ty, -tx)
        double nx_r = ty;
        double ny_r = -tx;
        ox[i] = x[i] + offset_m * nx_r;
        oy[i] = y[i] + offset_m * ny_r;
    }
    x.swap(ox);
    y.swap(oy);
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
    // When heading points directly at dest (sa≈0), the direction is ambiguous.
    // Resolve by using the actual heading change (end - start) instead.
    if (std::abs(sa) < 0.05) {
        sa = wrap_to_pi(th2 - th1);
    }
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
    // Reference 2025 uses: -44.7 % (2*pi) where -44.7 is in radians (NOT degrees).
    // This gives 5.5655 rad. The edge radii below were tuned for this heading.
    // Edge radii from reference: hub→2=0.0, 10→hub=1.48202, hub→1=0.866326
    double hub_heading = std::fmod(-44.7, 2.0 * M_PI);
    if (hub_heading < 0) hub_heading += 2.0 * M_PI;  // = 5.5655 rad
    add_node(-1.205, -0.83, hub_heading);  // node 24
    add_edge(24, 2, 0.0);
    // Note: reference uses radius=1.48202 for 10→24 but SCS geometry is infeasible
    // at heading 5.5655 (beta > tol). Use straight line; nodes are only 0.38m apart.
    add_edge(10, 24, 0.0);
    add_edge(24, 1, 0.866326);

    // Reference 2025 uses D* with +20 penalty on traffic-controlled edges
    // (indices 1,2,7,8,11,12,15,17,20,22,23,24). This steers routing away from
    // intersections with traffic lights/crosswalks. Apply the same penalties to
    // our A* — verified to produce identical routes as the reference D*.
    // Penalties are added AFTER edge creation so SCS waypoints are unaffected.
    static const std::set<int> penalized_edges = {1,2, 7,8, 11,12, 15,17, 20, 22,23,24};
    for (int idx : penalized_edges) {
        if (idx < static_cast<int>(edges_.size())) {
            edges_[idx].length += 20.0;
        }
    }
}

// -------------------------------------------------------------------------
// Spline resampling (C++ equivalent of scipy splprep/splev with s=0)
// Uses natural cubic spline for exact-interpolation uniform-spacing resampling.
// Replaces previous Catmull-Rom which overshoots at sharp direction changes.
// -------------------------------------------------------------------------

void resample_path(std::vector<double>& x, std::vector<double>& y, double spacing)
{
    size_t n = x.size();
    if (n < 2) return;

    // Compute cumulative arc lengths
    std::vector<double> arc(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        arc[i] = arc[i-1] + std::hypot(x[i] - x[i-1], y[i] - y[i-1]);
    }
    double total_length = arc.back();
    if (total_length < spacing) return;

    // Remove duplicate arc-length points (zero-length segments cause spline issues)
    std::vector<double> s_pts, x_pts, y_pts;
    s_pts.reserve(n);
    x_pts.reserve(n);
    y_pts.reserve(n);
    s_pts.push_back(arc[0]);
    x_pts.push_back(x[0]);
    y_pts.push_back(y[0]);
    for (size_t i = 1; i < n; ++i) {
        if (arc[i] - s_pts.back() > 1e-10) {
            s_pts.push_back(arc[i]);
            x_pts.push_back(x[i]);
            y_pts.push_back(y[i]);
        }
    }
    size_t m = s_pts.size();
    if (m < 2) return;

    // Linear fallback for 2 points
    if (m == 2) {
        int n_out = std::max(static_cast<int>(total_length / spacing), 2);
        std::vector<double> rx(n_out + 1), ry(n_out + 1);
        for (int i = 0; i <= n_out; ++i) {
            double t = static_cast<double>(i) / n_out;
            rx[i] = x_pts[0] + t * (x_pts[1] - x_pts[0]);
            ry[i] = y_pts[0] + t * (y_pts[1] - y_pts[0]);
        }
        x = std::move(rx);
        y = std::move(ry);
        return;
    }

    // Build natural cubic spline second derivatives M[] via Thomas algorithm.
    // Natural BC: M[0] = M[m-1] = 0.
    // For interior nodes j=1..m-2:
    //   h[j-1]*M[j-1] + 2*(h[j-1]+h[j])*M[j] + h[j]*M[j+1] = rhs[j]
    auto build_spline_M = [m](const std::vector<double>& s,
                               const std::vector<double>& f)
        -> std::vector<double>
    {
        std::vector<double> M(m, 0.0);
        if (m < 3) return M;

        std::vector<double> h(m - 1);
        for (size_t i = 0; i < m - 1; ++i)
            h[i] = s[i + 1] - s[i];

        size_t sz = m - 2;
        std::vector<double> diag(sz), sup(sz), rhs(sz);
        for (size_t k = 0; k < sz; ++k) {
            diag[k] = 2.0 * (h[k] + h[k + 1]);
            sup[k] = h[k + 1];
            rhs[k] = 6.0 * ((f[k+2] - f[k+1]) / h[k+1]
                           - (f[k+1] - f[k])   / h[k]);
        }

        // Thomas algorithm: forward elimination
        for (size_t k = 1; k < sz; ++k) {
            double w = h[k] / diag[k - 1];
            diag[k] -= w * sup[k - 1];
            rhs[k]  -= w * rhs[k - 1];
        }

        // Back substitution
        M[sz] = rhs[sz - 1] / diag[sz - 1];
        for (int k = static_cast<int>(sz) - 2; k >= 0; --k) {
            M[k + 1] = (rhs[k] - sup[k] * M[k + 2]) / diag[k];
        }

        return M;
    };

    auto Mx = build_spline_M(s_pts, x_pts);
    auto My = build_spline_M(s_pts, y_pts);

    // Evaluate spline at uniform arc-length intervals
    int n_out = static_cast<int>(total_length / spacing);
    if (n_out < 2) return;

    std::vector<double> rx, ry;
    rx.reserve(n_out + 1);
    ry.reserve(n_out + 1);

    size_t seg = 0;
    for (int i = 0; i <= n_out; ++i) {
        double t = static_cast<double>(i) / n_out * total_length;
        while (seg + 1 < m - 1 && s_pts[seg + 1] < t) ++seg;

        double hi = s_pts[seg + 1] - s_pts[seg];
        double a = (s_pts[seg + 1] - t) / hi;
        double b = (t - s_pts[seg]) / hi;

        rx.push_back(a * x_pts[seg] + b * x_pts[seg + 1] +
                     (hi * hi / 6.0) * ((a*a*a - a) * Mx[seg] +
                                        (b*b*b - b) * Mx[seg + 1]));
        ry.push_back(a * y_pts[seg] + b * y_pts[seg + 1] +
                     (hi * hi / 6.0) * ((a*a*a - a) * My[seg] +
                                        (b*b*b - b) * My[seg + 1]));
    }

    x = std::move(rx);
    y = std::move(ry);
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

static int find_closest_idx_directional(const std::vector<double>& wx,
                                        const std::vector<double>& wy,
                                        double px, double py,
                                        double pyaw)
{
    auto nearest_idx_fn = [&](double& nearest_dist_out) {
        double best = 1e18;
        int idx = 0;
        for (size_t i = 0; i < wx.size(); ++i) {
            double dx = wx[i] - px;
            double dy = wy[i] - py;
            double d2 = dx * dx + dy * dy;
            if (d2 < best) { best = d2; idx = static_cast<int>(i); }
        }
        nearest_dist_out = std::sqrt(best);
        return idx;
    };

    int n = static_cast<int>(wx.size());
    if (n < 3 || !std::isfinite(pyaw)) {
        double nearest_dist_unused = 0.0;
        return nearest_idx_fn(nearest_dist_unused);
    }

    double nearest_dist = 0.0;
    int nearest_idx = nearest_idx_fn(nearest_dist);
    double hx = std::cos(pyaw);
    double hy = std::sin(pyaw);

    double best_score = 1e18;
    int best_idx = nearest_idx;

    for (int i = 1; i < n - 1; ++i) {
        double dx = wx[i] - px;
        double dy = wy[i] - py;
        double dist = std::hypot(dx, dy);

        // Keep candidates local to current position to avoid long jumps.
        if (dist > nearest_dist + 0.60) continue;

        double tx = wx[i + 1] - wx[i - 1];
        double ty = wy[i + 1] - wy[i - 1];
        double tnorm = std::hypot(tx, ty);
        if (tnorm < 1e-6) continue;
        tx /= tnorm;
        ty /= tnorm;

        // Prefer tangent aligned with current vehicle heading.
        double heading_align = tx * hx + ty * hy;  // [-1, 1]

        // Also prefer direction that points toward route end.
        double gx = wx.back() - wx[i];
        double gy = wy.back() - wy[i];
        double gnorm = std::hypot(gx, gy);
        double goal_align = 1.0;
        if (gnorm > 1e-6) {
            gx /= gnorm;
            gy /= gnorm;
            goal_align = tx * gx + ty * gy;  // [-1, 1]
        }

        double reverse_penalty = (goal_align < -0.1) ? 10.0 : 0.0;
        double score = dist * dist
                     + 0.25 * (1.0 - heading_align)
                     + 0.35 * (1.0 - goal_align)
                     + reverse_penalty;
        if (score < best_score) {
            best_score = score;
            best_idx = i;
        }
    }

    return best_idx;
}

int RoadGraph::find_first_local_min(const std::vector<double>& wx,
                                    const std::vector<double>& wy,
                                    double px, double py,
                                    double threshold,
                                    int start_from)
{
    // Find the first contiguous region within threshold of (px,py),
    // then return the argmin within that region. This avoids picking
    // a later pass when the path visits the same area twice.
    double thresh_sq = threshold * threshold;
    bool in_region = false;
    int region_start = 0;
    int best_idx = find_closest_idx(wx, wy, px, py);  // fallback

    for (size_t i = static_cast<size_t>(start_from); i < wx.size(); ++i) {
        double d_sq = (wx[i] - px) * (wx[i] - px) + (wy[i] - py) * (wy[i] - py);
        if (d_sq < thresh_sq) {
            if (!in_region) {
                in_region = true;
                region_start = static_cast<int>(i);
            }
        } else {
            if (in_region) {
                // Exited the first close region — find min within it
                double best_d = 1e18;
                for (int j = region_start; j < static_cast<int>(i); ++j) {
                    double d = (wx[j] - px) * (wx[j] - px) + (wy[j] - py) * (wy[j] - py);
                    if (d < best_d) { best_d = d; best_idx = j; }
                }
                return best_idx;
            }
        }
    }
    // If we ended while still in region
    if (in_region) {
        double best_d = 1e18;
        for (int j = region_start; j < static_cast<int>(wx.size()); ++j) {
            double d = (wx[j] - px) * (wx[j] - px) + (wy[j] - py) * (wy[j] - py);
            if (d < best_d) { best_d = d; best_idx = j; }
        }
    }
    return best_idx;
}

RoadGraph::RoadGraph(double ds) : ds_(ds) {
    // Reference 2025 approach: single loop path through [24, 20, 9, 10]
    // This traverses Hub → (via node 20) → node 9 → node 10 → (back to hub area)
    // The D* pathfinder in the reference finds shortest weighted paths between
    // consecutive nodes in this sequence, penalizing traffic lights/crosswalks.
    // We use A* (equivalent for static graphs) with the same node sequence.
    std::vector<int> loop_sequence = {24, 20, 9, 10};

    auto loop_path = roadmap_.generate_path(loop_sequence);
    if (loop_path) {
        loop_x_ = std::move(loop_path->first);
        loop_y_ = std::move(loop_path->second);

        // Resample to uniform spacing (reference uses scipy splprep at 0.001m)
        resample_path(loop_x_, loop_y_, ds_);

        // Apply scale factor [1.01, 1.0] (reference 2025 uses this for slight
        // x-axis expansion to keep the car centered in lane)
        for (auto& xv : loop_x_) {
            xv *= 1.01;
        }
        // Determine mission split indices on the centerline geometry first.
        // The right-lane offset can move the first pickup pass farther away from
        // the raw pickup point and accidentally select the second pass.
        pickup_idx_ = find_first_local_min(loop_x_, loop_y_, PICKUP_X, PICKUP_Y, 0.8, 0);
        dropoff_idx_ = find_first_local_min(loop_x_, loop_y_, DROPOFF_X, DROPOFF_Y, 1.0, pickup_idx_);
        hub_idx_ = static_cast<int>(loop_x_.size()) - 1;

        // Shift the path to the right-lane center for controller tracking.
        // Keep the split indices from centerline ordering (same waypoint order).
        offset_path_to_right_lane_center(loop_x_, loop_y_, 0.05);
    }

    // Generate mission legs by slicing the canonical loop at pickup/dropoff.
    if (!loop_x_.empty() && pickup_idx_ >= 0 && dropoff_idx_ >= 0) {
        int n = static_cast<int>(loop_x_.size());

        int hp_end = std::min(pickup_idx_ + 1, n);
        routes_["hub_to_pickup"] = {
            std::vector<double>(loop_x_.begin(), loop_x_.begin() + hp_end),
            std::vector<double>(loop_y_.begin(), loop_y_.begin() + hp_end)
        };

        int pd_start = pickup_idx_;
        int pd_end = std::min(dropoff_idx_ + 1, n);
        if (pd_start < pd_end) {
            routes_["pickup_to_dropoff"] = {
                std::vector<double>(loop_x_.begin() + pd_start, loop_x_.begin() + pd_end),
                std::vector<double>(loop_y_.begin() + pd_start, loop_y_.begin() + pd_end)
            };
        }

        int dh_start = dropoff_idx_;
        routes_["dropoff_to_hub"] = {
            std::vector<double>(loop_x_.begin() + dh_start, loop_x_.end()),
            std::vector<double>(loop_y_.begin() + dh_start, loop_y_.end())
        };
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
                                     double cur_x, double cur_y,
                                     double cur_yaw) const
{
    auto it = routes_.find(route_name);
    if (it == routes_.end()) return std::nullopt;

    const auto& route = it->second;
    int start_idx = find_closest_idx_directional(route.x, route.y, cur_x, cur_y, cur_yaw);

    int n = static_cast<int>(route.x.size());
    int segment_start = start_idx;
    int segment_len = n - segment_start;

    // If we're too close to the route end, keep at least ~1m of path to avoid
    // near-zero-length local plans that can trap the controller in no-progress loops.
    int min_segment_pts = std::min(n, 1000);  // ds=0.001 -> ~1.0m
    if (segment_len < min_segment_pts) {
        segment_start = std::max(0, n - min_segment_pts);
        segment_len = n - segment_start;
    }

    std::vector<double> rx(route.x.begin() + segment_start,
                           route.x.begin() + segment_start + segment_len);
    std::vector<double> ry(route.y.begin() + segment_start,
                           route.y.begin() + segment_start + segment_len);

    // Prepend current position if not already close, but only if it doesn't
    // create a direction reversal (matching Python plan_path_for_mission_leg).
    double dist_to_start = std::hypot(cur_x - rx.front(), cur_y - ry.front());
    if (dist_to_start > 0.02 && rx.size() >= 2) {
        double prepend_dx = rx[0] - cur_x;
        double prepend_dy = ry[0] - cur_y;
        double route_dx = rx[1] - rx[0];
        double route_dy = ry[1] - ry[0];
        double dot = prepend_dx * route_dx + prepend_dy * route_dy;
        if (dot >= 0) {
            rx.insert(rx.begin(), cur_x);
            ry.insert(ry.begin(), cur_y);
        }
    }

    return std::make_pair(std::move(rx), std::move(ry));
}

std::optional<std::pair<std::vector<double>, std::vector<double>>>
RoadGraph::plan_path(double start_x, double start_y,
                     double goal_x, double goal_y) const
{
    // Find the route whose start/end is closest to the requested start/goal
    const Route* best_route = nullptr;
    double best_score = 1e18;

    for (auto& [name, route] : routes_) {
        if (route.x.empty()) continue;
        double start_dist = std::hypot(start_x - route.x.front(),
                                       start_y - route.y.front());
        double end_dist = std::hypot(goal_x - route.x.back(),
                                     goal_y - route.y.back());
        double score = start_dist + end_dist;
        if (score < best_score) {
            best_score = score;
            best_route = &route;
        }
    }

    if (!best_route) return std::nullopt;

    int start_idx = find_closest_idx(best_route->x, best_route->y, start_x, start_y);
    int goal_idx = find_closest_idx(best_route->x, best_route->y, goal_x, goal_y);

    if (goal_idx <= start_idx) {
        return std::make_pair(best_route->x, best_route->y);
    }
    return std::make_pair(
        std::vector<double>(best_route->x.begin() + start_idx,
                            best_route->x.begin() + goal_idx + 1),
        std::vector<double>(best_route->y.begin() + start_idx,
                            best_route->y.begin() + goal_idx + 1));
}

std::optional<std::pair<std::vector<double>, std::vector<double>>>
RoadGraph::plan_path_from_pose(double cur_x, double cur_y,
                               double goal_x, double goal_y) const
{
    // Find the route whose endpoint is closest to goal and has the vehicle nearby
    const Route* best_route = nullptr;
    double best_score = 1e18;

    for (auto& [name, route] : routes_) {
        if (route.x.empty()) continue;
        double end_dist = std::hypot(goal_x - route.x.back(),
                                     goal_y - route.y.back());
        int closest = find_closest_idx(route.x, route.y, cur_x, cur_y);
        double start_dist = std::hypot(cur_x - route.x[closest],
                                       cur_y - route.y[closest]);
        double score = start_dist + 2.0 * end_dist;  // weight goal proximity higher
        if (score < best_score) {
            best_score = score;
            best_route = &route;
        }
    }

    if (!best_route) return std::nullopt;

    int start_idx = find_closest_idx(best_route->x, best_route->y, cur_x, cur_y);
    int n = static_cast<int>(best_route->x.size());
    int segment_len = n - start_idx;

    int min_segment_pts = std::min(n, 1000);  // ds=0.001 -> ~1.0m
    if (segment_len < min_segment_pts) {
        start_idx = std::max(0, n - min_segment_pts);
        segment_len = n - start_idx;
    }

    std::vector<double> rx(best_route->x.begin() + start_idx,
                           best_route->x.begin() + start_idx + segment_len);
    std::vector<double> ry(best_route->y.begin() + start_idx,
                           best_route->y.begin() + start_idx + segment_len);

    // Prepend current position if not already close, with direction check
    double dist_to_start = std::hypot(cur_x - rx.front(), cur_y - ry.front());
    if (dist_to_start > 0.02 && rx.size() >= 2) {
        double prepend_dx = rx[0] - cur_x;
        double prepend_dy = ry[0] - cur_y;
        double route_dx = rx[1] - rx[0];
        double route_dy = ry[1] - ry[0];
        double dot = prepend_dx * route_dx + prepend_dy * route_dy;
        if (dot >= 0) {
            rx.insert(rx.begin(), cur_x);
            ry.insert(ry.begin(), cur_y);
        }
    }

    return std::make_pair(std::move(rx), std::move(ry));
}

}  // namespace acc
