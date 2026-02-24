/**
 * 2D cubic spline path with arc-length parameterization.
 *
 * Ported from pympc_core/spline_path.py CubicSplinePath.
 * Provides smooth C2-continuous path interpolation for MPCC.
 *
 * Uses Eigen for the tridiagonal system solve and vector operations.
 */

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace acc {

class CubicSplinePath {
public:
    /// Build cubic spline through waypoints (Nx2 stored as x[], y[]).
    /// smooth=true uses cubic spline; false uses piecewise linear.
    CubicSplinePath() = default;

    void build(const std::vector<double>& wx, const std::vector<double>& wy,
               bool smooth = true)
    {
        n_points_ = static_cast<int>(wx.size());
        if (n_points_ < 2) {
            throw std::runtime_error("CubicSplinePath: need at least 2 waypoints");
        }

        wp_x_ = wx;
        wp_y_ = wy;

        // Compute cumulative arc length
        s_values_.resize(n_points_);
        s_values_[0] = 0.0;
        for (int i = 1; i < n_points_; ++i) {
            double dx = wp_x_[i] - wp_x_[i - 1];
            double dy = wp_y_[i] - wp_y_[i - 1];
            s_values_[i] = s_values_[i - 1] + std::sqrt(dx * dx + dy * dy);
        }
        total_length_ = s_values_[n_points_ - 1];

        if (total_length_ < 1e-6) {
            throw std::runtime_error("CubicSplinePath: path has zero length");
        }

        if (smooth && n_points_ >= 4) {
            use_linear_ = false;
            compute_spline_coeffs(s_values_, wp_x_, cx_);
            compute_spline_coeffs(s_values_, wp_y_, cy_);
        } else {
            use_linear_ = true;
        }

        built_ = true;
    }

    /// Build from interleaved pairs: [(x0,y0), (x1,y1), ...]
    void build(const std::vector<std::pair<double, double>>& waypoints,
               bool smooth = true)
    {
        std::vector<double> wx(waypoints.size()), wy(waypoints.size());
        for (size_t i = 0; i < waypoints.size(); ++i) {
            wx[i] = waypoints[i].first;
            wy[i] = waypoints[i].second;
        }
        build(wx, wy, smooth);
    }

    bool is_built() const { return built_; }
    double total_length() const { return total_length_; }
    int n_points() const { return n_points_; }

    // --- Queries ---

    void get_position(double s, double& x, double& y) const {
        if (use_linear_) { linear_position(s, x, y); return; }
        int idx; double ds;
        find_segment(s, idx, ds);
        auto& cx = cx_[idx];
        auto& cy = cy_[idx];
        x = cx.a + cx.b * ds + cx.c * ds * ds + cx.d * ds * ds * ds;
        y = cy.a + cy.b * ds + cy.c * ds * ds + cy.d * ds * ds * ds;
    }

    double get_tangent(double s) const {
        if (use_linear_) { return linear_tangent(s); }
        int idx; double ds;
        find_segment(s, idx, ds);
        auto& cx = cx_[idx];
        auto& cy = cy_[idx];
        double dx_ds = cx.b + 2.0 * cx.c * ds + 3.0 * cx.d * ds * ds;
        double dy_ds = cy.b + 2.0 * cy.c * ds + 3.0 * cy.d * ds * ds;
        return std::atan2(dy_ds, dx_ds);
    }

    double get_curvature(double s) const {
        if (use_linear_) return 0.0;
        int idx; double ds;
        find_segment(s, idx, ds);
        auto& cx = cx_[idx];
        auto& cy = cy_[idx];
        double dx_ds  = cx.b + 2.0 * cx.c * ds + 3.0 * cx.d * ds * ds;
        double dy_ds  = cy.b + 2.0 * cy.c * ds + 3.0 * cy.d * ds * ds;
        double d2x_ds = 2.0 * cx.c + 6.0 * cx.d * ds;
        double d2y_ds = 2.0 * cy.c + 6.0 * cy.d * ds;
        double denom = std::pow(dx_ds * dx_ds + dy_ds * dy_ds, 1.5);
        if (std::abs(denom) < 1e-10) return 0.0;
        double kappa = (dx_ds * d2y_ds - dy_ds * d2x_ds) / denom;
        // Clamp curvature to prevent spikes at sharp corners between straight
        // segments (e.g. 58-degree corner at node 1 in hub_to_pickup route).
        // Max curvature ~5.0 corresponds to a minimum turn radius of 0.2m,
        // well within QCar2 capability but prevents solver over-reaction.
        return std::clamp(kappa, -max_curvature_, max_curvature_);
    }

    /// Find arc-length of closest point on path to (x, y).
    double find_closest_progress(double x, double y) const {
        // Coarse: closest waypoint
        double best_dist_sq = 1e18;
        int closest_idx = 0;
        for (int i = 0; i < n_points_; ++i) {
            double dx = wp_x_[i] - x;
            double dy = wp_y_[i] - y;
            double d2 = dx * dx + dy * dy;
            if (d2 < best_dist_sq) {
                best_dist_sq = d2;
                closest_idx = i;
            }
        }
        double best_s = s_values_[closest_idx];
        double best_dist = std::sqrt(best_dist_sq);

        // Fine: project onto adjacent segments
        int lo = std::max(0, closest_idx - 1);
        int hi = std::min(n_points_ - 1, closest_idx + 2);
        for (int i = lo; i < hi; ++i) {
            if (i >= n_points_ - 1) continue;
            double vx = wp_x_[i + 1] - wp_x_[i];
            double vy = wp_y_[i + 1] - wp_y_[i];
            double seg_sq = vx * vx + vy * vy;
            if (seg_sq < 1e-10) continue;
            double ux = x - wp_x_[i];
            double uy = y - wp_y_[i];
            double t = std::clamp((ux * vx + uy * vy) / seg_sq, 0.0, 1.0);
            double px = wp_x_[i] + t * vx;
            double py = wp_y_[i] + t * vy;
            double d = std::hypot(x - px, y - py);
            if (d < best_dist) {
                best_dist = d;
                best_s = s_values_[i] + t * std::sqrt(seg_sq);
            }
        }
        return best_s;
    }

    /// Compute contouring (lateral) and lag (longitudinal) errors.
    void compute_contouring_errors(double x, double y, double s,
                                   double& e_c, double& e_l) const
    {
        double ref_x, ref_y;
        get_position(s, ref_x, ref_y);
        double theta = get_tangent(s);
        double dx = x - ref_x;
        double dy = y - ref_y;
        double cos_t = std::cos(theta);
        double sin_t = std::sin(theta);
        e_l =  cos_t * dx + sin_t * dy;
        e_c = -sin_t * dx + cos_t * dy;
    }

    /// Get path reference: (x, y, cos_tangent, sin_tangent).
    void get_path_reference(double s,
                            double& ref_x, double& ref_y,
                            double& cos_t, double& sin_t) const
    {
        get_position(s, ref_x, ref_y);
        double theta = get_tangent(s);
        cos_t = std::cos(theta);
        sin_t = std::sin(theta);
    }

    /// Waypoint accessors
    double waypoint_x(int i) const { return wp_x_[i]; }
    double waypoint_y(int i) const { return wp_y_[i]; }
    double s_value(int i) const { return s_values_[i]; }

private:
    struct Coeffs { double a, b, c, d; };

    /// Natural cubic spline tridiagonal solver.
    static void compute_spline_coeffs(const std::vector<double>& t,
                                      const std::vector<double>& y,
                                      std::vector<Coeffs>& out)
    {
        int n = static_cast<int>(t.size()) - 1;

        // h[i] = t[i+1] - t[i]
        Eigen::VectorXd h(n);
        for (int i = 0; i < n; ++i)
            h(i) = t[i + 1] - t[i];

        // Set up tridiagonal system for second derivatives (c)
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n + 1, n + 1);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);

        A(0, 0) = 1.0;
        A(n, n) = 1.0;

        for (int i = 1; i < n; ++i) {
            A(i, i - 1) = h(i - 1);
            A(i, i) = 2.0 * (h(i - 1) + h(i));
            A(i, i + 1) = h(i);
            rhs(i) = 3.0 * ((y[i + 1] - y[i]) / h(i) -
                             (y[i] - y[i - 1]) / h(i - 1));
        }

        Eigen::VectorXd c_vec = A.colPivHouseholderQr().solve(rhs);

        out.resize(n);
        for (int i = 0; i < n; ++i) {
            Coeffs& cf = out[i];
            cf.a = y[i];
            cf.b = (y[i + 1] - y[i]) / h(i) -
                   h(i) * (2.0 * c_vec(i) + c_vec(i + 1)) / 3.0;
            cf.c = c_vec(i);
            cf.d = (c_vec(i + 1) - c_vec(i)) / (3.0 * h(i));
        }
    }

    void find_segment(double s, int& idx, double& ds) const {
        s = std::clamp(s, 0.0, total_length_);
        // Binary search for segment
        auto it = std::upper_bound(s_values_.begin(), s_values_.end(), s);
        idx = static_cast<int>(it - s_values_.begin()) - 1;
        idx = std::clamp(idx, 0, n_points_ - 2);
        ds = s - s_values_[idx];
    }

    void linear_position(double s, double& x, double& y) const {
        int idx; double ds;
        find_segment(s, idx, ds);
        double seg_len = s_values_[idx + 1] - s_values_[idx];
        if (seg_len < 1e-10) {
            x = wp_x_[idx]; y = wp_y_[idx]; return;
        }
        double alpha = ds / seg_len;
        x = wp_x_[idx] + alpha * (wp_x_[idx + 1] - wp_x_[idx]);
        y = wp_y_[idx] + alpha * (wp_y_[idx + 1] - wp_y_[idx]);
    }

    double linear_tangent(double s) const {
        int idx; double ds;
        find_segment(s, idx, ds);
        return std::atan2(wp_y_[idx + 1] - wp_y_[idx],
                          wp_x_[idx + 1] - wp_x_[idx]);
    }

    bool built_ = false;
    bool use_linear_ = true;
    int n_points_ = 0;
    double total_length_ = 0.0;
    double max_curvature_ = 5.0;  // ~0.2m minimum turn radius

    std::vector<double> wp_x_, wp_y_;
    std::vector<double> s_values_;
    std::vector<Coeffs> cx_, cy_;
};

}  // namespace acc
