/**
 * 2D cubic spline path with arc-length parameterization.
 *
 * Ported from pympc_core/spline_path.py CubicSplinePath.
 * Provides smooth C2-continuous path interpolation for MPCC.
 *
 * Uses Eigen for the tridiagonal system solve and vector operations.
 */

#pragma once

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

        // Smooth waypoints to limit curvature to vehicle capability.
        // Sharp corners (e.g. SCS path segment junctions with curvature >> 2.0)
        // are physically impossible to track and cause sustained CTE spikes.
        // This pass averages points near high-curvature regions to round corners,
        // while preserving smooth sections exactly.
        wp_x_ = wx;
        wp_y_ = wy;
        smooth_sharp_corners(wp_x_, wp_y_);

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
        // Clamp curvature to vehicle's physical capability.
        // Max steering δ_max = π/6 (30°), wheelbase L = 0.256m.
        // Max achievable curvature: κ_max = tan(δ_max)/L = tan(30°)/0.256 ≈ 2.26.
        // Higher curvature causes physically impossible reference points that
        // the solver can never track, leading to sustained CTE spikes.
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

    /// Find closest point on path to (x,y), searching forward from s_min.
    /// Returns arc-length of closest point. Searches within a window of ~1m.
    /// Used by solver for fast adaptive path re-projection during SQP.
    double find_closest_progress_from(double x, double y, double s_min) const {
        // Find starting waypoint index at s_min
        int start_idx = 0;
        for (int i = 0; i < n_points_ - 1; i++) {
            if (s_values_[i + 1] >= s_min) { start_idx = i; break; }
        }
        // Allow small backward search (5 points) for accuracy
        start_idx = std::max(0, start_idx - 5);

        // Search forward up to ~1.5m from s_min (covers full horizon)
        double s_end = s_min + 1.5;
        int end_idx = n_points_ - 1;
        for (int i = start_idx; i < n_points_; i++) {
            if (s_values_[i] > s_end) { end_idx = i; break; }
        }

        double best_dist_sq = 1e18;
        int closest_idx = start_idx;
        for (int i = start_idx; i <= end_idx; ++i) {
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
        int lo = std::max(start_idx, closest_idx - 1);
        int hi = std::min(end_idx, closest_idx + 2);
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
        // Enforce monotonicity: result must be >= s_min
        return std::max(best_s, s_min);
    }

    /// Waypoint accessors
    double waypoint_x(int i) const { return wp_x_[i]; }
    double waypoint_y(int i) const { return wp_y_[i]; }
    double s_value(int i) const { return s_values_[i]; }

private:
    struct Coeffs { double a, b, c, d; };

    /// Natural cubic spline using Thomas algorithm (tridiagonal solver, O(n)).
    /// The system is tridiagonal: only diagonals -1, 0, +1 are non-zero.
    /// Using a dense matrix + QR decomposition would be O(n^3) and allocate
    /// n^2 doubles — catastrophic for large paths (e.g. 9769 points = 760 MB).
    static void compute_spline_coeffs(const std::vector<double>& t,
                                      const std::vector<double>& y,
                                      std::vector<Coeffs>& out)
    {
        int n = static_cast<int>(t.size()) - 1;

        // h[i] = t[i+1] - t[i]
        std::vector<double> h(n);
        for (int i = 0; i < n; ++i)
            h[i] = t[i + 1] - t[i];

        // Tridiagonal system for natural cubic spline second derivatives (c).
        // Boundary conditions: c[0] = 0, c[n] = 0 (natural spline).
        // Interior rows: h[i-1]*c[i-1] + 2*(h[i-1]+h[i])*c[i] + h[i]*c[i+1] = rhs[i]
        // Thomas algorithm: forward sweep then back substitution.
        std::vector<double> diag(n + 1, 0.0);   // main diagonal
        std::vector<double> upper(n + 1, 0.0);   // upper diagonal
        std::vector<double> rhs(n + 1, 0.0);     // right-hand side

        // Boundary rows
        diag[0] = 1.0;
        diag[n] = 1.0;

        // Interior rows
        for (int i = 1; i < n; ++i) {
            // lower[i] = h[i-1] (used inline during forward sweep)
            diag[i] = 2.0 * (h[i - 1] + h[i]);
            upper[i] = h[i];
            rhs[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] -
                             (y[i] - y[i - 1]) / h[i - 1]);
        }

        // Forward sweep (eliminate lower diagonal)
        for (int i = 1; i <= n; ++i) {
            double lower_i = (i <= n - 1) ? h[i - 1] : 0.0;
            if (i == n) lower_i = 0.0;  // boundary row has no lower
            double w = (std::abs(diag[i - 1]) > 1e-30) ? lower_i / diag[i - 1] : 0.0;
            diag[i] -= w * upper[i - 1];
            rhs[i] -= w * rhs[i - 1];
        }

        // Back substitution
        std::vector<double> c_vec(n + 1, 0.0);
        c_vec[n] = (std::abs(diag[n]) > 1e-30) ? rhs[n] / diag[n] : 0.0;
        for (int i = n - 1; i >= 0; --i) {
            c_vec[i] = (std::abs(diag[i]) > 1e-30)
                ? (rhs[i] - upper[i] * c_vec[i + 1]) / diag[i]
                : 0.0;
        }

        out.resize(n);
        for (int i = 0; i < n; ++i) {
            Coeffs& cf = out[i];
            cf.a = y[i];
            cf.b = (y[i + 1] - y[i]) / h[i] -
                   h[i] * (2.0 * c_vec[i] + c_vec[i + 1]) / 3.0;
            cf.c = c_vec[i];
            cf.d = (c_vec[i + 1] - c_vec[i]) / (3.0 * h[i]);
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

    /// Gaussian smoothing of path waypoints to spread out sharp curvature
    /// transitions. The cubic spline amplifies curvature at points where the
    /// path heading changes rapidly (e.g., SCS straight→arc junctions). Even
    /// though discrete curvature of raw waypoints may be within limits, the
    /// spline's C2 continuity creates overshoot at these transitions.
    ///
    /// Gaussian smoothing with σ=150 (~150mm at 1mm spacing) spreads curvature
    /// transitions over ~300mm, giving the solver ~3 control steps to adapt
    /// steering. On straight sections this has zero effect (averaging collinear
    /// points returns the same line). On constant-curvature arcs, it slightly
    /// reduces curvature (~σ²/(2R) ≈ 22mm displacement for R=0.5m).
    static void smooth_sharp_corners(std::vector<double>& x, std::vector<double>& y) {
        int n = static_cast<int>(x.size());
        if (n < 100) return;

        const int sigma = 150;  // Gaussian σ in waypoint indices (~150mm at 1mm spacing)
        const int half = sigma * 3;  // 3σ coverage for kernel

        // Build Gaussian kernel
        std::vector<double> kernel(2 * half + 1);
        double ksum = 0.0;
        for (int i = -half; i <= half; i++) {
            kernel[i + half] = std::exp(-0.5 * (double)(i * i) / (double)(sigma * sigma));
            ksum += kernel[i + half];
        }
        for (auto& k : kernel) k /= ksum;

        // Convolve — clamp at boundaries (extends edge values)
        std::vector<double> nx(n), ny(n);
        for (int i = 0; i < n; i++) {
            double sx = 0.0, sy = 0.0;
            for (int j = -half; j <= half; j++) {
                int idx = std::clamp(i + j, 0, n - 1);
                sx += x[idx] * kernel[j + half];
                sy += y[idx] * kernel[j + half];
            }
            nx[i] = sx;
            ny[i] = sy;
        }

        // Preserve exact start and end positions
        nx[0] = x[0]; ny[0] = y[0];
        nx[n - 1] = x[n - 1]; ny[n - 1] = y[n - 1];

        x = nx;
        y = ny;
    }

    bool built_ = false;
    bool use_linear_ = true;
    int n_points_ = 0;
    double total_length_ = 0.0;
    double max_curvature_ = 2.25;  // tan(30°)/0.256m ≈ vehicle's min turn radius

    std::vector<double> wp_x_, wp_y_;
    std::vector<double> s_values_;
    std::vector<Coeffs> cx_, cy_;
};

}  // namespace acc
