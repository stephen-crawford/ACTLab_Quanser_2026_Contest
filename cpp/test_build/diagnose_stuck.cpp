/**
 * Diagnostic: Analyze mission path at specific progress percentages where
 * the vehicle gets stuck (22% and 79%).
 *
 * Build:
 *   cd /home/stephen/quanser-acc/cpp/test_build
 *   g++ -std=c++17 -O2 -I.. -I/usr/include/eigen3 \
 *       -o diagnose_stuck diagnose_stuck.cpp ../road_graph.cpp
 *
 * Run:
 *   ./diagnose_stuck
 */

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "coordinate_transform.h"
#include "cubic_spline_path.h"
#include "road_graph.h"

// =========================================================================
// Analyze path at a given progress fraction
// =========================================================================
struct ProgressAnalysis {
    double progress_frac;
    double s;              // arc-length
    double x, y;           // position in map frame
    double tangent_deg;    // tangent angle in degrees
    double curvature;      // curvature at this point
    double curv_min_200mm; // min curvature in ±200mm window
    double curv_max_200mm; // max curvature in ±200mm window
    double curv_avg_200mm; // avg curvature in ±200mm window
};

ProgressAnalysis analyze_at_progress(const acc::CubicSplinePath& spline,
                                     double frac)
{
    ProgressAnalysis a;
    a.progress_frac = frac;
    double total = spline.total_length();
    a.s = frac * total;

    spline.get_position(a.s, a.x, a.y);
    double tangent_rad = spline.get_tangent(a.s);
    a.tangent_deg = tangent_rad * 180.0 / M_PI;
    a.curvature = spline.get_curvature(a.s);

    // Analyze curvature in ±200mm window
    double window = 0.200; // 200mm
    double s_lo = std::max(0.0, a.s - window);
    double s_hi = std::min(total, a.s + window);
    int n_samples = 100;
    double ds = (s_hi - s_lo) / (n_samples - 1);

    double kmin = 1e18, kmax = -1e18, ksum = 0.0;
    for (int i = 0; i < n_samples; i++) {
        double s_sample = s_lo + i * ds;
        double k = spline.get_curvature(s_sample);
        if (k < kmin) kmin = k;
        if (k > kmax) kmax = k;
        ksum += k;
    }
    a.curv_min_200mm = kmin;
    a.curv_max_200mm = kmax;
    a.curv_avg_200mm = ksum / n_samples;

    return a;
}

// =========================================================================
// Print analysis for a leg
// =========================================================================
void print_analysis(const std::string& leg_name,
                    const acc::CubicSplinePath& spline,
                    const std::vector<double>& fracs)
{
    double total = spline.total_length();

    std::printf("\n");
    std::printf("================================================================\n");
    std::printf("  LEG: %s\n", leg_name.c_str());
    std::printf("  Total length: %.4f m  (%d waypoints)\n", total, spline.n_points());
    std::printf("================================================================\n");

    // Start heading
    double start_tangent = spline.get_tangent(0.0) * 180.0 / M_PI;
    double start_x, start_y;
    spline.get_position(0.0, start_x, start_y);
    std::printf("\n  Start position:  (%.4f, %.4f) map frame\n", start_x, start_y);
    std::printf("  Start heading:   %.2f deg\n", start_tangent);

    // End heading
    double end_tangent = spline.get_tangent(total - 0.001) * 180.0 / M_PI;
    double end_x, end_y;
    spline.get_position(total - 0.001, end_x, end_y);
    std::printf("  End position:    (%.4f, %.4f) map frame\n", end_x, end_y);
    std::printf("  End heading:     %.2f deg\n", end_tangent);

    std::printf("\n  %-8s  %-10s  %-20s  %-12s  %-10s  %-36s\n",
                "Prog%", "Arc(m)", "Position(x,y)", "Tangent(deg)",
                "Curvature", "Window ±200mm (min/max/avg)");
    std::printf("  %-8s  %-10s  %-20s  %-12s  %-10s  %-36s\n",
                "------", "--------", "------------------", "----------",
                "--------", "----------------------------------");

    for (double frac : fracs) {
        ProgressAnalysis a = analyze_at_progress(spline, frac);
        std::printf("  %5.1f%%   %8.4f   (%+8.4f, %+8.4f)   %+8.2f      %+8.4f   (min=%+7.4f max=%+7.4f avg=%+7.4f)\n",
                    a.progress_frac * 100.0,
                    a.s,
                    a.x, a.y,
                    a.tangent_deg,
                    a.curvature,
                    a.curv_min_200mm, a.curv_max_200mm, a.curv_avg_200mm);
    }
}

// =========================================================================
// Also find the max curvature and its location on each leg
// =========================================================================
void find_max_curvature_regions(const std::string& leg_name,
                                const acc::CubicSplinePath& spline)
{
    double total = spline.total_length();
    int n_scan = 2000;
    double ds = total / (n_scan - 1);

    // Find top 5 curvature peaks
    struct Peak { double s; double k; double frac; double x, y; };
    std::vector<Peak> peaks;

    double prev_k = 0.0;
    bool rising = false;
    for (int i = 0; i < n_scan; i++) {
        double s = i * ds;
        double k = std::abs(spline.get_curvature(s));
        if (i > 0 && k < prev_k && rising && prev_k > 0.5) {
            // We just passed a peak
            double peak_s = (i - 1) * ds;
            double px, py;
            spline.get_position(peak_s, px, py);
            peaks.push_back({peak_s, prev_k, peak_s / total, px, py});
        }
        rising = (k >= prev_k);
        prev_k = k;
    }

    // Sort by curvature descending
    std::sort(peaks.begin(), peaks.end(),
              [](const Peak& a, const Peak& b) { return a.k > b.k; });

    std::printf("\n  Top curvature peaks (|k| > 0.5):\n");
    std::printf("  %-10s  %-8s  %-10s  %-20s\n",
                "Curvature", "Prog%", "Arc(m)", "Position(x,y)");
    int show = std::min(10, (int)peaks.size());
    for (int i = 0; i < show; i++) {
        std::printf("  %+8.4f    %5.1f%%   %8.4f   (%+8.4f, %+8.4f)\n",
                    peaks[i].k, peaks[i].frac * 100.0, peaks[i].s,
                    peaks[i].x, peaks[i].y);
    }
    if (peaks.empty()) {
        std::printf("  (no peaks with |k| > 0.5)\n");
    }
}

// =========================================================================
// Curvature profile: print curvature at every 5% progress
// =========================================================================
void print_curvature_profile(const std::string& leg_name,
                             const acc::CubicSplinePath& spline)
{
    double total = spline.total_length();
    std::printf("\n  Curvature profile (every 5%%):\n");
    std::printf("  ");
    for (int p = 0; p <= 100; p += 5) {
        double s = (p / 100.0) * total;
        double k = spline.get_curvature(s);
        std::printf("%3d%%:%+6.3f  ", p, k);
        if ((p / 5) % 5 == 4) std::printf("\n  ");
    }
    std::printf("\n");
}

// =========================================================================
// Main
// =========================================================================
int main()
{
    std::printf("=============================================================\n");
    std::printf("  MISSION PATH DIAGNOSTIC — Stuck-Point Analysis\n");
    std::printf("  Analyzing progress at 20%%, 22%%, 25%%, 75%%, 79%%, 82%%\n");
    std::printf("=============================================================\n");

    // 1. Build road graph
    std::printf("\nBuilding road graph (ds=0.001)...\n");
    acc::RoadGraph rg(0.001);

    // Print loop info
    std::printf("Loop path: %zu waypoints\n", rg.loop_x().size());
    std::printf("Pickup index: %d\n", rg.pickup_index());
    std::printf("Dropoff index: %d\n", rg.dropoff_index());
    std::printf("Hub (end) index: %d\n", rg.hub_index());

    // 2. Get mission legs
    auto routes = rg.get_route_names();
    std::printf("\nAvailable routes: ");
    for (const auto& r : routes) std::printf("%s  ", r.c_str());
    std::printf("\n");

    struct LegInfo {
        std::string name;
        std::string route_name;
    };

    std::vector<LegInfo> legs = {
        {"Hub -> Pickup", "hub_to_pickup"},
        {"Pickup -> Dropoff", "pickup_to_dropoff"},
        {"Dropoff -> Hub", "dropoff_to_hub"}
    };

    // Progress fractions to analyze
    std::vector<double> fracs = {0.10, 0.15, 0.20, 0.22, 0.25, 0.30,
                                  0.50,
                                  0.70, 0.75, 0.79, 0.80, 0.82, 0.85, 0.90, 0.95};

    acc::TransformParams tp;

    for (const auto& leg : legs) {
        auto route_opt = rg.get_route(leg.route_name);
        if (!route_opt) {
            std::printf("\nERROR: Route '%s' not found!\n", leg.route_name.c_str());
            continue;
        }

        auto& [qlabs_x, qlabs_y] = *route_opt;
        std::printf("\n  Route '%s': %zu waypoints (QLabs frame)\n",
                    leg.route_name.c_str(), qlabs_x.size());

        // Transform to map frame
        std::vector<double> map_x, map_y;
        acc::qlabs_path_to_map(qlabs_x, qlabs_y, tp, map_x, map_y);

        // Build cubic spline
        acc::CubicSplinePath spline;
        spline.build(map_x, map_y, true);

        // Print analysis at key progress points
        print_analysis(leg.name, spline, fracs);

        // Print curvature profile
        print_curvature_profile(leg.name, spline);

        // Find max curvature regions
        find_max_curvature_regions(leg.name, spline);

        // Additional: print the heading change rate at stuck points
        std::printf("\n  Heading change rate (deg/m) at stuck points:\n");
        for (double frac : {0.20, 0.22, 0.25, 0.79}) {
            double s = frac * spline.total_length();
            double ds_step = 0.010; // 10mm steps
            double t1 = spline.get_tangent(s - ds_step) * 180.0 / M_PI;
            double t2 = spline.get_tangent(s + ds_step) * 180.0 / M_PI;
            double rate = (t2 - t1) / (2.0 * ds_step);
            std::printf("    %5.1f%%: %.2f deg/m\n", frac * 100.0, rate);
        }

        // Additional: compute what steering angle the vehicle needs
        std::printf("\n  Required steering angle delta = atan(L*kappa) at key points:\n");
        double L = 0.256; // wheelbase
        for (double frac : fracs) {
            double s = frac * spline.total_length();
            double k = spline.get_curvature(s);
            double delta_rad = std::atan(L * k);
            double delta_deg = delta_rad * 180.0 / M_PI;
            std::printf("    %5.1f%%: kappa=%+.4f -> delta=%+.2f deg\n",
                        frac * 100.0, k, delta_deg);
        }
    }

    // 3. Print the QLabs-frame waypoint positions at key indices
    std::printf("\n\n================================================================\n");
    std::printf("  RAW LOOP PATH — Key Index Positions (QLabs Frame)\n");
    std::printf("================================================================\n");

    const auto& lx = rg.loop_x();
    const auto& ly = rg.loop_y();
    int pickup_idx = rg.pickup_index();
    int dropoff_idx = rg.dropoff_index();
    int hub_idx = rg.hub_index();
    int n_total = (int)lx.size();

    std::printf("  Pickup  idx=%d: QLabs (%.4f, %.4f)  target (%.4f, %.4f)  dist=%.4f m\n",
                pickup_idx, lx[pickup_idx], ly[pickup_idx],
                acc::PICKUP_X, acc::PICKUP_Y,
                std::hypot(lx[pickup_idx] - acc::PICKUP_X, ly[pickup_idx] - acc::PICKUP_Y));
    std::printf("  Dropoff idx=%d: QLabs (%.4f, %.4f)  target (%.4f, %.4f)  dist=%.4f m\n",
                dropoff_idx, lx[dropoff_idx], ly[dropoff_idx],
                acc::DROPOFF_X, acc::DROPOFF_Y,
                std::hypot(lx[dropoff_idx] - acc::DROPOFF_X, ly[dropoff_idx] - acc::DROPOFF_Y));
    std::printf("  Hub     idx=%d: QLabs (%.4f, %.4f)  target (%.4f, %.4f)  dist=%.4f m\n",
                hub_idx, lx[hub_idx], ly[hub_idx],
                acc::HUB_X, acc::HUB_Y,
                std::hypot(lx[hub_idx] - acc::HUB_X, ly[hub_idx] - acc::HUB_Y));

    // Print leg lengths
    std::printf("\n  Leg waypoint counts:\n");
    std::printf("    hub_to_pickup:      0 -> %d = %d waypoints\n", pickup_idx, pickup_idx);
    std::printf("    pickup_to_dropoff:  %d -> %d = %d waypoints\n",
                pickup_idx, dropoff_idx, dropoff_idx - pickup_idx);
    std::printf("    dropoff_to_hub:     %d -> %d = %d waypoints\n",
                dropoff_idx, hub_idx, hub_idx - dropoff_idx);
    std::printf("    total loop:         %d waypoints\n", n_total);

    std::printf("\n  Done.\n");
    return 0;
}
