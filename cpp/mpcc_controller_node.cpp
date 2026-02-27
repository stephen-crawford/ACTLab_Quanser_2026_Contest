/**
 * C++ MPCC Controller ROS2 Node
 *
 * Replaces the Python mpcc_controller.py with a high-performance C++ node.
 * Uses the acados MPCC solver for the core MPCC optimization.
 *
 * Subscriptions:
 *   /odom               - nav_msgs/Odometry (velocity)
 *   /plan               - nav_msgs/Path (reference path)
 *   /traffic_control_state - std_msgs/String (JSON traffic state)
 *   /motion_enable       - std_msgs/Bool
 *   /mission/hold        - std_msgs/Bool
 *
 * Publications:
 *   /cmd_vel_nav         - geometry_msgs/Twist
 *   /mpcc/status         - std_msgs/String
 *   /mpcc/predicted_path - visualization_msgs/MarkerArray
 *
 * TF:
 *   Listens for map->base_link transform for vehicle pose.
 *
 * Control loop runs at 20Hz.
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <qcar2_interfaces/msg/motor_commands.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>

#include "mpcc_solver_interface.h"
#include "cubic_spline_path.h"
#include "road_boundaries.h"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <mutex>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <filesystem>

// Coordinate transform angle (empirically calibrated, matches reference repo)
static constexpr double TRANSFORM_THETA = 0.7177;

namespace acc_mpcc {

/**
 * Reference path with cumulative distance, tangent angles, and curvature.
 */
struct ReferencePath {
    std::vector<double> x, y;
    std::vector<double> cumulative_dist;
    std::vector<double> tangent_angle;
    std::vector<double> curvature;
    double total_length = 0.0;
    int n_points = 0;

    void build(const std::vector<std::pair<double,double>>& waypoints) {
        n_points = static_cast<int>(waypoints.size());
        if (n_points < 2) return;

        x.resize(n_points);
        y.resize(n_points);
        cumulative_dist.resize(n_points, 0.0);
        tangent_angle.resize(n_points, 0.0);
        curvature.resize(n_points, 0.0);

        for (int i = 0; i < n_points; i++) {
            x[i] = waypoints[i].first;
            y[i] = waypoints[i].second;
        }

        for (int i = 1; i < n_points; i++) {
            double dx = x[i] - x[i-1];
            double dy = y[i] - y[i-1];
            cumulative_dist[i] = cumulative_dist[i-1] + std::sqrt(dx*dx + dy*dy);
            tangent_angle[i-1] = std::atan2(dy, dx);
        }
        tangent_angle[n_points-1] = tangent_angle[n_points-2];
        total_length = cumulative_dist[n_points-1];

        // Compute curvature from tangent angle changes
        for (int i = 1; i < n_points - 1; i++) {
            double ds = cumulative_dist[i+1] - cumulative_dist[i-1];
            if (ds > 1e-6) {
                double dtheta = tangent_angle[i+1] - tangent_angle[i-1];
                // Normalize
                while (dtheta > M_PI) dtheta -= 2*M_PI;
                while (dtheta < -M_PI) dtheta += 2*M_PI;
                curvature[i] = dtheta / ds;
            }
        }
        curvature[0] = curvature[1];
        curvature[n_points-1] = curvature[n_points-2];
    }

    int find_closest_index(double px, double py) const {
        double min_dist = 1e9;
        int best = 0;
        for (int i = 0; i < n_points; i++) {
            double dx = x[i] - px;
            double dy = y[i] - py;
            double d = dx*dx + dy*dy;
            if (d < min_dist) {
                min_dist = d;
                best = i;
            }
        }
        return best;
    }

    double find_closest_progress(double px, double py) const {
        int idx = find_closest_index(px, py);
        double best_progress = cumulative_dist[idx];
        double min_dist = std::hypot(x[idx]-px, y[idx]-py);

        // Check adjacent segments for interpolated progress
        if (idx > 0) {
            double vx = x[idx] - x[idx-1], vy = y[idx] - y[idx-1];
            double ux = px - x[idx-1], uy = py - y[idx-1];
            double seg_sq = vx*vx + vy*vy;
            if (seg_sq > 1e-10) {
                double t = std::clamp((ux*vx + uy*vy) / seg_sq, 0.0, 1.0);
                double proj_x = x[idx-1] + t*vx, proj_y = y[idx-1] + t*vy;
                double d = std::hypot(proj_x-px, proj_y-py);
                if (d < min_dist) {
                    min_dist = d;
                    best_progress = cumulative_dist[idx-1] + t*std::sqrt(seg_sq);
                }
            }
        }
        if (idx < n_points-1) {
            double vx = x[idx+1] - x[idx], vy = y[idx+1] - y[idx];
            double ux = px - x[idx], uy = py - y[idx];
            double seg_sq = vx*vx + vy*vy;
            if (seg_sq > 1e-10) {
                double t = std::clamp((ux*vx + uy*vy) / seg_sq, 0.0, 1.0);
                double proj_x = x[idx] + t*vx, proj_y = y[idx] + t*vy;
                double d = std::hypot(proj_x-px, proj_y-py);
                if (d < min_dist) {
                    best_progress = cumulative_dist[idx] + t*std::sqrt(seg_sq);
                }
            }
        }
        return best_progress;
    }

    // Get path references for the MPCC solver at progress offsets
    std::vector<mpcc::PathRef> get_path_refs(double start_progress, int horizon,
                                              double v_ref, double dt,
                                              double /*max_velocity*/ = 1.2) const {
        std::vector<mpcc::PathRef> refs(horizon + 1);
        double lookahead_speed = v_ref;  // Match reference: use v_ref for horizon spacing
        for (int k = 0; k <= horizon; k++) {
            double s = start_progress + k * lookahead_speed * dt;
            s = std::clamp(s, 0.0, total_length - 0.001);

            // Find segment
            int idx = 0;
            for (int i = 1; i < n_points; i++) {
                if (cumulative_dist[i] > s) { idx = i-1; break; }
                idx = i;
            }
            idx = std::min(idx, n_points-2);

            double seg_len = cumulative_dist[idx+1] - cumulative_dist[idx];
            double alpha = 0.0;
            if (seg_len > 1e-6) {
                alpha = (s - cumulative_dist[idx]) / seg_len;
            }

            refs[k].x = x[idx] + alpha * (x[idx+1] - x[idx]);
            refs[k].y = y[idx] + alpha * (y[idx+1] - y[idx]);
            double ta = tangent_angle[idx];
            refs[k].cos_theta = std::cos(ta);
            refs[k].sin_theta = std::sin(ta);
            refs[k].curvature = curvature[idx] + alpha * (curvature[idx+1] - curvature[idx]);
        }
        return refs;
    }
};

class MPCCControllerNode : public rclcpp::Node {
public:
    MPCCControllerNode() : Node("mpcc_controller_cpp") {
        // Parameters — tuned to fix swerving (Feb 2026)
        // Key changes: startup phase re-enabled (3s), steering_rate_weight 3.0→1.5,
        // velocity tracking at k=0 only (matching reference R_ref structure)
        this->declare_parameter("reference_velocity", 0.45);
        this->declare_parameter("contour_weight", 8.0);
        this->declare_parameter("lag_weight", 15.0);
        this->declare_parameter("horizon", 10);
        this->declare_parameter("boundary_weight", 0.0);
        this->declare_parameter("boundary_default_width", 0.22);
        this->declare_parameter("road_boundaries_config", std::string(""));
        this->declare_parameter("use_direct_motor", true);
        this->declare_parameter("use_state_estimator", false);
        this->declare_parameter("steering_slew_rate", 1.0);  // rad/s command limiter (anti-oversteer)
        // Startup weight overrides — during first 3s, use HIGHER steering damping
        // to prevent aggressive steering during path alignment. Previous values
        // (sr=0.05, contour=1.0) allowed 30x more aggressive steering → oversteering.
        this->declare_parameter("startup_contour_weight", 6.0);
        this->declare_parameter("startup_lag_weight", 12.0);
        this->declare_parameter("startup_velocity_weight", 15.0);
        this->declare_parameter("startup_heading_weight", 0.5);
        this->declare_parameter("startup_steering_rate_weight", 2.0);
        this->declare_parameter("startup_curvature_decay", -0.4);

        // Initialize solver with tuned config
        mpcc::Config cfg;
        cfg.horizon = this->get_parameter("horizon").as_int();
        cfg.dt = 0.1;
        cfg.wheelbase = 0.256;
        cfg.max_velocity = 0.55;  // Hard speed ceiling — close to v_ref to prevent solver hitting max
        cfg.min_velocity = 0.0;   // Reference uses u_min=[0.0, ...]
        cfg.max_steering = 0.45;  // ±25.8° — hardware servo limit (ref uses π/6=30° but servo clips at 0.45)
        cfg.max_acceleration = 1.5;   // Reference has no explicit limit; allow fast response
        cfg.max_steering_rate = 1.5;  // Reference has no explicit limit; allow fast steering
        cfg.reference_velocity = this->get_parameter("reference_velocity").as_double();
        cfg.contour_weight = this->get_parameter("contour_weight").as_double();
        cfg.lag_weight = this->get_parameter("lag_weight").as_double();
        cfg.velocity_weight = 15.0;  // Track v_ref strongly (ref: R_ref[0]=17.0)
        cfg.steering_weight = 0.05;   // Reference R_ref[1]=0.05, penalizes |δ| gently
        cfg.acceleration_weight = 0.01;  // Reference R_u[0]=0.005; near-zero for fast speed changes
        cfg.steering_rate_weight = 1.1; // Match reference R_u[1]=1.1 exactly
        cfg.jerk_weight = 0.0;       // Reference has no jerk penalty
        cfg.robot_radius = 0.13;
        cfg.safety_margin = 0.10;
        cfg.obstacle_weight = 200.0;
        cfg.boundary_weight = this->get_parameter("boundary_weight").as_double();
        cfg.boundary_default_width = this->get_parameter("boundary_default_width").as_double();
        cfg.max_sqp_iterations = 5;   // PyMPC uses 5; was 3 (insufficient for curve convergence)
        cfg.max_qp_iterations = 20;   // qpOASES iteration limit (acados multiplies by 10 internally)
        cfg.qp_tolerance = 1e-5;
        cfg.startup_contour_weight = this->get_parameter("startup_contour_weight").as_double();
        cfg.startup_lag_weight = this->get_parameter("startup_lag_weight").as_double();
        cfg.startup_velocity_weight = this->get_parameter("startup_velocity_weight").as_double();
        cfg.startup_heading_weight = this->get_parameter("startup_heading_weight").as_double();
        cfg.startup_steering_rate_weight = this->get_parameter("startup_steering_rate_weight").as_double();
        cfg.startup_curvature_decay = this->get_parameter("startup_curvature_decay").as_double();
        solver_.init(cfg);
        config_ = cfg;

        // Road boundaries (YAML-driven)
        auto rb_config = this->get_parameter("road_boundaries_config").as_string();
        if (rb_config.empty()) {
            try {
                auto pkg = ament_index_cpp::get_package_share_directory("acc_stage1_mission");
                rb_config = pkg + "/config/road_boundaries.yaml";
            } catch (...) {
                try {
                    auto pkg = ament_index_cpp::get_package_share_directory("acc_mpcc_controller_cpp");
                    rb_config = pkg + "/config/road_boundaries.yaml";
                } catch (...) {}
            }
        }
        if (!rb_config.empty()) {
            road_boundaries_ = std::make_unique<acc::RoadBoundarySpline>(rb_config);
            RCLCPP_INFO(this->get_logger(), "Road boundaries loaded from %s", rb_config.c_str());
        }

        // Direct motor mode (bypass nav2_qcar_command_convert)
        use_direct_motor_ = this->get_parameter("use_direct_motor").as_bool();

        use_state_estimator_ = this->get_parameter("use_state_estimator").as_bool();
        steering_slew_rate_ = this->get_parameter("steering_slew_rate").as_double();

        // CubicSplinePath (built when path received)
        spline_path_ = std::make_unique<acc::CubicSplinePath>();

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publishers
        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel_nav", 10);
        if (use_direct_motor_) {
            motor_pub_ = this->create_publisher<qcar2_interfaces::msg::MotorCommands>(
                "/qcar2_motor_speed_cmd", 1);
        }
        status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/mpcc/status", 10);
        viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/mpcc/predicted_path", 10);
        telemetry_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/mpcc/telemetry", 10);
        replan_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/mpcc/replan_request", 10);

        // Subscribers
        auto qos_be = rclcpp::QoS(1).best_effort();
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", qos_be,
            [this](nav_msgs::msg::Odometry::SharedPtr msg) { odom_callback(msg); });

        // Use reliable() but NOT transient_local() for /plan subscription.
        // Nav2's planner also publishes on /plan with volatile durability.
        // A transient_local subscriber rejects volatile publishers, causing
        // "incompatible QoS DURABILITY_QOS_POLICY" and dropping all messages.
        // The mission manager republishes every 2s, so latching isn't needed.
        auto qos_plan = rclcpp::QoS(10).reliable();
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/plan", qos_plan,
            [this](nav_msgs::msg::Path::SharedPtr msg) { path_callback(msg); });

        motion_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/motion_enable", 10,
            [this](std_msgs::msg::Bool::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                double now_s = this->now().seconds();
                if (msg->data) {
                    // Require sustained true before re-enabling (prevents
                    // intermittent detections from resetting auto-resume timer)
                    motion_enable_consecutive_++;
                    if (motion_enable_consecutive_ >= MOTION_ENABLE_HYSTERESIS) {
                        motion_enabled_ = true;
                        motion_disabled_time_ = 0.0;
                        motion_enable_consecutive_ = 0;
                    }
                    // Don't reset motion_disabled_time_ on single true messages
                } else {
                    motion_enable_consecutive_ = 0;  // Reset consecutive counter
                    // Ignore false during post-resume cooldown
                    if (motion_resume_cooldown_time_ > 0 &&
                        (now_s - motion_resume_cooldown_time_) < motion_resume_cooldown_s_) {
                        return;
                    }
                    if (motion_enabled_) {
                        // First time going false — record timestamp
                        motion_disabled_time_ = now_s;
                    }
                    motion_enabled_ = false;
                }
            });

        hold_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/mission/hold", 10,
            [this](std_msgs::msg::Bool::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                mission_hold_ = msg->data;
            });

        traffic_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/traffic_control_state", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                traffic_state_json_ = msg->data;
            });

        obstacle_pos_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/obstacle_positions", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                parse_obstacle_positions(msg->data);
            });

        // JointState subscriber for encoder-based velocity (more accurate than TF)
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/qcar2_joint", qos_be,
            [this](sensor_msgs::msg::JointState::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                joint_state_callback(msg);
            });

        // Vehicle state subscriber (from EKF state estimator)
        if (use_state_estimator_) {
            vehicle_state_sub_ = this->create_subscription<std_msgs::msg::String>(
                "/vehicle_state", 10,
                [this](std_msgs::msg::String::SharedPtr msg) {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    parse_vehicle_state(msg->data);
                });
        }

        // Obstacle map subscriber (from obstacle tracker)
        obstacle_map_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/obstacle_map", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                parse_obstacle_map(msg->data);
            });

        // Control timer — MUST match solver dt (0.1s = 10Hz)
        // Reference (PolyCtrl 2025) uses controllerUpdateRate=10 with dt=0.1.
        // Running at 20Hz with dt=0.1 causes warm-start mismatch: the solver
        // shifts trajectory by one step (0.1s) every 0.05s call, making the
        // predicted trajectory out of sync with reality by 2x.
        int control_rate_ms = static_cast<int>(cfg.dt * 1000.0);  // 100ms for dt=0.1
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(control_rate_ms),
            [this]() { control_loop(); });

        start_time_ = this->now();
        init_log_file();

        RCLCPP_INFO(this->get_logger(),
            "C++ MPCC Controller [v3 - reference-aligned tuning]: horizon=%d, v_ref=%.2f, max_v=%.2f, contour=%.1f, lag=%.1f, vel_w=%.1f, boundary=%.1f, direct_motor=%s",
            cfg.horizon, cfg.reference_velocity, cfg.max_velocity,
            cfg.contour_weight, cfg.lag_weight, cfg.velocity_weight, cfg.boundary_weight,
            use_direct_motor_ ? "true" : "false");
    }

private:
    void odom_callback(nav_msgs::msg::Odometry::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        odom_velocity_ = msg->twist.twist.linear.x;
        if (msg->header.frame_id == "map") {
            state_x_ = msg->pose.pose.position.x;
            state_y_ = msg->pose.pose.position.y;
            auto& q = msg->pose.pose.orientation;
            state_theta_ = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                                       1.0 - 2.0*(q.y*q.y + q.z*q.z));
            state_v_ = msg->twist.twist.linear.x;
            has_odom_ = true;
        }
    }

    void joint_state_callback(sensor_msgs::msg::JointState::SharedPtr msg) {
        // Compute actual velocity from encoder ticks (matches reference MPC_node.py:65)
        // v = (velocity[0] / (720*4)) * ((13*19)/(70*30)) * (2*pi) * 0.033
        if (!msg->velocity.empty()) {
            double encoder_vel = msg->velocity[0];
            double v = (encoder_vel / (720.0 * 4.0))
                       * ((13.0 * 19.0) / (70.0 * 30.0))
                       * (2.0 * M_PI) * 0.033;
            joint_velocity_ = v;
            has_joint_velocity_ = true;
        }
    }

    void path_callback(nav_msgs::msg::Path::SharedPtr msg) {
        if (msg->poses.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "Path too short (%zu poses)", msg->poses.size());
            return;
        }

        std::vector<std::pair<double,double>> waypoints;
        waypoints.reserve(msg->poses.size());
        for (auto& ps : msg->poses) {
            waypoints.emplace_back(ps.pose.position.x, ps.pose.position.y);
        }

        std::lock_guard<std::mutex> lock(state_mutex_);

        // Only rebuild path + reset solver if path actually changed
        // (mission_manager re-publishes the same path periodically)
        bool path_changed = false;
        if (!has_path_ ||
            static_cast<int>(waypoints.size()) != ref_path_.n_points) {
            path_changed = true;
        } else {
            // Check first and last waypoints for change
            double d0 = std::hypot(waypoints.front().first - ref_path_.x[0],
                                    waypoints.front().second - ref_path_.y[0]);
            double d1 = std::hypot(waypoints.back().first - ref_path_.x.back(),
                                    waypoints.back().second - ref_path_.y.back());
            if (d0 > 0.01 || d1 > 0.01) {
                path_changed = true;
            }
        }

        if (path_changed) {
            // Reset startup ramp timer on EVERY new path (not just the first).
            // Each mission leg starts from a stop with a new path, so it needs
            // its own startup ramp. Previously only reset on first path, causing
            // legs 2 and 3 to never get a startup ramp.
            start_time_ = this->now();

            ref_path_.build(waypoints);
            has_path_ = true;
            current_progress_ = ref_path_.find_closest_progress(state_x_, state_y_);
            solver_.reset();
            // Reset state_delta_ to match solver's reset u_prev_.
            state_delta_ = 0.0;

            // Reset ALL secondary state for clean leg transition.
            // Without this, stale timers/counters from previous leg cause:
            // - stuck_timer_: spurious solver reset at new leg start
            // - solver_failure_count_: premature replan request
            // - sustained_cross_track_start_: immediate CTE alert on new leg
            // - steering_saturated_count_: stale diagnostic counter
            // - replan_requested_: stale replan state
            stuck_timer_ = 0.0;
            solver_failure_count_ = 0;
            steering_saturated_count_ = 0;
            sustained_cross_track_start_ = 0.0;
            replan_requested_ = false;

            // Build cubic spline path too (for smooth curvature)
            try {
                std::vector<double> wx, wy;
                wx.reserve(waypoints.size());
                wy.reserve(waypoints.size());
                for (auto& [x, y] : waypoints) { wx.push_back(x); wy.push_back(y); }
                spline_path_->build(wx, wy, true);
            } catch (...) {
                // Fall back to piecewise-linear if spline fails
            }

            bool spline_ok = spline_path_ && spline_path_->is_built();

            // Set spline_path for acados solver's theta_A-based reference lookup
            if (spline_ok) {
                solver_.spline_path = spline_path_.get();
            } else {
                solver_.spline_path = nullptr;
            }

            RCLCPP_INFO(this->get_logger(), "Path received: %zu waypoints, length=%.2fm, spline=%s",
                         waypoints.size(), ref_path_.total_length,
                         spline_ok ? "smooth" : "linear");

            char buf[128];
            std::snprintf(buf, sizeof(buf), "New path: %zu waypoints, length=%.2fm",
                          waypoints.size(), ref_path_.total_length);
            log_event(buf);
        } else {
            // Same path re-published — just update progress, keep warm-start
            current_progress_ = ref_path_.find_closest_progress(state_x_, state_y_);
        }
    }

    bool update_state_from_tf() {
        try {
            auto t = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            state_x_ = t.transform.translation.x;
            state_y_ = t.transform.translation.y;
            auto& q = t.transform.rotation;
            state_theta_ = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                                       1.0 - 2.0*(q.y*q.y + q.z*q.z));

            // Estimate velocity from position change
            auto now = this->now().seconds();
            if (last_pose_time_ > 0) {
                double dt = now - last_pose_time_;
                if (dt > 0.01) {
                    double dx = state_x_ - last_x_;
                    double dy = state_y_ - last_y_;
                    state_v_ = std::sqrt(dx*dx + dy*dy) / dt;
                }
            }
            last_pose_time_ = now;
            last_x_ = state_x_;
            last_y_ = state_y_;
            has_odom_ = true;
            return true;
        } catch (const tf2::TransformException&) {
            return false;
        }
    }

    bool parse_traffic_should_stop() {
        // Simple JSON parsing for should_stop field
        if (traffic_state_json_.empty()) return false;
        auto pos = traffic_state_json_.find("\"should_stop\"");
        if (pos == std::string::npos) return false;
        auto val_pos = traffic_state_json_.find("true", pos);
        auto false_pos = traffic_state_json_.find("false", pos);
        if (val_pos != std::string::npos &&
            (false_pos == std::string::npos || val_pos < false_pos)) {
            return true;
        }
        return false;
    }

    // Parse obstacle positions from JSON string
    void parse_obstacle_positions(const std::string& json) {
        detected_obstacles_.clear();
        // Simple JSON array parsing for {"obstacles": [{x, y, radius}, ...]}
        size_t pos = json.find("\"obstacles\"");
        if (pos == std::string::npos) return;

        size_t arr_start = json.find('[', pos);
        if (arr_start == std::string::npos) return;

        size_t search = arr_start;
        while (true) {
            size_t obj_start = json.find('{', search);
            if (obj_start == std::string::npos) break;
            size_t obj_end = json.find('}', obj_start);
            if (obj_end == std::string::npos) break;

            std::string obj = json.substr(obj_start, obj_end - obj_start + 1);
            mpcc::Obstacle obs;
            obs.x = parse_json_double(obj, "\"x\"");
            obs.y = parse_json_double(obj, "\"y\"");
            obs.radius = parse_json_double(obj, "\"radius\"");
            if (obs.radius > 0.01) {
                detected_obstacles_.push_back(obs);
            }

            search = obj_end + 1;
            if (json.find(']', search) < json.find('{', search)) break;
        }
    }

    static double parse_json_double(const std::string& json, const std::string& key) {
        auto pos = json.find(key);
        if (pos == std::string::npos) return 0.0;
        auto colon = json.find(':', pos);
        if (colon == std::string::npos) return 0.0;
        try {
            return std::stod(json.substr(colon + 1));
        } catch (...) {
            return 0.0;
        }
    }

    void parse_vehicle_state(const std::string& json) {
        // Parse: {"x":..,"y":..,"theta":..,"v":..,"omega":..}
        double x = parse_json_double(json, "\"x\"");
        double y = parse_json_double(json, "\"y\"");
        double theta = parse_json_double(json, "\"theta\"");
        double v = parse_json_double(json, "\"v\"");

        estimator_x_ = x;
        estimator_y_ = y;
        estimator_theta_ = theta;
        estimator_v_ = v;
        has_estimator_state_ = true;
        last_estimator_time_ = this->now().seconds();
    }

    void parse_obstacle_map(const std::string& json) {
        // Parse mapped obstacles as solver obstacles (with velocity from Kalman tracker)
        // Merge with raw detections (mapped obstacles have filtered positions)
        mapped_obstacles_.clear();
        size_t pos = json.find("\"obstacles\"");
        if (pos == std::string::npos) return;
        size_t arr_start = json.find('[', pos);
        if (arr_start == std::string::npos) return;

        size_t search = arr_start;
        while (true) {
            size_t obj_start = json.find('{', search);
            if (obj_start == std::string::npos) break;
            size_t obj_end = json.find('}', obj_start);
            if (obj_end == std::string::npos) break;

            std::string obj = json.substr(obj_start, obj_end - obj_start + 1);
            mpcc::Obstacle obs;
            obs.x = parse_json_double(obj, "\"x\"");
            obs.y = parse_json_double(obj, "\"y\"");
            obs.radius = parse_json_double(obj, "\"radius\"");
            obs.vx = parse_json_double(obj, "\"vx\"");
            obs.vy = parse_json_double(obj, "\"vy\"");
            if (obs.radius > 0.01) {
                mapped_obstacles_.push_back(obs);
            }

            search = obj_end + 1;
            if (json.find(']', search) < json.find('{', search)) break;
        }
    }

    void publish_telemetry(double v_ref, double v_cmd, double cross_track,
                           double heading_err, double progress_pct,
                           double steering, const std::string& leg,
                           int n_obstacles, const std::string& state_str) {
        char buf[512];
        double v_meas = has_joint_velocity_ ? joint_velocity_ : state_v_;
        std::snprintf(buf, sizeof(buf),
            "{\"timestamp\":%.6f,\"v_reference\":%.4f,\"v_measured\":%.4f,"
            "\"v_command\":%.4f,\"cross_track_error\":%.4f,\"heading_error\":%.4f,"
            "\"progress_pct\":%.1f,\"steering_angle\":%.4f,\"mission_leg\":\"%s\","
            "\"obstacles\":%d,\"state\":\"%s\"}",
            this->now().seconds(), v_ref, v_meas, v_cmd,
            cross_track, heading_err, progress_pct, steering,
            leg.c_str(), n_obstacles, state_str.c_str());
        auto msg = std_msgs::msg::String();
        msg.data = buf;
        telemetry_pub_->publish(msg);
    }

    // Compute cross-track error (signed lateral distance from path)
    double compute_cross_track_error() const {
        if (ref_path_.n_points < 2) return 0.0;
        int idx = ref_path_.find_closest_index(state_x_, state_y_);
        int next = std::min(idx + 1, ref_path_.n_points - 1);
        int prev = std::max(idx - 1, 0);
        double tx = ref_path_.x[next] - ref_path_.x[prev];
        double ty = ref_path_.y[next] - ref_path_.y[prev];
        double tlen = std::sqrt(tx*tx + ty*ty);
        if (tlen < 1e-6) return 0.0;
        // Normal pointing left: (-ty, tx)
        double nx = -ty / tlen, ny = tx / tlen;
        double dx = state_x_ - ref_path_.x[idx];
        double dy = state_y_ - ref_path_.y[idx];
        return dx * nx + dy * ny;
    }

    // Compute heading error (angle between vehicle heading and path tangent)
    double compute_heading_error() const {
        if (ref_path_.n_points < 2) return 0.0;
        int idx = ref_path_.find_closest_index(state_x_, state_y_);
        double tangent = ref_path_.tangent_angle[std::min(idx, ref_path_.n_points - 1)];
        double err = state_theta_ - tangent;
        while (err > M_PI) err -= 2*M_PI;
        while (err < -M_PI) err += 2*M_PI;
        return err;
    }

    // Generate smooth spline-based path references for the MPCC solver.
    // Unlike piecewise-linear ReferencePath, CubicSplinePath provides C2-continuous
    // tangent angles and bounded curvature, preventing solver oscillation at sharp
    // path segment junctions where curvature can spike to >100.
    //
    // Uses curvature-adaptive lookahead: on curves, horizon points are spaced
    // closer together (matching actual vehicle capability), preventing the solver
    // from chasing distant references that race ahead of the vehicle.
    // This matches the reference MPCC (CasADi/IPOPT) where arc-length progress
    // is a decision variable that naturally slows on curves.
    std::vector<mpcc::PathRef> get_spline_path_refs(
        double start_progress, int horizon, double v_ref, double dt,
        double /*actual_v*/)
    {
        // Need horizon+1 refs: the solver evaluates cost at k=0..N (N+1 points)
        // and the condensed QP builds sensitivity for k=1..N
        std::vector<mpcc::PathRef> refs(horizon + 1);
        double total_len = spline_path_->total_length();
        // Use v_ref as base speed for lookahead. This ensures sufficient horizon
        // coverage (~0.65m) even during startup or low-speed situations.
        // Curvature decay tightens spacing on curves where the vehicle naturally
        // slows, preventing references from racing ahead of the vehicle.
        // (Don't use actual_v — it collapses the horizon to ~0.15m at startup,
        // making the solver blind to upcoming curves and causing loops.)
        double s = start_progress;
        for (int k = 0; k <= horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);

            double ref_x, ref_y, cos_t, sin_t;
            spline_path_->get_path_reference(s, ref_x, ref_y, cos_t, sin_t);

            refs[k].x = ref_x;
            refs[k].y = ref_y;
            refs[k].cos_theta = cos_t;
            refs[k].sin_theta = sin_t;
            refs[k].curvature = spline_path_->get_curvature(s);

            // Advance by curvature-adaptive speed at this point.
            // On straights (κ≈0): step = v_ref * dt = 0.065m (full lookahead)
            // On curves (κ=1.25): step = v_ref * exp(-0.5) * dt = 0.039m (40% tighter)
            // On tight curves (κ=2): step = v_ref * exp(-0.8) * dt = 0.029m (55% tighter)
            double curv = std::abs(refs[k].curvature);
            double step_speed = v_ref * std::exp(-0.4 * curv);
            step_speed = std::max(step_speed, 0.10);
            s += step_speed * dt;
        }
        return refs;
    }

    // Generate road boundary constraints from path geometry or YAML config.
    // Uses spline path for smooth tangent angles (avoids boundary discontinuities at
    // piecewise-linear segment junctions).
    // Uses curvature-adaptive spacing to match get_spline_path_refs — boundaries
    // are evaluated at the same arc-length positions as path references.
    std::vector<mpcc::BoundaryConstraint> generate_boundaries(
        double start_progress, int horizon, double v_ref, double dt,
        double /*actual_v*/)
    {
        std::vector<mpcc::BoundaryConstraint> boundaries(horizon);
        double half_width = config_.boundary_default_width;
        bool use_spline = spline_path_ && spline_path_->is_built();
        double total_len = use_spline ? spline_path_->total_length() : ref_path_.total_length;
        double s = start_progress;

        for (int k = 0; k < horizon; k++) {
            s = std::clamp(s, 0.0, total_len - 0.001);

            double cx, cy, ta;
            double curv = 0.0;

            if (use_spline) {
                // Smooth spline-based position and tangent
                spline_path_->get_position(s, cx, cy);
                ta = spline_path_->get_tangent(s);
                curv = std::abs(spline_path_->get_curvature(s));
            } else {
                // Fallback: piecewise-linear
                int idx = 0;
                for (int i = 1; i < ref_path_.n_points; i++) {
                    if (ref_path_.cumulative_dist[i] > s) { idx = i - 1; break; }
                    idx = i;
                }
                idx = std::min(idx, ref_path_.n_points - 2);

                ta = ref_path_.tangent_angle[idx];

                double seg_len = ref_path_.cumulative_dist[idx + 1] -
                                 ref_path_.cumulative_dist[idx];
                double alpha = 0.0;
                if (seg_len > 1e-6) {
                    alpha = (s - ref_path_.cumulative_dist[idx]) / seg_len;
                }
                cx = ref_path_.x[idx] + alpha * (ref_path_.x[idx + 1] - ref_path_.x[idx]);
                cy = ref_path_.y[idx] + alpha * (ref_path_.y[idx + 1] - ref_path_.y[idx]);
            }

            // Try YAML-driven road boundaries first
            if (road_boundaries_) {
                auto br = road_boundaries_->get_boundary_constraints(cx, cy, ta);
                boundaries[k].nx = br.nx;
                boundaries[k].ny = br.ny;
                boundaries[k].b_left = br.b_left;
                boundaries[k].b_right = br.b_right;
            } else {
                // Fallback: path-tangent-based boundaries
                double nx = -std::sin(ta);  // Normal pointing left
                double ny =  std::cos(ta);
                double center_proj = nx * cx + ny * cy;
                boundaries[k].nx = nx;
                boundaries[k].ny = ny;
                boundaries[k].b_left = center_proj + half_width;
                boundaries[k].b_right = -(center_proj - half_width);
            }

            // Advance by curvature-adaptive speed (matching get_spline_path_refs)
            double step_speed = v_ref * std::exp(-0.4 * curv);
            step_speed = std::max(step_speed, 0.10);
            s += step_speed * dt;
        }
        return boundaries;
    }

    void control_loop() {
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Use state estimator if available and recent (< 0.5s)
        double now_s = this->now().seconds();
        bool using_estimator = false;
        if (use_state_estimator_ && has_estimator_state_ &&
            (now_s - last_estimator_time_) < 0.5) {
            state_x_ = estimator_x_;
            state_y_ = estimator_y_;
            state_theta_ = estimator_theta_;
            state_v_ = estimator_v_;
            has_odom_ = true;
            using_estimator = true;
        } else {
            // Fallback: raw TF + encoder
            update_state_from_tf();

            // Use best available velocity source
            if (has_joint_velocity_) {
                state_v_ = joint_velocity_;
            } else if (std::abs(odom_velocity_) > 0.001) {
                state_v_ = odom_velocity_;
            }
        }

        // Log position source on first acquisition
        if (has_odom_ && !logged_first_odom_) {
            logged_first_odom_ = true;
            RCLCPP_INFO(this->get_logger(),
                "First vehicle position acquired via %s: (%.3f, %.3f, %.1f deg)",
                using_estimator ? "state_estimator" : "raw TF",
                state_x_, state_y_, state_theta_ * 180.0 / M_PI);
        }

        if (!has_odom_ || !has_path_ || ref_path_.n_points < 2) {
            // Log why we're not running (throttled to every 5s)
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Control loop waiting: has_odom=%d (estimator=%d, tf_fallback=%d), "
                "has_path=%d, n_points=%d",
                has_odom_, has_estimator_state_,
                (has_odom_ && !has_estimator_state_), has_path_, ref_path_.n_points);
            return;
        }

        // Check hold
        if (mission_hold_) {
            publish_stop();
            return;
        }

        // Check motion enable — instead of binary stop, use obstacle-aware speed
        // limiting. The MPCC solver should always run (with obstacles in its cost
        // function) so it can steer around obstacles. Only fully stop if the
        // obstacle is very close AND the controller has been trying to avoid
        // for a sustained duration.
        bool obstacle_speed_limit_active = false;
        if (!motion_enabled_) {
            double now_s = this->now().seconds();
            if (motion_disabled_time_ > 0 &&
                (now_s - motion_disabled_time_) > motion_disable_timeout_) {
                // Auto-resume: obstacle detector likely has a false positive
                RCLCPP_WARN(this->get_logger(),
                    "motion_enable false for %.1fs, auto-resuming (timeout=%.0fs)",
                    now_s - motion_disabled_time_, motion_disable_timeout_);
                {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf),
                        "Auto-resume: motion disabled for %.1fs (timeout=%.0fs)",
                        now_s - motion_disabled_time_, motion_disable_timeout_);
                    log_event(buf);
                }
                motion_enabled_ = true;
                motion_disabled_time_ = 0.0;
                motion_resume_cooldown_time_ = now_s;
            } else if (!detected_obstacles_.empty() || !mapped_obstacles_.empty()) {
                // Obstacles detected: don't fully stop, let the MPCC solver handle
                // avoidance. Apply speed limit instead.
                obstacle_speed_limit_active = true;
                // Allow the solver to run with reduced speed
            } else {
                // No obstacles detected but motion disabled (traffic sign/light stop)
                publish_stop();
                return;
            }
        }

        // Check traffic control (skip during post-resume cooldown)
        {
            double now_s = this->now().seconds();
            bool in_cooldown = motion_resume_cooldown_time_ > 0 &&
                (now_s - motion_resume_cooldown_time_) < motion_resume_cooldown_s_;
            if (!in_cooldown && parse_traffic_should_stop()) {
                publish_stop();
                return;
            }
        }

        // Update progress (monotonic forward-only, matching reference MPC_node.py)
        // Reference uses index-based monotonic progression: theta_idx only increases.
        // This prevents the solver from getting confused when the vehicle veers
        // laterally — closest-point can jump backward, causing oscillation.
        double new_progress;
        double path_total_len;
        if (spline_path_ && spline_path_->is_built()) {
            new_progress = spline_path_->find_closest_progress(state_x_, state_y_);
            path_total_len = spline_path_->total_length();
        } else {
            new_progress = ref_path_.find_closest_progress(state_x_, state_y_);
            path_total_len = ref_path_.total_length;
        }
        // Strictly monotonic: only advance, never go backward.
        // NEVER reset progress backward — it causes the solver to chase old references,
        // amplifying oscillation. If the vehicle overshoots, the solver will naturally
        // correct by continuing forward (lag weight > contour weight = progress-first).
        if (new_progress > current_progress_) {
            current_progress_ = new_progress;
            stuck_timer_ = 0.0;
        } else {
            stuck_timer_ += config_.dt;
            if (stuck_timer_ > 5.0) {
                RCLCPP_WARN(this->get_logger(),
                    "No progress for %.1fs (progress=%.1f%%, closest=%.1f%%)",
                    stuck_timer_, current_progress_ / path_total_len * 100.0,
                    new_progress / path_total_len * 100.0);
                // Only reset solver warm-start, NOT progress
                solver_.reset();
                stuck_timer_ = 0.0;
            }
        }

        // Check goal reached
        double remaining = path_total_len - current_progress_;
        if (remaining < 0.15) {
            publish_stop();
            publish_status("Goal reached");
            log_event("Goal reached — path complete");
            return;
        }

        // Build solver state
        mpcc::VecX x0;
        x0 << state_x_, state_y_, state_theta_, state_v_, state_delta_;

        // Compute startup elapsed time
        double elapsed = (this->now() - start_time_).seconds();
        config_.startup_elapsed_s = elapsed;
        // Enable diagnostics extraction always in deployment
        config_.diagnostics_enabled = true;
        // Write per-stage data adaptively (every 10th cycle or when CTE is high)
        config_.per_stage_logging = (log_cycle_counter_ % 10 == 0) ||
                                     (std::abs(compute_cross_track_error()) > 0.15);
        solver_.config = config_;

        // Get path references — prefer spline (smooth curvature) over piecewise-linear
        // Uses curvature-adaptive lookahead: reference spacing based on max of actual
        // vehicle speed and half of reference velocity. This prevents the horizon from
        // extending too far ahead at startup (when v≈0), which caused the solver to
        // see aggressive curves and command bang-bang steering oscillation.
        double lookahead_v = std::max(state_v_, config_.reference_velocity * 0.5);
        std::vector<mpcc::PathRef> path_refs;
        if (spline_path_ && spline_path_->is_built()) {
            path_refs = get_spline_path_refs(
                current_progress_, config_.horizon,
                lookahead_v, config_.dt, state_v_);
        } else {
            path_refs = ref_path_.get_path_refs(
                current_progress_, config_.horizon,
                lookahead_v, config_.dt);
        }

        // Generate road boundary constraints from path geometry
        auto boundaries = generate_boundaries(
            current_progress_, config_.horizon,
            lookahead_v, config_.dt, state_v_);

        // Merge detected obstacles with mapped obstacles (prefer mapped — filtered)
        auto merged_obstacles = mapped_obstacles_.empty()
            ? detected_obstacles_
            : mapped_obstacles_;
        // Add any raw detections not near a mapped obstacle
        if (!mapped_obstacles_.empty()) {
            for (auto& raw : detected_obstacles_) {
                bool near_mapped = false;
                for (auto& mapped : mapped_obstacles_) {
                    if (std::hypot(raw.x - mapped.x, raw.y - mapped.y) < 0.5) {
                        near_mapped = true;
                        break;
                    }
                }
                if (!near_mapped) merged_obstacles.push_back(raw);
            }
        }

        // Saturation monitoring: log when steering has been at max for many cycles,
        // but do NOT reset the solver warm-start. Resetting during sustained turns
        // causes a deadlock where the solver never completes the curve — each reset
        // throws away the accumulated trajectory and the fresh start also saturates,
        // creating an infinite loop of reset → saturate → reset.
        if (steering_saturated_count_ > SATURATION_RESET_THRESHOLD &&
            steering_saturated_count_ % 50 == 0) {
            RCLCPP_INFO(this->get_logger(),
                "Steering saturated for %d cycles (progress=%.1f%%) — maintaining warm-start",
                steering_saturated_count_,
                100.0 * new_progress / path_total_len);
        }

        auto result = solver_.solve(x0, path_refs, current_progress_,
                                     path_total_len,
                                     merged_obstacles, boundaries);

        if (!result.success) {
            solver_failure_count_++;
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                "MPCC solver failed at progress=%.2f/%.2f, pos=(%.3f, %.3f), consecutive=%d",
                current_progress_, ref_path_.total_length, state_x_, state_y_,
                solver_failure_count_);
            // Check if sustained solver failures warrant a replan request
            double progress_pct = 100.0 * current_progress_ / path_total_len;
            check_replan_needed(compute_cross_track_error(), progress_pct);
            publish_stop();
            return;
        }

        // Reset solver failure counter on success
        solver_failure_count_ = 0;

        double v_cmd = result.v_cmd;
        double delta_cmd = result.delta_cmd;

        // Decelerate near goal
        if (remaining < 0.5) {
            double decel_factor = remaining / 0.5;
            v_cmd *= decel_factor;
            if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
        }

        // Apply obstacle speed limit: slow down but keep solver steering active
        if (obstacle_speed_limit_active) {
            v_cmd = std::min(v_cmd, 0.20);  // Creep at 0.20 m/s while avoiding
        }

        v_cmd = std::clamp(v_cmd, config_.min_velocity, config_.max_velocity);
        delta_cmd = std::clamp(delta_cmd, -config_.max_steering, config_.max_steering);

        // Apply a first-order steering slew-rate limit to prevent aggressive
        // command jumps that manifest as oversteer in deployment.
        if (steering_slew_rate_ > 0.0) {
            double max_step = steering_slew_rate_ * config_.dt;
            double lo = state_delta_ - max_step;
            double hi = state_delta_ + max_step;
            delta_cmd = std::clamp(delta_cmd, lo, hi);
            delta_cmd = std::clamp(delta_cmd, -config_.max_steering, config_.max_steering);
        }

        // Publish MotorCommands directly (bypass nav2_qcar_command_convert)
        if (use_direct_motor_ && motor_pub_) {
            qcar2_interfaces::msg::MotorCommands motor_cmd;
            motor_cmd.motor_names = {"steering_angle", "motor_throttle"};
            motor_cmd.values = {delta_cmd, v_cmd};  // Direct: delta in rad, v in m/s
            motor_pub_->publish(motor_cmd);
        }

        // Also publish Twist for debug/visualization (and legacy mode)
        double omega = v_cmd * std::tan(delta_cmd) / config_.wheelbase;
        omega = std::clamp(omega, -1.5, 1.5);

        auto cmd = geometry_msgs::msg::Twist();
        cmd.linear.x = v_cmd;
        cmd.angular.z = omega;
        if (!use_direct_motor_) {
            cmd_pub_->publish(cmd);  // Legacy: send Twist through nav2_qcar_command_convert
        } else {
            cmd_pub_->publish(cmd);  // Debug/visualization only
        }

        // Track steering for next iteration
        state_delta_ = delta_cmd;

        // Track steering saturation for recovery
        if (std::abs(std::abs(delta_cmd) - config_.max_steering) < 0.01) {
            steering_saturated_count_++;
        } else {
            steering_saturated_count_ = 0;
        }

        // Publish status
        double progress_pct = 100.0 * current_progress_ / path_total_len;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "v=%.2f, delta=%.1fdeg, omega=%.2f, progress=%.0f%%, solve=%.0fus, motor=%s",
            v_cmd, delta_cmd * 180.0 / M_PI, omega, progress_pct, result.solve_time_us,
            use_direct_motor_ ? "direct" : "twist");
        publish_status(std::string(buf));

        // Publish predicted path visualization
        publish_predicted_path(result);

        // Publish telemetry for dashboard
        double cross_track = compute_cross_track_error();
        double heading_err = compute_heading_error();
        int n_obs = static_cast<int>(merged_obstacles.size());
        publish_telemetry(config_.reference_velocity, v_cmd, cross_track,
                         heading_err, progress_pct, delta_cmd,
                         "", n_obs, "tracking");

        // Periodic status log (every 2s) so we can see the controller is active
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
            "MPCC: v=%.2f delta=%.1fdeg progress=%.1f%% solve=%.0fus pos=(%.3f,%.3f)",
            v_cmd, delta_cmd * 180.0 / M_PI, progress_pct,
            result.solve_time_us, state_x_, state_y_);

        // Persistent CSV log
        std::string state_src = using_estimator ? "estimator" : "tf+encoder";
        log_cycle(v_cmd, delta_cmd, progress_pct, result.solve_time_us,
                  cross_track, heading_err, n_obs, result, state_src);

        // Check if dynamic replanning is needed
        check_replan_needed(cross_track, progress_pct);
    }

    // ---------------------------------------------------------------
    // File logging (persistent CSV + event log)
    // ---------------------------------------------------------------
    void init_log_file() {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream ts;
        ts << std::put_time(&tm, "%Y%m%d_%H%M%S");

        // Write to the Docker-mounted workspace so logs appear on the host
        std::string log_dir = "/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/logs";
        try { std::filesystem::create_directories(log_dir); } catch (...) {
            // Fallback to /tmp if workspace mount not available
            log_dir = "/tmp/mission_logs";
            try { std::filesystem::create_directories(log_dir); } catch (...) {}
        }

        mpcc_csv_path_ = log_dir + "/mpcc_" + ts.str() + ".csv";
        mpcc_traj_csv_path_ = log_dir + "/mpcc_trajectory_" + ts.str() + ".csv";
        mpcc_event_path_ = log_dir + "/mpcc_events_" + ts.str() + ".log";

        {
            std::ofstream f(mpcc_csv_path_);
            f << "elapsed_s,x,y,theta,v_meas,v_cmd,delta_cmd,progress_pct,"
                 "solve_time_us,cross_track_err,heading_err,n_obstacles,"
                 "motion_enabled,traffic_stop,"
                 "acados_status,sqp_iter,kkt_norm,qp_status,"
                 "res_eq,res_ineq,res_comp,res_stat,"
                 "cost,acados_time_ms,"
                 "warmstart,warmstart_shifts,startup_progress,"
                 "eff_contour_w,eff_lag_w,eff_vel_w,eff_sr_w,eff_progress_w,eff_v_ref_k0,"
                 "obs_x,obs_y,obs_r,obs_dist,"
                 "state_source,tf_age_ms,steering_sat_count\n";
        }
        {
            std::ofstream f(mpcc_traj_csv_path_);
            f << "elapsed_s,stage,x,y,psi,theta_a,v,delta,v_theta,"
                 "ref_x,ref_y,ref_v,ref_curv\n";
        }
        {
            std::ofstream f(mpcc_event_path_);
            f << "# MPCC Controller Event Log - " << ts.str() << "\n";
            f << "# Config: horizon=" << config_.horizon
              << " v_ref=" << config_.reference_velocity
              << " contour=" << config_.contour_weight
              << " lag=" << config_.lag_weight
              << " boundary=" << config_.boundary_weight
              << " steer_rate=" << config_.steering_rate_weight
              << " heading=" << config_.heading_weight
              << " max_steer=" << config_.max_steering
              << " direct_motor=" << (use_direct_motor_ ? "true" : "false") << "\n\n";
        }
        log_start_time_ = this->now().seconds();
        RCLCPP_INFO(this->get_logger(), "=== MPCC LOGS ===");
        RCLCPP_INFO(this->get_logger(), "  CSV:        %s", mpcc_csv_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Trajectory: %s", mpcc_traj_csv_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "  Events:     %s", mpcc_event_path_.c_str());
        RCLCPP_INFO(this->get_logger(), "===============");
    }

    void log_cycle(double v_cmd, double delta_cmd, double progress_pct,
                   double solve_time_us, double cross_track, double heading_err,
                   int n_obstacles, const mpcc::Result& result,
                   const std::string& state_source) {
        // Log every control cycle for debugging (10Hz at dt=0.1s)
        log_cycle_counter_++;

        double elapsed = this->now().seconds() - log_start_time_;
        double v_meas = has_joint_velocity_ ? joint_velocity_ : state_v_;
        bool traffic_stop = parse_traffic_should_stop();
        double tf_age_ms = (last_pose_time_ > 0)
            ? (this->now().seconds() - last_pose_time_) * 1000.0
            : -1.0;

        const auto& d = result.diag;

        try {
            std::ofstream f(mpcc_csv_path_, std::ios::app);
            f << std::fixed << std::setprecision(3) << elapsed << ","
              << std::setprecision(4) << state_x_ << "," << state_y_ << ","
              << std::setprecision(4) << state_theta_ << ","
              << std::setprecision(4) << v_meas << "," << v_cmd << "," << delta_cmd << ","
              << std::setprecision(1) << progress_pct << ","
              << std::setprecision(0) << solve_time_us << ","
              << std::setprecision(4) << cross_track << "," << heading_err << ","
              << n_obstacles << ","
              << (motion_enabled_ ? 1 : 0) << ","
              << (traffic_stop ? 1 : 0) << ","
              // Solver diagnostics
              << d.acados_status << "," << d.sqp_iter << ","
              << std::setprecision(6) << d.kkt_norm_inf << "," << d.qp_status << ","
              << std::setprecision(6) << d.res_eq << "," << d.res_ineq << ","
              << d.res_comp << "," << d.res_stat << ","
              << std::setprecision(2) << result.cost << ","
              << std::setprecision(3) << d.acados_time_tot_ms << ","
              << (d.warmstart_used ? 1 : 0) << "," << d.warmstart_shift_count << ","
              << std::setprecision(3) << d.startup_progress << ","
              // Effective weights
              << std::setprecision(2) << d.eff_contour_w << "," << d.eff_lag_w << ","
              << d.eff_vel_w << "," << d.eff_sr_w << "," << d.eff_progress_w << ","
              << std::setprecision(4) << d.eff_v_ref_k0 << ","
              // Obstacle data
              << std::setprecision(3) << d.obs_x << "," << d.obs_y << ","
              << std::setprecision(4) << d.obs_r << ","
              << std::setprecision(3) << d.obs_dist << ","
              // State source and TF age
              << state_source << ","
              << std::setprecision(1) << tf_age_ms << ","
              << steering_saturated_count_ << "\n";
        } catch (...) {}

        // Write per-stage trajectory adaptively: every 10th cycle or high CTE
        bool write_traj = (log_cycle_counter_ % 10 == 0) ||
                           (std::abs(cross_track) > 0.15);
        if (write_traj && !d.stage_x.empty()) {
            try {
                std::ofstream f(mpcc_traj_csv_path_, std::ios::app);
                int N = config_.horizon;
                for (int k = 0; k <= N; k++) {
                    f << std::fixed << std::setprecision(3) << elapsed << ","
                      << k << ","
                      << std::setprecision(4) << d.stage_x[k] << "," << d.stage_y[k] << ","
                      << d.stage_psi[k] << "," << d.stage_theta_a[k] << ",";
                    if (k < N) {
                        f << d.stage_v[k] << "," << d.stage_delta[k] << "," << d.stage_v_theta[k];
                    } else {
                        f << ",,";
                    }
                    f << ",";
                    if (k < (int)d.ref_x.size()) {
                        f << d.ref_x[k] << "," << d.ref_y[k] << ","
                          << d.ref_v[k] << "," << d.ref_curv[k];
                    } else {
                        f << ",,,";
                    }
                    f << "\n";
                }
            } catch (...) {}
        }
    }

    void log_event(const std::string& event) {
        double elapsed = this->now().seconds() - log_start_time_;
        try {
            std::ofstream f(mpcc_event_path_, std::ios::app);
            f << "[+" << std::fixed << std::setprecision(1) << elapsed << "s] "
              << event << "\n";
        } catch (...) {}
    }

    void check_replan_needed(double cross_track, double progress_pct) {
        double now_s = this->now().seconds();

        // Cooldown: don't request replan too frequently
        if ((now_s - last_replan_request_time_) < REPLAN_COOLDOWN_S) return;

        bool should_request = false;
        std::string reason;
        double ct_abs = std::abs(cross_track);

        // Trigger 1: Sustained cross-track violation
        if (ct_abs > REPLAN_CROSS_TRACK_THRESHOLD) {
            if (sustained_cross_track_start_ <= 0.0) {
                sustained_cross_track_start_ = now_s;
            } else if (!replan_requested_ &&
                       (now_s - sustained_cross_track_start_) > REPLAN_SUSTAINED_DURATION) {
                should_request = true;
                reason = "cross_track_violation";
                replan_requested_ = true;  // debounce: one request per violation episode
            }
        } else {
            // Cross-track recovered below threshold
            sustained_cross_track_start_ = 0.0;
            replan_requested_ = false;
        }

        // Trigger 2: Consecutive solver failures (checked separately in control_loop)
        if (solver_failure_count_ >= REPLAN_SOLVER_FAILURE_THRESHOLD) {
            should_request = true;
            reason = "solver_failure";
            solver_failure_count_ = 0;  // reset after requesting
        }

        if (should_request) {
            char buf[512];
            std::snprintf(buf, sizeof(buf),
                "{\"reason\":\"%s\",\"cross_track_error\":%.4f,"
                "\"progress_pct\":%.1f,\"vehicle_x\":%.4f,\"vehicle_y\":%.4f}",
                reason.c_str(), cross_track, progress_pct, state_x_, state_y_);
            auto msg = std_msgs::msg::String();
            msg.data = buf;
            replan_pub_->publish(msg);
            last_replan_request_time_ = now_s;

            RCLCPP_WARN(this->get_logger(),
                "Replan requested: reason=%s, cross_track=%.3f, progress=%.1f%%",
                reason.c_str(), cross_track, progress_pct);
            log_event("Replan requested: " + reason +
                      ", cross_track=" + std::to_string(cross_track));
        }
    }

    void publish_stop() {
        auto cmd = geometry_msgs::msg::Twist();
        cmd_pub_->publish(cmd);
        if (use_direct_motor_ && motor_pub_) {
            qcar2_interfaces::msg::MotorCommands motor_cmd;
            motor_cmd.motor_names = {"steering_angle", "motor_throttle"};
            motor_cmd.values = {0.0, 0.0};
            motor_pub_->publish(motor_cmd);
        }
    }

    void publish_status(const std::string& status) {
        auto msg = std_msgs::msg::String();
        msg.data = status;
        status_pub_->publish(msg);
    }

    void publish_predicted_path(const mpcc::Result& result) {
        auto markers = visualization_msgs::msg::MarkerArray();
        auto marker = visualization_msgs::msg::Marker();
        marker.header.frame_id = "map";
        marker.header.stamp = this->now();
        marker.ns = "mpcc_predicted";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.scale.x = 0.05;
        marker.color.g = 1.0;
        marker.color.a = 0.8;

        for (size_t i = 0; i < result.predicted_x.size(); i++) {
            auto p = geometry_msgs::msg::Point();
            p.x = result.predicted_x[i];
            p.y = result.predicted_y[i];
            p.z = 0.1;
            marker.points.push_back(p);
        }
        markers.markers.push_back(marker);
        viz_pub_->publish(markers);
    }

    // Solver
    mpcc::ActiveSolver solver_;
    mpcc::Config config_;

    // Road boundary spline (YAML-driven)
    std::unique_ptr<acc::RoadBoundarySpline> road_boundaries_;
    // Cubic spline path
    std::unique_ptr<acc::CubicSplinePath> spline_path_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Publisher<qcar2_interfaces::msg::MotorCommands>::SharedPtr motor_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr telemetry_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr replan_pub_;

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr motion_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr hold_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr traffic_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr obstacle_pos_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr vehicle_state_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr obstacle_map_sub_;

    // Timer
    rclcpp::TimerBase::SharedPtr control_timer_;

    // State (protected by mutex)
    std::mutex state_mutex_;
    double state_x_ = 0.0, state_y_ = 0.0, state_theta_ = 0.0;
    double state_v_ = 0.0, state_delta_ = 0.0;
    double odom_velocity_ = 0.0;
    bool has_odom_ = false;
    bool logged_first_odom_ = false;
    bool has_path_ = false;
    bool motion_enabled_ = true;
    double motion_disabled_time_ = 0.0;
    double motion_disable_timeout_ = 8.0;  // Auto-resume after 8s of false motion_enable
    double motion_resume_cooldown_time_ = 0.0;
    double motion_resume_cooldown_s_ = 10.0;  // Ignore re-disabling for 10s after auto-resume (must outlast sign_detector 8s suppression)
    int motion_enable_consecutive_ = 0;       // Consecutive true messages needed to re-enable
    static constexpr int MOTION_ENABLE_HYSTERESIS = 2;  // ~200ms at 100ms publish rate (fast re-enable)
    int steering_saturated_count_ = 0;
    static constexpr int SATURATION_RESET_THRESHOLD = 10;  // Reset warm-start after 10 cycles (~0.5s) of saturated steering

    // Replan request state
    double sustained_cross_track_start_ = 0.0;
    bool replan_requested_ = false;
    double last_replan_request_time_ = 0.0;
    int solver_failure_count_ = 0;
    static constexpr double REPLAN_CROSS_TRACK_THRESHOLD = 0.40;  // meters
    static constexpr double REPLAN_SUSTAINED_DURATION = 1.0;       // seconds
    static constexpr double REPLAN_COOLDOWN_S = 5.0;               // min between requests
    static constexpr int REPLAN_SOLVER_FAILURE_THRESHOLD = 20;     // 1s at 20Hz

    bool mission_hold_ = false;
    std::string traffic_state_json_;
    double current_progress_ = 0.0;
    double stuck_timer_ = 0.0;
    ReferencePath ref_path_;
    std::vector<mpcc::Obstacle> detected_obstacles_;

    // JointState velocity
    double joint_velocity_ = 0.0;
    bool has_joint_velocity_ = false;

    // Direct motor mode
    bool use_direct_motor_ = true;
    double steering_slew_rate_ = 1.0;  // rad/s

    // State estimator
    bool use_state_estimator_ = false;
    bool has_estimator_state_ = false;
    double estimator_x_ = 0, estimator_y_ = 0, estimator_theta_ = 0, estimator_v_ = 0;
    double last_estimator_time_ = 0;

    // Mapped obstacles (from obstacle tracker, Kalman-filtered)
    std::vector<mpcc::Obstacle> mapped_obstacles_;

    // TF state
    double last_pose_time_ = 0.0;
    double last_x_ = 0.0, last_y_ = 0.0;

    // Timing
    rclcpp::Time start_time_;

    // File logging
    std::string mpcc_csv_path_;
    std::string mpcc_traj_csv_path_;
    std::string mpcc_event_path_;
    double log_start_time_ = 0.0;
    int log_cycle_counter_ = 0;
    int control_skip_counter_ = 0;
};

}  // namespace acc_mpcc

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<acc_mpcc::MPCCControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
