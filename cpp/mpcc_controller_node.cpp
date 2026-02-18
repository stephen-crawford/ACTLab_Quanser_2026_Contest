/**
 * C++ MPCC Controller ROS2 Node
 *
 * Replaces the Python mpcc_controller.py with a high-performance C++ node.
 * Uses the existing mpcc_solver.h for the core MPCC optimization.
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
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>

#include "mpcc_solver.h"

#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <mutex>

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
                                              double v_ref, double dt) const {
        std::vector<mpcc::PathRef> refs(horizon);
        for (int k = 0; k < horizon; k++) {
            double s = start_progress + k * v_ref * dt;
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
        // Parameters (defaults match tuned values from root cause analysis)
        this->declare_parameter("reference_velocity", 0.35);
        this->declare_parameter("contour_weight", 25.0);
        this->declare_parameter("lag_weight", 5.0);
        this->declare_parameter("horizon", 20);
        this->declare_parameter("boundary_weight", 30.0);
        this->declare_parameter("boundary_default_width", 0.22);

        // Initialize solver with tuned config
        mpcc::Config cfg;
        cfg.horizon = this->get_parameter("horizon").as_int();
        cfg.dt = 0.1;
        cfg.wheelbase = 0.256;
        cfg.max_velocity = 0.40;
        cfg.min_velocity = 0.0;
        cfg.max_steering = 0.45;
        cfg.max_acceleration = 0.6;
        cfg.max_steering_rate = 0.6;
        cfg.reference_velocity = this->get_parameter("reference_velocity").as_double();
        cfg.contour_weight = this->get_parameter("contour_weight").as_double();
        cfg.lag_weight = this->get_parameter("lag_weight").as_double();
        cfg.velocity_weight = 2.0;
        cfg.steering_weight = 3.0;
        cfg.acceleration_weight = 1.5;
        cfg.steering_rate_weight = 4.0;
        cfg.jerk_weight = 0.5;
        cfg.robot_radius = 0.13;
        cfg.safety_margin = 0.10;
        cfg.obstacle_weight = 200.0;
        cfg.boundary_weight = this->get_parameter("boundary_weight").as_double();
        cfg.boundary_default_width = this->get_parameter("boundary_default_width").as_double();
        cfg.max_sqp_iterations = 3;
        cfg.max_qp_iterations = 10;
        cfg.qp_tolerance = 1e-5;
        solver_.init(cfg);
        config_ = cfg;

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publishers
        cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/cmd_vel_nav", 10);
        status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/mpcc/status", 10);
        viz_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/mpcc/predicted_path", 10);

        // Subscribers
        auto qos_be = rclcpp::QoS(1).best_effort();
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", qos_be,
            [this](nav_msgs::msg::Odometry::SharedPtr msg) { odom_callback(msg); });

        auto qos_tl = rclcpp::QoS(10).transient_local().reliable();
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/plan", qos_tl,
            [this](nav_msgs::msg::Path::SharedPtr msg) { path_callback(msg); });

        motion_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/motion_enable", 10,
            [this](std_msgs::msg::Bool::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(state_mutex_);
                motion_enabled_ = msg->data;
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

        // Control timer (20 Hz)
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            [this]() { control_loop(); });

        start_time_ = this->now();

        RCLCPP_INFO(this->get_logger(),
            "C++ MPCC Controller initialized: horizon=%d, v_ref=%.2f, contour=%.1f, lag=%.1f, boundary=%.1f",
            cfg.horizon, cfg.reference_velocity, cfg.contour_weight, cfg.lag_weight, cfg.boundary_weight);
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
        ref_path_.build(waypoints);
        has_path_ = true;
        current_progress_ = ref_path_.find_closest_progress(state_x_, state_y_);
        solver_.reset();

        RCLCPP_INFO(this->get_logger(), "Path received: %zu waypoints, length=%.2fm",
                     waypoints.size(), ref_path_.total_length);
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

    // Generate road boundary constraints from path geometry
    std::vector<mpcc::BoundaryConstraint> generate_boundaries(
        double start_progress, int horizon, double v_ref, double dt)
    {
        std::vector<mpcc::BoundaryConstraint> boundaries(horizon);
        double half_width = config_.boundary_default_width;

        for (int k = 0; k < horizon; k++) {
            double s = start_progress + k * v_ref * dt;
            s = std::clamp(s, 0.0, ref_path_.total_length - 0.001);

            // Find segment index for this arc length
            int idx = 0;
            for (int i = 1; i < ref_path_.n_points; i++) {
                if (ref_path_.cumulative_dist[i] > s) { idx = i - 1; break; }
                idx = i;
            }
            idx = std::min(idx, ref_path_.n_points - 2);

            // Get tangent angle -> normal is perpendicular
            double ta = ref_path_.tangent_angle[idx];
            double nx = -std::sin(ta);  // Normal pointing left
            double ny =  std::cos(ta);

            // Interpolate path center position
            double seg_len = ref_path_.cumulative_dist[idx + 1] -
                             ref_path_.cumulative_dist[idx];
            double alpha = 0.0;
            if (seg_len > 1e-6) {
                alpha = (s - ref_path_.cumulative_dist[idx]) / seg_len;
            }
            double cx = ref_path_.x[idx] + alpha * (ref_path_.x[idx + 1] - ref_path_.x[idx]);
            double cy = ref_path_.y[idx] + alpha * (ref_path_.y[idx + 1] - ref_path_.y[idx]);

            // Boundary: n.dot(pos) in [center - half_width, center + half_width]
            double center_proj = nx * cx + ny * cy;
            boundaries[k].nx = nx;
            boundaries[k].ny = ny;
            boundaries[k].b_left = center_proj + half_width;
            boundaries[k].b_right = -(center_proj - half_width);
        }
        return boundaries;
    }

    void control_loop() {
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Try TF first for map-frame position
        update_state_from_tf();

        if (!has_odom_ || !has_path_ || ref_path_.n_points < 2) return;

        // Use odom velocity if available (more accurate)
        if (std::abs(odom_velocity_) > 0.001) {
            state_v_ = odom_velocity_;
        }

        // Check hold
        if (mission_hold_) {
            auto cmd = geometry_msgs::msg::Twist();
            cmd_pub_->publish(cmd);
            return;
        }

        // Check motion enable
        if (!motion_enabled_) {
            auto cmd = geometry_msgs::msg::Twist();
            cmd_pub_->publish(cmd);
            return;
        }

        // Check traffic control
        if (parse_traffic_should_stop()) {
            auto cmd = geometry_msgs::msg::Twist();
            cmd_pub_->publish(cmd);
            return;
        }

        // Update progress (forward-only)
        double new_progress = ref_path_.find_closest_progress(state_x_, state_y_);
        if (new_progress >= current_progress_ - 0.05) {
            current_progress_ = new_progress;
        }

        // Check goal reached
        double remaining = ref_path_.total_length - current_progress_;
        if (remaining < 0.15) {
            auto cmd = geometry_msgs::msg::Twist();
            cmd_pub_->publish(cmd);
            publish_status("Goal reached");
            return;
        }

        // Build solver state
        mpcc::VecX x0;
        x0 << state_x_, state_y_, state_theta_, state_v_, state_delta_;

        // Compute startup elapsed time
        double elapsed = (this->now() - start_time_).seconds();
        config_.startup_elapsed_s = elapsed;
        solver_.config = config_;

        // Get path references
        auto path_refs = ref_path_.get_path_refs(
            current_progress_, config_.horizon,
            config_.reference_velocity, config_.dt);

        // Generate road boundary constraints from path geometry
        auto boundaries = generate_boundaries(
            current_progress_, config_.horizon,
            config_.reference_velocity, config_.dt);

        // Use detected obstacles
        auto result = solver_.solve(x0, path_refs, current_progress_,
                                     ref_path_.total_length,
                                     detected_obstacles_, boundaries);

        if (!result.success) {
            auto cmd = geometry_msgs::msg::Twist();
            cmd_pub_->publish(cmd);
            return;
        }

        double v_cmd = result.v_cmd;
        double delta_cmd = result.delta_cmd;

        // Decelerate near goal
        if (remaining < 0.5) {
            double decel_factor = remaining / 0.5;
            v_cmd *= decel_factor;
            if (remaining > 0.2) v_cmd = std::max(v_cmd, 0.08);
        }

        v_cmd = std::clamp(v_cmd, 0.0, config_.max_velocity);
        delta_cmd = std::clamp(delta_cmd, -config_.max_steering, config_.max_steering);

        // Convert to Twist
        double omega = v_cmd * std::tan(delta_cmd) / config_.wheelbase;
        omega = std::clamp(omega, -1.5, 1.5);

        auto cmd = geometry_msgs::msg::Twist();
        cmd.linear.x = v_cmd;
        cmd.angular.z = omega;
        cmd_pub_->publish(cmd);

        // Track steering for next iteration
        state_delta_ = delta_cmd;

        // Publish status
        double progress_pct = 100.0 * current_progress_ / ref_path_.total_length;
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "v=%.2f, omega=%.2f, delta=%.1fdeg, progress=%.0f%%, solve=%.0fus",
            v_cmd, omega, delta_cmd * 180.0 / M_PI, progress_pct, result.solve_time_us);
        publish_status(std::string(buf));

        // Publish predicted path visualization
        publish_predicted_path(result);
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
    mpcc::Solver solver_;
    mpcc::Config config_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publishers
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr viz_pub_;

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr motion_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr hold_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr traffic_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr obstacle_pos_sub_;

    // Timer
    rclcpp::TimerBase::SharedPtr control_timer_;

    // State (protected by mutex)
    std::mutex state_mutex_;
    double state_x_ = 0.0, state_y_ = 0.0, state_theta_ = 0.0;
    double state_v_ = 0.0, state_delta_ = 0.0;
    double odom_velocity_ = 0.0;
    bool has_odom_ = false;
    bool has_path_ = false;
    bool motion_enabled_ = true;
    bool mission_hold_ = false;
    std::string traffic_state_json_;
    double current_progress_ = 0.0;
    ReferencePath ref_path_;
    std::vector<mpcc::Obstacle> detected_obstacles_;

    // TF state
    double last_pose_time_ = 0.0;
    double last_x_ = 0.0, last_y_ = 0.0;

    // Timing
    rclcpp::Time start_time_;
};

}  // namespace acc_mpcc

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<acc_mpcc::MPCCControllerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
