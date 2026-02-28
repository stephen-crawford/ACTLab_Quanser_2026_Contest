/**
 * C++ Mission Manager Node
 *
 * Ported from mission_manager.py. Full state machine for the ACC competition:
 *   WAIT_FOR_NAV -> CAPTURING_HUB -> GO_LEG -> PAUSED_OBSTACLE -> DWELL ->
 *   RECOVERING -> DONE / ABORTED
 *
 * Features:
 * - Nav2 action clients (ComputePathToPose / NavigateToPose)
 * - Road graph path planning (MPCC mode)
 * - MPCC goal checking via TF distance
 * - LED control via parameter service
 * - Recovery strategies: retry, backup, clear costmap, skip, restart
 * - Obstacle pause/resume with cooldown
 * - YAML mission config loading
 * - Behavior and coordinate logging
 */

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <nav2_msgs/action/navigate_to_pose.hpp>
#include <nav2_msgs/action/compute_path_to_pose.hpp>
#include <nav2_msgs/srv/clear_entire_costmap.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>
#include <rcl_interfaces/srv/set_parameters.hpp>
#include <rcl_interfaces/msg/parameter.hpp>
#include <rcl_interfaces/msg/parameter_value.hpp>
#include <rcl_interfaces/msg/parameter_type.hpp>
#include <action_msgs/msg/goal_status.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "coordinate_transform.h"
#include "road_graph.h"
#include "yaml_config.h"
#include "path_modifier.h"

#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <array>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace acc_mission {

using NavigateToPose = nav2_msgs::action::NavigateToPose;
using ComputePathToPose = nav2_msgs::action::ComputePathToPose;
using GoalStatus = action_msgs::msg::GoalStatus;

// LED color IDs (match qcar2_hardware)
enum LedColor {
    LED_RED = 0, LED_GREEN = 1, LED_BLUE = 2, LED_YELLOW = 3,
    LED_CYAN = 4, LED_MAGENTA = 5, LED_ORANGE = 6
};
constexpr int LED_HUB = LED_MAGENTA;
constexpr int LED_DRIVING = LED_GREEN;
constexpr int LED_PICKUP = LED_BLUE;
constexpr int LED_DROPOFF = LED_ORANGE;
constexpr int LED_OBSTACLE = LED_RED;
constexpr int LED_RECOVERY = LED_CYAN;

enum class MissionState {
    WAIT_FOR_NAV, CAPTURING_HUB, GO_LEG, PAUSED_OBSTACLE,
    DWELL, RECOVERING, DONE, ABORTED
};

enum class RecoveryStrategy {
    RETRY_SAME, CLEAR_COSTMAP, BACKUP_AND_RETRY,
    SKIP_WAYPOINT, RESTART_MISSION
};

struct MissionLeg {
    double target_x, target_y, target_yaw;
    std::string label;
    double dwell_s;
    bool is_skippable;
    std::string route_name;  // for MPCC mode
};

/// Helpers
static std::string state_name(MissionState s) {
    switch (s) {
        case MissionState::WAIT_FOR_NAV:     return "WAIT_FOR_NAV";
        case MissionState::CAPTURING_HUB:    return "CAPTURING_HUB";
        case MissionState::GO_LEG:           return "GO_LEG";
        case MissionState::PAUSED_OBSTACLE:  return "PAUSED_OBSTACLE";
        case MissionState::DWELL:            return "DWELL";
        case MissionState::RECOVERING:       return "RECOVERING";
        case MissionState::DONE:             return "DONE";
        case MissionState::ABORTED:          return "ABORTED";
    }
    return "UNKNOWN";
}

static double now_sec() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void yaw_to_quat(double yaw, double& qz, double& qw) {
    qz = std::sin(yaw / 2.0);
    qw = std::cos(yaw / 2.0);
}

static double quat_to_yaw(double qx, double qy, double qz, double qw) {
    return std::atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz));
}

// =========================================================================
class MissionManagerNode : public rclcpp::Node {
public:
    MissionManagerNode() : Node("mission_manager") {
        // Declare parameters
        this->declare_parameter("config_file", std::string(""));
        this->declare_parameter("use_tf_hub", true);
        this->declare_parameter("hub_tf_timeout_s", 15.0);
        this->declare_parameter("goal_timeout_s", 120.0);
        this->declare_parameter("max_retries_per_leg", 2);
        this->declare_parameter("enable_led", true);
        this->declare_parameter("goal_tol_m", 0.35);
        this->declare_parameter("enable_obstacle_detection", true);
        this->declare_parameter("obstacle_pause_timeout_s", 10.0);
        this->declare_parameter("backup_distance_m", 0.15);
        this->declare_parameter("backup_speed", 0.1);
        this->declare_parameter("mpcc_mode", false);

        auto config_file = this->get_parameter("config_file").as_string();
        use_tf_hub_ = this->get_parameter("use_tf_hub").as_bool();
        hub_tf_timeout_s_ = this->get_parameter("hub_tf_timeout_s").as_double();
        goal_timeout_s_ = this->get_parameter("goal_timeout_s").as_double();
        max_retries_ = this->get_parameter("max_retries_per_leg").as_int();
        enable_led_ = this->get_parameter("enable_led").as_bool();
        goal_tol_m_ = this->get_parameter("goal_tol_m").as_double();
        enable_obstacle_ = this->get_parameter("enable_obstacle_detection").as_bool();
        obstacle_pause_timeout_s_ = this->get_parameter("obstacle_pause_timeout_s").as_double();
        backup_distance_m_ = this->get_parameter("backup_distance_m").as_double();
        backup_speed_ = this->get_parameter("backup_speed").as_double();
        mpcc_mode_ = this->get_parameter("mpcc_mode").as_bool();

        // Load config
        load_mission_config(config_file);

        // Callback group
        cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publishers
        status_pub_ = this->create_publisher<std_msgs::msg::String>("mission/status", 10);
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel_nav", 10);

        if (mpcc_mode_) {
            // Use reliable() but NOT transient_local() for /plan publisher.
            // transient_local causes QoS DURABILITY incompatibility with Nav2's
            // planner (which also publishes on /plan with volatile durability),
            // leading to the MPCC controller rejecting all /plan messages.
            // The mission manager republishes every 2s, so latching isn't needed.
            auto path_qos = rclcpp::QoS(10).reliable();
            path_pub_ = this->create_publisher<nav_msgs::msg::Path>("/plan", path_qos);
            hold_pub_ = this->create_publisher<std_msgs::msg::Bool>("/mission/hold", 10);

            // Road graph (reference 2025 style: single loop [24,20,9,10])
            road_graph_ = std::make_unique<acc::RoadGraph>(0.001);
            auto routes = road_graph_->get_route_names();
            std::string route_list;
            for (auto& r : routes) route_list += r + " ";
            RCLCPP_INFO(this->get_logger(), "Road graph initialized: %s", route_list.c_str());
            RCLCPP_INFO(this->get_logger(), "  Loop path: %zu pts, pickup_idx=%d, dropoff_idx=%d, hub_idx=%d",
                         road_graph_->loop_x().size(),
                         road_graph_->pickup_index(),
                         road_graph_->dropoff_index(),
                         road_graph_->hub_index());
            if (!road_graph_->loop_x().empty()) {
                double lx_min = *std::min_element(road_graph_->loop_x().begin(), road_graph_->loop_x().end());
                double lx_max = *std::max_element(road_graph_->loop_x().begin(), road_graph_->loop_x().end());
                double ly_min = *std::min_element(road_graph_->loop_y().begin(), road_graph_->loop_y().end());
                double ly_max = *std::max_element(road_graph_->loop_y().begin(), road_graph_->loop_y().end());
                RCLCPP_INFO(this->get_logger(), "  Loop QLabs bbox: x=[%.3f..%.3f] y=[%.3f..%.3f]",
                             lx_min, lx_max, ly_min, ly_max);
            }
        }

        // Nav2 action client
        rclcpp::SubscriptionOptions sub_opts;
        sub_opts.callback_group = cb_group_;
        if (mpcc_mode_) {
            compute_path_client_ = rclcpp_action::create_client<ComputePathToPose>(
                this, "compute_path_to_pose", cb_group_);
        } else {
            navigate_client_ = rclcpp_action::create_client<NavigateToPose>(
                this, "navigate_to_pose", cb_group_);
        }

        // LED client
        if (enable_led_) {
            led_client_ = this->create_client<rcl_interfaces::srv::SetParameters>(
                "/qcar2_hardware/set_parameters");
        }

        // Motion enable subscriber
        if (enable_obstacle_) {
            auto qos = rclcpp::QoS(10).best_effort();
            motion_sub_ = this->create_subscription<std_msgs::msg::Bool>(
                "/motion_enable", qos,
                [this](std_msgs::msg::Bool::SharedPtr msg) {
                    bool was = motion_enabled_;
                    motion_enabled_ = msg->data;
                    if (was && !motion_enabled_) {
                        RCLCPP_INFO(this->get_logger(), "Obstacle/sign detected - motion DISABLED");
                        log_behavior("OBSTACLE_DETECTED", "motion disabled");
                        set_led(LED_OBSTACLE);
                    } else if (!was && motion_enabled_) {
                        RCLCPP_INFO(this->get_logger(), "Obstacle cleared - motion ENABLED");
                        log_behavior("OBSTACLE_CLEARED", "motion re-enabled");
                    }
                }, sub_opts);
        }

        // MPCC status subscriber
        if (mpcc_mode_) {
            mpcc_status_sub_ = this->create_subscription<std_msgs::msg::String>(
                "/mpcc/status", 10,
                [this](std_msgs::msg::String::SharedPtr msg) {
                    on_mpcc_status(msg->data);
                }, sub_opts);

            // Replan request subscriber (from MPCC controller)
            replan_sub_ = this->create_subscription<std_msgs::msg::String>(
                "/mpcc/replan_request", 10,
                [this](std_msgs::msg::String::SharedPtr msg) {
                    on_replan_request(msg->data);
                }, sub_opts);
        }

        // Traffic control state subscriber
        traffic_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/traffic_control_state", 10,
            [this](std_msgs::msg::String::SharedPtr /*msg*/) {
                // Traffic logging handled implicitly by sign detector
            }, sub_opts);

        // Obstacle map subscriber (from obstacle tracker)
        obstacle_map_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/obstacle_map", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                parse_obstacle_map(msg->data);
            }, sub_opts);

        // Main tick timer (10Hz)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&MissionManagerNode::tick, this), cb_group_);

        // Coordinate logging timer (1Hz)
        coord_timer_ = this->create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&MissionManagerNode::log_coordinates, this), cb_group_);

        log_start_time_ = now_sec();
        init_log_files();

        // Print config
        RCLCPP_INFO(this->get_logger(), "============================================================");
        RCLCPP_INFO(this->get_logger(), "MissionManager (C++) - ACC Competition");
        RCLCPP_INFO(this->get_logger(), "  Mode: %s", mpcc_mode_ ? "MPCC (path only)" : "Nav2");
        RCLCPP_INFO(this->get_logger(), "  Legs: %zu", legs_.size());
        RCLCPP_INFO(this->get_logger(), "  Goal tolerance: %.2fm", goal_tol_m_);
        RCLCPP_INFO(this->get_logger(), "  Transform: origin=(%.3f, %.3f), theta=%.4f rad (%.1f deg)",
                     tp_.origin_x, tp_.origin_y, tp_.origin_heading_rad,
                     tp_.origin_heading_rad * 180.0 / M_PI);
        for (size_t i = 0; i < legs_.size(); ++i) {
            auto& leg = legs_[i];
            RCLCPP_INFO(this->get_logger(), "  %zu: %-20s -> (%.2f, %.2f) dwell=%.1fs %s",
                i, leg.label.c_str(), leg.target_x, leg.target_y,
                leg.dwell_s, leg.is_skippable ? "(skip)" : "(req)");
        }

        // Validate coordinate transform: pickup in map frame should be around (-2.43, 4.81)
        // If it's at (4.44, 3.07), the transform is using R(-theta) instead of R(+theta)
        {
            double mx, my, myaw;
            acc::qlabs_to_map(0.125, 4.395, 0, tp_, mx, my, myaw);
            RCLCPP_INFO(this->get_logger(), "  Transform check: Pickup QLabs(0.125, 4.395) -> Map(%.2f, %.2f)", mx, my);
            if (mx > 2.0 || my < 0.0) {
                RCLCPP_ERROR(this->get_logger(),
                    "COORDINATE TRANSFORM BUG: Pickup mapped to (%.2f, %.2f) — "
                    "expected ~(-2.43, 4.81). The qlabs_to_map in coordinate_transform.h "
                    "may be using R(-theta) instead of R(+theta). REBUILD REQUIRED!", mx, my);
            }
        }

        RCLCPP_INFO(this->get_logger(), "============================================================");

        publish_status("WAIT_FOR_NAV");
    }

private:
    // -----------------------------------------------------------------
    // Config loading
    // -----------------------------------------------------------------
    void load_mission_config(std::string config_file) {
        if (config_file.empty()) {
            try {
                auto pkg_share = ament_index_cpp::get_package_share_directory("acc_stage1_mission");
                config_file = pkg_share + "/config/mission.yaml";
            } catch (...) {
                // Try alternative package name
                try {
                    auto pkg_share = ament_index_cpp::get_package_share_directory("acc_mpcc_controller_cpp");
                    config_file = pkg_share + "/config/mission.yaml";
                } catch (...) {
                    RCLCPP_WARN(this->get_logger(), "No config package found, using defaults");
                    setup_default_legs();
                    return;
                }
            }
        }

        YAML::Node cfg;
        try {
            cfg = YAML::LoadFile(config_file);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load %s: %s", config_file.c_str(), e.what());
            setup_default_legs();
            return;
        }

        bool use_qlabs = acc::yaml_bool(cfg, "use_qlabs_coords", false);
        if (cfg["transform"]) {
            auto tf = cfg["transform"];
            tp_.origin_x = acc::yaml_double(tf, "origin_x", -1.205);
            tp_.origin_y = acc::yaml_double(tf, "origin_y", -0.83);
            if (tf["origin_heading_rad"]) {
                tp_.origin_heading_rad = tf["origin_heading_rad"].as<double>();
            } else {
                double deg = acc::yaml_double(tf, "origin_heading_deg", -44.7);
                tp_.origin_heading_rad = -deg * M_PI / 180.0;
            }
        }

        auto read_xyz = [&](const std::string& key) -> std::array<double, 3> {
            auto node = cfg[key];
            return {node["x"].as<double>(), node["y"].as<double>(),
                    node["yaw"].as<double>()};
        };

        auto pickup = read_xyz("pickup");
        auto dropoff = read_xyz("dropoff");
        auto hub = read_xyz("hub");
        double dwell_s = acc::yaml_double(cfg, "dwell_s", 3.0);

        if (cfg["goal_tol_m"]) {
            goal_tol_m_ = cfg["goal_tol_m"].as<double>();
        }

        if (use_qlabs) {
            auto transform = [&](std::array<double, 3>& p) {
                double mx, my, myaw;
                acc::qlabs_to_map(p[0], p[1], p[2], tp_, mx, my, myaw);
                p = {mx, my, myaw};
            };
            transform(pickup);
            transform(dropoff);
            transform(hub);
        }

        hub_ = hub;

        if (mpcc_mode_) {
            legs_.push_back({pickup[0], pickup[1], pickup[2], "pickup", dwell_s, false, "hub_to_pickup"});
            legs_.push_back({dropoff[0], dropoff[1], dropoff[2], "dropoff", dwell_s, false, "pickup_to_dropoff"});
            legs_.push_back({hub[0], hub[1], hub[2], "hub", 0.0, false, "dropoff_to_hub"});
        } else {
            // Nav2 mode: could add waypoints here, keeping simple for now
            legs_.push_back({pickup[0], pickup[1], pickup[2], "pickup", dwell_s, false, ""});
            legs_.push_back({dropoff[0], dropoff[1], dropoff[2], "dropoff", dwell_s, false, ""});
            legs_.push_back({hub[0], hub[1], hub[2], "hub", 0.0, false, ""});
        }

        RCLCPP_INFO(this->get_logger(), "Config loaded: %s", config_file.c_str());
    }

    void setup_default_legs() {
        // QLabs defaults
        double px, py, pyaw, dx, dy, dyaw, hx, hy, hyaw;
        acc::qlabs_to_map(0.125, 4.395, 1.57, tp_, px, py, pyaw);
        acc::qlabs_to_map(-0.905, 0.800, 3.14, tp_, dx, dy, dyaw);
        acc::qlabs_to_map(-1.205, -0.83, -0.78, tp_, hx, hy, hyaw);
        hub_ = {hx, hy, hyaw};

        legs_.push_back({px, py, pyaw, "pickup", 3.0, false, "hub_to_pickup"});
        legs_.push_back({dx, dy, dyaw, "dropoff", 3.0, false, "pickup_to_dropoff"});
        legs_.push_back({hx, hy, hyaw, "hub", 0.0, false, "dropoff_to_hub"});
    }

    // -----------------------------------------------------------------
    // Logging
    // -----------------------------------------------------------------
    void init_log_files() {
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream ts;
        ts << std::put_time(&tm, "%Y%m%d_%H%M%S");

        // Write to the Docker-mounted workspace so logs appear on the host
        std::string log_dir = "/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/logs";
        try {
            std::filesystem::create_directories(log_dir);
        } catch (...) {
            // Fallback to /tmp if workspace mount not available
            log_dir = "/tmp/mission_logs";
            try { std::filesystem::create_directories(log_dir); } catch (...) {}
        }

        behavior_log_path_ = log_dir + "/behavior_" + ts.str() + ".log";
        coord_log_path_ = log_dir + "/coordinates_" + ts.str() + ".csv";

        {
            std::ofstream f(behavior_log_path_);
            f << "# Mission Behavior Log (C++) - " << ts.str() << "\n\n";
        }
        {
            std::ofstream f(coord_log_path_);
            f << "time,elapsed_s,state,map_x,map_y,map_yaw_deg,target_label,dist_to_target\n";
        }
        RCLCPP_INFO(this->get_logger(), "Logs: %s", behavior_log_path_.c_str());
    }

    void log_behavior(const std::string& event, const std::string& details = "") {
        double elapsed = now_sec() - log_start_time_;
        RCLCPP_INFO(this->get_logger(), "EVENT: %s | %s", event.c_str(), details.c_str());
        try {
            std::ofstream f(behavior_log_path_, std::ios::app);
            f << "[+" << std::fixed << std::setprecision(1) << elapsed << "s] "
              << event << " | " << details << "\n";
        } catch (...) {}
    }

    void log_coordinates() {
        auto pose = get_current_pose();
        if (!pose) return;
        auto [mx, my, myaw] = *pose;
        double elapsed = now_sec() - log_start_time_;
        std::string label;
        double dist = -1.0;
        if (leg_index_ < static_cast<int>(legs_.size())) {
            auto& leg = legs_[leg_index_];
            label = leg.label;
            dist = std::hypot(leg.target_x - mx, leg.target_y - my);
        }
        try {
            std::ofstream f(coord_log_path_, std::ios::app);
            f << std::fixed << std::setprecision(1) << elapsed << ","
              << state_name(state_) << ","
              << std::setprecision(4) << mx << "," << my << ","
              << std::setprecision(1) << (myaw * 180.0 / M_PI) << ","
              << label << "," << std::setprecision(3) << dist << "\n";
        } catch (...) {}
    }

    // -----------------------------------------------------------------
    // TF Utilities
    // -----------------------------------------------------------------
    std::optional<std::array<double, 3>> get_current_pose() {
        try {
            auto t = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero,
                tf2::durationFromSec(0.2));
            double x = t.transform.translation.x;
            double y = t.transform.translation.y;
            auto& q = t.transform.rotation;
            double yaw = quat_to_yaw(q.x, q.y, q.z, q.w);
            return std::array<double, 3>{x, y, yaw};
        } catch (const tf2::TransformException&) {
            return std::nullopt;
        }
    }

    bool capture_hub_from_tf() {
        auto pose = get_current_pose();
        if (!pose) return false;
        hub_ = *pose;
        RCLCPP_INFO(this->get_logger(), "Hub captured from TF: (%.3f, %.3f, %.3f)",
                     hub_[0], hub_[1], hub_[2]);
        log_behavior("HUB_CAPTURED_TF", "map=(" + std::to_string(hub_[0]) + ", " + std::to_string(hub_[1]) + ")");
        if (!legs_.empty()) {
            auto& last = legs_.back();
            last.target_x = hub_[0];
            last.target_y = hub_[1];
            last.target_yaw = hub_[2];
        }
        return true;
    }

    // -----------------------------------------------------------------
    // Status & LED
    // -----------------------------------------------------------------
    void publish_status(const std::string& status) {
        auto msg = std_msgs::msg::String();
        msg.data = status;
        status_pub_->publish(msg);
    }

    void publish_hold(bool hold) {
        if (mpcc_mode_ && hold_pub_) {
            auto msg = std_msgs::msg::Bool();
            msg.data = hold;
            hold_pub_->publish(msg);
        }
    }

    void set_led(int color_id) {
        if (!enable_led_ || !led_client_) return;
        if (!led_client_->service_is_ready()) return;

        auto req = std::make_shared<rcl_interfaces::srv::SetParameters::Request>();
        rcl_interfaces::msg::Parameter param;
        param.name = "led_color_id";
        param.value.type = rcl_interfaces::msg::ParameterType::PARAMETER_INTEGER;
        param.value.integer_value = color_id;
        req->parameters.push_back(param);
        led_client_->async_send_request(req);
    }

    // -----------------------------------------------------------------
    // MPCC status callback
    // -----------------------------------------------------------------
    void on_mpcc_status(const std::string& data) {
        if (data.find("Goal reached") != std::string::npos) {
            if (state_ == MissionState::GO_LEG && mpcc_mode_) {
                auto pose = get_current_pose();
                if (pose && leg_index_ < static_cast<int>(legs_.size())) {
                    auto& leg = legs_[leg_index_];
                    double dist = std::hypot(leg.target_x - (*pose)[0],
                                            leg.target_y - (*pose)[1]);
                    if (dist < goal_tol_m_ * 2.0) {
                        RCLCPP_INFO(this->get_logger(), "MPCC goal confirmed (dist=%.3f)", dist);
                        goal_sent_time_ = 0;
                        on_goal_success();
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Dynamic Replanning
    // -----------------------------------------------------------------
    void on_replan_request(const std::string& json) {
        // Guards: only replan during GO_LEG with road graph available
        if (state_ != MissionState::GO_LEG || !mpcc_mode_ || !road_graph_) {
            RCLCPP_DEBUG(this->get_logger(), "Replan request ignored: state=%s, mpcc=%d",
                         state_name(state_).c_str(), mpcc_mode_);
            return;
        }

        // Cooldown between replans
        double now = now_sec();
        if ((now - last_replan_time_) < REPLAN_COOLDOWN_S) {
            RCLCPP_DEBUG(this->get_logger(), "Replan request ignored: cooldown (%.1fs remaining)",
                         REPLAN_COOLDOWN_S - (now - last_replan_time_));
            return;
        }

        // Per-leg cap to prevent infinite replan cycles
        if (replan_count_ >= MAX_REPLANS_PER_LEG) {
            RCLCPP_WARN(this->get_logger(),
                "Replan request ignored: max %d replans reached for leg %d",
                MAX_REPLANS_PER_LEG, leg_index_);
            return;
        }

        replan_count_++;
        last_replan_time_ = now;

        RCLCPP_INFO(this->get_logger(),
            "Replanning path for leg %d (%s) — replan #%d, request: %s",
            leg_index_,
            (leg_index_ < static_cast<int>(legs_.size()) ? legs_[leg_index_].label.c_str() : "?"),
            replan_count_, json.c_str());
        log_behavior("REPLAN_TRIGGERED", "leg=" + std::to_string(leg_index_) +
                     " count=" + std::to_string(replan_count_) + " " + json);

        send_road_graph_path(leg_index_);
        // Reset periodic republish timer so the old cached path isn't re-published
        last_path_pub_time_ = now;
    }

    /**
     * Detect which mission leg the vehicle is currently on based on its
     * position. Used at startup to allow missions from arbitrary start positions.
     *
     * Returns the leg index (0=hub_to_pickup, 1=pickup_to_dropoff, 2=dropoff_to_hub).
     */
    int detect_current_leg() {
        auto pose = get_current_pose();
        if (!pose) {
            RCLCPP_WARN(this->get_logger(), "detect_current_leg: no TF available, defaulting to leg 0");
            return 0;
        }

        // Transform current position to QLabs frame
        double cur_qx, cur_qy;
        acc::map_to_qlabs_2d((*pose)[0], (*pose)[1], tp_, cur_qx, cur_qy);

        // Check if at hub (normal start)
        double hub_dist = std::hypot(cur_qx - tp_.origin_x, cur_qy - tp_.origin_y);
        if (hub_dist < goal_tol_m_ * 2.0) {
            RCLCPP_INFO(this->get_logger(),
                "detect_current_leg: at hub (dist=%.3f), starting from leg 0", hub_dist);
            return 0;
        }

        // Check each route to find the closest one with enough remaining path
        struct RouteCandidate {
            int leg_idx;
            double distance;       // distance from vehicle to closest point on route
            double remaining_frac; // fraction of route remaining
        };
        std::vector<RouteCandidate> candidates;

        for (int i = 0; i < static_cast<int>(legs_.size()); i++) {
            auto route = road_graph_->get_route(legs_[i].route_name);
            if (!route || route->first.size() < 2) continue;

            auto& [rx, ry] = *route;
            // Find closest point on this route
            int closest_idx = 0;
            double min_dist_sq = 1e18;
            for (int j = 0; j < static_cast<int>(rx.size()); j++) {
                double dsq = (rx[j] - cur_qx) * (rx[j] - cur_qx) +
                             (ry[j] - cur_qy) * (ry[j] - cur_qy);
                if (dsq < min_dist_sq) { min_dist_sq = dsq; closest_idx = j; }
            }
            double dist = std::sqrt(min_dist_sq);
            double remaining_frac = 1.0 - static_cast<double>(closest_idx) / rx.size();

            candidates.push_back({i, dist, remaining_frac});

            RCLCPP_INFO(this->get_logger(),
                "detect_current_leg: route '%s' (leg %d): closest_dist=%.3f, remaining=%.0f%%",
                legs_[i].route_name.c_str(), i, dist, remaining_frac * 100.0);
        }

        // Filter to routes with >20% remaining (viable)
        std::vector<RouteCandidate> viable;
        for (auto& c : candidates) {
            if (c.remaining_frac > 0.20) {
                viable.push_back(c);
            }
        }

        if (!viable.empty()) {
            // Among viable routes, select the one with minimum distance to vehicle
            auto best = std::min_element(viable.begin(), viable.end(),
                [](const RouteCandidate& a, const RouteCandidate& b) {
                    return a.distance < b.distance;
                });
            RCLCPP_INFO(this->get_logger(),
                "detect_current_leg: selected leg %d ('%s'), dist=%.3f, remaining=%.0f%%",
                best->leg_idx, legs_[best->leg_idx].label.c_str(),
                best->distance, best->remaining_frac * 100.0);
            return best->leg_idx;
        }

        // No viable routes (all near completion) — advance to the next leg after
        // the one with least remaining fraction
        if (!candidates.empty()) {
            auto least = std::min_element(candidates.begin(), candidates.end(),
                [](const RouteCandidate& a, const RouteCandidate& b) {
                    return a.remaining_frac < b.remaining_frac;
                });
            int next_leg = (least->leg_idx + 1) % static_cast<int>(legs_.size());
            RCLCPP_INFO(this->get_logger(),
                "detect_current_leg: all routes near completion, advancing to leg %d", next_leg);
            return next_leg;
        }

        RCLCPP_WARN(this->get_logger(), "detect_current_leg: no routes found, defaulting to leg 0");
        return 0;
    }

    // -----------------------------------------------------------------
    // Navigation Goal Management
    // -----------------------------------------------------------------
    void send_goal(int leg_idx) {
        if (leg_idx >= static_cast<int>(legs_.size())) return;
        auto& leg = legs_[leg_idx];
        goal_sent_time_ = now_sec();
        goal_result_received_ = false;
        last_goal_status_ = -1;
        current_target_ = {leg.target_x, leg.target_y, leg.target_yaw};

        RCLCPP_INFO(this->get_logger(), "Sending goal (%s): (%.3f, %.3f, %.3f)",
                     leg.label.c_str(), leg.target_x, leg.target_y, leg.target_yaw);
        log_behavior("GOAL_SENT", leg.label + " -> map=(" +
                     std::to_string(leg.target_x) + ", " + std::to_string(leg.target_y) + ")");
        publish_status("GO_LEG_" + leg.label);
        set_led(LED_DRIVING);

        if (mpcc_mode_ && road_graph_ && !leg.route_name.empty()) {
            send_road_graph_path(leg_idx);
        } else if (mpcc_mode_) {
            send_compute_path_goal(leg_idx);
        } else {
            send_navigate_goal(leg_idx);
        }
    }

    /**
     * Smooth the start of a map-frame path so it is tangent to the
     * vehicle's current heading.  Without this, a straight-line road
     * graph edge whose direction differs from the vehicle heading
     * creates an initial heading mismatch that the MPCC cannot recover
     * from during the low-speed startup ramp.
     *
     * Inserts a short Hermite-spline transition (0.3 m) from the
     * vehicle pose toward the path, then rejoins the original path.
     */
    void align_path_to_vehicle_heading(
        std::vector<double>& mx, std::vector<double>& my,
        double veh_x, double veh_y, double veh_yaw)
    {
        if (mx.size() < 2) return;

        // Path initial heading
        double path_dx = mx[1] - mx[0];
        double path_dy = my[1] - my[0];
        double path_yaw = std::atan2(path_dy, path_dx);

        double heading_err = acc::normalize_angle(path_yaw - veh_yaw);
        // Only fix if heading mismatch > 5 degrees
        if (std::abs(heading_err) < 5.0 * M_PI / 180.0) return;

        // Find the rejoin point: the first path point that is >= blend_dist
        // along the path from the start.
        // Scale blend distance with heading error — larger errors need longer arcs
        // to avoid extreme curvature in the Hermite transition.
        // IMPORTANT: blend must cover the MPCC solver's full horizon distance so
        // the solver only sees the blended heading transition, never the raw path
        // heading. At startup (v≈0.2, horizon=10, dt=0.1), the solver's effective
        // lookahead is ~0.5m. With v_ref=0.45 after ramp-up, it reaches ~1.0m.
        // Use at least 1.5m for large heading errors to provide safety margin.
        double base_blend = 0.50;
        double err_scale = std::abs(heading_err) / (M_PI / 6.0);  // 1.0 at 30°
        double blend_dist = base_blend + 1.0 * std::min(err_scale, 2.0);
        blend_dist = std::min(blend_dist, 2.5);  // cap at 2.5m
        double cum = 0.0;
        int rejoin_idx = 1;
        for (size_t i = 1; i < mx.size(); i++) {
            cum += std::hypot(mx[i] - mx[i-1], my[i] - my[i-1]);
            if (cum >= blend_dist) { rejoin_idx = static_cast<int>(i); break; }
        }
        if (rejoin_idx <= 0) return;

        // Cubic Hermite from (veh_x, veh_y, veh_yaw) to (mx[rejoin], my[rejoin], path tangent)
        double rx = mx[rejoin_idx], ry = my[rejoin_idx];
        double rtx, rty;
        if (rejoin_idx + 1 < static_cast<int>(mx.size())) {
            rtx = mx[rejoin_idx + 1] - mx[rejoin_idx];
            rty = my[rejoin_idx + 1] - my[rejoin_idx];
        } else {
            rtx = mx[rejoin_idx] - mx[rejoin_idx - 1];
            rty = my[rejoin_idx] - my[rejoin_idx - 1];
        }
        double rlen = std::hypot(rtx, rty);
        if (rlen > 1e-6) { rtx /= rlen; rty /= rlen; }

        // Scale tangent vectors to match chord length for smooth curvature
        double chord = std::hypot(rx - veh_x, ry - veh_y);
        double tang_scale = chord;  // Hermite tangent magnitude

        double p0x = veh_x, p0y = veh_y;
        double m0x = std::cos(veh_yaw) * tang_scale;
        double m0y = std::sin(veh_yaw) * tang_scale;
        double p1x = rx, p1y = ry;
        double m1x = rtx * tang_scale, m1y = rty * tang_scale;

        // Generate Hermite points at ds=0.001 spacing (matching path waypoint density)
        double ds = 0.001;
        int n_pts = std::max(static_cast<int>(chord / ds), 10);
        std::vector<double> new_x, new_y;
        new_x.reserve(n_pts + mx.size());
        new_y.reserve(n_pts + mx.size());

        for (int i = 0; i <= n_pts; i++) {
            double t = static_cast<double>(i) / n_pts;
            double t2 = t * t, t3 = t2 * t;
            // Hermite basis
            double h00 = 2*t3 - 3*t2 + 1;
            double h10 = t3 - 2*t2 + t;
            double h01 = -2*t3 + 3*t2;
            double h11 = t3 - t2;
            new_x.push_back(h00*p0x + h10*m0x + h01*p1x + h11*m1x);
            new_y.push_back(h00*p0y + h10*m0y + h01*p1y + h11*m1y);
        }

        // Append remaining original path after rejoin point
        for (size_t i = rejoin_idx + 1; i < mx.size(); i++) {
            new_x.push_back(mx[i]);
            new_y.push_back(my[i]);
        }

        mx = std::move(new_x);
        my = std::move(new_y);

        RCLCPP_INFO(this->get_logger(),
            "Path heading aligned: vehicle=%.1fdeg, path=%.1fdeg, err=%.1fdeg, blend=%.2fm",
            veh_yaw * 180.0/M_PI, path_yaw * 180.0/M_PI,
            heading_err * 180.0/M_PI, blend_dist);
    }

    void send_road_graph_path(int leg_idx) {
        auto& leg = legs_[leg_idx];

        // Get current position in QLabs frame
        double cur_qx = tp_.origin_x, cur_qy = tp_.origin_y;
        double cur_qyaw = std::numeric_limits<double>::quiet_NaN();
        auto pose = get_current_pose();
        if (pose) {
            acc::map_to_qlabs((*pose)[0], (*pose)[1], (*pose)[2], tp_, cur_qx, cur_qy, cur_qyaw);
        }

        auto path = road_graph_->plan_path_for_mission_leg(leg.route_name, cur_qx, cur_qy, cur_qyaw);
        if (!path || path->first.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "Road graph returned no path for %s", leg.label.c_str());
            if (mpcc_mode_) {
                send_compute_path_goal(leg_idx);
            }
            return;
        }

        auto& [qx, qy] = *path;

        // Validate QLabs path bounds (track is ~3.5m x 6m)
        {
            double qx_min = *std::min_element(qx.begin(), qx.end());
            double qx_max = *std::max_element(qx.begin(), qx.end());
            double qy_min = *std::min_element(qy.begin(), qy.end());
            double qy_max = *std::max_element(qy.begin(), qy.end());
            RCLCPP_INFO(this->get_logger(), "Road graph QLabs path: %zu pts, bbox=(%.2f..%.2f, %.2f..%.2f)",
                         qx.size(), qx_min, qx_max, qy_min, qy_max);
        }

        // Transform to map frame
        std::vector<double> mx, my;
        acc::qlabs_path_to_map(qx, qy, tp_, mx, my);

        // Validate map-frame path bounds (should be similar magnitude)
        {
            double mx_min = *std::min_element(mx.begin(), mx.end());
            double mx_max = *std::max_element(mx.begin(), mx.end());
            double my_min = *std::min_element(my.begin(), my.end());
            double my_max = *std::max_element(my.begin(), my.end());
            RCLCPP_INFO(this->get_logger(), "Map frame path: bbox=(%.2f..%.2f, %.2f..%.2f)",
                         mx_min, mx_max, my_min, my_max);
            double span_x = mx_max - mx_min;
            double span_y = my_max - my_min;
            if (span_x > 8.0 || span_y > 10.0) {
                RCLCPP_ERROR(this->get_logger(),
                    "MAP PATH BOUNDS CHECK FAILED: spans %.1f x %.1fm (expected <8x10). "
                    "coordinate_transform.h qlabs_to_map may use wrong rotation direction!", span_x, span_y);
            }
        }

        // Align path start with vehicle heading to prevent initial mismatch
        if (pose) {
            align_path_to_vehicle_heading(mx, my, (*pose)[0], (*pose)[1], (*pose)[2]);
        }

        // Replan safety gate: route endpoint must be near mission target.
        // If the leg-sliced route snaps to a wrong branch, try a goal-directed
        // fallback path before publishing.
        if (!mx.empty() && !my.empty()) {
            double end_err = std::hypot(mx.back() - leg.target_x, my.back() - leg.target_y);
            if (end_err > 1.0) {
                double goal_qx = 0.0, goal_qy = 0.0;
                acc::map_to_qlabs_2d(leg.target_x, leg.target_y, tp_, goal_qx, goal_qy);
                auto goal_path = road_graph_->plan_path(cur_qx, cur_qy, goal_qx, goal_qy);
                if (goal_path && goal_path->first.size() >= 2) {
                    auto alt_qx = goal_path->first;
                    auto alt_qy = goal_path->second;
                    std::vector<double> alt_mx, alt_my;
                    acc::qlabs_path_to_map(alt_qx, alt_qy, tp_, alt_mx, alt_my);
                    double alt_end_err = std::hypot(alt_mx.back() - leg.target_x,
                                                    alt_my.back() - leg.target_y);
                    if (alt_end_err <= 1.0) {
                        RCLCPP_WARN(this->get_logger(),
                            "Road-graph leg path endpoint %.2fm from target; "
                            "switching to goal-directed fallback (%.2fm)",
                            end_err, alt_end_err);
                        qx = std::move(alt_qx);
                        qy = std::move(alt_qy);
                        mx = std::move(alt_mx);
                        my = std::move(alt_my);
                        end_err = alt_end_err;
                    }
                }
                if (end_err > 1.0) {
                    if (end_err <= 3.0) {
                        // Bridge the final gap to the exact mission target so the
                        // controller cannot get trapped at a near-but-wrong endpoint.
                        const double bx0 = mx.back();
                        const double by0 = my.back();
                        int n_bridge = std::max(10, static_cast<int>(std::ceil(end_err / 0.05)));
                        for (int i = 1; i <= n_bridge; ++i) {
                            double a = static_cast<double>(i) / static_cast<double>(n_bridge);
                            mx.push_back(bx0 + a * (leg.target_x - bx0));
                            my.push_back(by0 + a * (leg.target_y - by0));
                        }
                        RCLCPP_WARN(this->get_logger(),
                            "Road-graph endpoint %.2fm from target for %s; appended %d-point bridge",
                            end_err, leg.label.c_str(), n_bridge);
                    } else {
                        RCLCPP_WARN(this->get_logger(),
                            "Rejected road-graph path for %s: endpoint %.2fm from target (max 3.0m bridge)",
                            leg.label.c_str(), end_err);
                        if (current_path_) {
                            RCLCPP_WARN(this->get_logger(),
                                "Keeping previous path to avoid replan thrash");
                            return;
                        }
                        if (mpcc_mode_) {
                            RCLCPP_WARN(this->get_logger(),
                                "Falling back to nav2 ComputePathToPose for %s", leg.label.c_str());
                            send_compute_path_goal(leg_idx);
                            return;
                        }
                    }
                }
            }
        }

        // Build Path message
        auto path_msg = nav_msgs::msg::Path();
        path_msg.header.frame_id = "map";
        path_msg.header.stamp = this->get_clock()->now();

        for (size_t i = 0; i < mx.size(); ++i) {
            auto ps = geometry_msgs::msg::PoseStamped();
            ps.header = path_msg.header;
            ps.pose.position.x = mx[i];
            ps.pose.position.y = my[i];
            double yaw;
            if (i + 1 < mx.size()) {
                yaw = std::atan2(my[i+1] - my[i], mx[i+1] - mx[i]);
            } else {
                yaw = leg.target_yaw;
            }
            double qz_val, qw_val;
            yaw_to_quat(yaw, qz_val, qw_val);
            ps.pose.orientation.z = qz_val;
            ps.pose.orientation.w = qw_val;
            path_msg.poses.push_back(ps);
        }

        base_path_ = path_msg;
        current_path_ = path_msg;
        path_pub_->publish(path_msg);
        goal_result_received_ = true;
        last_goal_status_ = GoalStatus::STATUS_SUCCEEDED;

        // Compute path length
        double path_length = 0;
        for (size_t i = 1; i < mx.size(); ++i) {
            path_length += std::hypot(mx[i] - mx[i-1], my[i] - my[i-1]);
        }

        RCLCPP_INFO(this->get_logger(), "Road graph path: %zu poses, %.2fm, route=%s",
                     path_msg.poses.size(), path_length, leg.route_name.c_str());
        log_behavior("ROAD_GRAPH_PATH", leg.label + ": " +
                     std::to_string(path_msg.poses.size()) + " poses, " +
                     std::to_string(path_length) + "m");
    }

    void send_compute_path_goal(int leg_idx) {
        auto& leg = legs_[leg_idx];
        if (!compute_path_client_) return;

        if (!compute_path_client_->wait_for_action_server(std::chrono::seconds(10))) {
            RCLCPP_ERROR(this->get_logger(), "ComputePathToPose server not available");
            return;
        }

        auto goal = ComputePathToPose::Goal();
        goal.goal.header.frame_id = "map";
        goal.goal.header.stamp = this->get_clock()->now();
        goal.goal.pose.position.x = leg.target_x;
        goal.goal.pose.position.y = leg.target_y;
        double qz_val, qw_val;
        yaw_to_quat(leg.target_yaw, qz_val, qw_val);
        goal.goal.pose.orientation.z = qz_val;
        goal.goal.pose.orientation.w = qw_val;
        goal.use_start = false;

        auto send_opts = rclcpp_action::Client<ComputePathToPose>::SendGoalOptions();
        send_opts.result_callback = [this](auto result) {
            on_compute_path_result(result);
        };
        compute_path_client_->async_send_goal(goal, send_opts);
    }

    void on_compute_path_result(
        const rclcpp_action::ClientGoalHandle<ComputePathToPose>::WrappedResult& result)
    {
        last_goal_status_ = static_cast<int>(result.code);
        goal_result_received_ = true;
        last_goal_result_time_ = now_sec();

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            auto& path = result.result->path;
            if (!path.poses.empty()) {
                RCLCPP_INFO(this->get_logger(), "Path computed: %zu poses", path.poses.size());
                base_path_ = path;
                current_path_ = path;
                path_pub_->publish(path);
            } else {
                RCLCPP_WARN(this->get_logger(), "Empty path received");
                base_path_.reset();
                current_path_.reset();
                on_goal_failure();
            }
        } else {
            base_path_.reset();
            current_path_.reset();
            on_goal_failure();
        }
    }

    void send_navigate_goal(int leg_idx) {
        auto& leg = legs_[leg_idx];
        if (!navigate_client_) return;

        if (!navigate_client_->wait_for_action_server(std::chrono::seconds(10))) {
            RCLCPP_ERROR(this->get_logger(), "NavigateToPose server not available");
            return;
        }

        auto goal = NavigateToPose::Goal();
        goal.pose.header.frame_id = "map";
        goal.pose.header.stamp = this->get_clock()->now();
        goal.pose.pose.position.x = leg.target_x;
        goal.pose.pose.position.y = leg.target_y;
        double qz_val, qw_val;
        yaw_to_quat(leg.target_yaw, qz_val, qw_val);
        goal.pose.pose.orientation.z = qz_val;
        goal.pose.pose.orientation.w = qw_val;

        auto send_opts = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();
        send_opts.result_callback = [this](auto result) {
            on_navigate_result(result);
        };
        nav_goal_handle_.reset();
        (void)navigate_client_->async_send_goal(goal, send_opts);
    }

    void on_navigate_result(
        const rclcpp_action::ClientGoalHandle<NavigateToPose>::WrappedResult& result)
    {
        last_goal_status_ = static_cast<int>(result.code);
        goal_result_received_ = true;
        last_goal_result_time_ = now_sec();
        nav_goal_handle_.reset();
        goal_sent_time_ = 0;

        if (result.code == rclcpp_action::ResultCode::SUCCEEDED) {
            on_goal_success();
        } else if (result.code == rclcpp_action::ResultCode::CANCELED) {
            RCLCPP_INFO(this->get_logger(), "Goal was cancelled");
        } else {
            on_goal_failure();
        }
    }

    // -----------------------------------------------------------------
    // Goal Success / Failure / Advance
    // -----------------------------------------------------------------
    void on_goal_success() {
        if (leg_index_ >= static_cast<int>(legs_.size())) return;
        auto& leg = legs_[leg_index_];

        if (mpcc_mode_) {
            publish_hold(true);
            auto cmd = geometry_msgs::msg::Twist();
            cmd_vel_pub_->publish(cmd);
        }

        if (leg.dwell_s > 0) {
            if (leg.label.find("pickup") != std::string::npos) {
                set_led(LED_PICKUP);
                log_behavior("PICKUP_ARRIVED", "dwelling " + std::to_string(leg.dwell_s) + "s");
            } else if (leg.label.find("dropoff") != std::string::npos) {
                set_led(LED_DROPOFF);
                log_behavior("DROPOFF_ARRIVED", "dwelling " + std::to_string(leg.dwell_s) + "s");
            } else if (leg.label.find("hub") != std::string::npos) {
                set_led(LED_HUB);
                log_behavior("HUB_ARRIVED", "");
            } else {
                set_led(LED_DRIVING);
            }
            state_ = MissionState::DWELL;
            stop_until_ = now_sec() + leg.dwell_s;
            publish_status("DWELL_" + leg.label);
            retry_count_ = 0;
            recovery_idx_ = 0;
            return;
        }

        if (mpcc_mode_) {
            log_behavior("WAYPOINT_SETTLING", leg.label + " 0.5s settle");
            state_ = MissionState::DWELL;
            stop_until_ = now_sec() + 0.5;
            retry_count_ = 0;
            recovery_idx_ = 0;
            return;
        }

        advance_to_next_leg();
    }

    void on_goal_failure() {
        RCLCPP_WARN(this->get_logger(), "Goal failed - entering recovery");
        log_behavior("GOAL_FAILED", "entering recovery");
        state_ = MissionState::RECOVERING;
        set_led(LED_RECOVERY);
        publish_status("RECOVERING");
    }

    void advance_to_next_leg() {
        leg_index_++;
        retry_count_ = 0;
        recovery_idx_ = 0;
        replan_count_ = 0;

        if (mpcc_mode_) publish_hold(false);

        if (leg_index_ >= static_cast<int>(legs_.size())) {
            state_ = MissionState::DONE;
            set_led(LED_HUB);
            publish_status("DONE");
            RCLCPP_INFO(this->get_logger(), "Mission COMPLETE!");
            log_behavior("MISSION_COMPLETE", "all legs finished");
            return;
        }

        state_ = MissionState::GO_LEG;
        send_goal(leg_index_);
    }

    // -----------------------------------------------------------------
    // Obstacle Pause / Resume
    // -----------------------------------------------------------------
    void pause_for_obstacle() {
        if (state_ != MissionState::GO_LEG) return;
        if (now_sec() - last_resume_time_ < resume_cooldown_s_) return;

        RCLCPP_INFO(this->get_logger(), "Pausing for obstacle");
        log_behavior("PAUSED_OBSTACLE", "");
        obstacle_pause_start_ = now_sec();
        state_ = MissionState::PAUSED_OBSTACLE;
        publish_status("PAUSED_OBSTACLE");
        set_led(LED_OBSTACLE);
    }

    void resume_from_pause() {
        RCLCPP_INFO(this->get_logger(), "Resuming after obstacle cleared");
        log_behavior("RESUMED_AFTER_OBSTACLE", "");
        obstacle_pause_start_ = 0;
        last_resume_time_ = now_sec();
        set_led(LED_DRIVING);

        if (mpcc_mode_ && road_graph_ && leg_index_ < static_cast<int>(legs_.size())) {
            // Re-anchor to current pose after stop events. Re-publishing a stale path
            // can leave the controller with a mismatched warm-start near sharp turns.
            RCLCPP_INFO(this->get_logger(), "Regenerating path from current pose after resume");
            state_ = MissionState::GO_LEG;
            send_road_graph_path(leg_index_);
            last_path_pub_time_ = now_sec();
        } else {
            state_ = MissionState::GO_LEG;
            send_goal(leg_index_);
        }
    }

    // -----------------------------------------------------------------
    // Recovery
    // -----------------------------------------------------------------
    std::vector<RecoveryStrategy> get_recovery_strategies() {
        std::vector<RecoveryStrategy> strats = {
            RecoveryStrategy::RETRY_SAME,
            RecoveryStrategy::CLEAR_COSTMAP,
            RecoveryStrategy::BACKUP_AND_RETRY,
        };
        if (leg_index_ < static_cast<int>(legs_.size()) && legs_[leg_index_].is_skippable)
            strats.push_back(RecoveryStrategy::SKIP_WAYPOINT);
        strats.push_back(RecoveryStrategy::RESTART_MISSION);
        return strats;
    }

    void execute_recovery() {
        auto strats = get_recovery_strategies();
        if (recovery_idx_ >= static_cast<int>(strats.size())) {
            RCLCPP_ERROR(this->get_logger(), "All recovery strategies exhausted - ABORTING");
            log_behavior("MISSION_ABORTED", "all recovery strategies exhausted");
            state_ = MissionState::ABORTED;
            set_led(LED_RED);
            publish_status("ABORTED");
            return;
        }

        auto strategy = strats[recovery_idx_];
        retry_count_++;

        switch (strategy) {
        case RecoveryStrategy::RETRY_SAME:
            RCLCPP_INFO(this->get_logger(), "Recovery: RETRY_SAME");
            state_ = MissionState::GO_LEG;
            send_goal(leg_index_);
            break;

        case RecoveryStrategy::BACKUP_AND_RETRY: {
            RCLCPP_INFO(this->get_logger(), "Recovery: BACKUP_AND_RETRY");
            auto cmd = geometry_msgs::msg::Twist();
            cmd.linear.x = -backup_speed_;
            cmd_vel_pub_->publish(cmd);
            backup_start_time_ = now_sec();
            state_ = MissionState::RECOVERING;
            break;
        }

        case RecoveryStrategy::CLEAR_COSTMAP: {
            RCLCPP_INFO(this->get_logger(), "Recovery: CLEAR_COSTMAP");
            try {
                auto clear_client = this->create_client<nav2_msgs::srv::ClearEntireCostmap>(
                    "/global_costmap/clear_entirely_global_costmap");
                if (clear_client->service_is_ready()) {
                    clear_client->async_send_request(
                        std::make_shared<nav2_msgs::srv::ClearEntireCostmap::Request>());
                }
            } catch (...) {}
            state_ = MissionState::GO_LEG;
            send_goal(leg_index_);
            break;
        }

        case RecoveryStrategy::SKIP_WAYPOINT:
            RCLCPP_WARN(this->get_logger(), "Recovery: SKIP_WAYPOINT");
            advance_to_next_leg();
            break;

        case RecoveryStrategy::RESTART_MISSION:
            RCLCPP_WARN(this->get_logger(), "Recovery: RESTART_MISSION");
            leg_index_ = 0;
            retry_count_ = 0;
            recovery_idx_ = 0;
            state_ = MissionState::GO_LEG;
            send_goal(0);
            break;
        }

        if (retry_count_ >= max_retries_) {
            recovery_idx_++;
            retry_count_ = 0;
        }
    }

    // -----------------------------------------------------------------
    // Obstacle Map + Path Avoidance
    // -----------------------------------------------------------------
    void parse_obstacle_map(const std::string& json) {
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
            acc::PathObstacle po;
            po.x = parse_json_double(obj, "\"x\"");
            po.y = parse_json_double(obj, "\"y\"");
            po.radius = parse_json_double(obj, "\"radius\"");
            po.obj_class = parse_json_string(obj, "\"class\"");
            // Parse static field
            auto static_pos = obj.find("\"static\"");
            po.is_static = (static_pos != std::string::npos &&
                           obj.find("true", static_pos) != std::string::npos);
            if (po.radius > 0.01) {
                mapped_obstacles_.push_back(po);
            }

            search = obj_end + 1;
            if (json.find(']', search) < json.find('{', search)) break;
        }

        // Check if current path needs avoidance re-planning.
        // Complements MPCC solver's soft penalty constraints with path-level modification.
        // Cooldown (2s) prevents compounding modifications from causing spiral paths.
        if (enable_path_avoidance_ && state_ == MissionState::GO_LEG && mpcc_mode_ &&
            base_path_ && !mapped_obstacles_.empty() &&
            (now_sec() - last_avoidance_time_) >= AVOIDANCE_COOLDOWN_S) {
            check_path_avoidance();
        }
    }

    void check_path_avoidance() {
        // Always compute from base_path_ (never from already-modified current_path_)
        if (!base_path_ || base_path_->poses.size() < 3) return;

        // Extract base path as xy vectors
        std::vector<double> px, py;
        px.reserve(base_path_->poses.size());
        py.reserve(base_path_->poses.size());
        for (auto& ps : base_path_->poses) {
            px.push_back(ps.pose.position.x);
            py.push_back(ps.pose.position.y);
        }

        int blocked_idx = acc::PathModifier::check_path_blocked(px, py, mapped_obstacles_);
        if (blocked_idx < 0) {
            // Path is clear — revert to base path if it was previously modified
            if (current_path_ && base_path_ &&
                current_path_->poses.size() == base_path_->poses.size()) {
                bool differs = false;
                for (size_t i = 0; i < current_path_->poses.size() && !differs; i++) {
                    if (current_path_->poses[i].pose.position.x != base_path_->poses[i].pose.position.x ||
                        current_path_->poses[i].pose.position.y != base_path_->poses[i].pose.position.y) {
                        differs = true;
                    }
                }
                if (differs) {
                    current_path_ = base_path_;
                    current_path_->header.stamp = this->get_clock()->now();
                    path_pub_->publish(*current_path_);
                    log_behavior("PATH_AVOIDANCE_CLEARED", "Reverted to base path");
                }
            }
            return;
        }

        // Find which obstacle blocks the path
        double min_d = 1e9;
        const acc::PathObstacle* blocking = nullptr;
        for (auto& obs : mapped_obstacles_) {
            double d = std::hypot(px[blocked_idx] - obs.x, py[blocked_idx] - obs.y);
            if (d < min_d) {
                min_d = d;
                blocking = &obs;
            }
        }
        if (!blocking) return;

        RCLCPP_INFO(this->get_logger(),
            "Path blocked at idx %d by %s (%.2f, %.2f) - generating avoidance",
            blocked_idx, blocking->obj_class.c_str(), blocking->x, blocking->y);

        if (acc::PathModifier::generate_avoidance_path(px, py, *blocking)) {
            // Build a new current_path_ from modified waypoints + base orientations
            auto modified = *base_path_;
            for (size_t i = 0; i < px.size() && i < modified.poses.size(); i++) {
                modified.poses[i].pose.position.x = px[i];
                modified.poses[i].pose.position.y = py[i];
            }
            modified.header.stamp = this->get_clock()->now();
            current_path_ = modified;
            path_pub_->publish(*current_path_);
            last_avoidance_time_ = now_sec();
            log_behavior("PATH_AVOIDANCE",
                "Modified path around " + blocking->obj_class +
                " at (" + std::to_string(blocking->x) + ", " + std::to_string(blocking->y) + ")");
        }
    }

    static double parse_json_double(const std::string& json, const std::string& key) {
        auto pos = json.find(key);
        if (pos == std::string::npos) return 0.0;
        auto colon = json.find(':', pos);
        if (colon == std::string::npos) return 0.0;
        try { return std::stod(json.substr(colon + 1)); }
        catch (...) { return 0.0; }
    }

    static std::string parse_json_string(const std::string& json, const std::string& key) {
        auto pos = json.find(key);
        if (pos == std::string::npos) return "";
        auto colon = json.find(':', pos);
        if (colon == std::string::npos) return "";
        auto q1 = json.find('"', colon + 1);
        if (q1 == std::string::npos) return "";
        auto q2 = json.find('"', q1 + 1);
        if (q2 == std::string::npos) return "";
        return json.substr(q1 + 1, q2 - q1 - 1);
    }

    // -----------------------------------------------------------------
    // Main Tick
    // -----------------------------------------------------------------
    void tick() {
        switch (state_) {
        case MissionState::WAIT_FOR_NAV: {
            bool ready = false;
            if (mpcc_mode_ && compute_path_client_) {
                ready = compute_path_client_->action_server_is_ready();
            } else if (navigate_client_) {
                ready = navigate_client_->action_server_is_ready();
            }
            // In MPCC mode with road graph, we can proceed even without Nav2
            if (mpcc_mode_ && road_graph_) ready = true;

            if (ready) {
                if (use_tf_hub_) {
                    state_ = MissionState::CAPTURING_HUB;
                    hub_capture_start_ = now_sec();
                } else {
                    start_mission();
                }
            }
            break;
        }

        case MissionState::CAPTURING_HUB:
            if (capture_hub_from_tf()) {
                start_mission();
            } else if (now_sec() - hub_capture_start_ > hub_tf_timeout_s_) {
                RCLCPP_WARN(this->get_logger(), "Hub TF timeout - using config hub");
                start_mission();
            }
            break;

        case MissionState::GO_LEG: {
            // Obstacle pause
            if (enable_obstacle_ && !mpcc_mode_ && !motion_enabled_) {
                bool has_path = !mpcc_mode_ || current_path_.has_value();
                if (has_path) {
                    pause_for_obstacle();
                    return;
                }
            }

            // MPCC path republishing (every 2s)
            if (mpcc_mode_ && current_path_) {
                double now = now_sec();
                if (now - last_path_pub_time_ > 2.0) {
                    path_pub_->publish(*current_path_);
                    last_path_pub_time_ = now;
                }
            }

            // MPCC goal checking via TF
            if (mpcc_mode_ && goal_result_received_ &&
                leg_index_ < static_cast<int>(legs_.size()))
            {
                auto pose = get_current_pose();
                if (pose) {
                    auto& leg = legs_[leg_index_];
                    double dist = std::hypot(leg.target_x - (*pose)[0],
                                            leg.target_y - (*pose)[1]);

                    // Periodic position logging
                    double now = now_sec();
                    if (now - last_pos_log_time_ > 2.0) {
                        RCLCPP_INFO(this->get_logger(),
                            "MPCC tracking: pos=(%.2f, %.2f) target=(%.2f, %.2f) dist=%.2fm",
                            (*pose)[0], (*pose)[1], leg.target_x, leg.target_y, dist);
                        last_pos_log_time_ = now;
                    }

                    if (dist < goal_tol_m_) {
                        RCLCPP_INFO(this->get_logger(),
                            "MPCC reached goal (dist=%.3f < tol=%.3f)", dist, goal_tol_m_);
                        log_behavior("MPCC_GOAL_REACHED", "dist=" + std::to_string(dist));
                        goal_sent_time_ = 0;
                        on_goal_success();
                        return;
                    }
                }
            }

            // Goal timeout
            if (goal_sent_time_ > 0 && (now_sec() - goal_sent_time_) > goal_timeout_s_) {
                if (mpcc_mode_) {
                    // In MPCC mode, keep tracking instead of forcing recovery timeout loops.
                    // The controller publishes explicit "Goal reached" and mission manager
                    // also checks TF distance each cycle.
                    goal_sent_time_ = now_sec();
                    RCLCPP_WARN(this->get_logger(),
                        "Goal timeout ignored in MPCC mode — continuing tracking");
                    log_behavior("GOAL_TIMEOUT_IGNORED_MPCC", "continuing");
                    break;
                }
                RCLCPP_WARN(this->get_logger(), "Goal timeout (%.0fs)", goal_timeout_s_);
                log_behavior("GOAL_TIMEOUT", "");
                goal_sent_time_ = 0;
                on_goal_failure();
            }
            break;
        }

        case MissionState::PAUSED_OBSTACLE:
            if (motion_enabled_) {
                resume_from_pause();
            } else if (obstacle_pause_start_ > 0 &&
                       (now_sec() - obstacle_pause_start_) > obstacle_pause_timeout_s_) {
                // Force resume instead of entering recovery - likely a false positive
                RCLCPP_WARN(this->get_logger(),
                    "Obstacle pause timeout (%.0fs) - force resuming (likely false positive)",
                    obstacle_pause_timeout_s_);
                log_behavior("OBSTACLE_TIMEOUT_RESUME",
                    "forced resume after " + std::to_string(obstacle_pause_timeout_s_) + "s");
                motion_enabled_ = true;
                resume_from_pause();
            }
            break;

        case MissionState::DWELL: {
            if (mpcc_mode_) {
                auto cmd = geometry_msgs::msg::Twist();
                cmd_vel_pub_->publish(cmd);
            }
            if (now_sec() >= stop_until_) {
                std::string label = (leg_index_ < static_cast<int>(legs_.size()))
                    ? legs_[leg_index_].label : "?";
                log_behavior("DWELL_COMPLETE", label + " - advancing");
                advance_to_next_leg();
            }
            break;
        }

        case MissionState::RECOVERING: {
            if (backup_start_time_ > 0) {
                double backup_duration = backup_distance_m_ / backup_speed_;
                if (now_sec() - backup_start_time_ >= backup_duration) {
                    auto cmd = geometry_msgs::msg::Twist();
                    cmd_vel_pub_->publish(cmd);
                    backup_start_time_ = 0;
                    state_ = MissionState::GO_LEG;
                    send_goal(leg_index_);
                }
                return;
            }
            execute_recovery();
            break;
        }

        case MissionState::DONE:
        case MissionState::ABORTED:
            break;
        }
    }

    void start_mission() {
        RCLCPP_INFO(this->get_logger(), "Starting mission with %zu legs", legs_.size());
        log_behavior("MISSION_START", std::to_string(legs_.size()) + " legs");
        set_led(LED_HUB);

        // Detect starting leg from current position (supports arbitrary start)
        if (mpcc_mode_ && road_graph_) {
            leg_index_ = detect_current_leg();
            RCLCPP_INFO(this->get_logger(), "Auto-detected start leg: %d (%s)",
                         leg_index_,
                         leg_index_ < static_cast<int>(legs_.size())
                             ? legs_[leg_index_].label.c_str() : "?");
            log_behavior("START_LEG_DETECTED", "leg=" + std::to_string(leg_index_));
        } else {
            leg_index_ = 0;
        }

        replan_count_ = 0;
        state_ = MissionState::GO_LEG;
        send_goal(leg_index_);
    }

    // -----------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------
    // Config
    acc::TransformParams tp_;
    bool use_tf_hub_ = true;
    double hub_tf_timeout_s_ = 15.0;
    double goal_timeout_s_ = 120.0;
    int max_retries_ = 2;
    bool enable_led_ = true;
    double goal_tol_m_ = 0.35;
    bool enable_obstacle_ = true;
    double obstacle_pause_timeout_s_ = 10.0;
    double backup_distance_m_ = 0.15;
    double backup_speed_ = 0.1;
    bool mpcc_mode_ = false;

    // Mission data
    std::vector<MissionLeg> legs_;
    std::array<double, 3> hub_ = {0, 0, 0};
    std::array<double, 3> current_target_ = {0, 0, 0};

    // State machine
    MissionState state_ = MissionState::WAIT_FOR_NAV;
    int leg_index_ = 0;
    int retry_count_ = 0;
    int recovery_idx_ = 0;
    double stop_until_ = 0;
    double obstacle_pause_start_ = 0;
    double backup_start_time_ = 0;
    double hub_capture_start_ = 0;
    double last_resume_time_ = 0;
    double resume_cooldown_s_ = 2.0;
    double goal_sent_time_ = 0;
    bool goal_result_received_ = false;
    int last_goal_status_ = -1;
    double last_goal_result_time_ = 0;
    bool motion_enabled_ = true;
    double last_path_pub_time_ = 0;
    double last_pos_log_time_ = 0;
    std::optional<nav_msgs::msg::Path> current_path_;
    std::optional<nav_msgs::msg::Path> base_path_;  // unmodified road_graph path
    double last_avoidance_time_ = 0;
    static constexpr double AVOIDANCE_COOLDOWN_S = 2.0;
    bool enable_path_avoidance_ = false;  // disabled: lidar-only obstacle tracker has too many
                                          // false positives (yield signs, walls) causing the path
                                          // to balloon and vehicle to get stuck. The MPCC solver's
                                          // built-in halfspace obstacle constraints handle avoidance
                                          // at the control level (same as reference implementation).

    // Dynamic replanning
    double last_replan_time_ = 0.0;
    int replan_count_ = 0;
    static constexpr double REPLAN_COOLDOWN_S = 5.0;
    static constexpr int MAX_REPLANS_PER_LEG = 5;

    // Road graph
    std::unique_ptr<acc::RoadGraph> road_graph_;

    // Obstacle map (from obstacle tracker)
    std::vector<acc::PathObstacle> mapped_obstacles_;

    // Callback group
    rclcpp::CallbackGroup::SharedPtr cb_group_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Action clients
    rclcpp_action::Client<ComputePathToPose>::SharedPtr compute_path_client_;
    rclcpp_action::Client<NavigateToPose>::SharedPtr navigate_client_;
    rclcpp_action::ClientGoalHandle<NavigateToPose>::SharedPtr nav_goal_handle_;

    // Publishers
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr hold_pub_;

    // Subscribers
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr motion_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr mpcc_status_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr replan_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr traffic_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr obstacle_map_sub_;

    // Service client
    rclcpp::Client<rcl_interfaces::srv::SetParameters>::SharedPtr led_client_;

    // Timers
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr coord_timer_;

    // Logging
    std::string behavior_log_path_, coord_log_path_;
    double log_start_time_ = 0;
};

}  // namespace acc_mission

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<acc_mission::MissionManagerNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
