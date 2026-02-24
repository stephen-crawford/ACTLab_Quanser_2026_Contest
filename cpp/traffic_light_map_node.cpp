/**
 * C++ Traffic Light Map Node
 *
 * Maps detected traffic light colors to known physical locations from
 * road_boundaries.yaml. Maintains a persistent traffic light state map.
 *
 * When obstacle_detector/sign_detector detects a traffic light color,
 * this node associates it with the nearest known location based on:
 *   1. Vehicle position + detection bearing -> estimated light position
 *   2. Nearest known traffic_control of type "traffic_light" within 1.5m
 *
 * Subscribes:
 *   /traffic_control_state (std_msgs/String, JSON) — light detections
 *   TF (map->base_link) — vehicle pose
 *
 * Publishes:
 *   /traffic_light_map (std_msgs/String, JSON) — persistent light states
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "road_boundaries.h"
#include "yaml_config.h"

#include <ament_index_cpp/get_package_share_directory.hpp>

#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace acc_tlm {

struct KnownLight {
    std::string name;
    double x, y;  // map frame
    double stop_line_distance;
};

struct LightState {
    std::string state;    // "red", "green", "yellow", "unknown"
    double last_update;
    std::string light_name;
};

class TrafficLightMapNode : public rclcpp::Node {
public:
    TrafficLightMapNode() : Node("traffic_light_map") {
        this->declare_parameter("association_radius", 1.5);
        this->declare_parameter("state_timeout", 10.0);
        association_radius_ = this->get_parameter("association_radius").as_double();
        state_timeout_ = this->get_parameter("state_timeout").as_double();

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Load known traffic light positions from road_boundaries.yaml
        load_known_lights();

        // Publisher
        map_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/traffic_light_map", 10);

        // Subscriber
        traffic_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/traffic_control_state", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                on_traffic_state(msg->data);
            });

        // 5Hz publish timer
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(200),
            [this]() { publish_map(); });

        RCLCPP_INFO(this->get_logger(),
            "Traffic Light Map initialized: %zu known lights, assoc_radius=%.1fm",
            known_lights_.size(), association_radius_);
    }

private:
    void load_known_lights() {
        std::string config_path;
        try {
            auto pkg = ament_index_cpp::get_package_share_directory("acc_stage1_mission");
            config_path = pkg + "/config/road_boundaries.yaml";
        } catch (...) {
            try {
                auto pkg = ament_index_cpp::get_package_share_directory("acc_mpcc_controller_cpp");
                config_path = pkg + "/config/road_boundaries.yaml";
            } catch (...) {
                RCLCPP_WARN(this->get_logger(), "Could not find road_boundaries.yaml");
                return;
            }
        }

        try {
            YAML::Node config = YAML::LoadFile(config_path);
            acc::TransformParams tp;
            if (config["transform"]) {
                auto tf = config["transform"];
                tp.origin_x = acc::yaml_double(tf, "origin_x", -1.205);
                tp.origin_y = acc::yaml_double(tf, "origin_y", -0.83);
                tp.origin_heading_rad = acc::yaml_double(tf, "origin_heading_rad", 0.7177);
            }

            if (config["traffic_controls"]) {
                for (const auto& tc : config["traffic_controls"]) {
                    std::string type = acc::yaml_str(tc, "type", "");
                    if (type != "traffic_light") continue;

                    KnownLight light;
                    light.name = acc::yaml_str(tc, "name", "unnamed");
                    double qx = acc::yaml_double(tc, "x", 0);
                    double qy = acc::yaml_double(tc, "y", 0);
                    light.stop_line_distance = acc::yaml_double(tc, "stop_line_distance", 0.2);

                    // Transform to map frame
                    double map_yaw;
                    acc::qlabs_to_map(qx, qy, 0, tp, light.x, light.y, map_yaw);

                    known_lights_.push_back(light);

                    // Initialize state
                    LightState ls;
                    ls.state = "unknown";
                    ls.last_update = 0;
                    ls.light_name = light.name;
                    light_states_[light.name] = ls;

                    RCLCPP_INFO(this->get_logger(),
                        "Known light: %s at map (%.2f, %.2f)",
                        light.name.c_str(), light.x, light.y);
                }
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Failed to load traffic controls: %s", e.what());
        }
    }

    void on_traffic_state(const std::string& json) {
        // Parse control_type and light_state from JSON
        std::string control_type = parse_json_string(json, "\"control_type\"");
        if (control_type != "traffic_light") return;

        std::string light_state = parse_json_string(json, "\"light_state\"");
        if (light_state.empty() || light_state == "unknown") return;

        // Get vehicle pose
        double veh_x, veh_y, veh_theta;
        if (!get_vehicle_pose(veh_x, veh_y, veh_theta)) return;

        double now = this->now().seconds();

        // Find nearest known traffic light to the vehicle's forward direction
        const KnownLight* best_light = nullptr;
        double best_dist = association_radius_;

        for (auto& kl : known_lights_) {
            double dx = kl.x - veh_x;
            double dy = kl.y - veh_y;
            double dist = std::sqrt(dx*dx + dy*dy);

            // Only consider lights roughly ahead (within 90 degrees of heading)
            double bearing = std::atan2(dy, dx);
            double angle_diff = normalize_angle(bearing - veh_theta);
            if (std::abs(angle_diff) > M_PI / 2) continue;

            if (dist < best_dist) {
                best_dist = dist;
                best_light = &kl;
            }
        }

        if (best_light) {
            auto& ls = light_states_[best_light->name];
            ls.state = light_state;
            ls.last_update = now;
            ls.light_name = best_light->name;
        }
    }

    bool get_vehicle_pose(double& x, double& y, double& theta) {
        try {
            auto t = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            x = t.transform.translation.x;
            y = t.transform.translation.y;
            auto& q = t.transform.rotation;
            theta = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                               1.0 - 2.0*(q.y*q.y + q.z*q.z));
            return true;
        } catch (const tf2::TransformException&) {
            return false;
        }
    }

    void publish_map() {
        std::lock_guard<std::mutex> lock(mutex_);
        double now = this->now().seconds();

        // Decay stale states
        for (auto& [name, ls] : light_states_) {
            if (ls.last_update > 0 && (now - ls.last_update) > state_timeout_) {
                ls.state = "unknown";
            }
        }

        // Build JSON
        std::string json = "{\"lights\":{";
        bool first = true;
        for (auto& [name, ls] : light_states_) {
            if (!first) json += ",";
            first = false;

            // Find known light position
            double lx = 0, ly = 0;
            for (auto& kl : known_lights_) {
                if (kl.name == name) { lx = kl.x; ly = kl.y; break; }
            }

            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "\"%s\":{\"state\":\"%s\",\"x\":%.4f,\"y\":%.4f,\"last_update\":%.2f}",
                name.c_str(), ls.state.c_str(), lx, ly, ls.last_update);
            json += buf;
        }
        json += "},\"stamp\":" + std::to_string(now) + "}";

        auto msg = std_msgs::msg::String();
        msg.data = json;
        map_pub_->publish(msg);
    }

    static double normalize_angle(double a) {
        while (a > M_PI)  a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
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

    std::mutex mutex_;

    // Known traffic lights from config
    std::vector<KnownLight> known_lights_;
    std::unordered_map<std::string, LightState> light_states_;

    // Parameters
    double association_radius_ = 1.5;
    double state_timeout_ = 10.0;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publisher
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr map_pub_;

    // Subscriber
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr traffic_sub_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace acc_tlm

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<acc_tlm::TrafficLightMapNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
