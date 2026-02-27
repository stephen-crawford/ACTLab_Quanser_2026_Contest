/**
 * C++ Multi-Class Obstacle Tracker Node
 *
 * Generalizes the Python PedestrianKalmanTracker to track ALL obstacle types
 * (cones, pedestrians, vehicles) with per-class collision radii.
 * Adds lidar scan processing for range-based obstacle confirmation.
 *
 * Subscribes:
 *   /obstacle_positions (std_msgs/String, JSON) — camera-based detections
 *   /scan (sensor_msgs/LaserScan) — lidar for confirmation/close-range detection
 *
 * Publishes:
 *   /tracked_obstacles (std_msgs/String, JSON) — Kalman-filtered obstacle positions
 *   /obstacle_map (std_msgs/String, JSON) — persistent obstacle map
 *
 * Uses constant-velocity Kalman filter per track (state: [x, y, vx, vy]).
 * Greedy nearest-neighbor data association.
 * Per-class collision radii: person=0.25, cone=0.10, vehicle=0.40
 */

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace acc_tracker {

// ============================================================================
// Per-class collision radii
// ============================================================================
static double class_radius(const std::string& cls) {
    if (cls == "person") return 0.25;
    if (cls == "cone" || cls == "traffic_cone" || cls == "sports ball") return 0.10;
    if (cls == "car" || cls == "vehicle") return 0.40;
    return 0.20;  // default
}

// ============================================================================
// Single Kalman Track
// ============================================================================
struct KalmanTrack {
    int track_id;
    std::string obj_class;
    double radius;
    std::string source;       // "camera", "lidar", "fused"
    bool confirmed_by_lidar = false;

    // State: [x, y, vx, vy]
    Eigen::Vector4d state;
    Eigen::Matrix4d P;

    int hits = 1;
    int misses = 0;
    double created_at = 0;
    double last_update = 0;

    // Noise parameters
    static constexpr double q_pos = 0.01;
    static constexpr double q_vel = 0.5;

    KalmanTrack(double x, double y, const std::string& cls, int id, double now)
        : track_id(id), obj_class(cls), radius(class_radius(cls)),
          source("camera"), created_at(now), last_update(now)
    {
        state << x, y, 0.0, 0.0;
        P = Eigen::Matrix4d::Identity();
        P(0,0) = 0.1; P(1,1) = 0.1;
        P(2,2) = 1.0; P(3,3) = 1.0;
    }

    double x() const { return state(0); }
    double y() const { return state(1); }
    double vx() const { return state(2); }
    double vy() const { return state(3); }
    double speed() const { return std::sqrt(vx()*vx() + vy()*vy()); }

    void predict(double dt) {
        if (dt <= 0) return;

        Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
        F(0,2) = dt; F(1,3) = dt;

        Eigen::Matrix4d Q = Eigen::Matrix4d::Zero();
        double dt3 = dt*dt*dt / 3.0;
        double dt2 = dt*dt / 2.0;
        Q(0,0) = q_pos * dt + q_vel * dt3;
        Q(1,1) = q_pos * dt + q_vel * dt3;
        Q(0,2) = q_vel * dt2; Q(2,0) = q_vel * dt2;
        Q(1,3) = q_vel * dt2; Q(3,1) = q_vel * dt2;
        Q(2,2) = q_vel * dt;
        Q(3,3) = q_vel * dt;

        state = F * state;
        P = F * P * F.transpose() + Q;
    }

    void update(double mx, double my) {
        Eigen::Matrix<double, 2, 4> H = Eigen::Matrix<double, 2, 4>::Zero();
        H(0,0) = 1.0; H(1,1) = 1.0;

        Eigen::Matrix2d R;
        R << 0.15, 0, 0, 0.15;  // ~15cm measurement noise

        Eigen::Vector2d z(mx, my);
        Eigen::Vector2d y_inn = z - H * state;
        Eigen::Matrix2d S = H * P * H.transpose() + R;
        Eigen::Matrix<double, 4, 2> K = P * H.transpose() * S.inverse();

        state = state + K * y_inn;
        P = (Eigen::Matrix4d::Identity() - K * H) * P;

        hits++;
        misses = 0;
    }
};

// ============================================================================
// Lidar cluster
// ============================================================================
struct LidarCluster {
    double angle;      // radians, center of cluster
    double distance;   // meters
    double width;      // angular width in radians
    double x_map, y_map;  // map-frame position (filled after transform)
};

// ============================================================================
// Mapped obstacle (persistent)
// ============================================================================
struct MappedObstacle {
    double x, y;
    double radius;
    std::string obj_class;
    double first_seen;
    double last_seen;
    double confidence;
    bool is_static;
    double vx = 0, vy = 0;
    int id;
};

// ============================================================================
// ObstacleTrackerNode
// ============================================================================
class ObstacleTrackerNode : public rclcpp::Node {
public:
    ObstacleTrackerNode() : Node("obstacle_tracker") {
        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publishers
        tracked_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/tracked_obstacles", 10);
        obstacle_map_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/obstacle_map", 10);

        // Subscribers
        obstacle_pos_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/obstacle_positions", 10,
            [this](std_msgs::msg::String::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                on_obstacle_positions(msg->data);
            });

        auto scan_qos = rclcpp::QoS(1).best_effort();
        scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", scan_qos,
            [this](sensor_msgs::msg::LaserScan::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                on_lidar_scan(msg);
            });

        // 20Hz update timer (predict + publish)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),
            [this]() { tick(); });

        last_predict_time_ = this->now().seconds();
        RCLCPP_INFO(this->get_logger(),
            "Obstacle Tracker (C++) initialized: multi-class Kalman + lidar fusion");
    }

private:
    // Parse obstacle positions JSON
    void on_obstacle_positions(const std::string& json) {
        // Parse JSON: {"obstacles": [{"x":..,"y":..,"radius":..,"obj_class":..,"frame":"map|base_link"}, ...]}
        std::vector<std::tuple<double, double, std::string>> detections;

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
            double x = parse_json_double(obj, "\"x\"");
            double y_val = parse_json_double(obj, "\"y\"");
            std::string cls = parse_json_string(obj, "\"obj_class\"");
            std::string frame = parse_json_string(obj, "\"frame\"");
            if (cls.empty()) cls = "unknown";

            // Frame handling:
            // - frame=="map": coordinates already in map frame.
            // - otherwise (legacy/default): interpret as vehicle frame (x=fwd, y=left)
            if (frame == "map" || frame == "world") {
                detections.emplace_back(x, y_val, cls);
            } else {
                double map_x, map_y;
                if (vehicle_to_map(x, y_val, map_x, map_y)) {
                    detections.emplace_back(map_x, map_y, cls);
                }
            }

            search = obj_end + 1;
            if (json.find(']', search) < json.find('{', search)) break;
        }

        update_camera(detections);
    }

    bool vehicle_to_map(double vx, double vy, double& mx, double& my) {
        try {
            auto t = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            double px = t.transform.translation.x;
            double py = t.transform.translation.y;
            auto& q = t.transform.rotation;
            double theta = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                                       1.0 - 2.0*(q.y*q.y + q.z*q.z));
            double ct = std::cos(theta), st = std::sin(theta);
            mx = px + vx * ct - vy * st;
            my = py + vx * st + vy * ct;
            return true;
        } catch (const tf2::TransformException&) {
            return false;
        }
    }

    void update_camera(const std::vector<std::tuple<double, double, std::string>>& detections) {
        double now = this->now().seconds();

        if (detections.empty()) {
            for (auto& track : tracks_) track.misses++;
            return;
        }

        if (tracks_.empty()) {
            for (auto& [dx, dy, cls] : detections) {
                int new_id = next_id_++;
                tracks_.emplace_back(dx, dy, cls, new_id, now);
                RCLCPP_INFO(this->get_logger(),
                    "New track #%d: class=%s at (%.2f, %.2f)",
                    new_id, cls.c_str(), dx, dy);
            }
            return;
        }

        // Greedy nearest-neighbor association
        std::vector<bool> used_tracks(tracks_.size(), false);
        std::vector<bool> used_dets(detections.size(), false);

        struct Pair { double dist; size_t ti, di; };
        std::vector<Pair> pairs;
        for (size_t ti = 0; ti < tracks_.size(); ti++) {
            for (size_t di = 0; di < detections.size(); di++) {
                auto& [dx, dy, cls] = detections[di];
                double d = std::hypot(tracks_[ti].x() - dx, tracks_[ti].y() - dy);
                pairs.push_back({d, ti, di});
            }
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b) { return a.dist < b.dist; });

        for (auto& p : pairs) {
            if (used_tracks[p.ti] || used_dets[p.di]) continue;
            if (p.dist > max_association_dist_) break;
            auto& [dx, dy, cls] = detections[p.di];
            tracks_[p.ti].update(dx, dy);
            tracks_[p.ti].last_update = now;
            // Update class if detection is more specific
            if (cls != "unknown" && tracks_[p.ti].obj_class == "unknown") {
                tracks_[p.ti].obj_class = cls;
                tracks_[p.ti].radius = class_radius(cls);
            }
            used_tracks[p.ti] = true;
            used_dets[p.di] = true;
        }

        // Increment miss for unmatched tracks
        for (size_t ti = 0; ti < tracks_.size(); ti++) {
            if (!used_tracks[ti]) tracks_[ti].misses++;
        }

        // Create new tracks for unmatched detections
        for (size_t di = 0; di < detections.size(); di++) {
            if (!used_dets[di]) {
                auto& [dx, dy, cls] = detections[di];
                int new_id = next_id_++;
                tracks_.emplace_back(dx, dy, cls, new_id, now);
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "New track #%d: class=%s at (%.2f, %.2f)",
                    new_id, cls.c_str(), dx, dy);
            }
        }
    }

    void on_lidar_scan(sensor_msgs::msg::LaserScan::SharedPtr msg) {
        auto clusters = extract_lidar_clusters(msg);

        // Try to confirm existing camera tracks with lidar data
        for (auto& cluster : clusters) {
            for (auto& track : tracks_) {
                double d = std::hypot(track.x() - cluster.x_map,
                                       track.y() - cluster.y_map);
                if (d < 0.3) {
                    track.confirmed_by_lidar = true;
                    track.source = "fused";
                }
            }
        }

        // Create lidar-only tracks for clusters not matching any camera track
        double now = this->now().seconds();
        for (auto& cluster : clusters) {
            if (cluster.distance > 2.0) continue;  // only close obstacles
            bool matched = false;
            for (auto& track : tracks_) {
                if (std::hypot(track.x() - cluster.x_map,
                               track.y() - cluster.y_map) < 0.5) {
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                int new_id = next_id_++;
                auto track = KalmanTrack(cluster.x_map, cluster.y_map,
                                         "unknown", new_id, now);
                track.source = "lidar";
                tracks_.push_back(std::move(track));
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Lidar-only track #%d at (%.2f, %.2f), dist=%.2fm",
                    new_id, cluster.x_map, cluster.y_map, cluster.distance);
            }
        }
    }

    std::vector<LidarCluster> extract_lidar_clusters(
        sensor_msgs::msg::LaserScan::SharedPtr msg)
    {
        std::vector<LidarCluster> clusters;

        // Get vehicle pose for map transform
        double veh_x = 0, veh_y = 0, veh_theta = 0;
        bool has_pose = false;
        try {
            auto t = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            veh_x = t.transform.translation.x;
            veh_y = t.transform.translation.y;
            auto& q = t.transform.rotation;
            veh_theta = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                                    1.0 - 2.0*(q.y*q.y + q.z*q.z));
            has_pose = true;
        } catch (const tf2::TransformException&) {
            return clusters;
        }

        // Extract clusters: group consecutive readings within 0.15m
        std::vector<std::pair<int, double>> valid_points;
        for (size_t i = 0; i < msg->ranges.size(); i++) {
            double r = msg->ranges[i];
            if (r > 0.15 && r < 3.0 && std::isfinite(r)) {
                valid_points.emplace_back(static_cast<int>(i), r);
            }
        }

        if (valid_points.empty()) return clusters;

        // Simple clustering
        std::vector<std::vector<std::pair<int, double>>> raw_clusters;
        raw_clusters.push_back({valid_points[0]});
        for (size_t i = 1; i < valid_points.size(); i++) {
            auto& prev = raw_clusters.back().back();
            auto& curr = valid_points[i];
            if (curr.first - prev.first <= 2 && std::abs(curr.second - prev.second) < 0.15) {
                raw_clusters.back().push_back(curr);
            } else {
                raw_clusters.push_back({curr});
            }
        }

        // Convert clusters to map frame
        for (auto& rc : raw_clusters) {
            if (rc.size() < 2) continue;  // skip single-point clusters

            double sum_angle = 0, sum_dist = 0;
            for (auto& [idx, range] : rc) {
                double angle = msg->angle_min + idx * msg->angle_increment;
                sum_angle += angle;
                sum_dist += range;
            }
            double avg_angle = sum_angle / rc.size();
            double avg_dist = sum_dist / rc.size();
            double width = (rc.back().first - rc.front().first) * msg->angle_increment;

            LidarCluster cluster;
            cluster.angle = avg_angle;
            cluster.distance = avg_dist;
            cluster.width = width;

            if (has_pose) {
                double global_angle = veh_theta + avg_angle;
                cluster.x_map = veh_x + avg_dist * std::cos(global_angle);
                cluster.y_map = veh_y + avg_dist * std::sin(global_angle);
            }
            clusters.push_back(cluster);
        }

        return clusters;
    }

    void tick() {
        std::lock_guard<std::mutex> lock(mutex_);

        double now = this->now().seconds();
        double dt = now - last_predict_time_;
        last_predict_time_ = now;

        // Predict all tracks
        for (auto& track : tracks_) {
            track.predict(dt);
        }

        // Prune stale tracks
        size_t before_size = tracks_.size();
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                [now, this](const KalmanTrack& t) {
                    if ((now - t.last_update) > max_coast_time_) {
                        RCLCPP_INFO(this->get_logger(),
                            "Track #%d lost: class=%s, hits=%d, age=%.1fs",
                            t.track_id, t.obj_class.c_str(), t.hits,
                            now - t.created_at);
                        return true;
                    }
                    return false;
                }),
            tracks_.end());

        // Periodic summary (every 5s)
        if (now - last_summary_time_ > 5.0) {
            last_summary_time_ = now;
            int confirmed = 0;
            for (auto& t : tracks_) { if (t.hits >= min_hits_to_confirm_) confirmed++; }
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Tracker: %zu active tracks (%d confirmed), %zu mapped obstacles",
                tracks_.size(), confirmed, obstacle_map_.size());
        }

        // Update persistent obstacle map
        update_obstacle_map(now);

        // Publish tracked obstacles
        publish_tracked(now);

        // Publish obstacle map
        publish_obstacle_map(now);
    }

    void update_obstacle_map(double now) {
        // Update existing mapped obstacles from tracks
        for (auto& track : tracks_) {
            if (track.hits < min_hits_to_confirm_) continue;

            bool found = false;
            for (auto& mapped : obstacle_map_) {
                if (std::hypot(mapped.x - track.x(), mapped.y - track.y()) < 0.3 &&
                    mapped.obj_class == track.obj_class) {
                    // Update existing
                    mapped.x = track.x();
                    mapped.y = track.y();
                    mapped.vx = track.vx();
                    mapped.vy = track.vy();
                    mapped.last_seen = now;
                    mapped.confidence = std::min(1.0, mapped.confidence + 0.1);
                    mapped.is_static = (track.speed() < 0.05 &&
                                        (now - mapped.first_seen) > 2.0);
                    found = true;
                    break;
                }
            }
            if (!found) {
                MappedObstacle mo;
                mo.x = track.x();
                mo.y = track.y();
                mo.radius = track.radius;
                mo.obj_class = track.obj_class;
                mo.first_seen = now;
                mo.last_seen = now;
                mo.confidence = 0.3;
                mo.is_static = false;
                mo.vx = track.vx();
                mo.vy = track.vy();
                mo.id = track.track_id;
                obstacle_map_.push_back(mo);
            }
        }

        // Prune stale mapped obstacles
        obstacle_map_.erase(
            std::remove_if(obstacle_map_.begin(), obstacle_map_.end(),
                [now](const MappedObstacle& m) {
                    return (now - m.last_seen) > 30.0;
                }),
            obstacle_map_.end());
    }

    void publish_tracked(double now) {
        std::string json = "{\"tracked_obstacles\":[";
        bool first = true;
        for (auto& track : tracks_) {
            if (track.hits < min_hits_to_confirm_) continue;
            if ((now - track.last_update) > 0.5) continue;

            if (!first) json += ",";
            first = false;

            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "{\"id\":%d,\"x\":%.4f,\"y\":%.4f,\"vx\":%.4f,\"vy\":%.4f,"
                "\"radius\":%.3f,\"class\":\"%s\",\"source\":\"%s\","
                "\"lidar_confirmed\":%s,\"hits\":%d}",
                track.track_id, track.x(), track.y(), track.vx(), track.vy(),
                track.radius, track.obj_class.c_str(), track.source.c_str(),
                track.confirmed_by_lidar ? "true" : "false", track.hits);
            json += buf;
        }
        json += "],\"stamp\":" + std::to_string(now) + "}";

        auto msg = std_msgs::msg::String();
        msg.data = json;
        tracked_pub_->publish(msg);
    }

    void publish_obstacle_map(double now) {
        std::string json = "{\"obstacles\":[";
        bool first = true;
        for (auto& m : obstacle_map_) {
            if (!first) json += ",";
            first = false;

            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "{\"id\":%d,\"x\":%.4f,\"y\":%.4f,\"radius\":%.3f,"
                "\"class\":\"%s\",\"confidence\":%.2f,\"static\":%s,"
                "\"vx\":%.4f,\"vy\":%.4f}",
                m.id, m.x, m.y, m.radius, m.obj_class.c_str(),
                m.confidence, m.is_static ? "true" : "false",
                m.vx, m.vy);
            json += buf;
        }
        json += "],\"stamp\":" + std::to_string(now) + "}";

        auto msg = std_msgs::msg::String();
        msg.data = json;
        obstacle_map_pub_->publish(msg);
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

    std::mutex mutex_;

    // Tracks
    std::vector<KalmanTrack> tracks_;
    int next_id_ = 0;
    double last_predict_time_ = 0;

    // Persistent obstacle map
    std::vector<MappedObstacle> obstacle_map_;

    // Parameters
    double max_association_dist_ = 1.0;
    double max_coast_time_ = 2.0;
    int min_hits_to_confirm_ = 2;
    double last_summary_time_ = 0;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publishers
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tracked_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr obstacle_map_pub_;

    // Subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr obstacle_pos_sub_;
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace acc_tracker

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<acc_tracker::ObstacleTrackerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
