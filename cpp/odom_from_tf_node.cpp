/**
 * C++ Odom From TF Publisher
 *
 * Ported from odom_from_tf.py.
 * Converts TF transforms (odom -> base_link) to Odometry messages on /odom.
 * Broadcasts TF frames when Cartographer doesn't provide them.
 *
 * Subscriptions: TF (via tf2_ros)
 * Publications:  /odom (nav_msgs/Odometry, BEST_EFFORT QoS)
 * TF Broadcasts: map -> odom, odom -> base_link (when needed)
 *
 * Timer runs at 50Hz (configurable).
 */

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/utils.h>

#include <cmath>
#include <string>

class OdomFromTFNode : public rclcpp::Node {
public:
    OdomFromTFNode() : Node("odom_from_tf") {
        // Parameters
        this->declare_parameter("odom_frame", "odom");
        this->declare_parameter("base_frame", "base_link");
        this->declare_parameter("map_frame", "map");
        this->declare_parameter("publish_rate", 50.0);
        this->declare_parameter("broadcast_tf", true);

        odom_frame_ = this->get_parameter("odom_frame").as_string();
        base_frame_ = this->get_parameter("base_frame").as_string();
        map_frame_ = this->get_parameter("map_frame").as_string();
        double rate = this->get_parameter("publish_rate").as_double();
        broadcast_tf_ = this->get_parameter("broadcast_tf").as_bool();

        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);

        // Publisher - BEST_EFFORT QoS
        auto qos = rclcpp::QoS(10).best_effort();
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", qos);

        // Timer
        double period_ms = 1000.0 / rate;
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(period_ms)),
            std::bind(&OdomFromTFNode::publish_odom, this));

        RCLCPP_INFO(this->get_logger(),
            "OdomFromTF (C++) started: %s -> %s -> /odom @ %.0fHz",
            odom_frame_.c_str(), base_frame_.c_str(), rate);
    }

private:
    enum class TFSource { NONE, ODOM_TO_BASE, MAP_TO_BASE };

    struct LookupResult {
        geometry_msgs::msg::TransformStamped transform;
        std::string frame_id;
        bool needs_broadcast = false;
        bool valid = false;
    };

    LookupResult lookup_transform() {
        LookupResult lr;

        // Primary: odom -> base_link
        try {
            lr.transform = tf_buffer_->lookupTransform(
                odom_frame_, base_frame_, tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            if (using_map_frame_) {
                RCLCPP_INFO(this->get_logger(),
                    "Switched to native %s -> %s TF",
                    odom_frame_.c_str(), base_frame_.c_str());
            }
            using_map_frame_ = false;
            lr.frame_id = odom_frame_;
            lr.needs_broadcast = false;
            lr.valid = true;
            return lr;
        } catch (const tf2::TransformException&) {}

        // Fallback: map -> base_link
        try {
            lr.transform = tf_buffer_->lookupTransform(
                map_frame_, base_frame_, tf2::TimePointZero,
                tf2::durationFromSec(0.05));
            if (!using_map_frame_) {
                RCLCPP_INFO(this->get_logger(),
                    "Using fallback %s -> %s TF",
                    map_frame_.c_str(), base_frame_.c_str());
            }
            using_map_frame_ = true;
            lr.frame_id = odom_frame_;
            lr.needs_broadcast = true;
            lr.valid = true;
            return lr;
        } catch (const tf2::TransformException&) {}

        return lr;  // invalid
    }

    void broadcast_odom_tf(const geometry_msgs::msg::TransformStamped& map_to_base) {
        auto now = this->get_clock()->now();

        // map -> odom (identity)
        geometry_msgs::msg::TransformStamped map_to_odom;
        map_to_odom.header.stamp = now;
        map_to_odom.header.frame_id = map_frame_;
        map_to_odom.child_frame_id = odom_frame_;
        map_to_odom.transform.rotation.w = 1.0;

        // odom -> base_link (same as map -> base_link since odom = map)
        geometry_msgs::msg::TransformStamped odom_to_base;
        odom_to_base.header.stamp = now;
        odom_to_base.header.frame_id = odom_frame_;
        odom_to_base.child_frame_id = base_frame_;
        odom_to_base.transform = map_to_base.transform;

        tf_broadcaster_->sendTransform({map_to_odom, odom_to_base});
    }

    void publish_odom() {
        auto lr = lookup_transform();
        if (!lr.valid) {
            double now = this->now().seconds();
            if (now - last_warn_time_ > 5.0) {
                RCLCPP_WARN(this->get_logger(),
                    "TF not available: tried %s->%s and %s->%s",
                    odom_frame_.c_str(), base_frame_.c_str(),
                    map_frame_.c_str(), base_frame_.c_str());
                last_warn_time_ = now;
            }
            return;
        }

        // Broadcast TF if needed
        if (lr.needs_broadcast && broadcast_tf_) {
            broadcast_odom_tf(lr.transform);
        }

        auto& t = lr.transform.transform;
        double x = t.translation.x;
        double y = t.translation.y;
        double z = t.translation.z;
        double qx = t.rotation.x;
        double qy = t.rotation.y;
        double qz = t.rotation.z;
        double qw = t.rotation.w;

        // Yaw from quaternion
        double siny_cosp = 2.0 * (qw * qz + qx * qy);
        double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
        double theta = std::atan2(siny_cosp, cosy_cosp);

        // Estimate velocity
        double current_time = this->now().seconds();
        double vx = 0.0, vy = 0.0, vtheta = 0.0;

        if (has_prev_) {
            double dt = current_time - last_time_;
            if (dt > 0.001) {
                double dx = x - last_x_;
                double dy = y - last_y_;
                double cos_t = std::cos(theta);
                double sin_t = std::sin(theta);
                vx = (dx * cos_t + dy * sin_t) / dt;
                vy = (-dx * sin_t + dy * cos_t) / dt;

                double dtheta = theta - last_theta_;
                while (dtheta > M_PI)  dtheta -= 2.0 * M_PI;
                while (dtheta < -M_PI) dtheta += 2.0 * M_PI;
                vtheta = dtheta / dt;
            }
        }

        last_x_ = x;
        last_y_ = y;
        last_theta_ = theta;
        last_time_ = current_time;
        has_prev_ = true;

        // Build Odometry message
        auto msg = nav_msgs::msg::Odometry();
        msg.header.stamp = lr.transform.header.stamp;
        msg.header.frame_id = lr.frame_id;
        msg.child_frame_id = base_frame_;

        msg.pose.pose.position.x = x;
        msg.pose.pose.position.y = y;
        msg.pose.pose.position.z = z;
        msg.pose.pose.orientation.x = qx;
        msg.pose.pose.orientation.y = qy;
        msg.pose.pose.orientation.z = qz;
        msg.pose.pose.orientation.w = qw;

        msg.twist.twist.linear.x = vx;
        msg.twist.twist.linear.y = vy;
        msg.twist.twist.angular.z = vtheta;

        // Covariance
        std::fill(msg.pose.covariance.begin(), msg.pose.covariance.end(), 0.01);
        msg.pose.covariance[0] = 0.01;   // x
        msg.pose.covariance[7] = 0.01;   // y
        msg.pose.covariance[35] = 0.01;  // yaw

        std::fill(msg.twist.covariance.begin(), msg.twist.covariance.end(), 0.01);
        msg.twist.covariance[0] = 0.1;   // vx
        msg.twist.covariance[35] = 0.1;  // vyaw

        odom_pub_->publish(msg);
    }

    // Parameters
    std::string odom_frame_, base_frame_, map_frame_;
    bool broadcast_tf_ = true;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // Publisher
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // State
    bool using_map_frame_ = false;
    bool has_prev_ = false;
    double last_x_ = 0, last_y_ = 0, last_theta_ = 0, last_time_ = 0;
    double last_warn_time_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdomFromTFNode>());
    rclcpp::shutdown();
    return 0;
}
