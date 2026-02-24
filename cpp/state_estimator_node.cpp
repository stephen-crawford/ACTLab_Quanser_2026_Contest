/**
 * C++ Vehicle State Estimator Node
 *
 * Extended Kalman Filter (EKF) fusing TF position, encoder velocity,
 * and /odom into a smooth, filtered vehicle state published at 50Hz.
 *
 * State vector: [x, y, theta, v, omega] (5 states)
 *   - (x, y): map-frame position
 *   - theta: heading
 *   - v: forward velocity
 *   - omega: yaw rate
 *
 * Measurement sources:
 *   - TF (map->base_link): x, y, theta
 *   - Encoder (/qcar2_joint): v
 *   - Odom (/odom): v, omega (backup)
 *
 * Publishes:
 *   /vehicle_state (std_msgs/String, JSON) at 50Hz
 *
 * Process model: constant-velocity Ackermann
 *   x_k+1 = x_k + v*cos(theta)*dt
 *   y_k+1 = y_k + v*sin(theta)*dt
 *   theta_k+1 = theta_k + omega*dt
 *   v_k+1 = v_k
 *   omega_k+1 = omega_k
 */

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>

#include <Eigen/Dense>

#include <cmath>
#include <chrono>
#include <mutex>
#include <string>

namespace acc_ekf {

/**
 * 5-state Extended Kalman Filter for vehicle state estimation.
 */
class VehicleEKF {
public:
    static constexpr int N = 5;  // state dimension
    using VecN = Eigen::Matrix<double, N, 1>;
    using MatN = Eigen::Matrix<double, N, N>;

    VehicleEKF() {
        x_.setZero();
        P_ = MatN::Identity();
        P_(0,0) = 1.0; P_(1,1) = 1.0; P_(2,2) = 0.5;
        P_(3,3) = 0.1; P_(4,4) = 0.1;

        // Process noise
        Q_ = MatN::Zero();
        Q_(0,0) = 0.01;  // x position
        Q_(1,1) = 0.01;  // y position
        Q_(2,2) = 0.005; // heading
        Q_(3,3) = 0.1;   // velocity
        Q_(4,4) = 0.05;  // yaw rate
    }

    void predict(double dt) {
        if (dt <= 0 || dt > 1.0) return;

        double v = x_(3);
        double omega = x_(4);
        double theta = x_(2);
        double ct = std::cos(theta);
        double st = std::sin(theta);

        // State prediction (constant velocity Ackermann)
        VecN x_pred;
        x_pred(0) = x_(0) + v * ct * dt;
        x_pred(1) = x_(1) + v * st * dt;
        x_pred(2) = normalize_angle(x_(2) + omega * dt);
        x_pred(3) = x_(3);  // constant velocity
        x_pred(4) = x_(4);  // constant yaw rate

        // Jacobian F = df/dx
        MatN F = MatN::Identity();
        F(0,2) = -v * st * dt;
        F(0,3) =  ct * dt;
        F(1,2) =  v * ct * dt;
        F(1,3) =  st * dt;
        F(2,4) = dt;

        // Scale process noise by dt
        MatN Q_scaled = Q_ * dt;

        x_ = x_pred;
        P_ = F * P_ * F.transpose() + Q_scaled;
    }

    /** Update with TF pose measurement: [x, y, theta] */
    void update_pose(double mx, double my, double mtheta) {
        Eigen::Matrix<double, 3, 1> z;
        z << mx, my, mtheta;

        Eigen::Matrix<double, 3, N> H = Eigen::Matrix<double, 3, N>::Zero();
        H(0,0) = 1.0;
        H(1,1) = 1.0;
        H(2,2) = 1.0;

        Eigen::Matrix<double, 3, 3> R;
        R << 0.02, 0, 0,
             0, 0.02, 0,
             0, 0, 0.01;

        Eigen::Matrix<double, 3, 1> y = z - H * x_;
        y(2) = normalize_angle(y(2));

        Eigen::Matrix<double, 3, 3> S = H * P_ * H.transpose() + R;
        Eigen::Matrix<double, N, 3> K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y;
        x_(2) = normalize_angle(x_(2));
        P_ = (MatN::Identity() - K * H) * P_;
    }

    /** Update with encoder velocity measurement: [v] */
    void update_velocity(double mv) {
        Eigen::Matrix<double, 1, 1> z;
        z << mv;

        Eigen::Matrix<double, 1, N> H = Eigen::Matrix<double, 1, N>::Zero();
        H(0,3) = 1.0;

        Eigen::Matrix<double, 1, 1> R;
        R << 0.005;  // encoder is very accurate

        Eigen::Matrix<double, 1, 1> y_inn = z - H * x_;
        Eigen::Matrix<double, 1, 1> S = H * P_ * H.transpose() + R;
        Eigen::Matrix<double, N, 1> K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y_inn;
        x_(2) = normalize_angle(x_(2));
        P_ = (MatN::Identity() - K * H) * P_;
    }

    /** Update with odom measurement: [v, omega] */
    void update_odom(double mv, double momega) {
        Eigen::Matrix<double, 2, 1> z;
        z << mv, momega;

        Eigen::Matrix<double, 2, N> H = Eigen::Matrix<double, 2, N>::Zero();
        H(0,3) = 1.0;
        H(1,4) = 1.0;

        Eigen::Matrix<double, 2, 2> R;
        R << 0.05, 0,
             0, 0.02;

        Eigen::Matrix<double, 2, 1> y_inn = z - H * x_;
        Eigen::Matrix<double, 2, 2> S = H * P_ * H.transpose() + R;
        Eigen::Matrix<double, N, 2> K = P_ * H.transpose() * S.inverse();

        x_ = x_ + K * y_inn;
        x_(2) = normalize_angle(x_(2));
        P_ = (MatN::Identity() - K * H) * P_;
    }

    double x() const { return x_(0); }
    double y() const { return x_(1); }
    double theta() const { return x_(2); }
    double v() const { return x_(3); }
    double omega() const { return x_(4); }
    const VecN& state() const { return x_; }
    const MatN& covariance() const { return P_; }

private:
    static double normalize_angle(double a) {
        while (a > M_PI)  a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    VecN x_;
    MatN P_;
    MatN Q_;
};


class StateEstimatorNode : public rclcpp::Node {
public:
    StateEstimatorNode() : Node("state_estimator") {
        // TF
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Publisher
        state_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/vehicle_state", 10);

        // Subscribers
        auto qos_be = rclcpp::QoS(1).best_effort();
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", qos_be,
            [this](nav_msgs::msg::Odometry::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(mutex_);
                ekf_.update_odom(msg->twist.twist.linear.x,
                                 msg->twist.twist.angular.z);
                last_odom_time_ = this->now().seconds();
            });

        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/qcar2_joint", qos_be,
            [this](sensor_msgs::msg::JointState::SharedPtr msg) {
                if (msg->velocity.empty()) return;
                // Compute velocity from encoder (matches reference MPC_node.py:65)
                double encoder_vel = msg->velocity[0];
                double v = (encoder_vel / (720.0 * 4.0))
                           * ((13.0 * 19.0) / (70.0 * 30.0))
                           * (2.0 * M_PI) * 0.033;
                std::lock_guard<std::mutex> lock(mutex_);
                ekf_.update_velocity(v);
                last_encoder_time_ = this->now().seconds();
            });

        // 50Hz predict + publish timer
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            [this]() { predict_and_publish(); });

        last_time_ = this->now().seconds();
        RCLCPP_INFO(this->get_logger(), "State Estimator (EKF) initialized at 50Hz");
        RCLCPP_INFO(this->get_logger(),
            "Initial covariance diag: [%.3f, %.3f, %.3f, %.3f, %.3f]",
            ekf_.covariance()(0,0), ekf_.covariance()(1,1),
            ekf_.covariance()(2,2), ekf_.covariance()(3,3),
            ekf_.covariance()(4,4));
    }

private:
    void predict_and_publish() {
        std::lock_guard<std::mutex> lock(mutex_);

        double now = this->now().seconds();
        double dt = now - last_time_;
        last_time_ = now;

        // TF pose update
        try {
            auto t = tf_buffer_->lookupTransform(
                "map", "base_link", tf2::TimePointZero,
                tf2::durationFromSec(0.02));
            double x = t.transform.translation.x;
            double y = t.transform.translation.y;
            auto& q = t.transform.rotation;
            double theta = std::atan2(2.0*(q.w*q.z + q.x*q.y),
                                       1.0 - 2.0*(q.y*q.y + q.z*q.z));

            // Check for large innovation (potential TF jump)
            double dx = x - ekf_.x();
            double dy = y - ekf_.y();
            double innovation = std::sqrt(dx*dx + dy*dy);
            if (innovation > 0.3) {
                RCLCPP_WARN(this->get_logger(),
                    "Large TF innovation: %.3fm (TF: %.3f,%.3f vs EKF: %.3f,%.3f)",
                    innovation, x, y, ekf_.x(), ekf_.y());
            }

            ekf_.update_pose(x, y, theta);
            last_tf_time_ = now;

            if (!has_tf_) {
                has_tf_ = true;
                RCLCPP_INFO(this->get_logger(),
                    "First TF pose received: (%.3f, %.3f, %.1f deg)",
                    x, y, theta * 180.0 / M_PI);
            }
        } catch (const tf2::TransformException&) {
            // No TF available — predict-only
            if (has_tf_ && (now - last_tf_time_) > 1.0) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "TF lost for %.1fs — running on prediction only", now - last_tf_time_);
            }
        }

        // Predict
        ekf_.predict(dt);

        // Periodic state summary
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "EKF state: pos=(%.3f,%.3f) v=%.3f omega=%.3f cov_diag=(%.4f,%.4f,%.4f)",
            ekf_.x(), ekf_.y(), ekf_.v(), ekf_.omega(),
            ekf_.covariance()(0,0), ekf_.covariance()(1,1), ekf_.covariance()(3,3));

        // Publish JSON state
        auto msg = std_msgs::msg::String();
        char buf[512];
        auto& P = ekf_.covariance();
        std::snprintf(buf, sizeof(buf),
            "{\"x\":%.6f,\"y\":%.6f,\"theta\":%.6f,\"v\":%.6f,\"omega\":%.6f,"
            "\"cov\":[%.6f,%.6f,%.6f,%.6f,%.6f],"
            "\"stamp\":%.6f}",
            ekf_.x(), ekf_.y(), ekf_.theta(), ekf_.v(), ekf_.omega(),
            P(0,0), P(1,1), P(2,2), P(3,3), P(4,4),
            now);
        msg.data = buf;
        state_pub_->publish(msg);
    }

    VehicleEKF ekf_;
    std::mutex mutex_;

    // TF
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // Publisher
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;

    // Timer
    rclcpp::TimerBase::SharedPtr timer_;

    double last_time_ = 0;
    double last_tf_time_ = 0;
    double last_odom_time_ = 0;
    double last_encoder_time_ = 0;
    bool has_tf_ = false;
};

}  // namespace acc_ekf

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<acc_ekf::StateEstimatorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
