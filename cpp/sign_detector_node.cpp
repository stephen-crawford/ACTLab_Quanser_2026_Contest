/**
 * C++ Traffic Sign Detector for QCar2 (QLabs Virtual Environment)
 *
 * Fast HSV color segmentation + contour shape analysis.
 * No ML model required - QLabs has clean, well-defined sign colors.
 *
 * Detection pipeline:
 *   1. Convert BGR -> HSV
 *   2. Threshold for red (stop signs, red lights), green (green lights),
 *      orange (cones), yellow (yield signs)
 *   3. Find contours, classify shape (octagon, triangle, circle)
 *   4. State machine: stop sign timing, traffic light state (yield detected but not acted on)
 *   5. Publish /traffic_control_state (JSON) and /motion_enable (Bool)
 *
 * Subscriptions:
 *   /camera/color_image  - sensor_msgs/Image (BGR8 or RGB8)
 *
 * Publications:
 *   /traffic_control_state - std_msgs/String (JSON)
 *   /motion_enable         - std_msgs/Bool
 *   /obstacle_positions    - std_msgs/String (JSON, cone positions)
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

// ============================================================================
// Configuration
// ============================================================================
struct DetectorConfig {
    // HSV ranges for red (two ranges needed since red wraps around H=0/180)
    // S≥120 rejects antialiased white-on-blue edges (roundabout sign arrows)
    cv::Scalar red_lower1{0, 120, 80};
    cv::Scalar red_upper1{10, 255, 255};
    cv::Scalar red_lower2{170, 120, 80};
    cv::Scalar red_upper2{180, 255, 255};

    // HSV ranges for green traffic light
    cv::Scalar green_lower{40, 80, 80};
    cv::Scalar green_upper{85, 255, 255};

    // HSV ranges for orange cones
    cv::Scalar orange_lower{5, 150, 150};
    cv::Scalar orange_upper{18, 255, 255};

    // HSV ranges for yellow (yield sign border, yellow light)
    cv::Scalar yellow_lower{18, 100, 100};
    cv::Scalar yellow_upper{35, 255, 255};

    // Minimum contour area (pixels^2) to consider
    double min_contour_area = 400.0;
    double max_contour_area = 80000.0;

    // Shape classification thresholds
    int stop_sign_min_vertices = 6;   // Octagon approx: 6-10 vertices
    int stop_sign_max_vertices = 12;
    int yield_min_vertices = 3;       // Triangle: 3-5 vertices
    int yield_max_vertices = 5;

    // Aspect ratio limits for traffic lights (taller than wide)
    double light_min_aspect = 0.3;
    double light_max_aspect = 3.5;

    // Minimum circularity for traffic light blobs (raised to reduce false positives)
    double light_min_circularity = 0.5;

    // Distance estimation: focal_length * real_size / pixel_size
    // QCar2 camera: ~640x480, ~60 deg FOV
    double focal_length_px = 554.0;   // Approximate
    double stop_sign_real_width = 0.08; // ~8cm in QLabs (1:10 scale)
    double light_real_width = 0.05;
    double cone_real_width = 0.05;

    // Timing (seconds)
    double stop_sign_pause = 3.0;
    double stop_sign_cooldown = 5.0;
    double traffic_light_cooldown = 6.0;
    double cross_waiting_timeout = 8.0;   // Max wait at red light (short to prevent deadlock)

    // Detection persistence
    int detection_threshold = 2;     // Consecutive frames to trigger
    int clear_threshold = 5;         // Consecutive clear frames to resume

    // Camera ROI: ignore bottom fraction (vehicle hood)
    double mask_bottom_fraction = 0.18;

    // Stop distance: only react to signs within this range (meters)
    double sign_stop_distance = 1.0;
    double cone_stop_distance = 0.7;

    // Spatial cooldown for stop signs (pixel Y difference for "new" sign)
    int stop_sign_bbox_threshold = 50;

    // Stop-sign geometric gating to avoid side-of-road false triggers in turns.
    // Require the sign to be reasonably centered in the image and not too high.
    double stop_sign_center_tolerance = 0.35;  // normalized x offset from center
    double stop_sign_min_y_fraction = 0.30;    // bbox center y / image height
};

// ============================================================================
// Detection result
// ============================================================================
enum class SignType {
    NONE,
    STOP_SIGN,
    YIELD_SIGN,
    TRAFFIC_LIGHT_RED,
    TRAFFIC_LIGHT_GREEN,
    TRAFFIC_LIGHT_YELLOW,
    TRAFFIC_CONE
};

struct Detection {
    SignType type = SignType::NONE;
    cv::Rect bbox;
    double confidence = 0.0;
    double distance = 0.0;
};

// Tracked object: once classified with ≥90% confidence, class is locked
struct TrackedObject {
    SignType locked_type = SignType::NONE;
    double best_confidence = 0.0;
    int bbox_center_x = 0;
    int bbox_center_y = 0;
    double last_seen_time = 0.0;
    bool is_locked = false;  // True once confidence ≥ lock threshold

    static constexpr double CONFIDENCE_LOCK_THRESHOLD = 0.90;
    static constexpr double EXPIRY_S = 5.0;  // Forget after 5s unseen
    static constexpr int SPATIAL_MATCH_PX = 60;  // Pixel distance to consider same object
};

// ============================================================================
// Sign Detector Node
// ============================================================================
class SignDetectorNode : public rclcpp::Node {
public:
    SignDetectorNode() : Node("sign_detector") {
        // Publishers
        traffic_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/traffic_control_state", 10);
        motion_pub_ = this->create_publisher<std_msgs::msg::Bool>(
            "/motion_enable", 10);
        obstacle_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/obstacle_positions", 10);

        // Subscriber
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/color_image", rclcpp::SensorDataQoS(),
            std::bind(&SignDetectorNode::image_callback, this,
                      std::placeholders::_1));

        // TF listener for computing obstacle positions in map frame
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Periodic publisher for traffic state (even when no detections)
        publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&SignDetectorNode::publish_state, this));

        node_start_time_ = this->now().seconds();

        RCLCPP_INFO(this->get_logger(),
            "Sign detector started (HSV + contour) [v3 - anti-deadlock + startup gate]");
    }

private:
    // ---- Image callback ----
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            if (msg->encoding == "rgb8") {
                cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            } else {
                cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            }
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "cv_bridge error: %s", e.what());
            return;
        }

        cv::Mat& frame = cv_ptr->image;
        if (frame.empty()) return;

        // Store image dimensions for bearing computation
        image_width_ = frame.cols;
        image_height_ = frame.rows;

        // Mask out bottom of image (vehicle hood)
        int mask_rows = static_cast<int>(frame.rows * cfg_.mask_bottom_fraction);
        if (mask_rows > 0) {
            frame(cv::Rect(0, frame.rows - mask_rows, frame.cols, mask_rows))
                .setTo(cv::Scalar(0, 0, 0));
        }

        // Convert to HSV
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

        // Detect all sign types
        std::vector<Detection> detections;
        detect_red_objects(hsv, frame, detections);
        detect_green_lights(hsv, frame, detections);
        detect_orange_cones(hsv, frame, detections);
        detect_yield_signs(hsv, frame, detections);

        // Apply class persistence: lock high-confidence detections
        std::lock_guard<std::mutex> lock(state_mutex_);
        apply_class_persistence(detections);

        // Process detections through state machine
        process_detections(detections);
    }

    // ---- Detection class persistence ----
    // Once an object reaches ≥90% confidence as one class, its class is locked
    // and subsequent detections at that spatial location cannot reclassify it.
    void apply_class_persistence(std::vector<Detection>& detections) {
        double now = this->now().seconds();

        // Expire old tracked objects
        tracked_objects_.erase(
            std::remove_if(tracked_objects_.begin(), tracked_objects_.end(),
                [now](const TrackedObject& t) {
                    return (now - t.last_seen_time) > TrackedObject::EXPIRY_S;
                }),
            tracked_objects_.end());

        for (auto& det : detections) {
            if (det.type == SignType::NONE) continue;

            int cx = det.bbox.x + det.bbox.width / 2;
            int cy = det.bbox.y + det.bbox.height / 2;

            // Find matching tracked object
            TrackedObject* match = nullptr;
            for (auto& tracked : tracked_objects_) {
                int dx = cx - tracked.bbox_center_x;
                int dy = cy - tracked.bbox_center_y;
                if (dx * dx + dy * dy <
                    TrackedObject::SPATIAL_MATCH_PX * TrackedObject::SPATIAL_MATCH_PX) {
                    match = &tracked;
                    break;
                }
            }

            if (match) {
                match->last_seen_time = now;
                // Update position (objects move in frame as vehicle approaches)
                match->bbox_center_x = cx;
                match->bbox_center_y = cy;

                if (match->is_locked) {
                    // Class is locked — override detection to locked type
                    if (det.type != match->locked_type) {
                        det.type = match->locked_type;
                    }
                } else {
                    // Not yet locked — update if this detection has higher confidence
                    if (det.confidence > match->best_confidence) {
                        match->best_confidence = det.confidence;
                        match->locked_type = det.type;
                    }
                    // Check if we should lock
                    if (match->best_confidence >= TrackedObject::CONFIDENCE_LOCK_THRESHOLD) {
                        match->is_locked = true;
                        det.type = match->locked_type;
                    }
                }
            } else {
                // New object — start tracking
                TrackedObject t;
                t.locked_type = det.type;
                t.best_confidence = det.confidence;
                t.bbox_center_x = cx;
                t.bbox_center_y = cy;
                t.last_seen_time = now;
                t.is_locked = det.confidence >= TrackedObject::CONFIDENCE_LOCK_THRESHOLD;
                tracked_objects_.push_back(t);
            }
        }
    }

    // ---- Red object detection (stop signs + red lights) ----
    void detect_red_objects(const cv::Mat& hsv, const cv::Mat& frame,
                           std::vector<Detection>& detections) {
        // Red wraps around H=0/180, so combine two ranges
        cv::Mat mask1, mask2, red_mask;
        cv::inRange(hsv, cfg_.red_lower1, cfg_.red_upper1, mask1);
        cv::inRange(hsv, cfg_.red_lower2, cfg_.red_upper2, mask2);
        red_mask = mask1 | mask2;

        // Morphological cleanup
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(red_mask, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area < cfg_.min_contour_area || area > cfg_.max_contour_area)
                continue;

            cv::Rect bbox = cv::boundingRect(cnt);

            // Reject if dominant color in bounding box region is blue (roundabout sign).
            // Blue roundabout signs have white arrows whose antialiased edges can
            // leak into the red HSV range. Check the original HSV for blue content.
            {
                cv::Mat roi_hsv = hsv(bbox);
                cv::Mat blue_in_roi;
                cv::inRange(roi_hsv, cv::Scalar(90, 80, 80), cv::Scalar(140, 255, 255), blue_in_roi);
                double blue_ratio = cv::countNonZero(blue_in_roi) /
                                    static_cast<double>(bbox.area());
                if (blue_ratio > 0.15) continue;  // Skip — likely a blue roundabout sign
            }

            // Approximate polygon
            std::vector<cv::Point> approx;
            double peri = cv::arcLength(cnt, true);
            cv::approxPolyDP(cnt, approx, 0.03 * peri, true);
            int vertices = static_cast<int>(approx.size());

            // Classify shape
            Detection det;
            det.bbox = bbox;

            // Check circularity (4 * pi * area / perimeter^2)
            double circularity = (4.0 * M_PI * area) / (peri * peri);

            // Aspect ratio of bounding box
            double aspect = static_cast<double>(bbox.width) / std::max(1, bbox.height);

            if (vertices >= cfg_.stop_sign_min_vertices &&
                vertices <= cfg_.stop_sign_max_vertices &&
                circularity > 0.5 && aspect > 0.7 && aspect < 1.4) {
                // Reject if significant yellow content — yield sign red border, not a stop sign
                cv::Mat roi_hsv = hsv(bbox);
                cv::Mat yellow_in_roi;
                cv::inRange(roi_hsv, cfg_.yellow_lower, cfg_.yellow_upper, yellow_in_roi);
                double yellow_ratio = cv::countNonZero(yellow_in_roi) /
                                      static_cast<double>(bbox.area());
                if (yellow_ratio > 0.15) continue;
                // Octagonal/polygonal, roughly circular and square -> stop sign
                det.type = SignType::STOP_SIGN;
                det.distance = estimate_distance(bbox.width, cfg_.stop_sign_real_width);
                det.confidence = std::min(1.0, area / 2000.0);
            } else if (circularity > cfg_.light_min_circularity &&
                       bbox.width < frame.cols * 0.10 &&
                       vertices <= 8) {
                // Small, circular red blob with few vertices -> red traffic light
                // Exclude shapes with many vertices (likely stop signs at distance)
                // Exclude large blobs (likely signs, not light bulbs)
                // Check if it's in the upper portion of the frame
                double y_ratio = static_cast<double>(bbox.y) / frame.rows;
                if (y_ratio < 0.6) {
                    det.type = SignType::TRAFFIC_LIGHT_RED;
                    det.distance = estimate_distance(bbox.width, cfg_.light_real_width);
                    det.confidence = std::min(1.0, circularity);
                }
            }

            if (det.type != SignType::NONE) {
                detections.push_back(det);
            }
        }
    }

    // ---- Green traffic light detection ----
    void detect_green_lights(const cv::Mat& hsv, const cv::Mat& frame,
                            std::vector<Detection>& detections) {
        cv::Mat green_mask;
        cv::inRange(hsv, cfg_.green_lower, cfg_.green_upper, green_mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(green_mask, green_mask, cv::MORPH_OPEN, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(green_mask, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area < cfg_.min_contour_area * 0.5 || area > cfg_.max_contour_area)
                continue;

            cv::Rect bbox = cv::boundingRect(cnt);
            double peri = cv::arcLength(cnt, true);
            double circularity = (4.0 * M_PI * area) / (peri * peri);

            // Green circular blob in upper part of frame -> green light
            double y_ratio = static_cast<double>(bbox.y) / frame.rows;
            if (circularity > cfg_.light_min_circularity &&
                y_ratio < 0.6 &&
                bbox.width < frame.cols * 0.10) {
                Detection det;
                det.type = SignType::TRAFFIC_LIGHT_GREEN;
                det.bbox = bbox;
                det.distance = estimate_distance(bbox.width, cfg_.light_real_width);
                det.confidence = std::min(1.0, circularity);
                detections.push_back(det);
            }
        }
    }

    // ---- Orange cone detection ----
    void detect_orange_cones(const cv::Mat& hsv, const cv::Mat& /*frame*/,
                            std::vector<Detection>& detections) {
        cv::Mat orange_mask;
        cv::inRange(hsv, cfg_.orange_lower, cfg_.orange_upper, orange_mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(orange_mask, orange_mask, cv::MORPH_OPEN, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(orange_mask, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area < cfg_.min_contour_area * 0.5 || area > cfg_.max_contour_area)
                continue;

            cv::Rect bbox = cv::boundingRect(cnt);

            // Cones are typically taller than wide and in lower half of image
            double aspect = static_cast<double>(bbox.height) / std::max(bbox.width, 1);
            if (aspect > 0.8 && aspect < 4.0) {
                Detection det;
                det.type = SignType::TRAFFIC_CONE;
                det.bbox = bbox;
                det.distance = estimate_distance(bbox.width, cfg_.cone_real_width);
                det.confidence = std::min(1.0, area / 1500.0);
                detections.push_back(det);
            }
        }
    }

    // ---- Yellow yield sign detection ----
    void detect_yield_signs(const cv::Mat& hsv, const cv::Mat& /*frame*/,
                            std::vector<Detection>& detections) {
        cv::Mat yellow_mask;
        cv::inRange(hsv, cfg_.yellow_lower, cfg_.yellow_upper, yellow_mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(yellow_mask, yellow_mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(yellow_mask, yellow_mask, cv::MORPH_CLOSE, kernel);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(yellow_mask, contours, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_SIMPLE);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area < cfg_.min_contour_area || area > cfg_.max_contour_area)
                continue;

            cv::Rect bbox = cv::boundingRect(cnt);

            // Approximate polygon — yield signs are triangular
            std::vector<cv::Point> approx;
            double peri = cv::arcLength(cnt, true);
            cv::approxPolyDP(cnt, approx, 0.04 * peri, true);
            int vertices = static_cast<int>(approx.size());

            if (vertices >= cfg_.yield_min_vertices &&
                vertices <= cfg_.yield_max_vertices) {
                // Triangular yellow shape -> yield sign
                // Check aspect ratio (yield signs are roughly equilateral)
                double aspect = static_cast<double>(bbox.width) / std::max(bbox.height, 1);
                if (aspect > 0.5 && aspect < 2.0) {
                    Detection det;
                    det.type = SignType::YIELD_SIGN;
                    det.bbox = bbox;
                    det.distance = estimate_distance(bbox.width, cfg_.stop_sign_real_width);
                    det.confidence = std::min(1.0, area / 1500.0);
                    detections.push_back(det);
                }
            }
        }
    }

    // ---- Distance estimation from bounding box width ----
    double estimate_distance(int bbox_width, double real_width) {
        if (bbox_width <= 0) return 10.0;
        return (cfg_.focal_length_px * real_width) / bbox_width;
    }

    // ---- State machine for traffic control ----
    void process_detections(const std::vector<Detection>& detections) {
        double now = this->now().seconds();

        // Post-action suppression: after completing a stop/light action,
        // ignore all detections briefly so the vehicle can drive away from
        // the sign. This prevents the deadlock where the stopped vehicle
        // keeps seeing the same sign and never resumes.
        if (now < post_action_suppress_until_) {
            should_stop_ = false;
            motion_enabled_ = true;
            control_type_ = "none";
            light_state_ = "unknown";
            stop_distance_ = 0.0;
            stop_duration_ = 0.0;
            clear_frames_ = cfg_.clear_threshold;
            return;
        }

        // Find the most relevant detection of each type
        Detection best_stop, best_red, best_green, best_cone, best_yield;
        best_stop.distance = 999.0;
        best_red.distance = 999.0;
        best_green.distance = 999.0;
        best_cone.distance = 999.0;
        best_yield.distance = 999.0;

        for (const auto& det : detections) {
            switch (det.type) {
                case SignType::STOP_SIGN:
                    if (det.distance < best_stop.distance &&
                        det.distance < cfg_.sign_stop_distance) {
                        best_stop = det;
                    }
                    break;
                case SignType::YIELD_SIGN:
                    if (det.distance < best_yield.distance &&
                        det.distance < cfg_.sign_stop_distance) {
                        best_yield = det;
                    }
                    break;
                case SignType::TRAFFIC_LIGHT_RED:
                    // Only consider red lights that are NOT also near a stop sign
                    // (red octagonal stop signs can be misclassified as red lights)
                    if (det.distance < best_red.distance &&
                        det.distance < cfg_.sign_stop_distance &&
                        !(stop_sign_cooldown_start_ > 0 &&
                          (now - stop_sign_cooldown_start_) < cfg_.stop_sign_cooldown)) {
                        best_red = det;
                    }
                    break;
                case SignType::TRAFFIC_LIGHT_GREEN:
                    if (det.distance < best_green.distance &&
                        det.distance < cfg_.sign_stop_distance) {
                        best_green = det;
                    }
                    break;
                case SignType::TRAFFIC_CONE:
                    if (det.distance < best_cone.distance &&
                        det.distance < cfg_.cone_stop_distance) {
                        best_cone = det;
                    }
                    break;
                default:
                    break;
            }
        }

        // Track detection persistence
        bool any_stop_trigger = false;

        // --- Stop sign logic ---
        if (best_stop.type == SignType::STOP_SIGN) {
            // Check if this is a different stop sign (spatial cooldown)
            int bbox_center_x = best_stop.bbox.x + best_stop.bbox.width / 2;
            int bbox_center_y = best_stop.bbox.y + best_stop.bbox.height / 2;
            double half_w = std::max(1.0, image_width_ * 0.5);
            double x_offset_norm = std::abs(bbox_center_x - image_width_ * 0.5) / half_w;
            double y_frac = static_cast<double>(bbox_center_y) / std::max(1, image_height_);
            bool stop_geom_ok =
                (x_offset_norm <= cfg_.stop_sign_center_tolerance) &&
                (y_frac >= cfg_.stop_sign_min_y_fraction);
            bool is_new_sign = !last_stop_bbox_valid_ ||
                std::abs(bbox_center_y - last_stop_bbox_y_) > cfg_.stop_sign_bbox_threshold;

            // Only detect if not on cooldown
            bool on_cooldown = stop_sign_cooldown_start_ > 0 &&
                (now - stop_sign_cooldown_start_) < cfg_.stop_sign_cooldown;

            if ((is_new_sign || !on_cooldown) && !on_cooldown && stop_geom_ok) {
                stop_detect_frames_++;
                if (stop_detect_frames_ >= cfg_.detection_threshold && !at_stop_sign_) {
                    at_stop_sign_ = true;
                    stop_start_time_ = now;
                    stop_wait_complete_ = false;
                    last_stop_bbox_y_ = bbox_center_y;
                    last_stop_bbox_valid_ = true;
                    RCLCPP_INFO(this->get_logger(), "STOP SIGN detected at %.2fm",
                                best_stop.distance);
                }
            } else if (!stop_geom_ok) {
                stop_detect_frames_ = 0;
            }
        } else {
            stop_detect_frames_ = 0;
        }

        // Stop sign state machine
        if (at_stop_sign_) {
            double elapsed = now - stop_start_time_;
            if (elapsed >= cfg_.stop_sign_pause) {
                stop_wait_complete_ = true;
                at_stop_sign_ = false;
                stop_sign_cooldown_start_ = now;
                // Force immediate resume and suppress detections
                should_stop_ = false;
                motion_enabled_ = true;
                post_action_suppress_until_ = now + POST_ACTION_SUPPRESS_S;
                RCLCPP_INFO(this->get_logger(),
                    "Stop sign wait complete, resuming (suppressing for %.1fs)",
                    POST_ACTION_SUPPRESS_S);
                return;  // Skip rest of detection logic this frame
            } else {
                any_stop_trigger = true;
                should_stop_ = true;
                control_type_ = "stop_sign";
                light_state_ = "unknown";
                stop_distance_ = best_stop.distance;
                stop_duration_ = cfg_.stop_sign_pause - elapsed;
            }
        }

        // --- Yield sign logic ---
        // Reference code (PolyCtrl 2025) ignores yield signs entirely — the YOLO
        // model detects them but MPC_node.py never acts on them. The competition
        // "failure to yield" infraction only applies when conflicting traffic is
        // present, and the scenario has no conflicting vehicle traffic at yield signs.
        // The yield sign near the hub at QLabs (0.0, -1.3) was causing an unnecessary
        // 2-second stop at startup before the vehicle even began driving.
        //
        // We log yield detections for awareness but do NOT stop.
        if (best_yield.type == SignType::YIELD_SIGN && !any_stop_trigger) {
            yield_detect_frames_++;
            if (yield_detect_frames_ >= cfg_.detection_threshold) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Yield sign seen at %.2fm (not stopping — reference behavior)",
                    best_yield.distance);
                yield_detect_frames_ = 0;  // Reset to prevent log spam
            }
        } else {
            yield_detect_frames_ = 0;
        }

        // --- Traffic light logic ---
        // Startup gate: ignore red lights for the first N seconds (reference: self.jaman)
        // The vehicle starts at hub and needs to drive before encountering real traffic lights
        double time_since_start = now - node_start_time_;
        if (best_red.type == SignType::TRAFFIC_LIGHT_RED && !any_stop_trigger &&
            time_since_start <= RED_LIGHT_STARTUP_DELAY_S) {
            // Startup gate: ignore red light during startup period
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Ignoring red light during startup gate (%.0f/%.0fs)",
                time_since_start, RED_LIGHT_STARTUP_DELAY_S);
        } else if (best_red.type == SignType::TRAFFIC_LIGHT_RED && !any_stop_trigger &&
            time_since_start > RED_LIGHT_STARTUP_DELAY_S) {
            red_detect_frames_++;
            green_detect_frames_ = 0;

            if (red_detect_frames_ >= cfg_.detection_threshold) {
                if (!cross_waiting_) {
                    cross_waiting_ = true;
                    cross_waiting_start_ = now;
                    cross_no_red_frames_ = 0;
                    RCLCPP_INFO(this->get_logger(), "RED LIGHT detected at %.2fm (%.0fs after startup)",
                                best_red.distance, time_since_start);
                }
                // Timeout even while red is still visible — prevents permanent deadlock
                // when red light stays in view (e.g., approaching intersection on hub_to_pickup)
                if (cross_waiting_ &&
                    (now - cross_waiting_start_) > cfg_.cross_waiting_timeout) {
                    cross_waiting_ = false;
                    should_stop_ = false;
                    motion_enabled_ = true;
                    // Use longer suppression so vehicle drives well past the light
                    post_action_suppress_until_ = now + TRAFFIC_LIGHT_SUPPRESS_S;
                    cross_cooldown_ = true;
                    cross_cooldown_start_ = now;
                    RCLCPP_WARN(this->get_logger(),
                        "Red light timeout (%.0fs, still visible), "
                        "force-resuming (suppressing for %.1fs)",
                        cfg_.cross_waiting_timeout, TRAFFIC_LIGHT_SUPPRESS_S);
                    return;
                }
            }
        } else if (best_green.type == SignType::TRAFFIC_LIGHT_GREEN && !any_stop_trigger) {
            green_detect_frames_++;
            red_detect_frames_ = 0;

            if (green_detect_frames_ >= cfg_.detection_threshold && cross_waiting_) {
                cross_waiting_ = false;
                cross_cooldown_ = true;
                cross_cooldown_start_ = now;
                // Force immediate resume and suppress
                should_stop_ = false;
                motion_enabled_ = true;
                post_action_suppress_until_ = now + TRAFFIC_LIGHT_SUPPRESS_S;
                RCLCPP_INFO(this->get_logger(),
                    "GREEN LIGHT detected, resuming (suppressing for %.1fs)",
                    TRAFFIC_LIGHT_SUPPRESS_S);
                return;
            }
        } else if (!any_stop_trigger) {
            // No traffic light detected
            if (cross_waiting_) {
                cross_no_red_frames_++;
                // Auto-expire if no red detected for ~2s (60 frames at 30Hz)
                if (cross_no_red_frames_ > 60) {
                    cross_waiting_ = false;
                    should_stop_ = false;
                    motion_enabled_ = true;
                    post_action_suppress_until_ = now + TRAFFIC_LIGHT_SUPPRESS_S;
                    cross_cooldown_ = true;
                    cross_cooldown_start_ = now;
                    RCLCPP_INFO(this->get_logger(),
                        "Red light lost for 2s, resuming (suppressing for %.1fs)",
                        TRAFFIC_LIGHT_SUPPRESS_S);
                    return;
                }
                // Timeout safety
                if ((now - cross_waiting_start_) > cfg_.cross_waiting_timeout) {
                    cross_waiting_ = false;
                    should_stop_ = false;
                    motion_enabled_ = true;
                    post_action_suppress_until_ = now + TRAFFIC_LIGHT_SUPPRESS_S;
                    cross_cooldown_ = true;
                    cross_cooldown_start_ = now;
                    RCLCPP_WARN(this->get_logger(),
                        "Red light timeout (%.0fs, lost), resuming (suppressing for %.1fs)",
                        cfg_.cross_waiting_timeout, TRAFFIC_LIGHT_SUPPRESS_S);
                    return;
                }
            }
            red_detect_frames_ = 0;
            green_detect_frames_ = 0;
        }

        // Traffic light state output
        if (cross_waiting_ && !any_stop_trigger) {
            any_stop_trigger = true;
            should_stop_ = true;
            control_type_ = "traffic_light";
            light_state_ = "red";
            stop_distance_ = best_red.distance;
            stop_duration_ = 0.0;  // Wait for green
        }

        if (cross_cooldown_) {
            if ((now - cross_cooldown_start_) > cfg_.traffic_light_cooldown) {
                cross_cooldown_ = false;
            }
        }

        // --- Cone logic ---
        bool cone_blocking = false;
        if (best_cone.type == SignType::TRAFFIC_CONE &&
            best_cone.distance < cfg_.cone_stop_distance) {
            cone_detect_frames_++;
            if (cone_detect_frames_ >= cfg_.detection_threshold) {
                cone_blocking = true;
            }
            // Store cone info for obstacle position publishing
            cone_detected_ = true;
            cone_distance_ = best_cone.distance;
            cone_bbox_center_x_ = best_cone.bbox.x + best_cone.bbox.width / 2.0;
            cone_image_width_ = image_width_;
        } else {
            cone_detect_frames_ = 0;
            cone_detected_ = false;
        }

        // --- Final output ---
        if (!any_stop_trigger && !cone_blocking) {
            clear_frames_++;
            if (clear_frames_ >= cfg_.clear_threshold) {
                should_stop_ = false;
                control_type_ = "none";
                light_state_ = "unknown";
                stop_distance_ = 0.0;
                stop_duration_ = 0.0;
            }
        } else {
            clear_frames_ = 0;
            if (cone_blocking && !any_stop_trigger) {
                should_stop_ = true;
                control_type_ = "cone";
                stop_distance_ = best_cone.distance;
            }
        }

        // Motion enable: false when we should stop
        motion_enabled_ = !should_stop_;
    }

    // ---- Periodic state publisher ----
    void publish_state() {
        std::lock_guard<std::mutex> lock(state_mutex_);

        // Publish traffic control state (JSON)
        auto tcs_msg = std_msgs::msg::String();
        std::ostringstream json;
        json << "{\"control_type\": \"" << control_type_ << "\", "
             << "\"light_state\": \"" << light_state_ << "\", "
             << "\"distance\": " << stop_distance_ << ", "
             << "\"should_stop\": " << (should_stop_ ? "true" : "false") << ", "
             << "\"stop_duration\": " << stop_duration_ << ", "
             << "\"stop_line_x\": 0.0, "
             << "\"stop_line_y\": 0.0}";
        tcs_msg.data = json.str();
        traffic_pub_->publish(tcs_msg);

        // Publish motion enable
        auto motion_msg = std_msgs::msg::Bool();
        motion_msg.data = motion_enabled_;
        motion_pub_->publish(motion_msg);

        // Publish obstacle positions in map frame (for MPCC solver avoidance)
        publish_obstacle_positions();
    }

    /**
     * Compute cone/pedestrian position in map frame and publish to /obstacle_positions.
     * Uses TF (map → base_link) + camera bearing angle + estimated distance.
     */
    void publish_obstacle_positions() {
        if (!cone_detected_ || cone_distance_ <= 0.01) {
            // No cone detected — publish empty obstacles
            auto msg = std_msgs::msg::String();
            msg.data = "{\"obstacles\": []}";
            obstacle_pub_->publish(msg);
            return;
        }

        // Get vehicle pose from TF
        geometry_msgs::msg::TransformStamped tf;
        try {
            tf = tf_buffer_->lookupTransform("map", "base_link",
                                              tf2::TimePointZero,
                                              tf2::durationFromSec(0.05));
        } catch (const tf2::TransformException&) {
            return;  // No TF available — skip this cycle
        }

        double veh_x = tf.transform.translation.x;
        double veh_y = tf.transform.translation.y;
        double veh_theta = tf2::getYaw(tf.transform.rotation);

        // Bearing angle from camera center
        double image_center = cone_image_width_ / 2.0;
        double bearing = std::atan2(image_center - cone_bbox_center_x_,
                                     cfg_.focal_length_px);

        // Position in vehicle frame (forward = +x, left = +y)
        double fwd = cone_distance_ * std::cos(bearing);
        double left = cone_distance_ * std::sin(bearing);

        // Transform to map frame
        double obs_x = veh_x + fwd * std::cos(veh_theta) - left * std::sin(veh_theta);
        double obs_y = veh_y + fwd * std::sin(veh_theta) + left * std::cos(veh_theta);

        // Publish with obstacle radius (0.10m for cone/pedestrian in QLabs)
        // Include explicit frame metadata to avoid downstream double-transform.
        double radius = 0.10;
        auto msg = std_msgs::msg::String();
        std::ostringstream obs_json;
        obs_json << "{\"obstacles\": [{\"x\": " << obs_x
                 << ", \"y\": " << obs_y
                 << ", \"radius\": " << radius
                 << ", \"obj_class\": \"cone\""
                 << ", \"frame\": \"map\"}]}";
        msg.data = obs_json.str();
        obstacle_pub_->publish(msg);
    }

    // ---- Configuration ----
    DetectorConfig cfg_;

    // ---- ROS interfaces ----
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr traffic_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr motion_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr obstacle_pub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::TimerBase::SharedPtr publish_timer_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::mutex state_mutex_;

    // ---- State machine ----
    bool should_stop_ = false;
    bool motion_enabled_ = true;
    std::string control_type_ = "none";
    std::string light_state_ = "unknown";
    double stop_distance_ = 0.0;
    double stop_duration_ = 0.0;

    // Stop sign state
    bool at_stop_sign_ = false;
    double stop_start_time_ = 0.0;
    bool stop_wait_complete_ = false;
    double stop_sign_cooldown_start_ = 0.0;
    int stop_detect_frames_ = 0;
    int last_stop_bbox_y_ = 0;
    bool last_stop_bbox_valid_ = false;

    // Traffic light state
    bool cross_waiting_ = false;
    double cross_waiting_start_ = 0.0;
    int cross_no_red_frames_ = 0;
    bool cross_cooldown_ = false;
    double cross_cooldown_start_ = 0.0;
    int red_detect_frames_ = 0;
    int green_detect_frames_ = 0;

    // Yield sign state
    // Yield sign state removed — yield signs are detected but not acted on
    // (matches reference code behavior; see yield logic comment in process_detections)
    int yield_detect_frames_ = 0;

    // Detection class persistence: tracked objects with confidence locking
    std::vector<TrackedObject> tracked_objects_;

    // Cone state
    int cone_detect_frames_ = 0;
    bool cone_detected_ = false;         // Cone visible in current frame
    double cone_distance_ = 0.0;         // Estimated distance (m)
    double cone_bbox_center_x_ = 0.0;    // Bbox center x in image pixels
    int cone_image_width_ = 640;         // Image width for bearing computation
    int image_width_ = 640;              // Stored from last frame
    int image_height_ = 480;             // Stored from last frame

    // Clear hysteresis
    int clear_frames_ = 0;

    // Post-action suppression: after completing a stop or light wait,
    // suppress all detections briefly so the vehicle can drive away
    double post_action_suppress_until_ = 0.0;
    static constexpr double POST_ACTION_SUPPRESS_S = 3.0;
    static constexpr double TRAFFIC_LIGHT_SUPPRESS_S = 12.0;  // Suppression after traffic light action (must be > controller's 10s cooldown)

    // Startup gate: ignore red light detections for the first N seconds.
    // Matches reference code's approach (self.jaman) — the vehicle starts at hub
    // and needs to drive to the first intersection before red lights are relevant.
    // Also prevents false red-light detections on distant objects during startup.
    static constexpr double RED_LIGHT_STARTUP_DELAY_S = 15.0;
    double node_start_time_ = 0.0;
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SignDetectorNode>());
    rclcpp::shutdown();
    return 0;
}
