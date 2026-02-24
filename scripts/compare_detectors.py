#!/usr/bin/env python3
"""
Obstacle Detection Comparison for SDCS QCar2.

Benchmarks 5 detection approaches on synthetic QLabs-like test images:
  1. HSV + Contour (ported from C++ sign_detector_node)
  2. YOLOv8n COCO
  3. Hybrid HSV + YOLO (HSV pre-filter, YOLO verification)
  4. Hough Circles + HSV (HoughCircles for lights, HSV for signs/cones)
  5. Custom YOLOv8n (QLabs-trained best.pt, 8 classes)

Generates metrics (precision, recall, F1, latency, distance error) and
6 comparison figures in scripts/detection_results/.

No ROS 2 dependency - runs standalone.

Usage:
    python3 scripts/compare_detectors.py
"""

import sys
import os
import time
import json
import math
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = Path(__file__).parent / 'detection_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# Camera parameters (from sign_detector_node.cpp)
FOCAL_LENGTH_PX = 554.0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_HFOV_RAD = math.radians(60.0)

# Real object widths in meters (from sign_detector_node.cpp)
REAL_WIDTHS = {
    'stop_sign': 0.08,
    'traffic_light_red': 0.05,
    'traffic_light_green': 0.05,
    'cone': 0.05,
    'pedestrian': 0.18,
}

# IoU threshold for TP/FP determination
IOU_THRESHOLD = 0.3

# ============================================================================
# HSV Color Ranges (from sign_detector_node.cpp + obstacle_detector.py)
# ============================================================================
HSV_RANGES = {
    'red1': {'lower': np.array([0, 100, 80]), 'upper': np.array([10, 255, 255])},
    'red2': {'lower': np.array([170, 100, 80]), 'upper': np.array([180, 255, 255])},
    'green': {'lower': np.array([40, 80, 80]), 'upper': np.array([85, 255, 255])},
    'orange': {'lower': np.array([5, 150, 150]), 'upper': np.array([18, 255, 255])},
    'yellow': {'lower': np.array([18, 100, 100]), 'upper': np.array([35, 255, 255])},
}


# ============================================================================
# Synthetic Image Generator
# ============================================================================

def _pixel_size_from_distance(real_width: float, distance: float) -> int:
    """Object pixel width given real width and distance."""
    if distance <= 0:
        return 0
    return max(1, int(FOCAL_LENGTH_PX * real_width / distance))


def _road_background(width=IMAGE_WIDTH, height=IMAGE_HEIGHT) -> np.ndarray:
    """Create a basic road-like background (gray asphalt + blue sky)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = height // 3
    # Sky (light blue in BGR)
    img[:horizon, :] = (210, 180, 140)
    # Road (gray asphalt)
    img[horizon:, :] = (80, 80, 80)
    # Lane markings
    lane_y = horizon + (height - horizon) // 2
    for x in range(0, width, 40):
        cv2.line(img, (x, lane_y), (x + 20, lane_y), (200, 200, 200), 2)
    return img


def _draw_stop_sign(img: np.ndarray, cx: int, cy: int, size: int):
    """Draw a red octagon stop sign."""
    pts = []
    for i in range(8):
        angle = math.pi / 8 + i * math.pi / 4
        px = int(cx + size * math.cos(angle))
        py = int(cy + size * math.sin(angle))
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)
    # Fill with red in BGR
    cv2.fillConvexPoly(img, pts, (0, 0, 220))
    # White border
    cv2.polylines(img, [pts], True, (255, 255, 255), max(1, size // 10))
    # "STOP" text
    font_scale = max(0.2, size / 50.0)
    thickness = max(1, size // 15)
    text_size = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    tx = cx - text_size[0] // 2
    ty = cy + text_size[1] // 2
    cv2.putText(img, "STOP", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness)


def _draw_traffic_light(img: np.ndarray, cx: int, cy: int, size: int,
                         color: str):
    """Draw a traffic light (colored circle on dark rectangle)."""
    # Housing
    rect_w = int(size * 1.2)
    rect_h = int(size * 2.5)
    x1 = cx - rect_w // 2
    y1 = cy - rect_h // 2
    cv2.rectangle(img, (x1, y1), (x1 + rect_w, y1 + rect_h), (30, 30, 30), -1)
    cv2.rectangle(img, (x1, y1), (x1 + rect_w, y1 + rect_h), (60, 60, 60), 2)
    # Light circle
    if color == 'red':
        bgr = (0, 0, 255)
        light_cy = cy - int(size * 0.6)
    else:  # green
        bgr = (0, 255, 0)
        light_cy = cy + int(size * 0.6)
    cv2.circle(img, (cx, light_cy), size // 2, bgr, -1)
    # Dim circles for other lights
    dim_color = (40, 40, 40)
    if color == 'red':
        cv2.circle(img, (cx, cy + int(size * 0.6)), size // 2, dim_color, -1)
    else:
        cv2.circle(img, (cx, cy - int(size * 0.6)), size // 2, dim_color, -1)


def _draw_cone(img: np.ndarray, cx: int, cy: int, size: int):
    """Draw an orange traffic cone (triangle/trapezoid)."""
    # Trapezoid shape
    top_w = max(2, size // 4)
    bot_w = size
    h = int(size * 1.5)
    pts = np.array([
        [cx - top_w // 2, cy - h // 2],
        [cx + top_w // 2, cy - h // 2],
        [cx + bot_w // 2, cy + h // 2],
        [cx - bot_w // 2, cy + h // 2],
    ], dtype=np.int32)
    # Orange in BGR
    cv2.fillConvexPoly(img, pts, (0, 140, 255))
    # White stripe
    stripe_y = cy - h // 6
    stripe_w = int(size * 0.4)
    cv2.rectangle(img, (cx - stripe_w // 2, stripe_y - 2),
                  (cx + stripe_w // 2, stripe_y + 2), (255, 255, 255), -1)


def _draw_pedestrian(img: np.ndarray, cx: int, cy: int, size: int):
    """Draw a rectangular pedestrian silhouette."""
    w = size
    h = int(size * 2.5)
    x1 = cx - w // 2
    y1 = cy - h // 2
    # Body (dark blue/brown)
    cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (60, 40, 30), -1)
    # Head
    head_r = max(2, w // 3)
    cv2.circle(img, (cx, y1 - head_r), head_r, (130, 120, 110), -1)
    # Legs
    leg_w = max(1, w // 4)
    leg_h = h // 3
    cv2.rectangle(img, (cx - w // 3, y1 + h), (cx - w // 3 + leg_w, y1 + h + leg_h),
                  (50, 50, 80), -1)
    cv2.rectangle(img, (cx + w // 6, y1 + h), (cx + w // 6 + leg_w, y1 + h + leg_h),
                  (50, 50, 80), -1)


def _object_bbox(cx: int, cy: int, pixel_size: int, obj_class: str):
    """Get bounding box [x, y, w, h] for a drawn object."""
    if obj_class == 'stop_sign':
        s = pixel_size
        return [cx - s, cy - s, 2 * s, 2 * s]
    elif obj_class in ('traffic_light_red', 'traffic_light_green'):
        w = int(pixel_size * 1.2)
        h = int(pixel_size * 2.5)
        return [cx - w // 2, cy - h // 2, w, h]
    elif obj_class == 'cone':
        w = pixel_size
        h = int(pixel_size * 1.5)
        return [cx - w // 2, cy - h // 2, w, h]
    elif obj_class == 'pedestrian':
        w = pixel_size
        h = int(pixel_size * 2.5)
        head_r = max(2, w // 3)
        return [cx - w // 2, cy - h // 2 - head_r, w, h + head_r + h // 3]
    return [cx - pixel_size // 2, cy - pixel_size // 2, pixel_size, pixel_size]


def _place_object(img: np.ndarray, obj_class: str, distance: float,
                  x_frac: float = 0.5) -> dict:
    """Place an object on the image at given distance. Returns ground truth dict."""
    real_w = REAL_WIDTHS[obj_class]
    pixel_size = _pixel_size_from_distance(real_w, distance)
    if pixel_size < 3:
        pixel_size = 3

    cx = int(IMAGE_WIDTH * x_frac)
    # Vertical position: closer objects lower in frame
    horizon = IMAGE_HEIGHT // 3
    # Map distance [0.3, 3.0] -> y position [IMAGE_HEIGHT-50, horizon+20]
    t = min(1.0, max(0.0, (distance - 0.3) / 2.7))
    cy = int((IMAGE_HEIGHT - 50) * (1 - t) + (horizon + 20) * t)

    draw_funcs = {
        'stop_sign': _draw_stop_sign,
        'traffic_light_red': lambda img, cx, cy, s: _draw_traffic_light(img, cx, cy, s, 'red'),
        'traffic_light_green': lambda img, cx, cy, s: _draw_traffic_light(img, cx, cy, s, 'green'),
        'cone': _draw_cone,
        'pedestrian': _draw_pedestrian,
    }

    if obj_class in draw_funcs:
        draw_funcs[obj_class](img, cx, cy, pixel_size)

    bbox = _object_bbox(cx, cy, pixel_size, obj_class)
    return {
        'class': obj_class,
        'bbox': bbox,
        'distance': distance,
        'center': (cx, cy),
        'pixel_size': pixel_size,
    }


def generate_test_suite() -> list:
    """Generate 24 synthetic test images with ground truth.

    Returns list of dicts: {name, image, ground_truth: [{class, bbox, distance}]}
    """
    test_images = []
    classes = ['stop_sign', 'traffic_light_red', 'traffic_light_green', 'cone', 'pedestrian']
    distances = [0.3, 0.5, 0.8, 1.5, 3.0]

    # Single object at various distances (5 classes x 3 distances = 15 images)
    # Use 3 representative distances per class
    for obj_class in classes:
        for dist in [0.3, 0.8, 3.0]:
            img = _road_background()
            gt = _place_object(img, obj_class, dist)
            test_images.append({
                'name': f'{obj_class}_d{dist:.1f}',
                'image': img,
                'ground_truth': [gt],
            })

    # Multi-object scenes (4 images)
    # Scene 1: stop sign + cone
    img = _road_background()
    gt1 = _place_object(img, 'stop_sign', 0.8, x_frac=0.7)
    gt2 = _place_object(img, 'cone', 0.5, x_frac=0.3)
    test_images.append({
        'name': 'multi_stop_cone',
        'image': img,
        'ground_truth': [gt1, gt2],
    })

    # Scene 2: red light + pedestrian
    img = _road_background()
    gt1 = _place_object(img, 'traffic_light_red', 1.5, x_frac=0.5)
    gt2 = _place_object(img, 'pedestrian', 0.5, x_frac=0.35)
    test_images.append({
        'name': 'multi_redlight_ped',
        'image': img,
        'ground_truth': [gt1, gt2],
    })

    # Scene 3: green light + cone
    img = _road_background()
    gt1 = _place_object(img, 'traffic_light_green', 1.5, x_frac=0.5)
    gt2 = _place_object(img, 'cone', 0.8, x_frac=0.65)
    test_images.append({
        'name': 'multi_greenlight_cone',
        'image': img,
        'ground_truth': [gt1, gt2],
    })

    # Scene 4: multiple cones
    img = _road_background()
    gt1 = _place_object(img, 'cone', 0.5, x_frac=0.3)
    gt2 = _place_object(img, 'cone', 0.8, x_frac=0.5)
    gt3 = _place_object(img, 'cone', 1.5, x_frac=0.7)
    test_images.append({
        'name': 'multi_3cones',
        'image': img,
        'ground_truth': [gt1, gt2, gt3],
    })

    # Edge cases (5 images)
    # Object at left frame boundary
    img = _road_background()
    gt = _place_object(img, 'stop_sign', 0.8, x_frac=0.05)
    test_images.append({
        'name': 'edge_left_boundary',
        'image': img,
        'ground_truth': [gt],
    })

    # Object at right frame boundary
    img = _road_background()
    gt = _place_object(img, 'cone', 0.5, x_frac=0.95)
    test_images.append({
        'name': 'edge_right_boundary',
        'image': img,
        'ground_truth': [gt],
    })

    # Empty road (false positive test)
    img = _road_background()
    test_images.append({
        'name': 'edge_empty_road',
        'image': img,
        'ground_truth': [],
    })

    # Very close pedestrian (large bbox)
    img = _road_background()
    gt = _place_object(img, 'pedestrian', 0.3, x_frac=0.4)
    test_images.append({
        'name': 'edge_close_pedestrian',
        'image': img,
        'ground_truth': [gt],
    })

    # Distant stop sign (tiny)
    img = _road_background()
    gt = _place_object(img, 'stop_sign', 3.0, x_frac=0.8)
    test_images.append({
        'name': 'edge_distant_stop',
        'image': img,
        'ground_truth': [gt],
    })

    assert len(test_images) == 24, f"Expected 24 test images, got {len(test_images)}"
    return test_images


# ============================================================================
# Detector Interface
# ============================================================================

class DetectorInterface(ABC):
    """Common interface for all detection approaches."""

    name: str = "base"
    requires: list = []  # Dependencies beyond OpenCV

    @abstractmethod
    def detect(self, image: np.ndarray) -> list:
        """Run detection on a BGR image.

        Returns list of dicts: {class, confidence, bbox: [x,y,w,h], distance}
        """
        ...

    def warmup(self, image: np.ndarray):
        """Optional warmup run (e.g., for YOLO model loading)."""
        self.detect(image)


# ============================================================================
# Detector 1: HSV + Contour (ported from sign_detector_node.cpp)
# ============================================================================

class HSVContourDetector(DetectorInterface):
    name = "HSV+Contour"
    requires = []

    # Class mapping for consistent output
    CLASS_MAP = {
        'stop_sign': 'stop_sign',
        'traffic_light_red': 'traffic_light_red',
        'traffic_light_green': 'traffic_light_green',
        'cone': 'cone',
        'pedestrian': 'pedestrian',
    }

    def detect(self, image: np.ndarray) -> list:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        detections = []
        detections.extend(self._detect_red_objects(hsv))
        detections.extend(self._detect_green_lights(hsv))
        detections.extend(self._detect_orange_cones(hsv))
        # HSV can't detect pedestrians - no distinct color signature
        return detections

    def _detect_red_objects(self, hsv: np.ndarray) -> list:
        """Detect stop signs and red traffic lights."""
        dets = []
        mask1 = cv2.inRange(hsv, HSV_RANGES['red1']['lower'], HSV_RANGES['red1']['upper'])
        mask2 = cv2.inRange(hsv, HSV_RANGES['red2']['lower'], HSV_RANGES['red2']['upper'])
        red_mask = mask1 | mask2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            # Classify by shape
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            sides = len(approx)

            # Check if in upper portion (traffic light)
            is_upper = (y + h / 2) < hsv.shape[0] * 0.5

            if is_upper and 4 <= sides <= 20:
                # Check circularity
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                if circularity > 0.3:
                    dist = self._estimate_distance(w, REAL_WIDTHS['traffic_light_red'])
                    confidence = min(1.0, area / 2000.0)
                    dets.append({
                        'class': 'traffic_light_red',
                        'confidence': confidence,
                        'bbox': [x, y, w, h],
                        'distance': dist,
                    })
                    continue

            # Stop sign: 7-9 sides (octagon)
            if 6 <= sides <= 10 and not is_upper:
                dist = self._estimate_distance(w, REAL_WIDTHS['stop_sign'])
                aspect = h / w if w > 0 else 0
                if 0.5 < aspect < 2.0:
                    confidence = min(1.0, area / 3000.0)
                    dets.append({
                        'class': 'stop_sign',
                        'confidence': confidence,
                        'bbox': [x, y, w, h],
                        'distance': dist,
                    })
            elif sides <= 5 and not is_upper:
                # Could be a red light in lower area; skip
                pass

        return dets

    def _detect_green_lights(self, hsv: np.ndarray) -> list:
        dets = []
        green_mask = cv2.inRange(hsv, HSV_RANGES['green']['lower'], HSV_RANGES['green']['upper'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:
                continue
            x, y, w, h = cv2.boundingRect(cnt)

            # Traffic lights are in upper portion of image
            if (y + h / 2) > hsv.shape[0] * 0.6:
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity > 0.3:
                dist = self._estimate_distance(w, REAL_WIDTHS['traffic_light_green'])
                confidence = min(1.0, area / 1500.0)
                dets.append({
                    'class': 'traffic_light_green',
                    'confidence': confidence,
                    'bbox': [x, y, w, h],
                    'distance': dist,
                })

        return dets

    def _detect_orange_cones(self, hsv: np.ndarray) -> list:
        dets = []
        orange_mask = cv2.inRange(hsv, HSV_RANGES['orange']['lower'], HSV_RANGES['orange']['upper'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            dist = self._estimate_distance(w, REAL_WIDTHS['cone'])
            confidence = min(1.0, area / 2000.0)
            dets.append({
                'class': 'cone',
                'confidence': confidence,
                'bbox': [x, y, w, h],
                'distance': dist,
            })

        return dets

    @staticmethod
    def _estimate_distance(bbox_width: int, real_width: float) -> float:
        if bbox_width <= 0:
            return 10.0
        return FOCAL_LENGTH_PX * real_width / bbox_width


# ============================================================================
# Detector 2: YOLOv8n COCO
# ============================================================================

class YOLOCocoDetector(DetectorInterface):
    name = "YOLOv8n COCO"
    requires = ["ultralytics"]

    # COCO class ID -> our class name mapping
    COCO_MAP = {
        0: 'pedestrian',       # person
        9: 'traffic_light_red',  # traffic light (needs color check)
        11: 'stop_sign',       # stop sign
        33: 'cone',            # sports ball (cone proxy)
    }

    def __init__(self):
        self._model = None
        self._available = False
        try:
            from ultralytics import YOLO
            self._model = YOLO('yolov8n.pt')
            self._available = True
        except (ImportError, Exception) as e:
            print(f"  [YOLOv8n COCO] Not available: {e}")

    def detect(self, image: np.ndarray) -> list:
        if not self._available:
            return []
        dets = []
        results = self._model(image, verbose=False, conf=0.25, device='cpu')
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                if cls_id not in self.COCO_MAP:
                    continue
                cls_name = self.COCO_MAP[cls_id]
                conf = float(result.boxes.conf[i].item())
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                w = int(x2 - x1)
                h = int(y2 - y1)

                # For traffic lights, classify color by HSV
                if cls_id == 9:
                    cls_name = self._classify_light_color(image, int(x1), int(y1), int(x2), int(y2))
                    if cls_name is None:
                        continue

                real_w = REAL_WIDTHS.get(cls_name, 0.1)
                dist = FOCAL_LENGTH_PX * real_w / max(w, 1)

                dets.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), w, h],
                    'distance': dist,
                })
        return dets

    @staticmethod
    def _classify_light_color(image, x1, y1, x2, y2) -> str:
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 3 or y2 - y1 < 3:
            return None
        crop = image[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask_r1 = cv2.inRange(hsv, np.array([0, 120, 100]), np.array([10, 255, 255]))
        mask_r2 = cv2.inRange(hsv, np.array([160, 120, 100]), np.array([180, 255, 255]))
        red_px = cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2)
        mask_g = cv2.inRange(hsv, np.array([40, 80, 80]), np.array([90, 255, 255]))
        green_px = cv2.countNonZero(mask_g)
        if red_px >= 3 and red_px > green_px:
            return 'traffic_light_red'
        elif green_px >= 3 and green_px > red_px:
            return 'traffic_light_green'
        return None


# ============================================================================
# Detector 3: Hybrid HSV + YOLO
# ============================================================================

class HybridHSVYoloDetector(DetectorInterface):
    name = "Hybrid HSV+YOLO"
    requires = ["ultralytics"]

    def __init__(self):
        self._hsv_det = HSVContourDetector()
        self._yolo_model = None
        self._available = False
        try:
            from ultralytics import YOLO
            self._yolo_model = YOLO('yolov8n.pt')
            self._available = True
        except (ImportError, Exception) as e:
            print(f"  [Hybrid HSV+YOLO] YOLO not available, falling back to HSV-only: {e}")

    def detect(self, image: np.ndarray) -> list:
        # Step 1: HSV pre-filter finds candidate regions
        hsv_dets = self._hsv_det.detect(image)

        if not self._available or not hsv_dets:
            return hsv_dets

        # Step 2: For each HSV candidate, crop and run YOLO to verify
        verified = []
        h_img, w_img = image.shape[:2]

        for det in hsv_dets:
            bx, by, bw, bh = det['bbox']
            # Expand crop by 50% for YOLO context
            margin_x = bw // 2
            margin_y = bh // 2
            cx1 = max(0, bx - margin_x)
            cy1 = max(0, by - margin_y)
            cx2 = min(w_img, bx + bw + margin_x)
            cy2 = min(h_img, by + bh + margin_y)

            crop = image[cy1:cy2, cx1:cx2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                # Too small to verify, trust HSV
                verified.append(det)
                continue

            # Resize crop to minimum YOLO size
            scale = max(1, 64 / min(crop.shape[:2]))
            if scale > 1:
                crop = cv2.resize(crop, None, fx=scale, fy=scale)

            results = self._yolo_model(crop, verbose=False, conf=0.15, device='cpu')
            yolo_found = False
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    yolo_found = True
                    # Boost confidence since both HSV and YOLO agree
                    det['confidence'] = min(1.0, det['confidence'] * 1.3)
                    break

            if yolo_found:
                verified.append(det)
            else:
                # HSV only - keep with lower confidence
                det['confidence'] *= 0.7
                verified.append(det)

        return verified


# ============================================================================
# Detector 4: Hough Circles + HSV
# ============================================================================

class HoughCircleHSVDetector(DetectorInterface):
    name = "Hough+HSV"
    requires = []

    def detect(self, image: np.ndarray) -> list:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        dets = []

        # Use HoughCircles for traffic lights
        dets.extend(self._detect_lights_hough(image, hsv))
        # Use HSV+contour for signs and cones (same as Detector 1)
        dets.extend(self._detect_signs_hsv(hsv))
        dets.extend(self._detect_cones_hsv(hsv))

        return dets

    def _detect_lights_hough(self, image: np.ndarray, hsv: np.ndarray) -> list:
        """Use HoughCircles to find circular lights, then classify color."""
        dets = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Focus on upper half for traffic lights
        roi_h = image.shape[0] // 2
        gray_roi = gray[:roi_h]

        gray_roi = cv2.GaussianBlur(gray_roi, (9, 9), 2)

        circles = cv2.HoughCircles(
            gray_roi, cv2.HOUGH_GRADIENT, dp=1.2,
            minDist=20, param1=100, param2=30,
            minRadius=3, maxRadius=50
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0]:
                cx, cy, r = int(c[0]), int(c[1]), int(c[2])
                # Classify color in circle region
                x1 = max(0, cx - r)
                y1 = max(0, cy - r)
                x2 = min(image.shape[1], cx + r)
                y2 = min(roi_h, cy + r)
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue

                hsv_crop = hsv[y1:y2, x1:x2]
                mask_r1 = cv2.inRange(hsv_crop, HSV_RANGES['red1']['lower'], HSV_RANGES['red1']['upper'])
                mask_r2 = cv2.inRange(hsv_crop, HSV_RANGES['red2']['lower'], HSV_RANGES['red2']['upper'])
                red_px = cv2.countNonZero(mask_r1) + cv2.countNonZero(mask_r2)

                mask_g = cv2.inRange(hsv_crop, HSV_RANGES['green']['lower'], HSV_RANGES['green']['upper'])
                green_px = cv2.countNonZero(mask_g)

                total_px = (x2 - x1) * (y2 - y1)
                cls_name = None
                if red_px > 5 and red_px > green_px and red_px > total_px * 0.1:
                    cls_name = 'traffic_light_red'
                elif green_px > 5 and green_px > red_px and green_px > total_px * 0.1:
                    cls_name = 'traffic_light_green'

                if cls_name:
                    w = 2 * r
                    dist = FOCAL_LENGTH_PX * REAL_WIDTHS.get(cls_name, 0.05) / max(w, 1)
                    confidence = min(1.0, r / 15.0)
                    dets.append({
                        'class': cls_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'distance': dist,
                    })

        return dets

    def _detect_signs_hsv(self, hsv: np.ndarray) -> list:
        """Detect stop signs using HSV (same as Detector 1)."""
        dets = []
        mask1 = cv2.inRange(hsv, HSV_RANGES['red1']['lower'], HSV_RANGES['red1']['upper'])
        mask2 = cv2.inRange(hsv, HSV_RANGES['red2']['lower'], HSV_RANGES['red2']['upper'])
        red_mask = mask1 | mask2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            epsilon = 0.04 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            sides = len(approx)

            # Skip if in upper half (traffic light territory)
            if (y + h / 2) < hsv.shape[0] * 0.5:
                continue

            if 6 <= sides <= 10:
                dist = FOCAL_LENGTH_PX * REAL_WIDTHS['stop_sign'] / max(w, 1)
                aspect = h / w if w > 0 else 0
                if 0.5 < aspect < 2.0:
                    confidence = min(1.0, area / 3000.0)
                    dets.append({
                        'class': 'stop_sign',
                        'confidence': confidence,
                        'bbox': [x, y, w, h],
                        'distance': dist,
                    })

        return dets

    def _detect_cones_hsv(self, hsv: np.ndarray) -> list:
        """Detect orange cones (same as Detector 1)."""
        dets = []
        orange_mask = cv2.inRange(hsv, HSV_RANGES['orange']['lower'], HSV_RANGES['orange']['upper'])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            dist = FOCAL_LENGTH_PX * REAL_WIDTHS['cone'] / max(w, 1)
            confidence = min(1.0, area / 2000.0)
            dets.append({
                'class': 'cone',
                'confidence': confidence,
                'bbox': [x, y, w, h],
                'distance': dist,
            })
        return dets


# ============================================================================
# Detector 5: Custom YOLOv8n (QLabs-trained)
# ============================================================================

class CustomYoloDetector(DetectorInterface):
    name = "Custom YOLOv8n"
    requires = ["ultralytics", "models/best.pt"]

    # Custom model class mapping (from obstacle_detector.py)
    CUSTOM_MAP = {
        0: 'cone',
        1: 'traffic_light_green',
        2: 'pedestrian',
        3: 'traffic_light_red',
        4: 'round',          # roundabout sign (ignored)
        5: 'stop_sign',
        6: 'traffic_light_yellow',
        7: 'yield',          # yield sign (ignored)
    }

    def __init__(self):
        self._model = None
        self._available = False
        model_paths = [
            os.path.join(os.path.dirname(__file__), '..', 'models', 'best.pt'),
            '/workspaces/isaac_ros-dev/ros2/src/acc_stage1_mission/models/best.pt',
        ]
        try:
            from ultralytics import YOLO
            for p in model_paths:
                if os.path.isfile(p):
                    self._model = YOLO(p)
                    self._available = True
                    print(f"  [Custom YOLOv8n] Loaded model from {p}")
                    break
            if not self._available:
                print(f"  [Custom YOLOv8n] Model not found at any path")
        except (ImportError, Exception) as e:
            print(f"  [Custom YOLOv8n] Not available: {e}")

    def detect(self, image: np.ndarray) -> list:
        if not self._available:
            return []
        dets = []
        results = self._model(image, verbose=False, conf=0.25, device='cpu')
        for result in results:
            if result.boxes is None:
                continue
            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                cls_name = self.CUSTOM_MAP.get(cls_id)
                if cls_name is None or cls_name in ('round', 'yield', 'traffic_light_yellow'):
                    continue
                conf = float(result.boxes.conf[i].item())
                x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()
                w = int(x2 - x1)
                h = int(y2 - y1)

                real_w = REAL_WIDTHS.get(cls_name, 0.1)
                dist = FOCAL_LENGTH_PX * real_w / max(w, 1)

                dets.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), w, h],
                    'distance': dist,
                })
        return dets


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_iou(box1: list, box2: list) -> float:
    """Compute IoU between two [x, y, w, h] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _normalize_class(cls_name: str) -> str:
    """Normalize class names across detectors and ground truth."""
    mapping = {
        'stop sign': 'stop_sign',
        'stop': 'stop_sign',
        'traffic_cone': 'cone',
        'sports ball': 'cone',
        'person': 'pedestrian',
        'traffic_light': 'traffic_light_red',
    }
    return mapping.get(cls_name, cls_name)


def evaluate_detections(detections: list, ground_truth: list, iou_threshold: float = IOU_THRESHOLD) -> dict:
    """Evaluate detections against ground truth.

    Returns dict with TP, FP, FN, per-class breakdown, distance errors.
    """
    # Normalize classes
    for d in detections:
        d['class'] = _normalize_class(d['class'])

    gt_matched = [False] * len(ground_truth)
    det_matched = [False] * len(detections)
    tp = 0
    distance_errors = []

    # Match detections to ground truth (greedy, highest IoU first)
    matches = []
    for di, det in enumerate(detections):
        for gi, gt in enumerate(ground_truth):
            if det.get('bbox') and gt.get('bbox'):
                iou = compute_iou(det['bbox'], gt['bbox'])
            else:
                iou = 0.0
            if iou >= iou_threshold and _normalize_class(gt['class']) == det['class']:
                matches.append((iou, di, gi))

    matches.sort(key=lambda x: -x[0])  # highest IoU first

    for iou_val, di, gi in matches:
        if det_matched[di] or gt_matched[gi]:
            continue
        det_matched[di] = True
        gt_matched[gi] = True
        tp += 1
        # Distance error
        det_dist = detections[di].get('distance')
        gt_dist = ground_truth[gi].get('distance')
        if det_dist is not None and gt_dist is not None:
            distance_errors.append(abs(det_dist - gt_dist))

    fp = sum(1 for m in det_matched if not m)
    fn = sum(1 for m in gt_matched if not m)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-class breakdown
    all_classes = set()
    for gt in ground_truth:
        all_classes.add(_normalize_class(gt['class']))
    for det in detections:
        all_classes.add(det['class'])

    per_class = {}
    for cls in all_classes:
        cls_tp = sum(1 for i, (dm, gt) in enumerate(zip(det_matched, detections))
                     if i < len(detections) and dm and gt['class'] == cls)
        # Recount properly
        cls_gt = [g for g in ground_truth if _normalize_class(g['class']) == cls]
        cls_det = [d for d in detections if d['class'] == cls]
        cls_tp = 0
        for g in cls_gt:
            for d in cls_det:
                if g.get('bbox') and d.get('bbox'):
                    iou = compute_iou(d['bbox'], g['bbox'])
                    if iou >= iou_threshold:
                        cls_tp += 1
                        break
        cls_fp = len(cls_det) - cls_tp
        cls_fn = len(cls_gt) - cls_tp
        cls_p = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        cls_r = cls_tp / (cls_tp + cls_fn) if (cls_tp + cls_fn) > 0 else 0.0
        cls_f1 = 2 * cls_p * cls_r / (cls_p + cls_r) if (cls_p + cls_r) > 0 else 0.0
        per_class[cls] = {'tp': cls_tp, 'fp': cls_fp, 'fn': cls_fn,
                          'precision': cls_p, 'recall': cls_r, 'f1': cls_f1}

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'distance_errors': distance_errors,
        'per_class': per_class,
    }


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(detectors: list, test_images: list) -> dict:
    """Run all detectors on all test images. Returns structured results."""
    results = {}

    for det in detectors:
        print(f"\n  [{det.name}]")
        det_results = {
            'times_ms': [],
            'per_image': [],
            'aggregate': None,
        }

        # Warmup
        det.warmup(test_images[0]['image'])

        total_tp = total_fp = total_fn = 0
        all_dist_errors = []
        per_class_agg = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
        all_detections_for_images = []

        for img_data in test_images:
            image = img_data['image']
            gt = img_data['ground_truth']

            t0 = time.perf_counter()
            dets = det.detect(image)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            det_results['times_ms'].append(elapsed_ms)
            metrics = evaluate_detections(dets, gt)
            metrics['time_ms'] = elapsed_ms
            metrics['detections'] = dets
            metrics['name'] = img_data['name']
            det_results['per_image'].append(metrics)
            all_detections_for_images.append(dets)

            total_tp += metrics['tp']
            total_fp += metrics['fp']
            total_fn += metrics['fn']
            all_dist_errors.extend(metrics['distance_errors'])

            for cls, cls_data in metrics['per_class'].items():
                per_class_agg[cls]['tp'] += cls_data['tp']
                per_class_agg[cls]['fp'] += cls_data['fp']
                per_class_agg[cls]['fn'] += cls_data['fn']

        # Aggregate metrics
        agg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        agg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        agg_f1 = 2 * agg_p * agg_r / (agg_p + agg_r) if (agg_p + agg_r) > 0 else 0.0

        # Per-class aggregates
        for cls in per_class_agg:
            d = per_class_agg[cls]
            d['precision'] = d['tp'] / (d['tp'] + d['fp']) if (d['tp'] + d['fp']) > 0 else 0.0
            d['recall'] = d['tp'] / (d['tp'] + d['fn']) if (d['tp'] + d['fn']) > 0 else 0.0
            d['f1'] = (2 * d['precision'] * d['recall'] / (d['precision'] + d['recall'])
                       if (d['precision'] + d['recall']) > 0 else 0.0)

        det_results['aggregate'] = {
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': agg_p, 'recall': agg_r, 'f1': agg_f1,
            'mean_time_ms': np.mean(det_results['times_ms']),
            'median_time_ms': np.median(det_results['times_ms']),
            'std_time_ms': np.std(det_results['times_ms']),
            'distance_error_mean': np.mean(all_dist_errors) if all_dist_errors else float('nan'),
            'distance_error_std': np.std(all_dist_errors) if all_dist_errors else float('nan'),
            'per_class': dict(per_class_agg),
        }
        det_results['all_detections'] = all_detections_for_images

        times = det_results['times_ms']
        print(f"    F1={agg_f1:.3f}  P={agg_p:.3f}  R={agg_r:.3f}  "
              f"TP={total_tp} FP={total_fp} FN={total_fn}  "
              f"Time: {np.mean(times):.1f}ms avg ({np.median(times):.1f}ms median)")

        results[det.name] = det_results

    return results


# ============================================================================
# Figure Generation
# ============================================================================

DETECTOR_COLORS = {
    'HSV+Contour': '#2196F3',
    'YOLOv8n COCO': '#4CAF50',
    'Hybrid HSV+YOLO': '#FF9800',
    'Hough+HSV': '#9C27B0',
    'Custom YOLOv8n': '#F44336',
}

ALL_CLASSES = ['stop_sign', 'traffic_light_red', 'traffic_light_green', 'cone', 'pedestrian']
CLASS_LABELS = {
    'stop_sign': 'Stop Sign',
    'traffic_light_red': 'Red Light',
    'traffic_light_green': 'Green Light',
    'cone': 'Cone',
    'pedestrian': 'Pedestrian',
}


def figure1_detection_grid(test_images: list, results: dict):
    """Figure 1: Detection grid - 5 detector columns x 4 selected scene rows."""
    det_names = [n for n in results.keys()]
    if not det_names:
        return

    # Select 4 representative scenes
    scene_indices = []
    preferred = ['stop_sign_d0.8', 'multi_stop_cone', 'multi_redlight_ped', 'multi_3cones']
    for pref in preferred:
        for i, img_data in enumerate(test_images):
            if img_data['name'] == pref:
                scene_indices.append(i)
                break
    # Fill remaining with first available
    while len(scene_indices) < 4:
        for i in range(len(test_images)):
            if i not in scene_indices:
                scene_indices.append(i)
                break

    n_cols = len(det_names)
    n_rows = len(scene_indices)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    colors_map = {
        'stop_sign': (0, 0, 255),
        'traffic_light_red': (0, 0, 200),
        'traffic_light_green': (0, 200, 0),
        'cone': (0, 165, 255),
        'pedestrian': (255, 0, 255),
    }

    for row, si in enumerate(scene_indices):
        img_data = test_images[si]
        for col, det_name in enumerate(det_names):
            ax = axes[row, col]
            img_copy = img_data['image'].copy()

            # Draw ground truth boxes (green dashed)
            for gt in img_data['ground_truth']:
                if gt.get('bbox'):
                    bx, by, bw, bh = gt['bbox']
                    cv2.rectangle(img_copy, (bx, by), (bx + bw, by + bh), (0, 255, 0), 1)

            # Draw detector predictions
            det_results = results[det_name]
            if si < len(det_results.get('all_detections', [])):
                dets = det_results['all_detections'][si]
                for d in dets:
                    if d.get('bbox'):
                        bx, by, bw, bh = d['bbox']
                        cls = _normalize_class(d['class'])
                        color = colors_map.get(cls, (255, 255, 0))
                        cv2.rectangle(img_copy, (bx, by), (bx + bw, by + bh), color, 2)
                        label = f"{cls[:6]} {d['confidence']:.2f}"
                        cv2.putText(img_copy, label, (bx, max(12, by - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            # Convert BGR -> RGB for matplotlib
            ax.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
            if row == 0:
                ax.set_title(det_name, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(img_data['name'], fontsize=9)
            ax.axis('off')

    fig.suptitle('Detection Grid: GT (green) vs Predictions (colored)', fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig1_detection_grid.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {OUTPUT_DIR / 'fig1_detection_grid.png'}")
    plt.close(fig)


def figure2_precision_recall_bars(results: dict):
    """Figure 2: Per-class precision/recall grouped bars."""
    det_names = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for metric_idx, metric in enumerate(['precision', 'recall']):
        ax = axes[metric_idx]
        x = np.arange(len(ALL_CLASSES))
        bar_width = 0.15
        offset = -(len(det_names) - 1) / 2 * bar_width

        for i, det_name in enumerate(det_names):
            agg = results[det_name]['aggregate']
            values = []
            for cls in ALL_CLASSES:
                cls_data = agg['per_class'].get(cls, {})
                values.append(cls_data.get(metric, 0.0))
            color = DETECTOR_COLORS.get(det_name, '#333333')
            bars = ax.bar(x + offset + i * bar_width, values, bar_width,
                          label=det_name, color=color, alpha=0.85)
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.2f}', ha='center', va='bottom', fontsize=6)

        ax.set_xlabel('Object Class')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'Per-Class {metric.capitalize()}', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_LABELS.get(c, c) for c in ALL_CLASSES], rotation=15)
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('Per-Class Precision & Recall by Detector', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig2_precision_recall.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig2_precision_recall.png'}")
    plt.close(fig)


def figure3_inference_speed(results: dict):
    """Figure 3: Inference speed box plot with threshold lines."""
    det_names = list(results.keys())
    times_data = [results[n]['times_ms'] for n in det_names]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bp = ax.boxplot(times_data, labels=det_names, patch_artist=True, widths=0.5)

    for i, (patch, det_name) in enumerate(zip(bp['boxes'], det_names)):
        color = DETECTOR_COLORS.get(det_name, '#333333')
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Threshold lines
    ax.axhline(y=1000 / 30, color='#F44336', linestyle='--', alpha=0.7,
               label='30 Hz (33.3ms)')
    ax.axhline(y=1000 / 10, color='#FF9800', linestyle='--', alpha=0.7,
               label='10 Hz (100ms)')

    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Speed Distribution', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=15)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig3_inference_speed.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig3_inference_speed.png'}")
    plt.close(fig)


def figure4_f1_vs_latency(results: dict):
    """Figure 4: F1 vs latency scatter with Pareto frontier."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    points = []
    for det_name, res in results.items():
        agg = res['aggregate']
        f1 = agg['f1']
        latency = agg['mean_time_ms']
        color = DETECTOR_COLORS.get(det_name, '#333333')
        ax.scatter(latency, f1, s=200, c=color, zorder=5, edgecolors='black', linewidths=1)
        ax.annotate(det_name, (latency, f1), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight='bold')
        points.append((latency, f1, det_name))

    # Pareto frontier
    points.sort(key=lambda p: p[0])
    pareto = []
    max_f1 = -1
    for lat, f1, name in points:
        if f1 > max_f1:
            pareto.append((lat, f1))
            max_f1 = f1
    if len(pareto) > 1:
        pareto_x, pareto_y = zip(*pareto)
        ax.plot(pareto_x, pareto_y, '--', color='#F44336', alpha=0.5,
                linewidth=1.5, label='Pareto frontier')

    # Ideal region (high F1, low latency)
    ax.axhspan(0.7, 1.05, xmin=0, xmax=0.3, alpha=0.08, color='green', label='Ideal region')

    # Threshold lines
    ax.axvline(x=1000 / 30, color='#F44336', linestyle=':', alpha=0.5, label='30Hz budget')

    ax.set_xlabel('Mean Inference Time (ms)')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Inference Latency', fontsize=14)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_f1_vs_latency.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig4_f1_vs_latency.png'}")
    plt.close(fig)


def figure5_distance_accuracy(results: dict, test_images: list):
    """Figure 5: Distance estimation accuracy — estimated vs ground truth."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (det_name, res) in zip(axes, results.items()):
        gt_dists = []
        est_dists = []
        cls_colors = []
        color_map = {
            'stop_sign': '#F44336',
            'traffic_light_red': '#E91E63',
            'traffic_light_green': '#4CAF50',
            'cone': '#FF9800',
            'pedestrian': '#2196F3',
        }

        for img_idx, img_metrics in enumerate(res['per_image']):
            dets = img_metrics.get('detections', [])
            gts = test_images[img_idx]['ground_truth']

            for d in dets:
                d_cls = _normalize_class(d['class'])
                d_dist = d.get('distance')
                if d_dist is None:
                    continue
                # Find matching GT
                best_iou = 0
                best_gt = None
                for g in gts:
                    if _normalize_class(g['class']) == d_cls and d.get('bbox') and g.get('bbox'):
                        iou = compute_iou(d['bbox'], g['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = g
                if best_gt and best_iou >= IOU_THRESHOLD:
                    gt_dists.append(best_gt['distance'])
                    est_dists.append(d_dist)
                    cls_colors.append(color_map.get(d_cls, '#333333'))

        if gt_dists:
            ax.scatter(gt_dists, est_dists, c=cls_colors, s=60, alpha=0.7, edgecolors='black', linewidths=0.5)
            # Perfect line
            max_d = max(max(gt_dists), max(est_dists)) * 1.1
            ax.plot([0, max_d], [0, max_d], 'k--', alpha=0.3, label='Perfect')
            ax.set_xlim(0, max_d)
            ax.set_ylim(0, max_d)
        ax.set_xlabel('Ground Truth Distance (m)')
        ax.set_ylabel('Estimated Distance (m)')
        ax.set_title(det_name, fontsize=10)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Distance Estimation Accuracy', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_distance_accuracy.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig5_distance_accuracy.png'}")
    plt.close(fig)


def figure6_confusion_matrices(results: dict, test_images: list):
    """Figure 6: Confusion matrices — one heatmap per detector."""
    classes_with_bg = ALL_CLASSES + ['background']
    n_cls = len(classes_with_bg)

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    for ax, (det_name, res) in zip(axes, results.items()):
        matrix = np.zeros((n_cls, n_cls), dtype=int)

        for img_idx, img_metrics in enumerate(res['per_image']):
            dets = img_metrics.get('detections', [])
            gts = test_images[img_idx]['ground_truth']

            gt_matched = [False] * len(gts)

            for d in dets:
                d_cls = _normalize_class(d['class'])
                if d_cls not in ALL_CLASSES:
                    continue
                d_idx = ALL_CLASSES.index(d_cls)

                # Find best matching GT
                best_iou = 0
                best_gi = -1
                for gi, g in enumerate(gts):
                    g_cls = _normalize_class(g['class'])
                    if d.get('bbox') and g.get('bbox'):
                        iou = compute_iou(d['bbox'], g['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi

                if best_gi >= 0 and best_iou >= IOU_THRESHOLD:
                    g_cls = _normalize_class(gts[best_gi]['class'])
                    g_idx = ALL_CLASSES.index(g_cls) if g_cls in ALL_CLASSES else n_cls - 1
                    matrix[g_idx, d_idx] += 1
                    gt_matched[best_gi] = True
                else:
                    # False positive: background predicted as something
                    matrix[n_cls - 1, d_idx] += 1

            # False negatives: GT not matched
            for gi, matched in enumerate(gt_matched):
                if not matched:
                    g_cls = _normalize_class(gts[gi]['class'])
                    if g_cls in ALL_CLASSES:
                        g_idx = ALL_CLASSES.index(g_cls)
                        matrix[g_idx, n_cls - 1] += 1

        # Plot heatmap
        im = ax.imshow(matrix, cmap='Blues', aspect='auto')
        labels = [CLASS_LABELS.get(c, c)[:8] for c in ALL_CLASSES] + ['BG']
        ax.set_xticks(range(n_cls))
        ax.set_yticks(range(n_cls))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(det_name, fontsize=10)

        # Annotate cells
        for i in range(n_cls):
            for j in range(n_cls):
                val = matrix[i, j]
                if val > 0:
                    ax.text(j, i, str(val), ha='center', va='center',
                            fontsize=8, color='white' if val > matrix.max() / 2 else 'black')

    fig.suptitle('Confusion Matrices by Detector', fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig6_confusion_matrices.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig6_confusion_matrices.png'}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 72)
    print("Obstacle Detection Comparison — SDCS QCar2")
    print("=" * 72)

    # Generate test suite
    print("\nGenerating 24 synthetic test images...")
    test_images = generate_test_suite()
    print(f"  Generated {len(test_images)} images ({IMAGE_WIDTH}x{IMAGE_HEIGHT})")

    # Save sample images
    sample_dir = OUTPUT_DIR / 'samples'
    sample_dir.mkdir(exist_ok=True)
    for img_data in test_images[:6]:
        cv2.imwrite(str(sample_dir / f"{img_data['name']}.png"), img_data['image'])
    print(f"  Saved sample images to {sample_dir}/")

    # Initialize detectors
    print("\nInitializing detectors...")
    detectors = [
        HSVContourDetector(),
        HoughCircleHSVDetector(),
    ]

    # Try YOLO-based detectors (graceful skip)
    try:
        yolo_coco = YOLOCocoDetector()
        if yolo_coco._available:
            detectors.append(yolo_coco)
    except Exception as e:
        print(f"  Skipping YOLO COCO: {e}")

    try:
        hybrid = HybridHSVYoloDetector()
        detectors.append(hybrid)
    except Exception as e:
        print(f"  Skipping Hybrid: {e}")

    try:
        custom = CustomYoloDetector()
        if custom._available:
            detectors.append(custom)
    except Exception as e:
        print(f"  Skipping Custom YOLO: {e}")

    print(f"  Active detectors: {[d.name for d in detectors]}")

    # Run benchmark
    print("\n--- Running Benchmark ---")
    results = run_benchmark(detectors, test_images)

    # Print summary table
    print("\n" + "=" * 72)
    print("SUMMARY TABLE")
    print("=" * 72)
    header = (f"{'Detector':<20} {'F1':>6} {'Prec':>6} {'Recall':>6} "
              f"{'TP':>4} {'FP':>4} {'FN':>4} {'Time(ms)':>10} {'Dist Err':>10}")
    print(header)
    print("-" * len(header))

    for det_name, res in results.items():
        agg = res['aggregate']
        dist_err = f"{agg['distance_error_mean']:.3f}" if not np.isnan(agg['distance_error_mean']) else "N/A"
        print(f"{det_name:<20} {agg['f1']:>6.3f} {agg['precision']:>6.3f} {agg['recall']:>6.3f} "
              f"{agg['tp']:>4} {agg['fp']:>4} {agg['fn']:>4} "
              f"{agg['mean_time_ms']:>10.1f} {dist_err:>10}")

    # Per-class summary
    print("\n" + "-" * 72)
    print("PER-CLASS F1 SCORES")
    print("-" * 72)
    header2 = f"{'Detector':<20} " + " ".join(f"{CLASS_LABELS.get(c, c)[:10]:>10}" for c in ALL_CLASSES)
    print(header2)
    print("-" * len(header2))
    for det_name, res in results.items():
        agg = res['aggregate']
        vals = []
        for cls in ALL_CLASSES:
            f1 = agg['per_class'].get(cls, {}).get('f1', 0.0)
            vals.append(f"{f1:.3f}")
        print(f"{det_name:<20} " + " ".join(f"{v:>10}" for v in vals))

    # Generate figures
    print("\n--- Generating Figures ---")
    figure1_detection_grid(test_images, results)
    figure2_precision_recall_bars(results)
    figure3_inference_speed(results)
    figure4_f1_vs_latency(results)
    figure5_distance_accuracy(results, test_images)
    figure6_confusion_matrices(results, test_images)

    # Save metrics to JSON
    metrics_json = {}
    for det_name, res in results.items():
        agg = res['aggregate']
        agg_clean = {k: v for k, v in agg.items() if k != 'per_class'}
        agg_clean['per_class'] = {}
        for cls, cls_data in agg['per_class'].items():
            agg_clean['per_class'][cls] = {k: v for k, v in cls_data.items()}
        # Convert NaN to None for JSON
        for k, v in agg_clean.items():
            if isinstance(v, float) and np.isnan(v):
                agg_clean[k] = None
        metrics_json[det_name] = agg_clean

    with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Saved {OUTPUT_DIR / 'metrics.json'}")

    # Recommendation
    print("\n" + "=" * 72)
    print("RECOMMENDATION")
    print("=" * 72)
    best_f1_name = max(results.keys(), key=lambda n: results[n]['aggregate']['f1'])
    best_speed_name = min(results.keys(), key=lambda n: results[n]['aggregate']['mean_time_ms'])
    print(f"  Best F1: {best_f1_name} ({results[best_f1_name]['aggregate']['f1']:.3f})")
    print(f"  Fastest: {best_speed_name} ({results[best_speed_name]['aggregate']['mean_time_ms']:.1f}ms)")

    # Find best Pareto-optimal
    pareto_candidates = []
    for name, res in results.items():
        agg = res['aggregate']
        pareto_candidates.append((name, agg['f1'], agg['mean_time_ms']))
    pareto_candidates.sort(key=lambda x: x[2])
    pareto = []
    max_f1 = -1
    for name, f1, lat in pareto_candidates:
        if f1 > max_f1:
            pareto.append(name)
            max_f1 = f1
    print(f"  Pareto-optimal: {pareto}")
    print(f"  Suggested detection_mode: 'auto' (custom model > COCO YOLO > HSV fallback)")

    print(f"\nAll results saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
