"""
Detection backend interface for ACC self-driving stack.

Defines the abstract base class for all detection backends and the
Detection dataclass used as the common result format.

Available backends (selected via config/modules.yaml or detection_mode param):
  - auto:      Custom YOLO -> COCO YOLO -> HSV fallback (default)
  - hsv:       HSV+Contour color detection (CPU, ~1.8ms, F1=0.43)
  - yolo_coco: YOLOv8n COCO pretrained (GPU/CPU)
  - custom:    Custom QLabs-trained YOLOv8n (best.pt)
  - hybrid:    HSV pre-filter + YOLO verification
  - hough_hsv: HoughCircles for lights + HSV for signs/cones (~3.6ms, F1=0.44)

How to switch:
  ROS param:   detection_mode:=hsv
  Launch arg:  ros2 launch ... detection_mode:=hough_hsv
  Config file: config/modules.yaml -> detection.backend
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class Detection:
    """Single detection result."""
    object_class: str          # 'stop_sign', 'traffic_light_red', 'cone', 'pedestrian', etc.
    confidence: float          # 0.0 - 1.0
    bbox: Optional[tuple]      # (x, y, w, h) in pixels, or None
    distance: Optional[float]  # estimated distance in meters, or None


class DetectorBackend(ABC):
    """Abstract base for all detection backends.

    Subclasses implement detect() to run inference on a single BGR frame
    and return a list of Detection objects. The ObstacleDetector node
    handles all state management (persistence, cooldowns, traffic light
    state machine) on top of the raw detections returned here.
    """

    name: str = "base"

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Run detection on a BGR image.

        Args:
            image: OpenCV BGR image (HxWx3 uint8).

        Returns:
            List of Detection objects found in the frame.
        """
        ...

    def warmup(self, image: np.ndarray) -> None:
        """Optional warmup call (e.g. for GPU model loading)."""
        self.detect(image)
