"""
Road Boundary Spline Module for MPCC Controller.

Provides spline-based road boundary representation inspired by PyMPC.
Computes linearized halfspace constraints for the MPCC solver to ensure
the vehicle stays within road boundaries.

IMPORTANT: Coordinates in road_boundaries.yaml are in QLabs world frame.
The MPCC controller operates in Cartographer's map frame.
This module handles the transformation between these frames.
"""

import os
import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import yaml


class RoadSegment:
    """Represents a single road segment with centerline and width information."""

    def __init__(self, name: str, segment_data: Dict[str, Any],
                 transform_origin: np.ndarray = None,
                 transform_heading: float = 0.0):
        """
        Initialize a road segment from configuration data.

        Args:
            name: Segment name
            segment_data: Dictionary containing segment definition
            transform_origin: Origin offset for coordinate transform
            transform_heading: Heading offset for coordinate transform (radians)
        """
        self.name = name
        self.segment_type = segment_data.get('type', 'spline')
        self.transform_origin = transform_origin if transform_origin is not None else np.array([0.0, 0.0])
        self.transform_heading = transform_heading

        if self.segment_type == 'circular':
            # Circular segment (intersection, hub area)
            center_qlabs = np.array([
                segment_data['center']['x'],
                segment_data['center']['y']
            ])
            # Transform center to map frame
            self.center = self._qlabs_to_map(center_qlabs)
            self.radius = segment_data['radius']
            self.width = segment_data.get('width', 0.24)
            self.centerline = None
            self.s_values = None
            self.width_left = None
            self.width_right = None
        else:
            # Spline segment (road)
            centerline_qlabs = segment_data.get('centerline', [])
            if centerline_qlabs:
                self.s_values = np.array([pt['s'] for pt in centerline_qlabs])
                # Transform all centerline points to map frame
                self.centerline = np.array([
                    self._qlabs_to_map(np.array([pt['x'], pt['y']]))
                    for pt in centerline_qlabs
                ])
                self.width_left = np.array([pt.get('width_left', 0.24) for pt in centerline_qlabs])
                self.width_right = np.array([pt.get('width_right', 0.24) for pt in centerline_qlabs])
            else:
                self.s_values = np.array([])
                self.centerline = np.array([]).reshape(0, 2)
                self.width_left = np.array([])
                self.width_right = np.array([])
            self.center = None
            self.radius = None
            self.width = None

    def _qlabs_to_map(self, point: np.ndarray) -> np.ndarray:
        """Transform a point from QLabs to map frame."""
        # Translate relative to origin
        dx = point[0] - self.transform_origin[0]
        dy = point[1] - self.transform_origin[1]

        # Rotate by the calibrated transform angle
        theta = self.transform_heading
        cos_h = np.cos(theta)
        sin_h = np.sin(theta)
        x_map = cos_h * dx + sin_h * dy
        y_map = -sin_h * dx + cos_h * dy

        return np.array([x_map, y_map])

    def contains_point(self, x: float, y: float, margin: float = 0.5) -> bool:
        """
        Check if a point is within or near this segment.
        Point is expected in map frame (already transformed).

        Args:
            x, y: Point coordinates in map frame
            margin: Extra margin around segment

        Returns:
            True if point is within segment bounds
        """
        if self.segment_type == 'circular':
            dist = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
            return dist <= self.radius + self.width + margin
        else:
            if len(self.centerline) == 0:
                return False
            # Check if point is near any centerline point
            for i in range(len(self.centerline)):
                cx, cy = self.centerline[i]
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                max_width = max(self.width_left[i], self.width_right[i])
                if dist <= max_width + margin:
                    return True
            return False

    def get_nearest_point_and_tangent(self, x: float, y: float) -> Tuple[np.ndarray, float, float, float]:
        """
        Find the nearest point on the segment centerline and its tangent.
        Point is expected in map frame.

        Args:
            x, y: Query point coordinates in map frame

        Returns:
            Tuple of (nearest_point, tangent_angle, width_left, width_right)
            All in map frame.
        """
        if self.segment_type == 'circular':
            # For circular segments, project point onto circle
            dx = x - self.center[0]
            dy = y - self.center[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < 0.01:
                # Point at center, use arbitrary direction
                return self.center.copy(), 0.0, self.width, self.width

            # Nearest point on circle
            nearest = self.center + self.radius * np.array([dx/dist, dy/dist])
            # Tangent is perpendicular to radius (counter-clockwise)
            tangent = np.arctan2(-dx, dy)
            return nearest, tangent, self.width, self.width
        else:
            if len(self.centerline) < 2:
                if len(self.centerline) == 1:
                    return self.centerline[0].copy(), 0.0, self.width_left[0], self.width_right[0]
                return np.array([x, y]), 0.0, 0.24, 0.24

            # Find nearest segment
            min_dist = float('inf')
            best_point = self.centerline[0].copy()
            best_tangent = 0.0
            best_idx = 0
            interp_t = 0.0

            point = np.array([x, y])

            for i in range(len(self.centerline) - 1):
                p1 = self.centerline[i]
                p2 = self.centerline[i + 1]

                # Project point onto segment
                v = p2 - p1
                u = point - p1
                seg_len_sq = np.dot(v, v)

                if seg_len_sq < 1e-10:
                    # Degenerate segment
                    proj = p1.copy()
                    t = 0.0
                else:
                    t = np.clip(np.dot(u, v) / seg_len_sq, 0.0, 1.0)
                    proj = p1 + t * v

                dist = np.linalg.norm(point - proj)
                if dist < min_dist:
                    min_dist = dist
                    best_point = proj.copy()
                    best_tangent = np.arctan2(v[1], v[0])
                    best_idx = i
                    interp_t = t

            # Interpolate widths
            next_idx = min(best_idx + 1, len(self.width_left) - 1)
            w_left = (1 - interp_t) * self.width_left[best_idx] + interp_t * self.width_left[next_idx]
            w_right = (1 - interp_t) * self.width_right[best_idx] + interp_t * self.width_right[next_idx]

            return best_point, best_tangent, w_left, w_right


class TrafficControl:
    """Represents a traffic control (stop sign, traffic light, etc.)."""

    def __init__(self, control_data: Dict[str, Any],
                 transform_origin: np.ndarray = None,
                 transform_heading: float = 0.0):
        """Initialize from configuration data."""
        self.control_type = control_data['type']
        self.name = control_data.get('name', 'unnamed')

        # Store in QLabs frame
        position_qlabs = np.array([
            control_data['position']['x'],
            control_data['position']['y']
        ])

        # Transform to map frame
        origin = transform_origin if transform_origin is not None else np.array([0.0, 0.0])
        dx = position_qlabs[0] - origin[0]
        dy = position_qlabs[1] - origin[1]
        theta = transform_heading
        cos_h = np.cos(theta)
        sin_h = np.sin(theta)
        self.position = np.array([
            cos_h * dx + sin_h * dy,
            -sin_h * dx + cos_h * dy
        ])

        self.stop_line_distance = control_data.get('stop_line_distance', 0.2)
        self.approach_angles = control_data.get('approach_angles', [])


class ObstacleZone:
    """Represents a zone with special obstacle handling."""

    def __init__(self, zone_data: Dict[str, Any],
                 transform_origin: np.ndarray = None,
                 transform_heading: float = 0.0):
        """Initialize from configuration data."""
        self.name = zone_data['name']
        self.zone_type = zone_data['type']

        # Store in QLabs frame
        center_qlabs = np.array([
            zone_data['center']['x'],
            zone_data['center']['y']
        ])

        # Transform to map frame
        origin = transform_origin if transform_origin is not None else np.array([0.0, 0.0])
        dx = center_qlabs[0] - origin[0]
        dy = center_qlabs[1] - origin[1]
        theta = transform_heading
        cos_h = np.cos(theta)
        sin_h = np.sin(theta)
        self.center = np.array([
            cos_h * dx + sin_h * dy,
            -sin_h * dx + cos_h * dy
        ])

        self.max_velocity = zone_data.get('max_velocity', 0.4)

        if self.zone_type == 'circle':
            self.radius = zone_data['radius']
            self.width = None
            self.height = None
        elif self.zone_type == 'rectangle':
            self.radius = None
            self.width = zone_data['width']
            self.height = zone_data['height']

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (in map frame) is within this zone."""
        if self.zone_type == 'circle':
            dist = np.sqrt((x - self.center[0])**2 + (y - self.center[1])**2)
            return dist <= self.radius
        elif self.zone_type == 'rectangle':
            dx = abs(x - self.center[0])
            dy = abs(y - self.center[1])
            return dx <= self.width / 2 and dy <= self.height / 2
        return False


class RoadBoundarySpline:
    """
    Spline-based road boundary representation (inspired by PyMPC).

    Provides linearized halfspace constraints for MPCC controller to ensure
    the vehicle stays within road boundaries.

    All internal storage is in MAP FRAME after transformation from QLabs.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize road boundary spline from configuration file.

        Args:
            config_path: Path to road_boundaries.yaml config file
        """
        # Default parameters (updated for better driving room)
        self.road_width = 0.30
        self.vehicle_half_width = 0.08
        self.safety_margin = 0.02

        # Road segments (stored in map frame)
        self.segments: List[RoadSegment] = []

        # Traffic controls (stored in map frame)
        self.traffic_controls: List[TrafficControl] = []

        # Obstacle zones (stored in map frame)
        self.obstacle_zones: List[ObstacleZone] = []

        # Coordinate transform parameters (for reference)
        self.transform_origin = np.array([-1.205, -0.83])  # Default QLabs origin
        self.transform_heading = 0.7177  # Calibrated transform angle (radians)

        # Debug flag
        self.debug = False

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            print(f"[RoadBoundarySpline] Config not found: {config_path}")
            return

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load global parameters
        self.road_width = config.get('road_width', 0.24)
        self.vehicle_half_width = config.get('vehicle_half_width', 0.12)
        self.safety_margin = config.get('safety_margin', 0.05)

        # Load transform parameters
        if 'transform' in config:
            transform = config['transform']
            self.transform_origin = np.array([
                transform.get('origin_x', -1.205),
                transform.get('origin_y', -0.83)
            ])
            # Use calibrated radian value when available, fall back to degrees
            if 'origin_heading_rad' in transform:
                self.transform_heading = transform['origin_heading_rad']
            else:
                self.transform_heading = math.radians(-transform.get('origin_heading_deg', -44.7))

        print(f"[RoadBoundarySpline] Transform: origin=({self.transform_origin[0]:.3f}, {self.transform_origin[1]:.3f}), "
              f"heading={math.degrees(self.transform_heading):.1f}deg")

        # Load road segments (will be transformed to map frame internally)
        for segment_data in config.get('road_segments', []):
            name = segment_data.get('name', 'unnamed')
            segment = RoadSegment(name, segment_data,
                                  self.transform_origin, self.transform_heading)
            self.segments.append(segment)
            if self.debug and segment.centerline is not None and len(segment.centerline) > 0:
                print(f"  Segment '{name}': {len(segment.centerline)} points, "
                      f"first=({segment.centerline[0][0]:.2f}, {segment.centerline[0][1]:.2f})")

        # Load traffic controls
        for control_data in config.get('traffic_controls', []):
            control = TrafficControl(control_data,
                                     self.transform_origin, self.transform_heading)
            self.traffic_controls.append(control)

        # Load obstacle zones
        for zone_data in config.get('obstacle_zones', []):
            zone = ObstacleZone(zone_data,
                                self.transform_origin, self.transform_heading)
            self.obstacle_zones.append(zone)

        print(f"[RoadBoundarySpline] Loaded {len(self.segments)} segments, "
              f"{len(self.traffic_controls)} traffic controls, "
              f"{len(self.obstacle_zones)} obstacle zones")

    def get_active_segment(self, x: float, y: float) -> Optional[RoadSegment]:
        """
        Find the road segment that contains the given point.
        Point is expected in MAP FRAME.

        Args:
            x, y: Point coordinates in map frame

        Returns:
            The active RoadSegment or None
        """
        for segment in self.segments:
            if segment.contains_point(x, y):
                return segment
        return None

    def get_boundary_constraints(
        self,
        x: float,
        y: float,
        theta: float,
        use_map_frame: bool = True  # Kept for API compatibility, but ignored (always map frame)
    ) -> Tuple[np.ndarray, float, float]:
        """
        Get linearized boundary constraints at given position.
        Position is expected in MAP FRAME.

        The constraints are in the form:
            A @ position <= b_left  (left boundary)
            -A @ position <= b_right (right boundary)

        Where A is the normal vector pointing LEFT from the road direction.

        Args:
            x, y: Vehicle position in map frame
            theta: Vehicle heading in map frame (used as fallback)
            use_map_frame: Ignored (always expects map frame)

        Returns:
            A: Normal vector [ny, -nx] pointing LEFT (2D array)
            b_left: Left boundary constraint value
            b_right: Right boundary constraint value
        """
        # Find the containing segment
        segment = self.get_active_segment(x, y)

        if segment is None:
            # No segment found - use vehicle heading for constraint direction
            # Create wide constraints using default road width
            dx_norm = np.cos(theta)
            dy_norm = np.sin(theta)
            A = np.array([dy_norm, -dx_norm])  # Normal pointing LEFT

            # Allow generous width when off-road
            effective_width = self.road_width * 2 - self.vehicle_half_width
            position = np.array([x, y])
            b_left = np.dot(A, position) + effective_width
            b_right = -np.dot(A, position) + effective_width

            if self.debug:
                print(f"[RoadBoundary] No segment at ({x:.2f}, {y:.2f}), using vehicle heading")

            return A, b_left, b_right

        # Get nearest point on segment and road direction (all in map frame)
        nearest_point, road_tangent, width_left, width_right = segment.get_nearest_point_and_tangent(x, y)

        # Compute normal vector pointing LEFT (perpendicular to road direction)
        dx_norm = np.cos(road_tangent)
        dy_norm = np.sin(road_tangent)
        A = np.array([dy_norm, -dx_norm])

        # Effective widths (accounting for vehicle and safety margin)
        effective_left = width_left - self.vehicle_half_width - self.safety_margin
        effective_right = width_right - self.vehicle_half_width - self.safety_margin

        # Ensure reasonable driving room (minimum 10cm)
        effective_left = max(effective_left, 0.10)
        effective_right = max(effective_right, 0.10)

        # Compute constraint bounds
        # Left boundary: A @ [x,y] <= A @ nearest_point + effective_left
        # Right boundary: -A @ [x,y] <= -A @ nearest_point + effective_right
        b_left = np.dot(A, nearest_point) + effective_left
        b_right = -np.dot(A, nearest_point) + effective_right

        if self.debug:
            print(f"[RoadBoundary] At ({x:.2f}, {y:.2f}): segment='{segment.name}', "
                  f"nearest=({nearest_point[0]:.2f}, {nearest_point[1]:.2f}), "
                  f"tangent={math.degrees(road_tangent):.1f}deg, "
                  f"widths=({effective_left:.2f}, {effective_right:.2f})")

        return A, b_left, b_right

    def get_boundary_constraints_from_path(
        self,
        path_x: float,
        path_y: float,
        path_theta: float,
        default_width: float = 0.25
    ) -> Tuple[np.ndarray, float, float]:
        """
        Compute boundary constraints directly from the reference path.
        This is a simpler approach that doesn't require matching road segments.

        Uses the path tangent to define the road direction and applies
        fixed-width constraints perpendicular to the path.

        Args:
            path_x, path_y: Position on path (map frame)
            path_theta: Tangent angle at this path position (map frame)
            default_width: Half-width of road (default 0.25m for QCar2 roads)

        Returns:
            A: Normal vector pointing LEFT
            b_left: Left boundary constraint value
            b_right: Right boundary constraint value
        """
        # Compute normal vector pointing LEFT (perpendicular to path direction)
        dx_norm = np.cos(path_theta)
        dy_norm = np.sin(path_theta)
        A = np.array([dy_norm, -dx_norm])

        # Effective width (accounting for vehicle and safety margin)
        # QCar2 is small (~0.16m wide), so we can use tighter margins
        # With default_width=0.25, vehicle_half_width=0.08, safety_margin=0.02:
        # effective_width = 0.25 - 0.08 - 0.02 = 0.15m driving room each side
        effective_width = default_width - self.vehicle_half_width - self.safety_margin
        effective_width = max(effective_width, 0.08)  # Minimum 8cm driving room

        # Compute constraint bounds centered on path
        position = np.array([path_x, path_y])
        b_left = np.dot(A, position) + effective_width
        b_right = -np.dot(A, position) + effective_width

        return A, b_left, b_right

    def get_velocity_limit(self, x: float, y: float) -> float:
        """
        Get velocity limit at given position based on obstacle zones.
        Position is expected in MAP FRAME.

        Args:
            x, y: Position in map frame

        Returns:
            Maximum allowed velocity (default 0.6 m/s if no zone applies)
        """
        min_velocity = 0.6  # Default max velocity

        for zone in self.obstacle_zones:
            if zone.contains_point(x, y):
                min_velocity = min(min_velocity, zone.max_velocity)

        return min_velocity

    def get_nearby_traffic_controls(
        self,
        x: float,
        y: float,
        theta: float,
        max_distance: float = 1.5
    ) -> List[Tuple[TrafficControl, float]]:
        """
        Get traffic controls within range that the vehicle is approaching.
        Position is expected in MAP FRAME.

        Args:
            x, y: Vehicle position in map frame
            theta: Vehicle heading in map frame
            max_distance: Maximum distance to consider

        Returns:
            List of (TrafficControl, distance) tuples
        """
        results = []

        for control in self.traffic_controls:
            # Compute distance (both in map frame)
            dx = control.position[0] - x
            dy = control.position[1] - y
            dist = np.sqrt(dx**2 + dy**2)

            if dist > max_distance:
                continue

            # Check if approaching (control is in front of vehicle)
            angle_to_control = np.arctan2(dy, dx)
            angle_diff = abs(self._normalize_angle(angle_to_control - theta))

            # Must be within ~90 degrees of heading
            if angle_diff > np.pi / 2:
                continue

            results.append((control, dist))

        # Sort by distance
        results.sort(key=lambda x: x[1])

        return results

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def check_boundary_violation(self, x: float, y: float, theta: float) -> Tuple[bool, float]:
        """
        Check if position violates road boundaries.

        Args:
            x, y: Vehicle position in map frame
            theta: Vehicle heading in map frame

        Returns:
            (is_violated, violation_amount)
        """
        A, b_left, b_right = self.get_boundary_constraints(x, y, theta)

        position = np.array([x, y])

        # Check left boundary: A @ position <= b_left
        left_value = np.dot(A, position)
        left_violation = max(0, left_value - b_left)

        # Check right boundary: -A @ position <= b_right
        right_value = -np.dot(A, position)
        right_violation = max(0, right_value - b_right)

        total_violation = left_violation + right_violation
        is_violated = total_violation > 0.01  # Small tolerance

        return is_violated, total_violation
