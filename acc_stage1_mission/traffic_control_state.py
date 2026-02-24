"""
Traffic Control State data class.

Since acc_stage1_mission is a Python-only ROS2 package (ament_python),
we use a Python dataclass with JSON serialization instead of a custom .msg file.
This is published/subscribed via std_msgs/String.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TrafficControlState:
    """
    Traffic control state message (JSON-serialized via std_msgs/String).

    Attributes:
        control_type: Type of traffic control ("traffic_light", "stop_sign", "yield_sign", "none")
        light_state: State of traffic light ("red", "green", "yellow", "unknown")
        distance: Distance to the traffic control (meters)
        should_stop: Whether the vehicle should stop
        stop_duration: How long to stop (0 for traffic lights - wait for green)
        stop_line_x: X coordinate of stop line (map frame)
        stop_line_y: Y coordinate of stop line (map frame)
    """
    control_type: str = "none"
    light_state: str = "unknown"
    distance: float = 0.0
    should_stop: bool = False
    stop_duration: float = 0.0
    stop_line_x: float = 0.0
    stop_line_y: float = 0.0
    light_name: str = ""  # Name of matched known traffic light (from traffic_light_map)

    def to_json(self) -> str:
        """Serialize to JSON string for ROS2 std_msgs/String."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> 'TrafficControlState':
        """Deserialize from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(**data)
        except (json.JSONDecodeError, TypeError) as e:
            # Return default state on parse error
            return cls()


@dataclass
class ObstaclePosition:
    """
    Obstacle position for MPCC avoidance.

    Attributes:
        x, y: Position in map frame
        radius: Collision radius
        vx, vy: Velocity components (for prediction)
        obj_class: Object class (person, cone, vehicle, etc.)
    """
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.2
    vx: float = 0.0
    vy: float = 0.0
    obj_class: str = "unknown"

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ObstaclePositions:
    """
    Collection of obstacle positions (JSON-serialized via std_msgs/String).
    """
    obstacles: list = field(default_factory=list)  # List of ObstaclePosition dicts
    timestamp: float = 0.0  # ROS time in seconds

    def to_json(self) -> str:
        """Serialize to JSON string for ROS2 std_msgs/String."""
        return json.dumps({
            'obstacles': self.obstacles,
            'timestamp': self.timestamp
        })

    @classmethod
    def from_json(cls, json_str: str) -> 'ObstaclePositions':
        """Deserialize from JSON string."""
        try:
            data = json.loads(json_str)
            return cls(
                obstacles=data.get('obstacles', []),
                timestamp=data.get('timestamp', 0.0)
            )
        except (json.JSONDecodeError, TypeError):
            return cls()

    def add_obstacle(self, obs: ObstaclePosition) -> None:
        """Add an obstacle to the collection."""
        self.obstacles.append(obs.to_dict())
