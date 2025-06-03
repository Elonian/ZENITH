import math
import numpy as np
from utils.vector import Vector

class MovementController:
    def __init__(self, agent, communicator, config):
        self.agent = agent
        self.communicator = communicator
        self.config = config

    def move_to_waypoint(self, world_pos: Vector) -> bool:
        try:
            angle, turn_direction = self._get_angle_and_direction(world_pos)
            if angle > 5:  # If not facing the waypoint
                if turn_direction:
                    self.communicator.agent_rotate(self.agent.id, angle, turn_direction)
                    return False

            if self._align_direction(world_pos):
                step_time = 0.1  # Adjust step time as needed
                self.communicator.agent_step_forward(self.agent.id, step_time)

            return self._reached_waypoint(world_pos)

        except Exception as e:
            print(f"Error in movement: {e}")
            return False

    def _get_angle_and_direction(self, target_pos: Vector) -> tuple[float, str | None]:
        """Compute angle and turn direction to face the target."""
        to_target = Vector.subtract(target_pos, self.agent.position)
        angle = math.degrees(
            math.acos(np.clip(self.agent.direction.dot(to_target.normalize()), -1, 1))
        )
        cross = self.agent.direction.cross(to_target)
        turn_direction = 'left' if cross < 0 else 'right'
        if angle < 2:
            return 0.0, None
        return angle, turn_direction

    def _align_direction(self, target_pos: Vector) -> bool:
        """Return True if facing the target within 5 degrees."""
        to_target = Vector.subtract(target_pos, self.agent.position)
        angle = math.degrees(
            math.acos(np.clip(self.agent.direction.dot(to_target.normalize()), -1, 1))
        )
        return angle < 5

    def _reached_waypoint(self, waypoint: Vector) -> bool:
        """Return True if agent is within threshold of waypoint."""
        threshold = self.config.get('waypoint_distance_threshold', 1.0)
        return Vector.distance(self.agent.position, waypoint) < threshold