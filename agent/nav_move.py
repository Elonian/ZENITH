import math
import time

# heading version
def navigate_to_target_with_heading(controller, humanoid_id, start_pos, target_pos, heading_deg):
    """
    Navigate humanoid from start_pos to target_pos, considering current heading.

    Args:
        controller: UnrealCV interface controller
        humanoid_id: ID of the humanoid agent
        start_pos: [x, y] current position
        target_pos: [x, y] goal position
        heading_deg: current agent heading (degrees, -180 ~ 180)

    Returns:
        new_heading: updated heading after rotation (in degrees), or None if error occurs
    """
    try:
        DEFAULT_SPEED = 100
        MAX_STEP_DURATION = 10.0
        MIN_STEP_DURATION = 0.2

        sx, sy = start_pos
        tx, ty = target_pos

        # Step 1: Set agent speed
        controller.humanoid_set_speed(humanoid_id, DEFAULT_SPEED)

        # Step 2: Compute direction to target (UE5: X+ up, Y+ right)
        dx = tx - sx
        dy = ty - sy
        target_angle = math.degrees(math.atan2(dy, dx))

        # Step 3: Calculate shortest rotation from current heading
        delta_angle = target_angle - heading_deg
        delta_angle = (delta_angle + 180) % 360 - 180  # normalize to [-180, 180]

        # Step 4: Rotate agent if needed
        if abs(delta_angle) > 1e-2:
            direction = 'right' if delta_angle > 0 else 'left'
            print(direction, delta_angle)
            controller.humanoid_rotate(humanoid_id, abs(delta_angle), direction)
            time.sleep(1.0)

        # Step 5: Move forward
        distance = math.hypot(dx, dy)
        total_time = distance / DEFAULT_SPEED

        if total_time <= MIN_STEP_DURATION:
            steps = 1
            step_duration = total_time
        else:
            steps = math.ceil(total_time / MAX_STEP_DURATION)
            step_duration = total_time / steps

        for _ in range(steps):
            controller.humanoid_step_forward(humanoid_id, duration=step_duration, direction=0)
            time.sleep(step_duration)

        return

    except Exception as e:
        print(f"[Error] Failed to navigate: {e}")
        return None
    

    
# without heading version
def navigate_to_target(controller, humanoid_id, start_pos, target_pos):
    """
    Navigate humanoid from start_pos to target_pos with default speed,
    automatically computing rotation and duration.

    Args:
        controller: UnrealCV interface controller
        humanoid_id: ID of the humanoid agent
        start_pos: [x, y] initial position
        target_pos: [a, b] goal position
    """
    DEFAULT_SPEED = 200  # unit per second
    MAX_STEP_DURATION = 10.0  # seconds per step
    MIN_STEP_DURATION = 0.2

    sx, sy = start_pos
    tx, ty = target_pos

    # Set speed
    controller.humanoid_set_speed(humanoid_id, DEFAULT_SPEED)

    # Compute facing angle
    dx = tx - sx
    dy = ty - sy
    print(dx,dy)
    target_angle_deg = math.degrees(math.atan2(dy, dx))  # [-180, 180]
    print(target_angle_deg)

    # Determine rotation direction
    if target_angle_deg == 0:
        print("No rotation needed, already facing target.")
    else:
        if target_angle_deg > 0:
            direction = 'left'
            angle = target_angle_deg
        else:
            direction = 'right'
            angle = -target_angle_deg
        controller.humanoid_rotate(humanoid_id, angle, direction)
        time.sleep(1)

    # Compute distance and time needed
    if target_angle_deg == 0:
        # max(|dx|, |dy|)
        distance = max(abs(dx), abs(dy))
    else:
        distance = math.hypot(dx, dy)
    print("Distance to walk:", distance)
    total_time = distance / DEFAULT_SPEED

    # Choose step size
    if total_time <= MIN_STEP_DURATION:
        steps = 1
        step_duration = total_time
    else:
        steps = math.ceil(total_time / MAX_STEP_DURATION)
        step_duration = total_time / steps

    # Step forward repeatedly
    for _ in range(steps):
        controller.humanoid_step_forward(humanoid_id, duration=step_duration, direction=0)
        time.sleep(step_duration)