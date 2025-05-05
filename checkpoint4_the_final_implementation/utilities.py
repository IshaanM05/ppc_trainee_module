# utilities.py
import numpy as np

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0.0
        self.integral = 0.0

    def update(self, error, dt):
        """
        Update the PID controller.

        Args:
            error (float): The error between the desired and current values.
            dt (float): Time delta since the last update.

        Returns:
            float: The control output (throttle in this case).
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output


def stanley_steering(x, y, yaw, v, waypoints, k=1.0, ks=1e-2, max_steer=np.radians(30)):
    """
    Stanley steering controller.

    Args:
        x, y     : rear axle position of the car
        yaw      : vehicle heading angle (in radians)
        v        : vehicle speed
        waypoints: Nx2 array of path waypoints (ordered along the path)
        k        : cross-track gain
        ks       : softening term to prevent division by zero
        max_steer: max steering angle (in radians)

    Returns:
        steer       : steering angle in radians
        target_idx  : index of the nearest waypoint
    """

    # --- Step 1: Compute front axle position ---
    L = 2.5  # wheelbase
    fx = x + L * np.cos(yaw)
    fy = y + L * np.sin(yaw)

    # --- Step 2: Find nearest waypoint to front axle ---
    dx = waypoints[:, 0] - fx
    dy = waypoints[:, 1] - fy
    dists = np.hypot(dx, dy)
    target_idx = np.argmin(dists)

    # --- Step 3: Compute path heading (theta) at that point ---
    wp1 = waypoints[target_idx]
    wp2 = waypoints[min(target_idx + 1, len(waypoints) - 1)]  # lookahead to compute heading
    dx_path = wp2[0] - wp1[0]
    dy_path = wp2[1] - wp1[1]
    path_yaw = np.arctan2(dy_path, dx_path)

    # --- Step 4: Compute heading error (ψ = θ - yaw) ---
    heading_error = path_yaw - yaw
    # Normalize to [-π, π]
    heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

    # --- Step 5: Compute signed cross-track error ---
    dx_f = fx - wp1[0]
    dy_f = fy - wp1[1]
    # Perpendicular to path direction
    crosstrack_error = np.sin(path_yaw) * dx_f - np.cos(path_yaw) * dy_f

    # --- Step 6: Stanley control law ---
    steer = heading_error + np.arctan2(k * crosstrack_error, ks + v)

    # --- Step 7: Clip the steer angle to limits ---
    steer = np.clip(steer, -max_steer, max_steer)

    return steer, target_idx
