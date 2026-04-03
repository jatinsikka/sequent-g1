import numpy as np

class PID:
    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0
        
    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        
    def compute(self, error: float, dt: float) -> float:
        self.integral += error * dt
        
        if dt > 0:
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0
            
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        
        return output


class LocomotionPIDController:
    def __init__(self, 
                 kp_pos: float = 1.0, ki_pos: float = 0.0, kd_pos: float = 0.0, 
                 max_vel: float = 1.0, min_vel: float = 0.0):
        self.pid_x = PID(kp_pos, ki_pos, kd_pos)
        self.pid_y = PID(kp_pos, ki_pos, kd_pos)
        self.min_vel = min_vel
        
    def reset(self):
        self.pid_x.reset()
        self.pid_y.reset()
        
    def compute_action(self, current_pos: np.ndarray, current_yaw: float, 
                       target_pos: np.ndarray, target_yaw: float, dt: float) -> tuple:
        err_x = target_pos[0] - current_pos[0]
        err_y = target_pos[1] - current_pos[1]
        
        v_world_x = self.pid_x.compute(err_x, dt)
        v_world_y = self.pid_y.compute(err_y, dt)
        
        c = np.cos(current_yaw)
        s = np.sin(current_yaw)
        
        vx_cmd = c * v_world_x + s * v_world_y
        vy_cmd = -s * v_world_x + c * v_world_y
        
        dist = np.sqrt(err_x**2 + err_y**2)
        if dist > 0.1:
            if abs(vx_cmd) < self.min_vel:
                if vx_cmd >= 0:
                    vx_cmd = self.min_vel
                else:
                    vx_cmd = -self.min_vel
        
        return vx_cmd, vy_cmd, target_yaw