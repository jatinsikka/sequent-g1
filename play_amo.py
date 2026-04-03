# -----------------------------------------------------------------------------
# Copyright [2025] [Jialong Li, Xuxin Cheng, Tianshu Huang, Xiaolong Wang]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script is based on an initial draft generously provided by Zixuan Chen.
# -----------------------------------------------------------------------------

"""
G1 Humanoid Robot Simulation with AMO Policy

Controls:
    W/S     - Forward/backward velocity
    A/D     - Turn left/right (yaw)
    Q/E     - Strafe left/right
    Z/X     - Adjust height
    U/J     - Torso yaw
    I/K     - Torso pitch
    O/L     - Torso roll
    T       - Toggle random arm movements
    ESC     - Quit
"""

import types
from collections import deque

import glfw
import mujoco
import mujoco_viewer
import numpy as np
import torch


# =============================================================================
# Utility Functions
# =============================================================================

def quat_to_euler(quat):
    """Convert quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw)."""
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def create_key_callback(viewer):
    """Create keyboard callback for robot control."""
    def key_callback(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
            
        # Movement controls
        if key == glfw.KEY_W:
            self.commands[0] += 0.05  # Forward
        elif key == glfw.KEY_S:
            self.commands[0] -= 0.05  # Backward
        elif key == glfw.KEY_A:
            self.commands[1] += 0.1   # Turn left
        elif key == glfw.KEY_D:
            self.commands[1] -= 0.1   # Turn right
        elif key == glfw.KEY_Q:
            self.commands[2] += 0.05  # Strafe left
        elif key == glfw.KEY_E:
            self.commands[2] -= 0.05  # Strafe right
        elif key == glfw.KEY_Z:
            self.commands[3] += 0.05  # Height up
        elif key == glfw.KEY_X:
            self.commands[3] -= 0.05  # Height down
        # Torso controls
        elif key == glfw.KEY_J:
            self.commands[4] += 0.1   # Torso yaw+
        elif key == glfw.KEY_U:
            self.commands[4] -= 0.1   # Torso yaw-
        elif key == glfw.KEY_K:
            self.commands[5] += 0.05  # Torso pitch+
        elif key == glfw.KEY_I:
            self.commands[5] -= 0.05  # Torso pitch-
        elif key == glfw.KEY_L:
            self.commands[6] += 0.05  # Torso roll+
        elif key == glfw.KEY_O:
            self.commands[6] -= 0.1   # Torso roll-
        # Toggle arm control
        elif key == glfw.KEY_T:
            self.commands[7] = not self.commands[7]
            print(f"Arm control: {'ON' if self.commands[7] else 'OFF'}")
        # Quit
        elif key == glfw.KEY_ESCAPE:
            print("Quitting...")
            glfw.set_window_should_close(self.window, True)
            return
        
        # Print current state
        print(f"vx: {self.commands[0]:>6.2f}  "
              f"vy: {self.commands[2]:>6.2f}  "
              f"yaw: {self.commands[1]:>6.2f}  "
              f"height: {0.75 + self.commands[3]:>5.2f}  "
              f"torso: [{self.commands[4]:>5.2f}, {self.commands[5]:>5.2f}, {self.commands[6]:>5.2f}]")
    
    return types.MethodType(key_callback, viewer)


# =============================================================================
# Robot Configuration
# =============================================================================

G1_CONFIG = {
    "model_path": "g1.xml",
    "num_actions": 15,
    "num_dofs": 23,
    "stiffness": np.array([
        150, 150, 150, 300, 80, 20,   # Left leg
        150, 150, 150, 300, 80, 20,   # Right leg
        400, 400, 400,                 # Waist
        80, 80, 40, 60,                # Left arm
        80, 80, 40, 60,                # Right arm
    ]),
    "damping": np.array([
        2, 2, 2, 4, 2, 1,   # Left leg
        2, 2, 2, 4, 2, 1,   # Right leg
        15, 15, 15,          # Waist
        2, 2, 1, 1,          # Left arm
        2, 2, 1, 1,          # Right arm
    ]),
    "default_dof_pos": np.array([
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # Left leg
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # Right leg
        0.0, 0.0, 0.0,                     # Waist
        0.5, 0.0, 0.2, 0.3,                # Left arm
        0.5, 0.0, -0.2, 0.3,               # Right arm
    ]),
    "torque_limits": np.array([
        88, 139, 88, 139, 50, 50,   # Left leg
        88, 139, 88, 139, 50, 50,   # Right leg
        88, 50, 50,                  # Waist
        25, 25, 25, 25,              # Left arm
        25, 25, 25, 25,              # Right arm
    ]),
}


# =============================================================================
# Humanoid Environment
# =============================================================================

class HumanoidEnv:
    """MuJoCo environment for G1 humanoid robot with AMO policy."""
    
    def __init__(self, policy_jit, robot_type="g1", device="cuda", headless=False):
        self.device = device
        self.headless = headless
        self._load_robot_config(robot_type)
        self._init_simulation()
        self._init_viewer()
        self._init_policy(policy_jit)
        self._init_state()
    
    def _load_robot_config(self, robot_type):
        """Load robot-specific configuration."""
        if robot_type != "g1":
            raise ValueError(f"Robot type '{robot_type}' not supported!")
        
        config = G1_CONFIG
        self.model_path = config["model_path"]
        self.num_actions = config["num_actions"]
        self.num_dofs = config["num_dofs"]
        self.stiffness = config["stiffness"]
        self.damping = config["damping"]
        self.default_dof_pos = config["default_dof_pos"]
        self.torque_limits = config["torque_limits"]
        self.arm_dof_range = 0.4
    
    def _init_simulation(self):
        """Initialize MuJoCo simulation."""
        self.sim_duration = 2000.0
        self.sim_dt = 0.002
        self.sim_decimation = 10
        self.control_dt = self.sim_dt * self.sim_decimation
        
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)
        
        # Cache body ID for camera tracking
        self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
    
    def _init_viewer(self):
        """Initialize viewer with keyboard controls."""
        if self.headless:
            # Create mock viewer for headless mode
            class MockViewer:
                def __init__(self):
                    self.commands = np.zeros(8, dtype=np.float32)
                def render(self):
                    pass
                def close(self):
                    pass
            self.viewer = MockViewer()
            return
        
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.commands = np.zeros(8, dtype=np.float32)
        
        # Camera setup
        self.viewer.cam.distance = 4.0
        self.viewer.cam.elevation = -10.0
        self.viewer.cam.azimuth = -90.0
        
        # Keyboard callback
        self.viewer._key_callback = create_key_callback(self.viewer)
        glfw.set_key_callback(self.viewer.window, self.viewer._key_callback)
    
    def _init_policy(self, policy_jit):
        """Initialize policy and adapter networks."""
        self.policy_jit = policy_jit
        
        # Load adapter network
        self.adapter = torch.jit.load("adapter_jit.pt", map_location=self.device)
        self.adapter.eval()
        for param in self.adapter.parameters():
            param.requires_grad = False
        
        # Load normalization statistics
        norm_stats = torch.load("adapter_norm_stats.pt", weights_only=False)
        self.input_mean = torch.tensor(norm_stats['input_mean'], device=self.device, dtype=torch.float32)
        self.input_std = torch.tensor(norm_stats['input_std'], device=self.device, dtype=torch.float32)
        self.output_mean = torch.tensor(norm_stats['output_mean'], device=self.device, dtype=torch.float32)
        self.output_std = torch.tensor(norm_stats['output_std'], device=self.device, dtype=torch.float32)
    
    def _init_state(self):
        """Initialize state variables."""
        # Observation parameters
        self.n_priv = 3
        self.n_proprio = 3 + 2 + 2 + 23 * 3 + 2 + 15
        self.history_len = 10
        self.extra_history_len = 25
        self.n_demo_dof = 8
        
        # State buffers
        self.dof_pos = np.zeros(self.num_dofs, dtype=np.float32)
        self.dof_vel = np.zeros(self.num_dofs, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        
        # Action state
        self.action_scale = 0.25
        self.last_action = np.zeros(self.num_dofs, dtype=np.float32)
        self.arm_action = self.default_dof_pos[15:].copy()
        self.prev_arm_action = self.default_dof_pos[15:].copy()
        self.arm_blend = 0.0
        self.toggle_arm = False
        
        # Gait state
        self.gait_cycle = np.array([0.25, 0.25])
        self.gait_freq = 1.3
        self._in_place_stand = True
        
        # Observation scales
        self.scale_ang_vel = 0.25
        self.scale_dof_vel = 0.05
        
        # Demo observation template
        self.demo_obs_template = np.zeros(self.n_demo_dof + 9)
        self.demo_obs_template[:self.n_demo_dof] = self.default_dof_pos[15:]
        self.demo_obs_template[self.n_demo_dof + 6:self.n_demo_dof + 9] = 0.75
        
        # History buffers
        self.proprio_history = deque(maxlen=self.history_len)
        self.extra_history = deque(maxlen=self.extra_history_len)
        for _ in range(self.history_len):
            self.proprio_history.append(np.zeros(self.n_proprio))
        for _ in range(self.extra_history_len):
            self.extra_history.append(np.zeros(self.n_proprio))
    
    def _extract_state(self):
        """Extract robot state from simulation."""
        self.dof_pos = self.data.qpos.astype(np.float32)[-self.num_dofs:]
        self.dof_vel = self.data.qvel.astype(np.float32)[-self.num_dofs:]
        self.quat = self.data.sensor('orientation').data.astype(np.float32)
        self.ang_vel = self.data.sensor('angular-velocity').data.astype(np.float32)
    
    def _compute_observation(self):
        """Compute observation vector for policy."""
        commands = self.viewer.commands
        rpy = quat_to_euler(self.quat)
        
        # Compute yaw error
        dyaw = rpy[2] - commands[1]
        dyaw = np.remainder(dyaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand:
            dyaw = 0.0
        
        # Gait phase observation
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)
        
        # Adapter input: [height, torso_yaw, torso_pitch, torso_roll, arm_dofs]
        adapter_input = np.concatenate([
            [0.75 + commands[3], commands[4], commands[5], commands[6]],
            self.dof_pos[15:]
        ])
        adapter_input = torch.tensor(adapter_input, device=self.device, dtype=torch.float32).unsqueeze(0)
        adapter_input = (adapter_input - self.input_mean) / (self.input_std + 1e-8)
        
        with torch.no_grad():
            adapter_output = self.adapter(adapter_input)
            adapter_output = adapter_output * self.output_std + self.output_mean
        
        # Proprioceptive observation
        obs_prop = np.concatenate([
            self.ang_vel * self.scale_ang_vel,
            rpy[:2],
            [np.sin(dyaw), np.cos(dyaw)],
            self.dof_pos - self.default_dof_pos,
            self.dof_vel * self.scale_dof_vel,
            self.last_action,
            gait_obs,
            adapter_output.cpu().numpy().squeeze(),
        ])
        
        # Demo observation
        obs_demo = self.demo_obs_template.copy()
        obs_demo[:self.n_demo_dof] = self.dof_pos[15:]
        obs_demo[self.n_demo_dof] = commands[0]      # vx
        obs_demo[self.n_demo_dof + 1] = commands[2]  # vy
        obs_demo[self.n_demo_dof + 3:self.n_demo_dof + 6] = commands[4:7]  # torso
        obs_demo[self.n_demo_dof + 6:self.n_demo_dof + 9] = 0.75 + commands[3]  # height
        
        # Update standing flag
        self._in_place_stand = np.abs(commands[0]) < 0.1
        
        # Update history
        self.proprio_history.append(obs_prop)
        self.extra_history.append(obs_prop)
        
        # Full observation
        obs_priv = np.zeros(self.n_priv)
        obs_hist = np.array(self.proprio_history).flatten()
        
        return np.concatenate([obs_prop, obs_demo, obs_priv, obs_hist])
    
    def _update_gait(self):
        """Update gait cycle phase."""
        self.gait_cycle = np.remainder(self.gait_cycle + self.control_dt * self.gait_freq, 1.0)
        
        # Sync gait when standing
        if self._in_place_stand:
            if np.any(np.abs(self.gait_cycle - 0.25) < 0.05):
                self.gait_cycle = np.array([0.25, 0.25])
        else:
            if np.all(np.abs(self.gait_cycle - 0.25) < 0.05):
                self.gait_cycle = np.array([0.25, 0.75])
    
    def _update_arm_action(self, step):
        """Update arm action with random or default targets."""
        if step % 300 == 0 and step > 0 and self.viewer.commands[7]:
            # Random arm movement
            self.arm_blend = 0.0
            self.prev_arm_action = self.dof_pos[15:].copy()
            self.arm_action = np.random.uniform(-self.arm_dof_range, self.arm_dof_range, 8)
            self.toggle_arm = True
        elif not self.viewer.commands[7] and self.toggle_arm:
            # Return to default
            self.toggle_arm = False
            self.arm_blend = 0.0
            self.prev_arm_action = self.dof_pos[15:].copy()
            self.arm_action = self.default_dof_pos[15:]
        
        self.arm_blend = min(1.0, self.arm_blend + 0.01)
    
    def run(self):
        """Main simulation loop."""
        num_steps = int(self.sim_duration / self.sim_dt)
        pd_target = self.default_dof_pos.copy()
        
        for step in range(num_steps):
            self._extract_state()
            
            # Control at decimated rate
            if step % self.sim_decimation == 0:
                obs = self._compute_observation()
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                
                # Run policy
                with torch.no_grad():
                    extra_hist = torch.tensor(
                        np.array(self.extra_history).flatten(),
                        dtype=torch.float, device=self.device
                    ).view(1, -1)
                    raw_action = self.policy_jit(obs_tensor, extra_hist).cpu().numpy().squeeze()
                
                raw_action = np.clip(raw_action, -40.0, 40.0)
                scaled_action = raw_action * self.action_scale
                
                # Update action buffer
                self.last_action = np.concatenate([
                    raw_action,
                    (self.dof_pos[15:] - self.default_dof_pos[15:]) / self.action_scale
                ])
                
                # Compute PD target
                self._update_arm_action(step)
                pd_target[:15] = scaled_action + self.default_dof_pos[:15]
                pd_target[15:] = (1 - self.arm_blend) * self.prev_arm_action + self.arm_blend * self.arm_action
                
                # Update gait
                self._update_gait()
                
                # Update camera and render
                self.viewer.cam.lookat = self.data.xpos[self.pelvis_id].astype(np.float32)
                self.viewer.render()
            
            # PD control
            torque = (pd_target - self.dof_pos) * self.stiffness - self.dof_vel * self.damping
            torque = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.data.ctrl = torque
            
            mujoco.mj_step(self.model, self.data)
        
        self.viewer.close()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    policy_jit = torch.jit.load("amo_jit.pt", map_location=device)
    env = HumanoidEnv(policy_jit=policy_jit, robot_type="g1", device=device)
    env.run()
