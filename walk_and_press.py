"""CONTINUOUS walk-up-and-press in ONE simulation (no reset/cut):
  Phase WALK  - PID + AMO navigates the robot up to the control panel.
  Phase PRESS - on arrival the left arm reaches in (Jacobian IK) and presses the button,
                while AMO keeps the legs balanced. The forward approach carries the body
                in so the hand reaches without the stand-still recoil.
Renders the whole thing with a following camera -> walk_press.gif.
"""
import numpy as np, torch, mujoco, warnings; warnings.filterwarnings("ignore")
from play_amo import HumanoidEnv, quat_to_euler
from pid_controller import LocomotionPIDController
from walk_and_grasp import OBJECT_POSITIONS, compute_approach_waypoint
import imageio

dev = "cuda" if torch.cuda.is_available() else "cpu"
policy = torch.jit.load("amo_jit.pt", map_location=dev)
env = HumanoidEnv(policy_jit=policy, robot_type="g1", device=dev, headless=True)
pid = LocomotionPIDController(kp_pos=1.0, ki_pos=0.0, kd_pos=0.1, max_vel=0.6, min_vel=0.2)

button = OBJECT_POSITIONS["button_red"]
press_stop = 0.40            # start pressing when pelvis is this far (m) from the button (xy)
approach_wp = compute_approach_waypoint(button, approach_distance=press_stop)
m, d = env.model, env.data
hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_rubber_hand")
cap = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "push_button_red_top")
capjoint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "push_button_red_joint")
ARM = ['left_shoulder_pitch_joint','left_shoulder_roll_joint','left_shoulder_yaw_joint','left_elbow_joint']
arm_qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM]
arm_dofadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in ARM]
default = env.default_dof_pos.copy(); jacp = np.zeros((3, m.nv))

phase = "walk"; frames = []; pd = default.copy(); arm_scale = 2.5; max_press = 0.0
ren = mujoco.Renderer(m, 380, 500); cam = mujoco.MjvCamera()
for i in range(int(env.sim_duration / env.sim_dt)):
    env._extract_state()
    if i % env.sim_decimation == 0:
        rp = env.data.xpos[env.pelvis_id][:2]; rpy = quat_to_euler(env.quat)
        if env.data.xpos[env.pelvis_id][2] < 0.4: print("FELL"); break
        dist = float(np.linalg.norm(approach_wp - rp))
        if phase == "walk" and dist < press_stop + 0.04:
            phase = "press"; tgt_yaw = rpy[2]
        # --- AMO leg control (both phases) ---
        if phase == "walk":
            bearing = np.arctan2(approach_wp[1]-rp[1], approach_wp[0]-rp[0])
            vx, vy, hd = pid.compute_action(current_pos=rp, current_yaw=rpy[2], target_pos=approach_wp, target_yaw=bearing, dt=env.control_dt)
            env.viewer.commands[0]=vx; env.viewer.commands[2]=vy; env.viewer.commands[1]=hd
        else:
            env.viewer.commands[:] = 0.0; env.viewer.commands[1] = tgt_yaw
        obs = env._compute_observation(); ot = torch.from_numpy(obs).float().unsqueeze(0).to(dev)
        with torch.no_grad():
            eh = torch.tensor(np.array(env.extra_history).flatten().copy(), dtype=torch.float).view(1,-1).to(dev)
            raw = env.policy_jit(ot, eh).cpu().numpy().squeeze()
        raw = np.clip(raw, -40, 40)
        env.last_action = np.concatenate([raw.copy(), (env.dof_pos-env.default_dof_pos)[15:]/env.action_scale])
        pd = np.concatenate([raw*env.action_scale, np.zeros(8)]) + env.default_dof_pos
        # --- arms ---
        if phase == "walk":
            pd[15:] = (1-env.arm_blend)*env.prev_arm_action + env.arm_blend*env.arm_action
            env.arm_blend = min(1.0, env.arm_blend + 0.01)
        else:
            cappos = d.geom_xpos[cap].copy(); target = cappos.copy(); target[1] -= 0.04
            mujoco.mj_jacBody(m, d, jacp, None, hand); J = jacp[:, arm_dofadr]
            dq = np.linalg.lstsq(J, target - d.xpos[hand].copy(), rcond=None)[0]
            atgt = d.qpos[arm_qadr] + np.clip(dq, -0.8, 0.8)
            pd[15:19] = np.clip(atgt, default[15:19]-arm_scale, default[15:19]+arm_scale)
            max_press = max(max_press, float(d.qpos[m.jnt_qposadr[capjoint]]))
        env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt*env.gait_freq, 1.0)
        if env._in_place_stand and np.any(np.abs(env.gait_cycle-0.25)<0.05): env.gait_cycle=np.array([0.25,0.25])
        if not env._in_place_stand and np.all(np.abs(env.gait_cycle-0.25)<0.05): env.gait_cycle=np.array([0.25,0.75])
        rpz = env.data.xpos[env.pelvis_id]
        cam.lookat[:] = [rpz[0]*0.4+button[0]*0.6, rpz[1]*0.5+button[1]*0.5, 0.85]; cam.distance=3.0; cam.azimuth=135; cam.elevation=-12
        ren.update_scene(d, camera=cam); frames.append(ren.render())
    torque = (pd-env.dof_pos)*env.stiffness - env.dof_vel*env.damping
    env.data.ctrl = np.clip(torque, -env.torque_limits, env.torque_limits); mujoco.mj_step(m, d)
    if phase=="press" and i > (env.sim_decimation*220): break
print(f"phase={phase} max_press={max_press*100:.2f}cm frames={len(frames)}")
imageio.mimsave("walk_press.gif", frames, fps=22)
print("saved walk_press.gif")
