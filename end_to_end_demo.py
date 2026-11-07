"""END-TO-END: robot WALKS to the panel, then RL (curriculum policy) REACHES from the rest
pose and PRESSES the yellow button — one continuous sim, no reset/cut. The reach is RL, not IK.

  Phase WALK  - AMO locomotion (forward velocity) carries the robot up to the press stance.
  Phase PRESS - the trained curriculum policy drives the right arm: reach -> contact -> press,
                AMO balancing the legs. Arm starts at the rest pose (what the walk leaves it in).

Usage: python end_to_end_demo.py [checkpoint.zip]
"""
import sys, os
from demo_paths import PROJ, require
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, torch, mujoco, imageio
from stable_baselines3 import PPO
from env_wrapper_button import ButtonPressEnv, GRIP_CLOSED

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_button/curr_v2_latest.zip"
require(ckpt)
model = PPO.load(ckpt, device="cpu")
env = ButtonPressEnv(button_name="button_yellow", unified=True, reset_in_contact=False, curriculum=True, headless=True)
env.set_curriculum_frac(0.0)   # frac 0 -> NOMINAL stance (stance noise scales with frac) for the reference
obs, _ = env.reset(seed=0)     # robot at panel; this reset also caches the servo contact pose (wrist tilt)
e = env.env
dev = env.device               # AMO policy device — match it in the manual walk loop
press_wrist = env._cached_w3_c.copy()   # the wrist tilt the policy trained with (from the cached servo)
a4_rest = e.default_dof_pos[19:23].copy()
target_yaw = env.target_yaw

pel = e.model.jnt_qposadr[mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]
press_y = float(e.data.qpos[pel + 1])           # the arrival (press) y
e.data.qpos[pel + 1] += 0.60                     # teleport BACK (+Y) to start the walk
e.data.qpos[pel] += 0.02                         # pre-compensate the walk's consistent x-drift (~2cm)
e.data.qvel[:] = 0.0
mujoco.mj_forward(e.model, e.data)

def shot():
    return env.render_frame(640, 480)

def amo_step(vx, arm4, wrist):
    e.viewer.commands[:] = 0.0; e.viewer.commands[0] = vx; e.viewer.commands[1] = target_yaw
    e.wrist_target = np.asarray(wrist, float)
    e._extract_state(); ao = e._compute_observation()
    ot = torch.from_numpy(ao).float().unsqueeze(0).to(dev)
    with torch.no_grad():
        eh = torch.tensor(np.array(e.extra_history).flatten().copy(), dtype=torch.float).view(1, -1).to(dev)
        leg = np.clip(e.policy_jit(ot, eh).cpu().numpy().squeeze(), -40, 40)
    e.last_action = np.concatenate([leg.copy(), (e.dof_pos[15:] - e.default_dof_pos[15:]) / e.action_scale])
    pd = e.default_dof_pos.copy(); pd[:15] = leg * e.action_scale + e.default_dof_pos[:15]
    pd[19:23] = arm4
    e.gait_cycle = np.remainder(e.gait_cycle + e.control_dt * e.gait_freq, 1.0)
    if not e._in_place_stand and np.all(np.abs(e.gait_cycle - 0.25) < 0.05): e.gait_cycle = np.array([0.25, 0.75])
    for _ in range(e.sim_decimation):
        tau = np.clip((pd - e.dof_pos) * e.stiffness - e.dof_vel * e.damping, -e.torque_limits, e.torque_limits)
        e.apply_ctrl(tau, GRIP_CLOSED); mujoco.mj_step(e.model, e.data); e._extract_state()

frames = []
# settle a moment, then WALK forward until back at the press stance
e._in_place_stand = False
for _ in range(8): amo_step(0.0, a4_rest, np.zeros(3))
for t in range(400):
    # walk in until slightly CLOSER than the nominal stance: the walk's stopping momentum leaves
    # the pelvis ~4cm shy otherwise, putting the rest-pose reach beyond trained competence (~25cm)
    if float(e.data.qpos[pel + 1]) <= press_y - 0.035: break
    amo_step(0.34, a4_rest, np.zeros(3))
    if t % 2 == 0: frames.append(shot())
walk_frames = len(frames)
from scipy.spatial.transform import Rotation as _R
_p = e.data.xpos[e.pelvis_id]; _yaw = _R.from_matrix(e.data.xmat[e.pelvis_id].reshape(3,3)).as_euler('xyz')[2]
print(f"after walk: pelvis xy=({_p[0]:.3f},{_p[1]:.3f}) yaw={_yaw:.2f} | press-stance xy=({env.robot_start_pos[0]:.3f},{press_y:.3f}) yaw={target_yaw:.2f}")

# ARRIVAL HANDOFF: re-init the controller via the env's OWN reset, pinned AT the walked-in base
# pose (no repositioning, no stance noise, frac exactly 1.0 = arm from true rest). Manually
# replicating the training reset state failed 4 ways (settle stiffness, wrist timing, AMO
# history, filter state) — the env's reset IS the correct constructor for the policy's state.
cur = e.data.qpos[pel:pel+7].copy()
env.robot_start_pos = np.array([cur[0], cur[1], env.robot_start_pos[2]])
env.robot_start_quat = cur[3:7].copy()
env.stance_noise_k = 0.0
env.set_curriculum_frac(1.0); env.set_curriculum_frac_min(1.0)
# preserve SCENE-OBJECT state across the handoff reset (Jatin spotted the lever twitch ~10deg:
# the reset's keyframe restore was touching world objects — the robot may re-init its controller,
# but the WORLD must not blink)
_scene_joints = ["lever_handle_joint", "screwdriver_joint", "block_cube_joint", "small_box_joint",
                 "wrench_joint", "right_table_item2_joint", "battery_pack_joint"]
full_qpos = e.data.qpos.copy(); full_qvel = e.data.qvel.copy()   # the EXACT walked-in world+robot state
obs, _ = env.reset()                                              # re-init controller internals (histories,
press_wrist2 = env._solved_wrist.copy()                           #  biases, wrist) — physical state next:
e.data.qpos[:] = full_qpos; e.data.qvel[:] = full_qvel            # restore EVERYTHING (no pose snap, no
mujoco.mj_forward(e.model, e.data); e._extract_state()            #  scene twitch — Jatin saw both)
# now SETTLE ON CAMERA: the robot visibly gathers from its walk pose into the press-ready pose
for i in range(70):
    env._amo_arm_step(e.default_dof_pos[19:23], wrist_target=press_wrist2)
    if i % 2 == 0: frames.append(shot())
env._filt_arm[:] = 0; env._prev_action[:] = 0; env.episode_steps = 0
env.reward_fn.reset(); env._held_steps = 0; env._was_deep = False
env.initial_button_displacement = e.data.qpos[env.button_joint_id]
obs = env._get_obs()
maxpress = 0.0; dips = 0; was_deep = False
env.max_episode_steps = 400   # give the reach+press room (the walk-in start is farther than eval's)
for t in range(400):
    a, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(a)
    _d = env._get_button_displacement(); maxpress = max(maxpress, _d)
    if _d > 0.02: was_deep = True
    if was_deep and _d < 0.010: dips += 1; was_deep = False   # a full press->release cycle = one PUMP
    frames.append(shot())
    if t % 40 == 0:
        gp = env._right_hand_pos(); cap = e.data.geom_xpos[env._cap_gid]
        print(f"  press t{t}: gripper-cap {np.linalg.norm(gp - cap)*100:4.1f}cm  disp {env._get_button_displacement()*1000:4.1f}mm")
    if term or trunc: break

# JOB DONE -> scripted DISENGAGE (Jatin: after the press, the hand should just lower and rest —
# don't leave the RL policy in charge with nothing to do). PD eases the arm back to its rest pose.
for i in range(55):
    env._amo_arm_step(a4_rest, wrist_target=np.zeros(3))
    if i % 2 == 0: frames.append(shot())

imageio.mimsave(os.path.join(PROJ, "_e2e_demo.mp4"), frames, fps=30, quality=8)
print(f"saved _e2e_demo.mp4  frames={len(frames)} (walk {walk_frames}) MAX_press={maxpress*1000:.1f}mm PUMPS={dips}")
