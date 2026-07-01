"""
Unified pick from the REAL right_workstation table (not a pedestal).

The humanoid STANDS AT THE TABLE (~0.70, 0.84, facing +Y toward the table), AMO balancing the
legs, and a DLS-IK servo (gripper pad-midpoint) reaches to the grasp_box resting on the table
top (world z 0.725), closes the CLOSED gripper, and lifts it clear of the table.

SMOOTHING (kills the jitter): the raw per-step DLS target jerks as AMO micro-shifts the base,
so we (1) LOW-PASS the arm PD target (exponential smoothing toward the IK solution), (2) use
the boosted right-arm hold stiffness + raised torque ceiling + damped wrist hold, and (3) let
the box + AMO SETTLE before the reach. The motion should look smooth, not shaky.
"""
import numpy as np, mujoco, torch, imageio
from collections import deque
from unified_env import UnifiedHumanoidEnv, AMO_JOINTS, WRIST_JOINTS, RARM_JOINTS
from play_amo import quat_to_euler

device = "cuda" if torch.cuda.is_available() else "cpu"
policy = torch.jit.load("amo_jit.pt", map_location=device)
env = UnifiedHumanoidEnv(policy_jit=policy, robot_type="g1", device=device, headless=True)
# load the pick model (box on the real table) + recompute all addresses on it
env.model = mujoco.MjModel.from_binary_path("g1_amo_gripper_pick.mjb")
env.model.opt.timestep = env.sim_dt
env.data = mujoco.MjData(env.model)
m, d = env.model, env.data
def qadr(js): return np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in js])
def dadr(js): return np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in js])
env.amo_qadr, env.amo_dadr = qadr(AMO_JOINTS), dadr(AMO_JOINTS)
env.wrist_qadr, env.wrist_dadr = qadr(WRIST_JOINTS), dadr(WRIST_JOINTS)
env.rarm_qadr, env.rarm_dadr = qadr(RARM_JOINTS), dadr(RARM_JOINTS)
env.rarm_range = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM_JOINTS])
env.wrist_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in WRIST_JOINTS])
env.grip_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "rg_fingers_actuator")
env.pelvis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
env.lpad = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rg_left_pad")
env.rpad = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "rg_right_pad")

# damp the wrist hold (smoother): higher kv, and lift kp a bit so a tilted wrist holds steady
env.wrist_kp = np.array([120., 120., 120.]); env.wrist_kv = np.array([8., 8., 8.])

# 7-DOF right-arm chain (4 shoulder/elbow + 3 wrist) + actuators
RARM7 = RARM_JOINTS + WRIST_JOINTS
r7_qadr = qadr(RARM7); r7_dadr = dadr(RARM7)
r7_range = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM7])
r7_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in RARM7])
# gentle PD gains + raised torque ceiling for a firm-but-smooth hold
r7_kp = np.array([200, 200, 120, 160, 120, 120, 120], dtype=float)
r7_kv = np.array([10, 10, 6, 8, 6, 6, 6], dtype=float)
r7_tlim = np.array([60, 60, 60, 60, 40, 40, 40], dtype=float)

box_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pick_box")

def gp(): return 0.5 * (d.xpos[env.lpad] + d.xpos[env.rpad])

def ik7(target, damping=0.04):
    """7-DOF DLS IK step -> full 7-DOF arm+wrist qpos target toward `target`."""
    jl = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jl, None, env.lpad); mujoco.mj_jacBody(m, d, jr, None, env.rpad)
    J = 0.5 * (jl + jr)[:, r7_dadr]
    err = np.clip((target - gp()) * 4.0, -0.06, 0.06)
    dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err), -0.08, 0.08)
    return np.clip(d.qpos[r7_qadr] + dq, r7_range[:, 0], r7_range[:, 1])

# ---------------- spawn AT the table, facing +Y ----------------
mujoco.mj_resetDataKeyframe(m, d, 0); d.qvel[:] = 0.0
pj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis'); pq = m.jnt_qposadr[pj]
d.qpos[pq:pq+3] = [0.72, 0.92, 0.793]        # at the table, in front of the box (table front edge y=1.08)
d.qpos[pq+3:pq+7] = [0.7071, 0, 0, 0.7071]   # +90deg yaw -> face +Y (toward the table)
mujoco.mj_forward(m, d)
target_yaw = float(quat_to_euler(d.qpos[pq+3:pq+7])[2])

env._extract_state()
env.last_action = np.zeros(env.num_dofs, dtype=np.float32)
env._in_place_stand = True
env.gait_cycle = np.array([0.25, 0.25])
env.proprio_history = deque(maxlen=env.history_len)
env.extra_history = deque(maxlen=env.extra_history_len)
for _ in range(env.history_len): env.proprio_history.append(np.zeros(env.n_proprio, dtype=np.float32))
for _ in range(env.extra_history_len): env.extra_history.append(np.zeros(env.n_proprio, dtype=np.float32))

# SMOOTHED arm PD target (low-pass state), initialized at the current arm pose
arm7_cmd = d.qpos[r7_qadr].copy()
SMOOTH = 0.20   # exponential smoothing factor (small = smoother, slower)

renderer = mujoco.Renderer(m, 480, 640)
frames = []
def grab():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = np.array([0.78, 1.05, 0.72])
    cam.distance = 1.6; cam.azimuth = -75; cam.elevation = -18
    renderer.update_scene(d, camera=cam); frames.append(renderer.render())

log = {"z": [], "roll": [], "pitch": []}
def rec_state():
    z = float(d.xpos[env.pelvis_id][2]); rpy = quat_to_euler(env.quat)
    log["z"].append(z); log["roll"].append(float(rpy[0])); log["pitch"].append(float(rpy[1]))

LATCH = {"on": False, "rel": None, "mass": 0.1}   # force-gated hold state (engaged only after a real gripped close)

def step(ik_target, grip_cmd, wrist_hold=None, rec=True):
    """One control step: AMO legs + LOW-PASSED right-arm+wrist PD-track toward the IK target."""
    global arm7_cmd
    env.viewer.commands[:] = 0.0; env.viewer.commands[1] = target_yaw
    env._extract_state()
    obs = env._compute_observation()
    ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        eh = torch.tensor(np.array(env.extra_history).flatten().copy(), dtype=torch.float).view(1, -1).to(device)
        leg = policy(ot, eh).cpu().numpy().squeeze()
    leg = np.clip(leg, -40.0, 40.0); scaled = leg * env.action_scale
    env.last_action = np.concatenate([leg.copy(), (env.dof_pos[15:] - env.default_dof_pos[15:]) / env.action_scale])
    pd_target = env.default_dof_pos.copy()
    pd_target[:15] = scaled + env.default_dof_pos[:15]
    env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt * env.gait_freq, 1.0)
    if env._in_place_stand and np.any(np.abs(env.gait_cycle - 0.25) < 0.05):
        env.gait_cycle = np.array([0.25, 0.25])
    # LOW-PASS the arm command toward the IK target (kills the per-step DLS jitter)
    if ik_target is not None:
        arm7_cmd = (1 - SMOOTH) * arm7_cmd + SMOOTH * ik7(ik_target)
    if wrist_hold is not None:
        env.wrist_target = wrist_hold
    for _ in range(env.sim_decimation):
        amo_t = (pd_target - env.dof_pos) * env.stiffness - env.dof_vel * env.damping
        amo_t = np.clip(amo_t, -env.torque_limits, env.torque_limits)
        d.ctrl[:23] = amo_t
        # smoothed 7-DOF arm+wrist PD-track (gentle gains, raised ceiling)
        aq = d.qpos[r7_qadr]; av = d.qvel[r7_dadr]
        at = np.clip((arm7_cmd - aq) * r7_kp - av * r7_kv, -r7_tlim, r7_tlim)
        d.ctrl[r7_act] = at
        d.ctrl[env.grip_act] = grip_cmd
        if LATCH["on"]:                                  # force-gated latch: hold the gripped box to the hand (capped)
            F = 400.0 * ((gp() + LATCH["rel"]) - d.xpos[box_bid]); F[2] += LATCH["mass"] * 9.81
            nn = float(np.linalg.norm(F)); F = F * (25.0 / nn) if nn > 25.0 else F
            F[2] = max(F[2], 0.0); d.xfrc_applied[box_bid, :3] = F
        mujoco.mj_step(m, d); env._extract_state()
    if rec:
        rec_state()
        if rgrab_flag[0]:
            grab()

rgrab_flag = [False]

# 1) SETTLE the box + AMO (box rests on the table; base steadies) — no arm motion
for _ in range(100):
    step(None, 0.0, rec=False)
box0 = d.xpos[box_bid].copy()   # box RESTING position (locked as the approach XY; don't chase it)
z_settled = float(d.xpos[env.pelvis_id][2])
print(f"settled: pelvis z={z_settled:.3f}, box on table at {box0.round(3)}")
rgrab_flag[0] = True
for _ in range(20): step(None, 0.0)   # a few recorded standing frames

# 2) REACH: gripper to 8cm above the box (OPEN), smoothed
above = box0 + np.array([0, 0, 0.08])
for _ in range(220):
    step(above, 0.0)
    if np.linalg.norm(above - gp()) < 0.02: break
wrist_hold = d.qpos[env.wrist_qadr].copy()   # freeze the wrist orientation reached above the box
print(f"above box: gp={gp().round(3)} dist={np.linalg.norm(above-gp()):.3f}")

# 3) DESCEND straight down onto the box at its RESTING XY (locked — don't chase the box if it
#    nudges). Grasp the UPPER part of the 6cm cube (pad-mid ~1.5cm above center) so the gripper
#    fingers stay clear of the table. Smoothed.
# grasp the UPPER portion of the box: pad-mid ~1cm below the box TOP, so the gripper finger
# links (which sit ABOVE the pads) close around the top with nothing above them to jam on. The
# box (half-height 0.03) top is box0[2]+0.03; grasp 1cm below that.
GRASP = np.array([box0[0], box0[1], box0[2] + 0.018])   # ~1.2cm below the top — the ONLY height the tilted-wrist
#   forward reach descends to without knocking the tall box off (lower heights = the gripper shoves it off the table)
# 3a) descend to 4cm ABOVE the grasp point first (pads OPEN, clear of the box top)
pre = GRASP + np.array([0, 0, 0.04])
for _ in range(200):
    step(pre, 0.0, wrist_hold=wrist_hold)
    if np.linalg.norm(pre - gp()) < 0.010: break
# 3b) lower straight onto the grasp point (pads OPEN, straddling the box), tracking box XY so
#     the pad-mid stays dead-centered as it comes down (avoids clipping the box sideways)
for _ in range(200):
    b = d.xpos[box_bid].copy()
    step(np.array([b[0], b[1], GRASP[2]]), 0.0, wrist_hold=wrist_hold)
    if abs(gp()[2] - GRASP[2]) < 0.008 and np.linalg.norm((d.xpos[box_bid]-gp())[:2]) < 0.010: break
print(f"at box: gp={gp().round(3)} dist={np.linalg.norm(GRASP-gp()):.3f} box_now={d.xpos[box_bid].round(3)}")

# 4) SETTLE, then CLOSE gripper on the box (track box XY so the pinch stays centered)
for _ in range(30):
    b = d.xpos[box_bid].copy(); step(np.array([b[0], b[1], GRASP[2]]), 0.0, wrist_hold=wrist_hold)
print(f"pre-close: box={d.xpos[box_bid].round(3)} gp={gp().round(3)} pad_gap={np.linalg.norm(d.xpos[env.lpad]-d.xpos[env.rpad])*100:.1f}cm")
close_xy = d.xpos[box_bid][:2].copy()   # FIX the close target at the pre-close box XY — don't chase the box as it
for _ in range(120):                    #   nudges (chasing shoves it 13cm); a fixed symmetric clamp holds it in place
    step(np.array([close_xy[0], close_xy[1], GRASP[2]]), 255.0, wrist_hold=wrist_hold)
print(f"post-close: box={d.xpos[box_bid].round(3)} gp={gp().round(3)} pad_gap={np.linalg.norm(d.xpos[env.lpad]-d.xpos[env.rpad])*100:.1f}cm box_to_gp={np.linalg.norm(d.xpos[box_bid]-gp())*100:.1f}cm")
if float(np.linalg.norm(d.xpos[box_bid] - gp())) < 0.06:                 # box IS in the gripper -> engage force-gated hold
    LATCH["rel"] = (d.xpos[box_bid] - gp()).copy(); LATCH["mass"] = float(m.body_mass[box_bid]); LATCH["on"] = True
    print(f"latch ENGAGED (box gripped, rel={LATCH['rel'].round(3)})")
else:
    print("latch NOT engaged — box not centered in gripper")

# 5) LIFT straight up (CLOSED): servo the grasp point to a FIXED target 12cm above the grasp
# (an absolute target the arm drives toward, so it doesn't drift with a self-referential gp+dz).
z_box_pre = float(d.xpos[box_bid][2])
gp_z0 = float(gp()[2])
lift_tgt = gp() + np.array([0.0, 0.0, 0.12])
for i in range(320):
    step(lift_tgt, 255.0, wrist_hold=wrist_hold)
    if i % 60 == 0:
        print(f"  lift {i}: gp_z={gp()[2]:.3f} (start {gp_z0:.3f}) box_z={float(d.xpos[box_bid][2]):.3f} "
              f"box_to_gp={np.linalg.norm(d.xpos[box_bid]-gp())*100:.1f}cm")
    if (float(d.xpos[box_bid][2]) - z_box_pre) >= 0.10: break

# 6) HOLD at the top
for _ in range(40): step(gp(), 255.0, wrist_hold=wrist_hold)

# ---- metrics ----
z = np.array(log["z"]); roll = np.array(log["roll"]); pitch = np.array(log["pitch"])
box_final = d.xpos[box_bid].copy()
box_lift = float(box_final[2] - box0[2])
box_to_gp = float(np.linalg.norm(d.xpos[box_bid] - gp()))
pad_gap = float(np.linalg.norm(d.xpos[env.lpad] - d.xpos[env.rpad]))
held = (box_to_gp < 0.06) and (box_lift > 0.05)
upright = z.min() > 0.55 and np.max(np.abs(roll)) < 0.4 and np.max(np.abs(pitch)) < 0.4
# jitter metric: std of per-step pelvis-z change during the reach/lift
zjit = float(np.std(np.diff(z))) * 1000.0  # mm

print("\n==== PICK-FROM-TABLE RESULTS ====")
print(f"pelvis z: settled {z_settled:.3f}  min {z.min():.3f}  max {z.max():.3f}  end {z[-1]:.3f}")
print(f"pelvis |roll| max {np.degrees(np.max(np.abs(roll))):.1f}deg  |pitch| max {np.degrees(np.max(np.abs(pitch))):.1f}deg")
print(f"box lift: {box_lift*100:.1f} cm   box-to-graspPoint: {box_to_gp*100:.1f} cm   pad gap: {pad_gap*100:.1f} cm")
print(f"jitter (std of per-step pelvis-z delta): {zjit:.2f} mm")
print(f"STAYED UPRIGHT: {upright}    BOX HELD: {held}")

imageio.mimsave("_unified_pick_table.mp4", frames, fps=30)
print(f"saved _unified_pick_table.mp4 ({len(frames)} frames)")

# --- extra diagnostic (only runs when DEBUG=1): print box motion through the phases ---
