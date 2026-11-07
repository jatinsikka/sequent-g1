"""
G6-lite — ONE CONTINUOUS SIM, ONE TAKE: the G1 WALKS to the workstation table, PICKS the
red box (IK + force-gated latch), CARRIES it while WALKING ~2.5m (180-deg turn) to the
control panel, PLACES the box down clear of the press path (side release, short honest drop),
PRESSES the yellow button ONCE with the RL curriculum policy, then lowers
the arm to REST — box latched in the gripper through the carry. No cuts, no teleports after
the walk starts (the press handoff uses end_to_end_demo's reset + full-snapshot restore).

Model: g1_amo_gripper_pick.mjb (= press model + box) — the ButtonPressEnv is pointed at it
via a UnifiedHumanoidEnv._load_robot_config monkeypatch BEFORE construction, so the press
env, the pick machinery and the RL policy all live in the SAME sim.

Output: _g6_onetake.mp4 + _g6_f0..f3.png (pick / carry / place / press) + printed metrics.
"""
import sys, os
from demo_paths import PROJ, require
sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, torch, mujoco, imageio
from collections import deque
from stable_baselines3 import PPO

from unified_env import UnifiedHumanoidEnv, AMO_JOINTS, WRIST_JOINTS, RARM_JOINTS
from play_amo import quat_to_euler

# ---- point the unified env at the PICK model (press model + box) BEFORE construction ----
_orig_lrc = UnifiedHumanoidEnv._load_robot_config
def _lrc_pick(self, robot_type):
    _orig_lrc(self, robot_type)
    self.model_path = "g1_amo_gripper_pick.mjb"
UnifiedHumanoidEnv._load_robot_config = _lrc_pick

from env_wrapper_button import ButtonPressEnv, GRIP_CLOSED

ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints_button/curr_v2_latest.zip"
require(ckpt, "g1_amo_gripper_pick.mjb")
rl_model = PPO.load(ckpt, device="cpu")
env = ButtonPressEnv(button_name="button_yellow", unified=True, reset_in_contact=False,
                     curriculum=True, headless=True)
e = env.env
m, d = e.model, e.data
assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pick_box") >= 0, "pick model missing box"
device = env.device
policy = e.policy_jit

# ---- (a) reference reset at frac 0: caches the servo contact pose (wrist tilt) ----
env.set_curriculum_frac(0.0)
obs, _ = env.reset(seed=0)
a4_rest = e.default_dof_pos[19:23].copy()
press_target_yaw = env.target_yaw                     # -pi/2 (facing the panel)
pel = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]
pq = pel
press_y = float(d.qpos[pel + 1])                      # the arrival (press) y
press_x = float(env.robot_start_pos[0])               # 0.07
print(f"press stance: ({press_x:.3f},{press_y:.3f}) yaw={press_target_yaw:.2f}")

# ================= PICK MACHINERY (walk_pick_demo.py verbatim, on e's model) =================
def qadr(js): return np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in js])
def dadr(js): return np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in js])
# damp the wrist hold during the pick (walk_pick lesson); restored to trained gains pre-press
e.wrist_kp = np.array([120., 120., 120.]); e.wrist_kv = np.array([8., 8., 8.])

RARM7 = RARM_JOINTS + WRIST_JOINTS
r7_qadr = qadr(RARM7); r7_dadr = dadr(RARM7)
r7_range = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in RARM7])
r7_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in RARM7])
r7_kp = np.array([200, 200, 120, 160, 120, 120, 120], dtype=float)
r7_kv = np.array([10, 10, 6, 8, 6, 6, 6], dtype=float)
r7_tlim = np.array([60, 60, 60, 60, 40, 40, 40], dtype=float)

box_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pick_box")

def gp(): return 0.5 * (d.xpos[e.lpad] + d.xpos[e.rpad])

def ik7(target, damping=0.04):
    jl = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jl, None, e.lpad); mujoco.mj_jacBody(m, d, jr, None, e.rpad)
    J = 0.5 * (jl + jr)[:, r7_dadr]
    err = np.clip((target - gp()) * 4.0, -0.06, 0.06)
    dq = np.clip(J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), err), -0.08, 0.08)
    return np.clip(d.qpos[r7_qadr] + dq, r7_range[:, 0], r7_range[:, 1])

def wrap(a): return (a + np.pi) % (2 * np.pi) - np.pi

# ---- (b) teleport (pre-walk setup) to the pick approach spawn: 0.6m back from the table ----
STANCE = np.array([0.72, 0.92, 0.793])
WALK_BACK = 0.60
STOP_Y = STANCE[1] + 0.02
mujoco.mj_resetDataKeyframe(m, d, 0); d.qvel[:] = 0.0
d.qpos[env.button_joint_id] = 0.0
d.qpos[pq:pq+3] = [STANCE[0], STANCE[1] - WALK_BACK, STANCE[2]]
d.qpos[pq+3:pq+7] = [0.7071, 0, 0, 0.7071]           # +90deg yaw -> face +Y (toward the table)
mujoco.mj_forward(m, d)
table_yaw = float(quat_to_euler(d.qpos[pq+3:pq+7])[2])
e.wrist_target = np.zeros(3)

e._extract_state()
e.last_action = np.zeros(e.num_dofs, dtype=np.float32)
e._in_place_stand = True
e.gait_cycle = np.array([0.25, 0.25])
e.proprio_history = deque(maxlen=e.history_len)
e.extra_history = deque(maxlen=e.extra_history_len)
for _ in range(e.history_len): e.proprio_history.append(np.zeros(e.n_proprio, dtype=np.float32))
for _ in range(e.extra_history_len): e.extra_history.append(np.zeros(e.n_proprio, dtype=np.float32))

arm7_cmd = d.qpos[r7_qadr].copy()
SMOOTH = 0.20

# ================= camera: follow cam (pick) blending to the press view =================
renderer = mujoco.Renderer(m, 480, 640)
frames = []
Y0_CARRY = STANCE[1]
def cam_t():
    """0 at the table, 1 at the panel — drives the follow->press camera blend."""
    y = float(d.xpos[e.pelvis_id][1])
    return float(np.clip((Y0_CARRY - y) / (Y0_CARRY - press_y), 0.0, 1.0))
def grab(force_t=None):
    p = d.xpos[e.pelvis_id]
    t = cam_t() if force_t is None else force_t
    follow_look = np.array([p[0] + 0.06, min(p[1] + 0.13, 1.05), 0.72])
    press_look = np.array([0.15, -1.65, 0.85])
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (1 - t) * follow_look + t * press_look
    cam.distance = 1.9 + t * (1.6 - 1.9)
    cam.azimuth = -75 + t * 75
    cam.elevation = -18 + t * 3
    renderer.update_scene(d, camera=cam); frames.append(renderer.render())

log = {"z": [], "roll": [], "pitch": []}
def rec_state():
    z = float(d.xpos[e.pelvis_id][2]); rpy = quat_to_euler(e.quat)
    log["z"].append(z); log["roll"].append(float(rpy[0])); log["pitch"].append(float(rpy[1]))

LATCH = {"on": False, "rel_local": None, "mass": 0.1, "cap": 25.0, "dropped": False}

def pad_R():
    return d.xmat[e.lpad].reshape(3, 3)

def latch_engage():
    """(Re-)anchor the latch offset in the GRIPPER (pad) frame — a world-frame rel becomes
    wrong the moment the wrist reorients (run 1: it ripped the box out at the handoff).
    The anchor is CLAMPED to 1.5cm from the pad midpoint so the latch pulls the box INTO
    the pinch (run 2: anchoring at the already-slid-out 2.6cm offset just held it at the
    pinch margin, and the RL reach dynamics slid it the rest of the way out)."""
    rel = pad_R().T @ (d.xpos[box_bid] - gp())
    n = float(np.linalg.norm(rel))
    if n > 0.015:
        rel *= 0.015 / n
    LATCH["rel_local"] = rel
    LATCH["on"] = True

def latch_force():
    """Force-gated latch: hold the gripped box to the hand (capped), offset tracked in the
    pad frame. DROP-GATE: if the box has clearly left the gripper, disengage permanently
    instead of catapulting a 40g box with a capped-25N force (run 1's QACC NaN). Called
    every mj_step inside step(); once per CONTROL step around env.step/_amo_arm_step
    (xfrc persists across the 10 substeps)."""
    if not LATCH["on"]:
        return
    dist = float(np.linalg.norm(d.xpos[box_bid] - gp()))
    if dist > 0.12:
        LATCH["on"] = False; LATCH["dropped"] = True
        d.xfrc_applied[box_bid, :] = 0.0
        print(f"LATCH DISENGAGED — box left the gripper (dist {dist*100:.1f}cm)")
        return
    F = 400.0 * ((gp() + pad_R() @ LATCH["rel_local"]) - d.xpos[box_bid]); F[2] += LATCH["mass"] * 9.81
    nn = float(np.linalg.norm(F)); F = F * (LATCH["cap"] / nn) if nn > LATCH["cap"] else F
    F[2] = max(F[2], 0.0); d.xfrc_applied[box_bid, :3] = F

def step(ik_target, grip_cmd, wrist_hold=None, rec=True, vx=0.0, yaw=None):
    """One control step: AMO legs (vx + absolute yaw setpoint) + LOW-PASSED 7-DOF arm PD."""
    global arm7_cmd
    e.viewer.commands[:] = 0.0
    e.viewer.commands[0] = vx
    e.viewer.commands[1] = table_yaw if yaw is None else yaw
    e._extract_state()
    obs = e._compute_observation()
    ot = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    with torch.no_grad():
        eh = torch.tensor(np.array(e.extra_history).flatten().copy(), dtype=torch.float).view(1, -1).to(device)
        leg = policy(ot, eh).cpu().numpy().squeeze()
    leg = np.clip(leg, -40.0, 40.0); scaled = leg * e.action_scale
    e.last_action = np.concatenate([leg.copy(), (e.dof_pos[15:] - e.default_dof_pos[15:]) / e.action_scale])
    pd_target = e.default_dof_pos.copy()
    pd_target[:15] = scaled + e.default_dof_pos[:15]
    e.gait_cycle = np.remainder(e.gait_cycle + e.control_dt * e.gait_freq, 1.0)
    if e._in_place_stand and np.any(np.abs(e.gait_cycle - 0.25) < 0.05):
        e.gait_cycle = np.array([0.25, 0.25])
    if not e._in_place_stand and np.all(np.abs(e.gait_cycle - 0.25) < 0.05):
        e.gait_cycle = np.array([0.25, 0.75])
    if ik_target is not None:
        arm7_cmd = (1 - SMOOTH) * arm7_cmd + SMOOTH * ik7(ik_target)
    if wrist_hold is not None:
        e.wrist_target = wrist_hold
    for _ in range(e.sim_decimation):
        amo_t = (pd_target - e.dof_pos) * e.stiffness - e.dof_vel * e.damping
        amo_t = np.clip(amo_t, -e.torque_limits, e.torque_limits)
        d.ctrl[:23] = amo_t
        aq = d.qpos[r7_qadr]; av = d.qvel[r7_dadr]
        at = np.clip((arm7_cmd - aq) * r7_kp - av * r7_kv, -r7_tlim, r7_tlim)
        d.ctrl[r7_act] = at
        d.ctrl[e.grip_act] = grip_cmd
        if LATCH["on"]:
            latch_force()
        mujoco.mj_step(m, d); e._extract_state()
    if rec:
        rec_state()
        if rgrab_flag[0]:
            grab()

rgrab_flag = [False]
snap = {}
carry_b2g = []   # box-to-gripper distance samples during carry + press

# ============================== PHASE 1: WALK -> PICK (Gate 4 verbatim) ==============================
for _ in range(30):
    step(None, 0.0, rec=False)
rgrab_flag[0] = True

walk_steps = 0
for t in range(500):
    if float(d.qpos[pq + 1]) >= STOP_Y: break
    step(None, 0.0, rec=(t % 2 == 0), vx=0.30)
    walk_steps += 1
pxy = d.xpos[e.pelvis_id]
print(f"after walk ({walk_steps} steps): pelvis xy=({pxy[0]:.3f},{pxy[1]:.3f}) | stance xy=({STANCE[0]:.3f},{STANCE[1]:.3f})")

for _ in range(70):
    step(None, 0.0)
box0 = d.xpos[box_bid].copy()
pxy = d.xpos[e.pelvis_id]
print(f"settled: pelvis=({pxy[0]:.3f},{pxy[1]:.3f},{pxy[2]:.3f}), box at {box0.round(3)}")

above = box0 + np.array([0, 0, 0.08])
for _ in range(220):
    step(above, 0.0)
    if np.linalg.norm(above - gp()) < 0.02: break
wrist_hold = d.qpos[e.wrist_qadr].copy()
print(f"above box: gp={gp().round(3)} dist={np.linalg.norm(above-gp()):.3f}")

GRASP = np.array([box0[0], box0[1], box0[2] + 0.018])
pre = GRASP + np.array([0, 0, 0.04])
for _ in range(200):
    step(pre, 0.0, wrist_hold=wrist_hold)
    if np.linalg.norm(pre - gp()) < 0.010: break
for _ in range(200):
    b = d.xpos[box_bid].copy()
    step(np.array([b[0], b[1], GRASP[2]]), 0.0, wrist_hold=wrist_hold)
    if abs(gp()[2] - GRASP[2]) < 0.008 and np.linalg.norm((d.xpos[box_bid]-gp())[:2]) < 0.010: break
print(f"at box: gp={gp().round(3)} dist={np.linalg.norm(GRASP-gp()):.3f} box_now={d.xpos[box_bid].round(3)}")

for _ in range(30):
    b = d.xpos[box_bid].copy(); step(np.array([b[0], b[1], GRASP[2]]), 0.0, wrist_hold=wrist_hold)
close_xy = d.xpos[box_bid][:2].copy()
for _ in range(120):
    step(np.array([close_xy[0], close_xy[1], GRASP[2]]), 255.0, wrist_hold=wrist_hold)
print(f"post-close: box={d.xpos[box_bid].round(3)} gp={gp().round(3)} box_to_gp={np.linalg.norm(d.xpos[box_bid]-gp())*100:.1f}cm")
if float(np.linalg.norm(d.xpos[box_bid] - gp())) < 0.06:
    LATCH["mass"] = float(m.body_mass[box_bid]); latch_engage()
    print(f"latch ENGAGED (rel_local={LATCH['rel_local'].round(3)})")
else:
    print("latch NOT engaged — box not centered in gripper")

z_box_pre = float(d.xpos[box_bid][2])
lift_tgt = gp() + np.array([0.0, 0.0, 0.12])
for i in range(320):
    step(lift_tgt, 255.0, wrist_hold=wrist_hold)
    if (float(d.xpos[box_bid][2]) - z_box_pre) >= 0.10: break
for _ in range(60): step(lift_tgt, 255.0, wrist_hold=wrist_hold)
snap["pick"] = len(frames) - 1
box_lift_at_pick = float(d.xpos[box_bid][2] - box0[2])
print(f"PICK done: lift={box_lift_at_pick*100:.1f}cm box_to_gp={np.linalg.norm(d.xpos[box_bid]-gp())*100:.1f}cm")

# ============================== PHASE 2: CARRY — turn 180deg, walk to the panel ==============================
if not LATCH["on"]:
    print("FATAL: no latch — carrying would drop the box. Aborting after pick.")
    imageio.mimsave("_g6_onetake.mp4", frames, fps=30, quality=8); sys.exit(1)

# re-anchor the latch in the pad frame + stronger cap for the dynamic carry
latch_engage(); LATCH["cap"] = 40.0
e._in_place_stand = False
carry_fall = False

# 2a) TURN 180deg WHILE WALKING SLOWLY: AMO cannot turn in place (measured: yaw pinned at
# vx=0 for any ramp rate), but tracks a 30deg/s yaw ramp at vx=0.10 with ~5deg residual and
# ~0.15m arc drift. vx is gated off if the arc carries the pelvis toward the table edge.
TURN_RATE = np.radians(30.0) * e.control_dt            # 30 deg/s, per control step
N_TURN = int(np.pi / TURN_RATE)                        # ~300 steps for 180deg
for i in range(N_TURN):
    ycmd = np.pi/2 - (i + 1) * TURN_RATE               # clockwise, through 0 (facing +X)
    vturn = 0.10 if float(d.qpos[pq + 1]) < 1.00 else 0.0   # don't advance into the table
    step(None, 255.0, wrist_hold=wrist_hold, vx=vturn, yaw=ycmd, rec=(i % 2 == 0))
    carry_b2g.append(float(np.linalg.norm(d.xpos[box_bid] - gp())))
    if float(d.xpos[e.pelvis_id][2]) < 0.5:
        carry_fall = True; print(f"FELL during turn at i={i}"); break
cy = float(quat_to_euler(d.qpos[pq+3:pq+7])[2])
print(f"turn done: yaw={np.degrees(cy):.1f}deg (target -90) pelvis z={d.xpos[e.pelvis_id][2]:.3f} "
      f"box_to_gp={carry_b2g[-1]*100:.1f}cm")

# 2b) WALK to the press stance in TWO LEGS with a CLOSED-LOOP yaw servo. Run 3 measured an
# ~18deg systematic yaw tracking bias while walking (commanded heading never reached), so
# open-loop heading commands cannot close lateral error. The servo commands
#   yaw_cmd = actual_yaw + clip(err, +-0.35) + integrator(bias)
# — bounded lead the policy can follow, integrator kills the standing bias.
#   leg 1: head for a waypoint 1.0m up-track of the stance (close x early)
#   leg 2: cross-track law — heading = -90deg minus a term ~ x error; the correction clip
#          SHRINKS near arrival so it arrives square.
yaw_ib = 0.0
def steer(des):
    global yaw_ib
    ay = float(quat_to_euler(d.qpos[pq+3:pq+7])[2])
    err = wrap(des - ay)
    yaw_ib = float(np.clip(yaw_ib + 0.004 * err, -0.4, 0.4))
    return ay + float(np.clip(err, -0.35, 0.35)) + yaw_ib

carry_walk_steps = 0
mid_y = 0.5 * (STANCE[1] + press_y)
# TWO PURE-PURSUIT waypoints (run 4: a shrinking cross-track clip floored the lateral
# correction at the stop line — robot pinned at y=-1.62 with x 19cm off for 1800 steps):
#   wp1 = (press_x, 0.0): get ON the panel track while still 1.5m out
#   aim2 = 0.30m past the stance: the descent leg converges x geometrically as it closes
wp1 = np.array([press_x, 0.0])
aim2 = np.array([press_x, press_y - 0.30])
# NOTE (run 6, reverted): a fixed-clip cross-track final approach with slower vx made arrival
# WORSE (x err 6->21cm) — AMO drifts +X at ~3-4cm/s while descending with the box payload on
# the right arm (AMO does not compensate external payload), and the clipped correction
# saturated against that drift. Pure pursuit of aim2 (this code) is the best measured: run 5
# arrived x err 6.0cm / yaw 10.2deg.
leg = 1
stall = 0
for t in range(2500):
    if carry_fall: break
    x, y = float(d.qpos[pq]), float(d.qpos[pq + 1])
    if y <= press_y - 0.035 and (abs(x - press_x) < 0.06 or y <= press_y - 0.10): break
    if y <= press_y - 0.035:
        stall += 1
        if stall > 250:
            print(f"  stall break at ({x:.3f},{y:.3f}) — accepting this stance"); break
    if float(d.xpos[e.pelvis_id][2]) < 0.5:
        carry_fall = True; print(f"FELL during carry walk at t={t}"); break
    if leg == 1 and (np.hypot(x - wp1[0], y - wp1[1]) < 0.12 or y < wp1[1] - 0.05):
        leg = 2
        print(f"  leg 2 (descent) from ({x:.3f},{y:.3f}) x_err={abs(x-press_x)*100:.1f}cm")
    tgt = wp1 if leg == 1 else aim2
    des = np.arctan2(tgt[1] - y, tgt[0] - x)
    vx = 0.30 if (y - press_y) > 0.35 else 0.18
    step(None, 255.0, wrist_hold=wrist_hold, vx=vx, yaw=steer(des), rec=(t % 2 == 0))
    carry_b2g.append(float(np.linalg.norm(d.xpos[box_bid] - gp())))
    carry_walk_steps += 1
    if t % 200 == 0:
        print(f"  carry t{t}: ({x:.3f},{y:.3f}) yaw={np.degrees(float(quat_to_euler(d.qpos[pq+3:pq+7])[2])):.0f} "
              f"des={np.degrees(des):.0f} ib={np.degrees(yaw_ib):.0f}")
    if "carry" not in snap and y <= mid_y:
        snap["carry"] = len(frames) - 1
if "carry" not in snap: snap["carry"] = len(frames) - 1
pxy = d.xpos[e.pelvis_id]; cy = float(quat_to_euler(d.qpos[pq+3:pq+7])[2])
arr_xerr = abs(pxy[0] - press_x); arr_yawerr = abs(wrap(cy - press_target_yaw))
print(f"carry walk done ({carry_walk_steps} steps): pelvis=({pxy[0]:.3f},{pxy[1]:.3f}) yaw={np.degrees(cy):.1f}deg "
      f"| press stance=({press_x:.3f},{press_y:.3f}) yaw=-90deg "
      f"| arrival err: x={arr_xerr*100:.1f}cm yaw={np.degrees(arr_yawerr):.1f}deg "
      f"box_to_gp={carry_b2g[-1]*100:.1f}cm")
if arr_xerr > 0.10 or arr_yawerr > 0.21:
    print("WARNING: arrival outside the trained stance envelope (+-4cm/+-5deg) — press may miss")

# 2c) arrival settle (standing); keep the 40N cap — the press reach dynamics also need the
# stronger hold (run 2: box slid out 40 steps into the RL reach)
e._in_place_stand = True
for i in range(50):
    step(None, 255.0, wrist_hold=wrist_hold, vx=0.0, yaw=press_target_yaw, rec=(i % 2 == 0))
    carry_b2g.append(float(np.linalg.norm(d.xpos[box_bid] - gp())))

if carry_fall:
    imageio.imwrite("_g6_fail_carry.png", frames[-1])
    imageio.mimsave("_g6_onetake.mp4", frames, fps=30, quality=8)
    print("ABORT: fell during carry — saved _g6_fail_carry.png + partial video"); sys.exit(1)

# 2d) PLACE the box down before pressing (a human doesn't press with a full hand — and 'place'
# is one of the 7 skills). Release at the robot's SIDE (+X, y >= pelvis_y — AWAY from the
# machine face; run 7 aimed pelvis+[0.28,-0.18] and shoved the box INTO the panel), from
# whatever height the arm comfortably reaches — a short drop to the floor is honest, the arm
# cannot reach the floor from standing. CRITICAL: zero d.xfrc_applied on release — run 7's
# box hung "wedged" at z=0.58 because the last capped-40N latch force kept being applied on
# every mj_step after LATCH["on"]=False (latch_force only clears xfrc in its drop-gate), and
# the pinned box sat in the press path (press whiffed 0.0mm).
def stance_report(tag):
    p = d.xpos[e.pelvis_id]; yy = float(quat_to_euler(d.qpos[pq+3:pq+7])[2])
    print(f"  [{tag}] pelvis=({p[0]:.3f},{p[1]:.3f}) yaw={np.degrees(yy):.1f} "
          f"(stance err x={(p[0]-press_x)*100:+.1f}cm y={(p[1]-press_y)*100:+.1f}cm "
          f"yaw={np.degrees(abs(wrap(yy-press_target_yaw))):.1f}deg)")

stance_report("place start")
pelv = d.xpos[e.pelvis_id].copy()
side_xy = np.array([pelv[0] + 0.44, pelv[1] + 0.04])
Z_HI, Z_LO = 0.60, 0.42
for i in range(70):
    frac_i = min(1.0, i / 60.0)
    tgt = np.array([side_xy[0], side_xy[1], Z_HI - (Z_HI - Z_LO) * frac_i])
    step(tgt, 255.0, wrist_hold=wrist_hold, vx=0.0, yaw=press_target_yaw, rec=(i % 2 == 0))
    carry_b2g.append(float(np.linalg.norm(d.xpos[box_bid] - gp())))
place_idx = len(carry_b2g)          # b2g samples up to here = box latched (carry + lower)
LATCH["on"] = False
d.xfrc_applied[box_bid, :] = 0.0    # kill the residual latch force (persists across mj_step)
for i in range(40):                 # open + hold while the box drops clear
    step(np.array([side_xy[0], side_xy[1], Z_LO]), 0.0, wrist_hold=wrist_hold, vx=0.0,
         yaw=press_target_yaw, rec=(i % 2 == 0))
stance_report("after release")
snap["place"] = len(frames) - 1     # box just released, resting on the floor beside the robot
for i in range(40):                 # retract up/inward to a carry-like pose for the handoff
    step(np.array([pelv[0] + 0.10, pelv[1] - 0.15, 0.75]), 0.0, wrist_hold=wrist_hold, vx=0.0,
         yaw=press_target_yaw, rec=(i % 2 == 0))
stance_report("after retract")
# STANCE RECOVERY: the place lean creeps the pelvis toward the panel (run 8: y err -5.2cm ->
# -8.9cm, outside the policy's +-8cm trained stance noise -> the press flailed 15-30cm from the
# cap, 0.0mm). Walk BACKWARD a few steps to restore the y the working no-place run pressed from.
retract_tgt = np.array([pelv[0] + 0.10, pelv[1] - 0.15, 0.75])
if float(d.qpos[pq + 1]) < press_y - 0.030:
    e._in_place_stand = False
    for i in range(160):
        if float(d.qpos[pq + 1]) >= press_y - 0.020: break
        step(retract_tgt, 0.0, wrist_hold=wrist_hold, vx=-0.12, yaw=press_target_yaw, rec=(i % 2 == 0))
    e._in_place_stand = True
    for i in range(40):             # re-settle standing
        step(retract_tgt, 0.0, wrist_hold=wrist_hold, vx=0.0, yaw=press_target_yaw, rec=(i % 2 == 0))
    stance_report("after recovery")
box_jadr = m.jnt_dofadr[m.body_jntadr[box_bid]]
bp = d.xpos[box_bid]
print(f"placed: box at {bp.round(3)} | on floor={bp[2] < 0.15} "
      f"speed={np.linalg.norm(d.qvel[box_jadr:box_jadr+3]):.3f}m/s")

# ============================== PHASE 3: RL PRESS HANDOFF (end_to_end_demo verbatim) ==============================
e.wrist_kp = np.array([120., 120., 120.]); e.wrist_kv = np.array([4., 4., 4.])   # trained wrist gains
cur = d.qpos[pel:pel+7].copy()
env.robot_start_pos = np.array([cur[0], cur[1], env.robot_start_pos[2]])
env.robot_start_quat = cur[3:7].copy()
env.stance_noise_k = 0.0
env.set_curriculum_frac(1.0); env.set_curriculum_frac_min(1.0)
full_qpos = d.qpos.copy(); full_qvel = d.qvel.copy()   # the EXACT walked-in world+robot state
pel_before = d.xpos[e.pelvis_id].copy()
obs, _ = env.reset()                                    # re-init controller internals only
press_wrist2 = env._solved_wrist.copy()
d.qpos[:] = full_qpos; d.qvel[:] = full_qvel            # restore EVERYTHING (no pose snap, no scene twitch)
mujoco.mj_forward(m, d); e._extract_state()
pel_after = d.xpos[e.pelvis_id].copy()
print(f"handoff continuity: pelvis moved {np.linalg.norm(pel_after-pel_before)*1000:.2f}mm across the reset")

# on-camera settle: arm gathers from the carry pose into the press-ready rest pose (box latched)
for i in range(70):
    latch_force()
    env._amo_arm_step(e.default_dof_pos[19:23], wrist_target=press_wrist2)
    if i % 2 == 0:
        rec_state(); grab(force_t=1.0)
    carry_b2g.append(float(np.linalg.norm(d.xpos[box_bid] - gp())))
if LATCH["on"]:
    latch_engage()                                      # re-anchor for the press-arm pose
cap_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, env.cap_geom_name)
stance_report("post-settle")
print(f"  [post-settle] gripper-cap {np.linalg.norm(env._right_hand_pos() - d.geom_xpos[cap_gid])*100:.1f}cm "
      f"wrist_q={d.qpos[e.wrist_qadr].round(3)} press_wrist2={press_wrist2.round(3)}")

env._filt_arm[:] = 0; env._prev_action[:] = 0; env.episode_steps = 0
env.reward_fn.reset(); env._held_steps = 0; env._was_deep = False
env.initial_button_displacement = d.qpos[env.button_joint_id]
obs = env._get_obs()
maxpress = 0.0; dips = 0; was_deep = False; terminated = False
env.max_episode_steps = 400
press_frame_disp = -1.0
for t in range(400):
    latch_force()
    a, _ = rl_model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(a)
    _d = env._get_button_displacement(); maxpress = max(maxpress, _d)
    if _d > 0.02: was_deep = True
    if was_deep and _d < 0.010: dips += 1; was_deep = False
    rec_state(); grab(force_t=1.0)
    if _d > press_frame_disp:
        press_frame_disp = _d; snap["press"] = len(frames) - 1
    carry_b2g.append(float(np.linalg.norm(d.xpos[box_bid] - gp())))
    if t % 40 == 0:
        gpp = env._right_hand_pos(); cap = d.geom_xpos[env._cap_gid]
        print(f"  press t{t}: gripper-cap {np.linalg.norm(gpp - cap)*100:4.1f}cm  disp {_d*1000:4.1f}mm "
              f"box_to_gp {carry_b2g[-1]*100:.1f}cm")
    if term or trunc:
        terminated = bool(term); break

# ============================== PHASE 4: scripted DISENGAGE to rest ==============================
for i in range(55):
    latch_force()
    env._amo_arm_step(a4_rest, wrist_target=np.zeros(3))
    if i % 2 == 0:
        rec_state(); grab(force_t=1.0)
    carry_b2g.append(float(np.linalg.norm(d.xpos[box_bid] - gp())))
snap["rest"] = len(frames) - 1

# ============================== METRICS ==============================
z = np.array(log["z"]); roll = np.array(log["roll"]); pitch = np.array(log["pitch"])
b2g = np.array(carry_b2g)
upright = z.min() > 0.6 and np.max(np.abs(roll)) < 0.5 and np.max(np.abs(pitch)) < 0.5
held_carry = (not LATCH["dropped"]) and b2g[:place_idx].max() < 0.10
box_end = d.xpos[box_bid].copy()
box_speed = float(np.linalg.norm(d.qvel[box_jadr:box_jadr+3]))
btn_xy = d.geom_xpos[cap_gid][:2]
box_btn_dist = float(np.linalg.norm(box_end[:2] - btn_xy))
placed_ok = box_end[2] < 0.15 and box_btn_dist > 0.25 and box_speed < 0.05
press_ok = maxpress >= 0.020 and dips == 0 and terminated

print("\n==== G6 ONE-TAKE RESULTS ====")
print(f"pick lift: {box_lift_at_pick*100:.1f} cm  (dropped early: {LATCH['dropped']})")
print(f"box-to-gripper max over carry+lower: {b2g[:place_idx].max()*100:.1f}cm")
print(f"box placed at {box_end.round(3)}  dist-to-button={box_btn_dist*100:.1f}cm  speed={box_speed:.3f}m/s")
print(f"press: MAX={maxpress*1000:.1f}mm  PUMPS={dips}  terminated(press-once)={terminated}")
print(f"pelvis z: min={z.min():.3f}  |roll|max={np.degrees(np.max(np.abs(roll))):.1f}deg "
      f"|pitch|max={np.degrees(np.max(np.abs(pitch))):.1f}deg")
print(f"PASS box HELD thru carry : {held_carry}")
print(f"PASS box PLACED (resting z<0.15, >25cm from button): {placed_ok}")
print(f"PASS press>=20mm PUMPS=0 : {press_ok}")
print(f"PASS upright (z>0.6)     : {upright}")
print(f"frames: {len(frames)}")

imageio.mimsave("_g6_onetake.mp4", frames, fps=30, quality=8)
print(f"saved _g6_onetake.mp4 ({len(frames)} frames)")
for name, key in (("_g6_f0.png", "pick"), ("_g6_f1.png", "carry"), ("_g6_f2.png", "place"), ("_g6_f3.png", "press")):
    if key in snap:
        imageio.imwrite(name, frames[snap[key]])
        print(f"saved {name} (frame {snap[key]})")
