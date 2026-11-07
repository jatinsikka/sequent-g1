"""GATE 7 v0 — the FULL loop: incident TICKET -> BRAIN (SOP retrieval) -> PLAN -> SIM EXECUTION.

  1. INPUT   an incident ticket string (default: INC-00073 from the brain's held-out test set)
  2. BRAIN   subprocess into brain/benchmark/brain_query.py (SEQUENT_BRAIN_DIR) -> JSON
             (fine-tuned bi-encoder, R@1 0.577 over 5 seeds): top-5 SOPs + the top SOP's steps
  3. PLAN    map SOP 'skill entity' steps to sim skills:
               walk_to X      -> WALK phase (AMO locomotion)
               press_button X -> PRESS phase (RL curriculum policy)
               pick X         -> PICK phase (v0: not chained; captioned)
               read_sensor / notify / wait -> caption only
               anything else  -> SKIPPED (skill not embodied)
  4. EXECUTE v0: when the plan contains walk_to + press_button, run the proven walk->press
             chain (same seeds/stances as end_to_end_demo.py, checkpoints_button/curr_v2_latest.zip)
  5. RENDER  _g7_ticket.mp4 — title cards (ticket / retrieval / plan) + captioned execution

Usage: python ticket_demo.py ["custom ticket text"]
"""
import sys, os, json, subprocess, textwrap, time

from demo_paths import PROJ, BRAIN_DIR, BRAIN_PY, BRAIN_QUERY, require
CKPT = "checkpoints_button/curr_v2_latest.zip"
sys.path.insert(0, PROJ); os.chdir(PROJ)
require(CKPT)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 640, 480, 30

# The demo ticket: INC-00073 (held-out test incident, verified to retrieve its gold SOP-0031 at rank 1)
DEFAULT_TICKET = ("Sparks came out of the charging socket on the dock by the trim line — not just a "
                  "small crackle, a real discharge. The panel lit up red and the floor around the dock "
                  "smelled like something shorted badly. The docked vehicle still had its fault lights "
                  "going when maintenance arrived.")

# ---------------------------------------------------------------- rendering helpers
def _font(sz, bold=False):
    try:
        return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf", sz)
    except Exception:
        return ImageFont.load_default()

F_TITLE, F_HEAD, F_BODY, F_SMALL = _font(26, True), _font(18, True), _font(15), _font(13)

def title_card(lines, seconds=3.0):
    """lines: list of (text, font, color). Returns `seconds` worth of identical frames."""
    img = Image.new("RGB", (W, H), (18, 22, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 6], fill=(230, 179, 37))          # sequent gold strip
    y = 34
    for text, font, color in lines:
        for ln in (textwrap.wrap(text, width=int(W / (font.size * 0.52))) or [""]):
            d.text((30, y), ln, font=font, fill=color)
            y += int(font.size * 1.45)
        y += 6
    fr = np.asarray(img)
    return [fr] * int(seconds * FPS)

def overlay(frame, top_line, bottom_line=None):
    """Draw the running step banner onto a sim frame."""
    img = Image.fromarray(frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, 26], fill=(18, 22, 30, 215))
    d.text((8, 5), top_line, font=F_SMALL, fill=(230, 179, 37))
    if bottom_line:
        d.rectangle([0, H - 30, W, H], fill=(18, 22, 30, 215))
        d.text((8, H - 24), bottom_line, font=F_SMALL, fill=(235, 235, 235))
    return np.asarray(img)

def caption_card(base_frame, step_no, n_steps, step_text, status, seconds=1.6):
    """Freeze-frame with a big caption bar — for steps the sim narrates instead of embodies."""
    img = Image.fromarray(base_frame).convert("RGB")
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, H - 84, W, H], fill=(18, 22, 30, 235))
    d.text((14, H - 76), f"STEP {step_no}/{n_steps}: {step_text}", font=F_HEAD, fill=(230, 179, 37))
    d.text((14, H - 48), status, font=F_BODY, fill=(235, 235, 235))
    fr = np.asarray(img)
    return [fr] * int(seconds * FPS)

def vts(frames):
    return f"{len(frames)/FPS:6.1f}s"

# ---------------------------------------------------------------- 1. TICKET
ticket = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TICKET
print("=" * 78)
print("GATE 7 v0 — TICKET -> BRAIN -> PLAN -> EXECUTION")
print("=" * 78)
print(f"[TICKET] {ticket}\n")

# ---------------------------------------------------------------- 2. BRAIN (retrieval)
print("[BRAIN] querying sop_planner_baseline retriever (bi-encoder subprocess)...")
t0 = time.time()
r = subprocess.run([BRAIN_PY, BRAIN_QUERY, ticket, "5"], capture_output=True, text=True,
                   encoding="utf-8", cwd=BRAIN_DIR, timeout=600)
if r.returncode != 0:
    print(r.stderr[-2000:]); sys.exit(f"brain_query failed (rc={r.returncode})")
brain = json.loads(r.stdout)
print(f"[BRAIN] retrieval done in {time.time()-t0:.1f}s — top-5:")
for i, h in enumerate(brain["top5"], 1):
    print(f"    {i}. {h['sop_id']}  score={h['score']:.4f}  {h['title']}")
sop = brain["top_sop"]
print(f"[BRAIN] selected SOP: {sop['sop_id']} — {sop['title']}")
for s in sop["steps"]:
    print(f"          - {s}")
print()

# ---------------------------------------------------------------- 3. PLAN (skill mapping)
EMBODIED = {"walk_to": "WALK phase (AMO locomotion)",
            "press_button": "PRESS phase (RL curriculum policy)",
            "pick": "PICK phase (IK grasp)"}
CAPTION_ONLY = {"read_sensor", "notify", "wait"}

plan = []   # (step_text, verb, entity, mode)  mode: EXECUTE | CAPTION | SKIPPED
for st in sop["steps"]:
    parts = st.split()
    verb, entity = parts[0], " ".join(parts[1:])
    if verb in EMBODIED:
        plan.append((st, verb, entity, "EXECUTE"))
    elif verb in CAPTION_ONLY:
        plan.append((st, verb, entity, "CAPTION"))
    else:
        plan.append((st, verb, entity, "SKIPPED"))

verbs = [p[1] for p in plan]
print("[PLAN] SOP steps -> sim skills:")
for i, (st, verb, entity, mode) in enumerate(plan, 1):
    label = {"EXECUTE": EMBODIED.get(verb, ""), "CAPTION": "caption only",
             "SKIPPED": "SKIPPED (skill not embodied)"}[mode]
    print(f"    {i}. {st:<44s} -> {mode:<8s} {label}")
if not ("walk_to" in verbs and "press_button" in verbs):
    sys.exit("[PLAN] v0 executes only walk_to+press_button plans — this SOP has neither; aborting.")
print()

# ---------------------------------------------------------------- title cards
frames = []
frames += title_card([
    ("SEQUENT — GATE 7: TICKET -> SOP -> PLAN -> ROBOT", F_TITLE, (230, 179, 37)),
    ("INCIDENT TICKET", F_HEAD, (235, 235, 235)),
    (f"“{ticket}”", F_BODY, (200, 205, 215)),
], seconds=4.5)
frames += title_card(
    [("BRAIN: SOP RETRIEVAL (trained bi-encoder, 1012 SOPs)", F_TITLE, (230, 179, 37))] +
    [(f"{i}. {h['sop_id']}   {h['score']:.3f}   {h['title']}", F_BODY,
      (120, 220, 120) if i == 1 else (200, 205, 215)) for i, h in enumerate(brain["top5"], 1)] +
    [(f"SELECTED: {sop['sop_id']} — {sop['title']}", F_HEAD, (120, 220, 120))], seconds=4.0)
frames += title_card(
    [("PLAN: SOP STEPS -> SIM SKILLS", F_TITLE, (230, 179, 37))] +
    [(f"{i}. {st}   ->   " + {"EXECUTE": EMBODIED.get(v, ""), "CAPTION": "caption",
                              "SKIPPED": "SKIPPED (not embodied)"}[m], F_BODY,
      (120, 220, 120) if m == "EXECUTE" else (200, 205, 215))
     for i, (st, v, e, m) in enumerate(plan, 1)], seconds=4.0)

# ---------------------------------------------------------------- 4. EXECUTE (walk -> press)
# The chain below is end_to_end_demo.py's proven recipe verbatim (seed 0, +0.60y walk-in,
# handoff reset pinned at the walked-in base, frac 1.0 rest-pose reach). DO NOT tweak.
print("[EXEC] loading sim (walk -> press chain, ckpt " + CKPT + ")...")
import torch, mujoco, imageio
from stable_baselines3 import PPO
from env_wrapper_button import ButtonPressEnv, GRIP_CLOSED

model = PPO.load(CKPT, device="cpu")
env = ButtonPressEnv(button_name="button_yellow", unified=True, reset_in_contact=False,
                     curriculum=True, headless=True)
env.set_curriculum_frac(0.0)
obs, _ = env.reset(seed=0)
e = env.env
dev = env.device
a4_rest = e.default_dof_pos[19:23].copy()
target_yaw = env.target_yaw

pel = e.model.jnt_qposadr[mujoco.mj_name2id(e.model, mujoco.mjtObj.mjOBJ_JOINT, 'pelvis')]
press_y = float(e.data.qpos[pel + 1])
e.data.qpos[pel + 1] += 0.60
e.data.qpos[pel] += 0.02
e.data.qvel[:] = 0.0
mujoco.mj_forward(e.model, e.data)

def shot():
    return env.render_frame(W, H)

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

n_steps = len(plan)
i_walk = verbs.index("walk_to") + 1              # 1-based step numbers for the banner
i_press = verbs.index("press_button") + 1
step_walk = plan[i_walk - 1][0]
step_press = plan[i_press - 1][0]
ticker = f"TICKET {sop['sop_id']}: {sop['title']}"

# --- WALK phase
print(f"[EXEC] [video {vts(frames)}] STEP {i_walk}/{n_steps}: {step_walk}  -> WALK phase")
e._in_place_stand = False
for _ in range(8): amo_step(0.0, a4_rest, np.zeros(3))
for t in range(400):
    if float(e.data.qpos[pel + 1]) <= press_y - 0.035: break
    amo_step(0.34, a4_rest, np.zeros(3))
    if t % 2 == 0:
        frames.append(overlay(shot(), ticker, f"STEP {i_walk}/{n_steps}: {step_walk}   [EXECUTING: WALK — AMO locomotion]"))
from scipy.spatial.transform import Rotation as _R
_p = e.data.xpos[e.pelvis_id]; _yaw = _R.from_matrix(e.data.xmat[e.pelvis_id].reshape(3, 3)).as_euler('xyz')[2]
print(f"[EXEC] after walk: pelvis xy=({_p[0]:.3f},{_p[1]:.3f}) yaw={_yaw:.2f} | press-stance y={press_y:.3f} yaw={target_yaw:.2f}")

# --- ARRIVAL HANDOFF (env's own reset, pinned at the walked-in base; world state preserved)
cur = e.data.qpos[pel:pel + 7].copy()
env.robot_start_pos = np.array([cur[0], cur[1], env.robot_start_pos[2]])
env.robot_start_quat = cur[3:7].copy()
env.stance_noise_k = 0.0
env.set_curriculum_frac(1.0); env.set_curriculum_frac_min(1.0)
full_qpos = e.data.qpos.copy(); full_qvel = e.data.qvel.copy()
obs, _ = env.reset()
press_wrist2 = env._solved_wrist.copy()
e.data.qpos[:] = full_qpos; e.data.qvel[:] = full_qvel
mujoco.mj_forward(e.model, e.data); e._extract_state()
for i in range(70):
    env._amo_arm_step(e.default_dof_pos[19:23], wrist_target=press_wrist2)
    if i % 2 == 0:
        frames.append(overlay(shot(), ticker, f"STEP {i_press}/{n_steps}: {step_press}   [arrival — gathering press stance]"))
env._filt_arm[:] = 0; env._prev_action[:] = 0; env.episode_steps = 0
env.reward_fn.reset(); env._held_steps = 0; env._was_deep = False
env.initial_button_displacement = e.data.qpos[env.button_joint_id]
obs = env._get_obs()

# --- PRESS phase (RL policy)
print(f"[EXEC] [video {vts(frames)}] STEP {i_press}/{n_steps}: {step_press}  -> PRESS phase (RL)")
maxpress = 0.0; dips = 0; was_deep = False
env.max_episode_steps = 400
for t in range(400):
    a, _ = model.predict(obs, deterministic=True)
    obs, rwd, term, trunc, info = env.step(a)
    _d = env._get_button_displacement(); maxpress = max(maxpress, _d)
    if _d > 0.02: was_deep = True
    if was_deep and _d < 0.010: dips += 1; was_deep = False
    frames.append(overlay(shot(), ticker, f"STEP {i_press}/{n_steps}: {step_press}   [EXECUTING: PRESS — RL policy]  depth {_d*1000:4.1f}mm"))
    if t % 40 == 0:
        gp = env._right_hand_pos(); cap = e.data.geom_xpos[env._cap_gid]
        print(f"[EXEC]   press t{t}: gripper-cap {np.linalg.norm(gp - cap)*100:4.1f}cm  disp {_d*1000:4.1f}mm")
    if term or trunc: break
print(f"[EXEC] press done: MAX depth {maxpress*1000:.1f}mm, pumps {dips}")

# --- DISENGAGE (scripted; hand lowers to rest — job done)
for i in range(55):
    env._amo_arm_step(a4_rest, wrist_target=np.zeros(3))
    if i % 2 == 0:
        frames.append(overlay(shot(), ticker, f"STEP {i_press}/{n_steps}: {step_press}   [disengage — arm to rest]"))
last = shot()

# --- remaining plan steps -> caption cards (read_sensor / notify / wait / skipped / extra exec)
for idx, (st, verb, entity, mode) in enumerate(plan, 1):
    if idx in (i_walk, i_press):
        continue
    if mode == "CAPTION":
        status = {"read_sensor": f"reading {entity} — caption only (sensor I/O not embodied)",
                  "notify": f"notifying {entity} — caption only (comms not embodied)",
                  "wait": f"waiting {entity} — caption only (time-lapse)"}[verb]
    elif mode == "EXECUTE":
        status = "NOT RUN — v0 chain executes one walk->press pass"
    else:
        status = "SKIPPED (skill not embodied)"
    print(f"[EXEC] [video {vts(frames)}] STEP {idx}/{n_steps}: {st}  -> {status}")
    frames += caption_card(last, idx, n_steps, st, status)

# --- closing card
verdict = "PRESS VERIFIED" if maxpress >= 0.020 else "PRESS BELOW 20mm THRESHOLD"
frames += title_card([
    ("TICKET RESOLVED" if maxpress >= 0.020 else "TICKET INCOMPLETE", F_TITLE,
     (120, 220, 120) if maxpress >= 0.020 else (230, 100, 90)),
    (f"{sop['sop_id']} — {sop['title']}", F_HEAD, (235, 235, 235)),
    (f"{verdict}: max button depth {maxpress*1000:.1f}mm (threshold 20mm), pumps {dips}", F_BODY, (200, 205, 215)),
    ("walk_to + press_button embodied · read_sensor / notify / wait captioned", F_BODY, (200, 205, 215)),
], seconds=3.5)

out = os.path.join(PROJ, "_g7_ticket.mp4")
imageio.mimsave(out, frames, fps=FPS, quality=8)
print()
print(f"[DONE] saved {out}  frames={len(frames)} ({len(frames)/FPS:.1f}s)")
print(f"[DONE] MAX_press={maxpress*1000:.1f}mm  PUMPS={dips}  ->  {verdict}")
