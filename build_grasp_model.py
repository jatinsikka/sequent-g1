"""
Build the fixed-base grasp model: G1 (robot only) + Robotiq 2F-85 gripper grafted on the
right wrist. Compiles via MjSpec and saves a self-contained MJB binary (spec.to_xml segfaults
on attached models in mujoco 3.2.3, so we save the compiled model instead).

Run:  python build_grasp_model.py   ->  writes g1_robotiq.mjb
"""
import os
import re
import numpy as np
import mujoco

os.chdir(os.path.dirname(os.path.abspath(__file__)))
A = os.path.abspath("mujoco_menagerie/robotiq_2f85/assets")

import sys
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "pedestal"

# base scene from g1.xml. The "table" variant KEEPS interactive_objects.xml (the real screwdriver +
# objects on the existing workstations) — the original manipulation setup; only the standalone
# pedestal/bench variants strip them and drop in a synthetic object.
src = open("g1.xml", encoding="utf-8").read()
if VARIANT != "table":
    src = src.replace('<include file="interactive_objects.xml"/>', '')
src = re.sub(r"\s*<keyframe>.*?</keyframe>", "", src, flags=re.DOTALL)

# Restore the 3 right-wrist joints (the real G1 has them; this MJCF rigidified them) -> 7-DOF arm,
# so the gripper can orient for a natural, level grasp. Specs from the official Unitree G1 model.
_wrist = {
    '<body name="right_wrist_roll_link" pos="0.1 -0.00188791 -0.01">':
        '<joint name="right_wrist_roll_joint" axis="1 0 0" range="-1.97222 1.97222" actuatorfrcrange="-40 40" armature="0.01"/>',
    '<body name="right_wrist_pitch_link" pos="0.038 0 0">':
        '<joint name="right_wrist_pitch_joint" axis="0 1 0" range="-1.61443 1.61443" actuatorfrcrange="-40 40" armature="0.01"/>',
    '<body name="right_wrist_yaw_link" pos="0.046 0 0">':
        '<joint name="right_wrist_yaw_joint" axis="0 0 1" range="-1.61443 1.61443" actuatorfrcrange="-40 40" armature="0.01"/>',
}
for _tag, _jt in _wrist.items():
    assert src.count(_tag) == 1, f"wrist anchor not unique: {_tag}"
    src = src.replace(_tag, _tag + "\n                        " + _jt)
_wrist_act = (
    '    <position name="rwrist_roll" joint="right_wrist_roll_joint" kp="200" kv="5" ctrlrange="-1.97222 1.97222" forcerange="-40 40"/>\n'
    '    <position name="rwrist_pitch" joint="right_wrist_pitch_joint" kp="200" kv="5" ctrlrange="-1.61443 1.61443" forcerange="-40 40"/>\n'
    '    <position name="rwrist_yaw" joint="right_wrist_yaw_joint" kp="200" kv="5" ctrlrange="-1.61443 1.61443" forcerange="-40 40"/>\n'
)
src = src.replace("  </actuator>", _wrist_act + "  </actuator>")

# Uniform POSITION control for the whole right arm (robust tracking like the wrist) + force
# authority for the sim (real G1 arm spec is +-25 Nm; we raise it so control can drive poses).
src = src.replace('actuatorfrcrange="-25 25"', 'actuatorfrcrange="-60 60"')
_arm_pos = {"right_shoulder_pitch_joint": "-3.0892 2.6704", "right_shoulder_roll_joint": "-2.2515 1.5882",
            "right_shoulder_yaw_joint": "-2.618 2.618", "right_elbow_joint": "-1.0472 2.0944"}
for _j, _rng in _arm_pos.items():
    _motor = f'<motor name="{_j}" joint="{_j}" gear="1" ctrllimited="true" ctrlrange="-200 200"/>'
    assert src.count(_motor) == 1, f"arm motor not found: {_j}"
    src = src.replace(_motor, f'<position name="{_j}" joint="{_j}" kp="300" kv="15" ctrlrange="{_rng}"/>')
open("g1_base.xml", "w", encoding="utf-8").write(src)

# merge: attach FIRST (brings bodies + materials), THEN add meshes (attach doesn't copy meshes)
spec = mujoco.MjSpec(); spec.from_file("g1_base.xml")
grip = mujoco.MjSpec(); grip.from_file("mujoco_menagerie/robotiq_2f85/2f85.xml")
wrist = spec.find_body("right_wrist_yaw_link")
frame = wrist.add_frame()
frame.pos = [0.05, 0.0, 0.0]
frame.quat = [0.5, 0.5, 0.5, 0.5]   # +Z->+X (distal) plus a 90deg roll so the jaw opens horizontally
frame.attach_body(grip.find_body("base_mount"), "rg_", "")
for nm in ["base_mount", "base", "driver", "coupler", "follower", "pad", "silicone_pad", "spring_link"]:
    me = spec.add_mesh(); me.name = "rg_" + nm; me.file = os.path.join(A, nm + ".stl"); me.scale = [0.001, 0.001, 0.001]

# hide the old rigid hand (invisible + non-colliding) so only the gripper remains
_g = spec.find_body("right_rubber_hand").first_geom()
if _g is not None:
    _g.rgba = [0, 0, 0, 0]; _g.contype = 0; _g.conaffinity = 0

wb = spec.worldbody

def addbox(body, name, pos, size, rgba, col=True, fric=(1.0, 0.05, 0.005)):
    g = body.add_geom(); g.name = name; g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = pos; g.size = size; g.rgba = rgba
    g.contype = 1 if col else 0; g.conaffinity = 15 if col else 0
    if col: g.friction = list(fric)
    return g

if VARIANT == "table":
    # ---- manipulation at the real workstation table: ONE clean grasp-target box on the table top ----
    # (the existing screwdriver/clutter is hidden by the env; this is the single, reliable target).
    gb = wb.add_body(); gb.name = "grasp_box"; gb.pos = [0.80, 1.13, 0.80]
    gb.add_freejoint()
    bg = addbox(gb, "grasp_box_geom", [0, 0, 0], [0.025, 0.025, 0.025], [0.85, 0.30, 0.20, 1])
    bg.mass = 0.10
    OUT = "g1_robotiq_table.mjb"
elif VARIANT == "lever":
    # ---- LEVER on the workstation table: a fixed mount + a hinged handle the arm rotates ----
    # Pivot placed where the grasp box sits (within the validated right-hand reach band). Hinge axis
    # = world X (horizontal), so the handle swings in the Y-Z plane (toward/away from the robot).
    # grasp_box must EXIST (the table env code path looks it up), but for the lever task it would
    # collide with the handle, so park it off to the side / out of the workspace (env also disables it).
    gb = wb.add_body(); gb.name = "grasp_box"; gb.pos = [1.40, 1.40, 0.80]
    gb.add_freejoint()
    bg = addbox(gb, "grasp_box_geom", [0, 0, 0], [0.025, 0.025, 0.025], [0.85, 0.30, 0.20, 1])
    bg.mass = 0.10

    PIV = [0.80, 1.15, 0.74]                     # hinge pivot AT the box's proven-reachable spot (the arm
    #   reaches the grasp_box at ~(0.80,1.20,0.75) and lifts to ~0.87, so a handle whose tip sits ~(0.80,1.15,
    #   0.80) is inside the demonstrated reach band). The prior (0.78,0.90,0.79) was too close+high: an IK
    #   operability probe showed the hand stalling 17cm short of the handle, so the lever never moved. [reach fix]
    # fixed mount: a small post the hinge body is anchored to
    mnt = wb.add_body(); mnt.name = "lever_mount"; mnt.pos = [PIV[0], PIV[1], PIV[2]]
    mg = mnt.add_geom(); mg.name = "lever_mount_geom"; mg.type = mujoco.mjtGeom.mjGEOM_BOX
    mg.pos = [0, 0, -0.025]; mg.size = [0.03, 0.03, 0.025]; mg.rgba = [0.35, 0.37, 0.42, 1]
    mg.contype = 2; mg.conaffinity = 2
    # hinged handle body: rotates about world X at the pivot
    lv = mnt.add_body(); lv.name = "lever_arm"; lv.pos = [0, 0, 0]
    jt = lv.add_joint(); jt.name = "lever_hinge"; jt.type = mujoco.mjtJoint.mjJNT_HINGE
    jt.axis = [-1, 0, 0]; jt.range = [0.0, 1.4]   # +angle swings the top toward +Y = AWAY from the robot,
    #   which is the only direction the robot (standing at low Y) can push it. (axis +X made +angle go toward
    #   the robot, so the arm could only drive it into the 0 clamp.)
    jt.damping = 0.5; jt.stiffness = 0.6; jt.springref = 0.0   # mild: holds at 0 but the arm can drive it
    jt.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE       # avoid inherited bad actfrcrange default
    # handle = a capsule ~12cm long standing up (+Z at angle 0). Its geom frame is offset so the
    # capsule body extends from the pivot upward; pushing the top in -Y rotates the hinge toward +angle.
    hg = lv.add_geom(); hg.name = "lever_handle"; hg.type = mujoco.mjtGeom.mjGEOM_CAPSULE
    hg.fromto = [0, 0, 0, 0, 0, 0.08]; hg.size = [0.014, 0, 0]   # radius 1.4cm, length 8cm
    hg.rgba = [0.90, 0.55, 0.10, 1]; hg.mass = 0.05
    hg.contype = 2; hg.conaffinity = 2; hg.friction = [1.0, 0.05, 0.005]
    OUT = "g1_robotiq_lever.mjb"
elif VARIANT == "bench":
    # ---- reachable WORKBENCH in front of the fixed robot + the original SCREWDRIVER lying on top ----
    CX, CY, TOP = -0.415, -1.10, 0.78        # bench top-surface height (chosen near the validated reach band)
    bench = wb.add_body(); bench.name = "task_bench"; bench.pos = [CX, CY, 0.0]
    addbox(bench, "task_bench_top", [0, 0, TOP - 0.02], [0.30, 0.22, 0.02], [0.86, 0.88, 0.92, 1])
    legh = (TOP - 0.04) / 2
    for lx in (0.26, -0.26):
        for ly in (0.18, -0.18):
            addbox(bench, f"bench_leg_{'p' if lx>0 else 'm'}{'p' if ly>0 else 'm'}",
                   [lx, ly, legh], [0.02, 0.02, legh], [0.30, 0.33, 0.38, 1])
    # screwdriver (handle = grasp target = task_tool_geom; + thin shaft), lying on its side
    sd = wb.add_body(); sd.name = "task_tool"; sd.pos = [CX, CY, TOP + 0.02]
    sd.quat = [0.70710678, 0.0, 0.70710678, 0.0]    # rotate shaft from +Z to +X -> lies flat
    sd.add_freejoint()
    h = sd.add_geom(); h.name = "task_tool_geom"; h.type = mujoco.mjtGeom.mjGEOM_BOX
    h.size = [0.014, 0.014, 0.045]; h.rgba = [0.90, 0.60, 0.10, 1]; h.mass = 0.08
    h.contype = 1; h.conaffinity = 15; h.friction = [1.0, 0.05, 0.005]
    sh = sd.add_geom(); sh.name = "task_tool_shaft"; sh.type = mujoco.mjtGeom.mjGEOM_BOX
    sh.pos = [0, 0, 0.10]; sh.size = [0.005, 0.005, 0.06]; sh.rgba = [0.72, 0.72, 0.72, 1]; sh.mass = 0.02
    sh.contype = 1; sh.conaffinity = 15; sh.friction = [1.0, 0.05, 0.005]
    OUT = "g1_robotiq_bench.mjb"
else:
    # static pedestal + a free-floating box tool at the reachable spot
    SPOT = [-0.415, -1.117, 0.821]
    ped = wb.add_body(); ped.name = "task_pedestal"; ped.pos = [SPOT[0], SPOT[1], SPOT[2] - 0.035 - 0.05]
    pg = addbox(ped, "task_pedestal_geom", [0, 0, 0], [0.06, 0.06, 0.05], [0.4, 0.4, 0.46, 1], fric=(3, 0.1, 0.01))
    tool = wb.add_body(); tool.name = "task_tool"; tool.pos = [SPOT[0], SPOT[1], SPOT[2] + 0.01]
    tool.add_freejoint()
    addbox(tool, "task_tool_geom", [0, 0, 0], [0.02, 0.02, 0.035], [0.85, 0.30, 0.20, 1], fric=(3, 0.1, 0.01)).mass = 0.05
    OUT = "g1_robotiq.mjb"

model = spec.compile()
print(f"[{VARIANT}] compiled: nu={model.nu} nbody={model.nbody} nq={model.nq}")
mujoco.mj_saveModel(model, OUT, None)
print("saved", OUT)

# ---- verify the gripper closes, with the robot frozen so it can't explode ----
m0 = mujoco.MjModel.from_xml_path("g1.xml"); d0 = mujoco.MjData(m0)
mujoco.mj_resetDataKeyframe(m0, d0, 0); mujoco.mj_forward(m0, d0)
stand = {}
for i in range(m0.njnt):
    nm = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, i)
    if nm and m0.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        stand[nm] = float(d0.qpos[m0.jnt_qposadr[i]])
pid0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
base_q = d0.qpos[m0.jnt_qposadr[[i for i in range(m0.njnt) if m0.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and m0.jnt_bodyid[i] == pid0][0]]:][:7]

d = mujoco.MjData(model)
pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
pjm = [i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and model.jnt_bodyid[i] == pid][0]
d.qpos[model.jnt_qposadr[pjm]:model.jnt_qposadr[pjm] + 7] = base_q
for nm, v in stand.items():
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, nm)
    if j >= 0:
        d.qpos[model.jnt_qposadr[j]] = v
mujoco.mj_forward(model, d)
qhome = d.qpos.copy()
# free = gripper joints (rg_*); held = everything else
gjoints = [i for i in range(model.njnt) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("rg_")]
free_q = set(); free_d = set()
for i in gjoints:
    free_q.add(model.jnt_qposadr[i]); free_d.add(model.jnt_dofadr[i])
held_q = [i for i in range(model.nq) if i not in free_q]
held_d = [i for i in range(model.nv) if i not in free_d]
lp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rg_left_pad")
rp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rg_right_pad")
ai = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "rg_fingers_actuator")
o = np.linalg.norm(d.xpos[lp] - d.xpos[rp])
d.ctrl[ai] = 255
for _ in range(300):
    mujoco.mj_step(model, d)
    d.qpos[held_q] = qhome[held_q]; d.qvel[held_d] = 0
c = np.linalg.norm(d.xpos[lp] - d.xpos[rp])
print(f"gripper pad gap: open={o*100:.1f}cm -> closed={c*100:.1f}cm  ({'WORKS' if o-c>0.02 else 'FAILED'})")
