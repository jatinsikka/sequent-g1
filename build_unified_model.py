"""
Build the UNIFIED gripper-humanoid model: g1.xml (the AMO walking humanoid, with rubber
hands + keyframe + interactive_objects) PLUS the Robotiq 2F-85 gripper grafted on the right
wrist -- but KEEPING THE AMO CONTROL SCHEME (torque/motor actuators driven by the PD loop).

Differs from build_grasp_model.py (fixed-base grasp) in the ways that matter for AMO:
  * restores the 3 right-wrist joints but as MOTOR (torque) actuators, gear=1, ctrlrange
    matching the leg/arm motors (-200..200) -- NOT kp=300 position actuators.
  * does NOT convert the right arm to position control (no kp=300, no actuatorfrcrange bump).
  * KEEPS the <keyframe> (AMO home pose) and the interactive_objects.xml include.
  * appends the wrist+gripper DOFs at the END of the right-arm chain, so the 23 AMO DOFs
    (leg+waist+arm) keep their qpos/dof addresses relative to plain g1.xml.

Saves g1_amo_gripper.mjb (spec.to_xml segfaults on attached models in mujoco 3.2.3).
Run:  python build_unified_model.py
"""
import os
import re
import numpy as np
import mujoco

os.chdir(os.path.dirname(os.path.abspath(__file__)))
A = os.path.abspath("mujoco_menagerie/robotiq_2f85/assets")

# base scene from g1.xml -- KEEP interactive_objects.xml and the keyframe (AMO needs the home pose)
src = open("g1.xml", encoding="utf-8").read()

# --- restore the 3 right-wrist joints -> 7-DOF right arm (so the gripper can orient) ---
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

# --- wrist actuators as MOTOR (torque), consistent with the AMO PD scheme (NOT position) ---
# gear=1 so ctrl == torque, same as every other robot motor. ctrlrange -40..40 (wrist frc range).
_wrist_act = (
    '    <motor name="right_wrist_roll_joint" joint="right_wrist_roll_joint" gear="1" ctrllimited="true" ctrlrange="-40 40"/>\n'
    '    <motor name="right_wrist_pitch_joint" joint="right_wrist_pitch_joint" gear="1" ctrllimited="true" ctrlrange="-40 40"/>\n'
    '    <motor name="right_wrist_yaw_joint" joint="right_wrist_yaw_joint" gear="1" ctrllimited="true" ctrlrange="-40 40"/>\n'
)
src = src.replace("  </actuator>", _wrist_act + "  </actuator>")

# --- extend the keyframe qpos: append 11 zeros (3 wrist + 8 gripper joints, all at rest=0) ---
# the keyframe qpos is position-indexed; adding DOFs at the end of qpos means we must append
# their home values. wrist rest = 0; the 8 rg_ gripper joints rest = 0 (gripper open).
def _extend_keyframe(s):
    m = re.search(r'(<key name="home"\s+qpos=")(.*?)("\s*/>)', s, flags=re.DOTALL)
    assert m, "keyframe not found"
    body = m.group(2).rstrip()
    body = body + "\n            0 0 0" + "\n            0 0 0 0 0 0 0 0"   # 3 wrist + 8 gripper
    return s[:m.start()] + m.group(1) + body + "\n            " + m.group(3) + s[m.end():]
src = _extend_keyframe(src)

open("g1_amo_gripper_base.xml", "w", encoding="utf-8").write(src)

# --- merge: attach gripper FIRST (brings bodies+materials), THEN add its meshes ---
spec = mujoco.MjSpec(); spec.from_file("g1_amo_gripper_base.xml")
grip = mujoco.MjSpec(); grip.from_file("mujoco_menagerie/robotiq_2f85/2f85.xml")
wrist = spec.find_body("right_wrist_yaw_link")
frame = wrist.add_frame()
frame.pos = [0.05, 0.0, 0.0]
frame.quat = [0.5, 0.5, 0.5, 0.5]   # +Z->+X (distal) plus 90deg roll so the jaw opens horizontally
frame.attach_body(grip.find_body("base_mount"), "rg_", "")
for nm in ["base_mount", "base", "driver", "coupler", "follower", "pad", "silicone_pad", "spring_link"]:
    me = spec.add_mesh(); me.name = "rg_" + nm; me.file = os.path.join(A, nm + ".stl"); me.scale = [0.001, 0.001, 0.001]

# --- hide the old rigid right hand (invisible + non-colliding) so only the gripper remains ---
_g = spec.find_body("right_rubber_hand").first_geom()
if _g is not None:
    _g.rgba = [0, 0, 0, 0]; _g.contype = 0; _g.conaffinity = 0

# --- make the gripper PADS collide with the mask-2 interactive objects (buttons/lever) ---
# The Robotiq pads come in on collision mask 1 (contype=1,conaffinity=1) while the panel
# buttons + lever use mask 2 (contype=2,conaffinity=2). MuJoCo pairs geoms only when
# (contype_a & conaffinity_b) or (contype_b & conaffinity_a) is nonzero, so mask-1 pads
# pass STRAIGHT THROUGH the mask-2 buttons -> no press force. Widen the finger pad geoms to
# both masks (contype/conaffinity bit 1 AND 2) so the CLOSED gripper is a solid pusher for
# press/lever while still colliding with the mask-1 world (floor/pick box).
for _pb in ["rg_left_pad", "rg_right_pad"]:
    _b = spec.find_body(_pb)
    if _b is None:
        continue
    _g2 = _b.first_geom()
    while _g2 is not None:
        _g2.contype = 3; _g2.conaffinity = 3   # bits {1,2}: collide with mask-1 world AND mask-2 buttons/lever
        _g2 = _b.next_geom(_g2)

# --- strengthen the gripper grip force so a held box doesn't slip during the lift ---
# The Robotiq tendon actuator ships with forcerange -5..5, marginal for holding a lifted box.
# Raise it so the CLOSED gripper clamps firmly (still a compliant tendon pinch, just more force).
for _a in spec.actuators:
    if _a.name == "rg_fingers_actuator":
        _a.forcerange = [-40, 40]

# --- optional: a grasp box on the REAL right_workstation table (no pedestal) ---
# The unified pick must use the actual workstation, matching the fixed-base grasp: the
# right_workstation is at (0.9,1.4,0) with its table-top surface at world z=0.725 (top geom
# local z=0.7, half-height 0.025). The box sits on the top at a spot within the humanoid's
# right-hand reach when it STANDS AT THE TABLE facing +Y (like the fixed-base grasp base).
import sys
WITH_BOX = ("--pick" in sys.argv)
if WITH_BOX:
    wb = spec.worldbody
    TABLE_TOP = 0.725                      # right_workstation top surface (world z)
    # a stable ~6cm CUBE: wide enough NOT to tip when touched (a tall thin box tips over),
    # and tall enough that its CENTER (grasp point) sits ~3cm above the table — at the top-down
    # gripper's reachable floor, so the pads bracket the box without the fingers hitting the
    # table (a short flat box forced the pad-mid 2cm high and shoved the box on close).
    # 4.4cm-wide (grippable — well inside the 85mm Robotiq stroke), 6cm-tall block: the height
    # lifts the grasp center ~3cm above the table (clear of the fingers-hit-table problem) while
    # the 4.4cm base keeps it stable (a 3.6cm-wide tall box tipped when touched).
    BOX_HW = 0.019    # half-width (3.8cm): narrow enough that the closing Robotiq pads CLAMP it
    #                   (firm normal force -> friction hold), wide enough that it doesn't slip
    #                   through the pads during the close.
    BOX_HZ = 0.030    # half-height (6cm) -> grasp center clear of the table
    SPOT = [0.80, 1.20, TABLE_TOP + BOX_HZ]   # box center on the table top
    bx = wb.add_body(); bx.name = "pick_box"; bx.pos = SPOT
    bx.add_freejoint()
    bg = bx.add_geom(); bg.name = "pick_box_geom"; bg.type = mujoco.mjtGeom.mjGEOM_BOX
    # narrow in X (the jaw axis) for a firm clamp, wider in Y for a stable base (won't tip)
    bg.size = [BOX_HW, 0.025, BOX_HZ]; bg.rgba = [0.85,0.30,0.20,1]; bg.mass = 0.04
    bg.contype = 1; bg.conaffinity = 15; bg.friction = [4.0, 0.2, 0.02]
    bg.solref = [0.003, 1.0]; bg.solimp = [0.96, 0.97, 0.001, 0.5, 2.0]   # STIFF contact -> box rests ON the table, no sink-through
    # extend the keyframe qpos with the box freejoint home (pos + identity quat)
    key = spec.key[0]
    key.qpos = list(key.qpos) + [SPOT[0], SPOT[1], SPOT[2], 1.0, 0.0, 0.0, 0.0]

model = spec.compile()
OUT = "g1_amo_gripper_pick.mjb" if WITH_BOX else "g1_amo_gripper.mjb"
print(f"compiled: nu={model.nu} nbody={model.nbody} nq={model.nq} nv={model.nv} nkey={model.nkey}")
mujoco.mj_saveModel(model, OUT, None)
print("saved", OUT)

# ---- sanity: leg/arm DOF addresses must MATCH plain g1.xml (AMO reads them) ----
g1 = mujoco.MjModel.from_xml_path("g1.xml")
AMO_JOINTS = [
    "left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint",
    "left_ankle_pitch_joint","left_ankle_roll_joint",
    "right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint",
    "right_ankle_pitch_joint","right_ankle_roll_joint",
    "waist_yaw_joint","waist_roll_joint","waist_pitch_joint",
    "left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
    "right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
]
print("\n=== AMO DOF address check (qposadr / dofadr): g1 vs unified ===")
ok = True
q_g1, q_uni = [], []
for jn in AMO_JOINTS:
    a = mujoco.mj_name2id(g1, mujoco.mjtObj.mjOBJ_JOINT, jn)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    qa, qb = g1.jnt_qposadr[a], model.jnt_qposadr[b]
    da, db = g1.jnt_dofadr[a], model.jnt_dofadr[b]
    q_g1.append(qa); q_uni.append(qb)
    same = (qa == qb) and (da == db)
    ok = ok and same
    if not same:
        print(f"  MISMATCH {jn}: g1 q{qa}/d{da} vs uni q{qb}/d{db}")
# contiguity + last-23 check
q_uni = np.array(q_uni)
print("unified AMO qpos addrs:", q_uni.min(), "..", q_uni.max(), "contiguous:", np.all(np.diff(q_uni) == 1))
print("plain-g1 uses qpos[-23:]  -> in g1 that is", g1.nq-23, "..", g1.nq-1,
      "(min AMO qpos in g1 =", min(q_g1), ")")
print("unified nq:", model.nq, " qpos[-23:] would start at", model.nq-23,
      "but AMO DOFs start at", q_uni.min(), "-> MUST read explicit addrs, NOT [-23:]")
print("ADDRESSES MATCH g1:", ok)
