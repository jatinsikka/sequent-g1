"""
Static viewer for the G1 + Robotiq 2F-85 gripper (g1_robotiq.mjb).
Shows the robot standing with the right arm reaching forward so you can inspect the gripper
mount/orientation. Pose held static (no stepping) so you can orbit freely.

Run:  & "C:/Users/sikka/miniconda3/envs/amo/python.exe" Sequent-robotics/view_grasp.py
Mouse: drag to orbit, scroll to zoom, right-drag to pan.
"""
import os
import time
import numpy as np
import mujoco
import mujoco.viewer

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# standing pose (joint angles by name) from the original model's home keyframe
m0 = mujoco.MjModel.from_xml_path("g1.xml"); d0 = mujoco.MjData(m0)
mujoco.mj_resetDataKeyframe(m0, d0, 0); mujoco.mj_forward(m0, d0)
stand = {}
for i in range(m0.njnt):
    nm = mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_JOINT, i)
    if nm and m0.jnt_type[i] in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
        stand[nm] = float(d0.qpos[m0.jnt_qposadr[i]])
pid0 = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
base_q = d0.qpos[m0.jnt_qposadr[[i for i in range(m0.njnt)
        if m0.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and m0.jnt_bodyid[i] == pid0][0]]:][:7].copy()
base_q[0] -= 0.15   # robot stands ~1.5 steps left (box to its right) -> natural reach

# the merged model (G1 + Robotiq gripper)
m = mujoco.MjModel.from_binary_path("g1_robotiq.mjb"); d = mujoco.MjData(m)
pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
pjm = [i for i in range(m.njnt) if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_bodyid[i] == pid][0]
d.qpos[m.jnt_qposadr[pjm]:m.jnt_qposadr[pjm] + 7] = base_q
for nm, v in stand.items():
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)
    if j >= 0:
        d.qpos[m.jnt_qposadr[j]] = v
# right arm at the level, collision-free grasp config (7-DOF IK); gripper open, straddling the tool
for jn, val in [("right_shoulder_pitch_joint", 0.097), ("right_shoulder_roll_joint", -0.040),
                ("right_shoulder_yaw_joint", -0.097), ("right_elbow_joint", 0.391),
                ("right_wrist_roll_joint", 0.056), ("right_wrist_pitch_joint", -0.479),
                ("right_wrist_yaw_joint", 0.129)]:
    d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]] = val
# place the tool on its pedestal at the grasp spot
_tb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "task_tool")
_ta = m.jnt_qposadr[[i for i in range(m.njnt) if m.jnt_bodyid[i] == _tb][0]]
d.qpos[_ta:_ta + 3] = [-0.415, -1.117, 0.821]; d.qpos[_ta + 3:_ta + 7] = [1, 0, 0, 0]
mujoco.mj_forward(m, d)

print("Viewer open — drag to orbit. Inspect the gripper on the right arm.")
with mujoco.viewer.launch_passive(m, d, show_left_ui=False, show_right_ui=False) as viewer:
    # start framed on the robot + gripper so no panning needed
    viewer.cam.lookat[:] = [-0.50, -1.20, 0.90]
    viewer.cam.distance = 1.7
    viewer.cam.azimuth = 270
    viewer.cam.elevation = -12
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)
