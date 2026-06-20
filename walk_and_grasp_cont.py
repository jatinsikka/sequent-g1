"""CONTINUOUS walk-up-and-grasp in ONE simulation (no reset/cut):
  WALK   - PID + AMO walks the robot forward (+Y) up to the workbench.
  SETTLE - a few stand steps to stabilize on arrival.
  GRASP  - the RL grasp policy (v5.5) reaches and grasps the screwdriver, AMO balancing.
Renders a gif AND exports the trajectory for the 3D viewer (traj_walkgrasp.json — same
scene/meshes as the grasp export, so the viewer just loads ?traj=walkgrasp).
"""
import os, json, numpy as np, torch, mujoco, warnings; warnings.filterwarnings("ignore")
from stable_baselines3 import PPO
from env_wrapper import G1RLEnv
from play_amo import quat_to_euler
from pid_controller import LocomotionPIDController
import imageio

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = os.path.join(os.path.dirname(__file__), "..", "sequent-site", "viewer")
g1 = G1RLEnv(policy_jit_path="amo_jit.pt", robot_type="g1", device=DEVICE, reward_fn=None, headless=True)
model = PPO.load("checkpoints/v55_final.zip", device="cpu")
g1.reset()
env = g1.env
m, d = env.model, env.data

GRASP_POS = np.array([0.6, 0.90])          # where the grasp policy expects to stand
START = np.array([0.6, -0.55])             # ~1.45m behind, same +Y facing
_rq = 46
d.qpos[_rq:_rq+2] = START; d.qvel[:] = 0.0
mujoco.mj_step(m, d); env._extract_state()
pid = LocomotionPIDController(kp_pos=1.0, ki_pos=0.0, kd_pos=0.1, max_vel=0.55, min_vel=0.2)

# geom set for the viewer trajectory — identical selection/order to export_scene.py
root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); robot = set()
for b in range(m.nbody):
    x = b
    while x > 0:
        if x == root: robot.add(b); break
        x = m.body_parentid[x]
gidx = [i for i in range(m.ngeom) if m.geom_rgba[i][3] >= 0.05 and
        not (int(m.geom_bodyid[i]) in robot and m.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH)]

def mat2quat(mat9):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=np.float64)); return q

ren = mujoco.Renderer(m, 380, 500); cam = mujoco.MjvCamera()
frames, traj = [], []

def record():
    fr = [[round(float(v), 4) for v in (*d.geom_xpos[i], *mat2quat(d.geom_xmat[i]))] for i in gidx]
    traj.append(fr)
    rp = d.xpos[env.pelvis_id]
    cam.lookat[:] = [rp[0], rp[1]*0.5 + GRASP_POS[1]*0.5, 0.85]; cam.distance = 2.8; cam.azimuth = 150; cam.elevation = -12
    ren.update_scene(d, camera=cam); frames.append(ren.render())

# ---- WALK phase ----
for i in range(int(env.sim_duration / env.sim_dt)):
    env._extract_state()
    if i % env.sim_decimation == 0:
        rp = d.xpos[env.pelvis_id][:2]; rpy = quat_to_euler(env.quat)
        if float(d.xpos[env.pelvis_id][2]) < 0.4: print("FELL in walk"); break
        if np.linalg.norm(GRASP_POS - rp) < 0.04: break
        vx, vy, hd = pid.compute_action(current_pos=rp, current_yaw=rpy[2], target_pos=GRASP_POS, target_yaw=np.pi/2, dt=env.control_dt)
        env.viewer.commands[0] = vx; env.viewer.commands[2] = vy; env.viewer.commands[1] = hd
        obs = env._compute_observation(); ot = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            eh = torch.tensor(np.array(env.extra_history).flatten().copy(), dtype=torch.float).view(1, -1).to(DEVICE)
            raw = np.clip(env.policy_jit(ot, eh).cpu().numpy().squeeze(), -40, 40)
        env.last_action = np.concatenate([raw.copy(), (env.dof_pos - env.default_dof_pos)[15:] / env.action_scale])
        pd = np.concatenate([raw * env.action_scale, np.zeros(8)]) + env.default_dof_pos
        pd[15:] = (1 - env.arm_blend) * env.prev_arm_action + env.arm_blend * env.arm_action
        env.arm_blend = min(1.0, env.arm_blend + 0.01)
        env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt * env.gait_freq, 1.0)
        if np.all(np.abs(env.gait_cycle - 0.25) < 0.05): env.gait_cycle = np.array([0.25, 0.75])
        record()
    torque = (pd - env.dof_pos) * env.stiffness - env.dof_vel * env.damping
    env.data.ctrl = np.clip(torque, -env.torque_limits, env.torque_limits); mujoco.mj_step(m, d)
print(f"arrived at {d.xpos[env.pelvis_id][:2].round(2)}; walk frames={len(frames)}")

# ---- SETTLE: brief stand to zero the gait (short, so it barely drifts) then grasp ----
for s in range(8):
    env._extract_state()
    env.viewer.commands[:] = 0.0; env.viewer.commands[1] = np.pi/2
    obs = env._compute_observation(); ot = torch.from_numpy(obs).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        eh = torch.tensor(np.array(env.extra_history).flatten().copy(), dtype=torch.float).view(1, -1).to(DEVICE)
        raw = np.clip(env.policy_jit(ot, eh).cpu().numpy().squeeze(), -40, 40)
    env.last_action = np.concatenate([raw.copy(), (env.dof_pos - env.default_dof_pos)[15:] / env.action_scale])
    pd = np.concatenate([raw * env.action_scale, np.zeros(8)]) + env.default_dof_pos
    pd[15:] = env.default_dof_pos[15:]
    env.gait_cycle = np.remainder(env.gait_cycle + env.control_dt * env.gait_freq, 1.0)
    for k in range(env.sim_decimation):
        torque = (pd - env.dof_pos) * env.stiffness - env.dof_vel * env.damping
        env.data.ctrl = np.clip(torque, -env.torque_limits, env.torque_limits); mujoco.mj_step(m, d)
    record()
print(f"settled at {d.xpos[env.pelvis_id][:2].round(2)}; frames={len(frames)}")

# ---- GRASP phase (RL policy via the grasp env's own step) ----
env._in_place_stand = True; env.gait_cycle = np.array([0.25, 0.25])
g1.target_object_pos = d.xpos[g1.screwdriver_body_id].copy()
g1.object_grasped = False; g1.grasp_rewarded = False
z0 = float(d.xpos[g1.screwdriver_body_id][2]); grasped = False; max_lift = 0.0
for t in range(180):
    a, _ = model.predict(g1._get_grasp_obs(), deterministic=True)
    g1.step(a)
    if getattr(g1, "object_grasped", False): grasped = True
    max_lift = max(max_lift, float(d.xpos[g1.screwdriver_body_id][2]) - z0)
    record()
print(f"GRASP: grasped={grasped} max_lift={max_lift*100:.1f}cm total frames={len(frames)}")

imageio.mimsave("walk_grasp.gif", frames, fps=22)
json.dump({"fps": 22, "frames": traj}, open(os.path.join(OUT, "traj_walkgrasp.json"), "w"))
print(f"saved walk_grasp.gif + traj_walkgrasp.json ({len(traj)} frames)")
