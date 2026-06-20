"""Export a MuJoCo rollout for the in-browser 3D viewer. Robust approach: export MuJoCo's
own CANONICAL mesh vertices/faces (m.mesh_vert/mesh_face) — the exact geometry MuJoCo
renders at geom_xpos/geom_xmat — so there is no STL-recentering ambiguity. Plus static geom
defs + per-frame world transforms of every visible geom.
Usage: python export_scene.py grasp
"""
import sys, os, json, numpy as np, mujoco, torch, warnings; warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "..", "sequent-site", "viewer")
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
kind = sys.argv[1] if len(sys.argv) > 1 else "grasp"

def mat2quat(mat9):
    q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(mat9, dtype=np.float64)); return q  # w,x,y,z

if kind == "grasp":
    from stable_baselines3 import PPO
    from env_wrapper import G1RLEnv
    env = G1RLEnv(policy_jit_path="amo_jit.pt", robot_type="g1", device=DEVICE, reward_fn=None, headless=True)
    model = PPO.load("checkpoints/v55_final.zip", device="cpu")
    m, d = env.env.model, env.env.data
    step_fn = lambda obs: env.step(model.predict(obs, deterministic=True)[0])
    obs, _ = env.reset(); HORIZON = 90
else:
    raise SystemExit("only 'grasp' supported here")

# robot bodies = subtree of the pelvis (so we can drop the robot's collision primitives)
root = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
robot_bodies = set()
if root >= 0:
    for b in range(m.nbody):
        x = b
        while x > 0:
            if x == root: robot_bodies.add(b); break
            x = m.body_parentid[x]

# static geom table + collect canonical mesh geometry
geoms, meshdata = [], {}
for i in range(m.ngeom):
    rgba = m.geom_rgba[i].tolist()
    if rgba[3] < 0.05: continue
    t = int(m.geom_type[i])
    is_robot = int(m.geom_bodyid[i]) in robot_bodies
    if is_robot and t != mujoco.mjtGeom.mjGEOM_MESH:
        continue  # robot collision primitives -> drop; keep only its visual meshes
    meshname = None
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[i]); meshname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid)
        if meshname not in meshdata:
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
            meshdata[meshname] = {"v": [round(float(x), 4) for x in m.mesh_vert[va:va+vn].flatten()],
                                  "f": [int(x) for x in m.mesh_face[fa:fa+fn].flatten()]}
    geoms.append({"idx": i, "type": t, "size": [round(float(x), 4) for x in m.geom_size[i]],
                  "rgba": [round(float(x), 3) for x in rgba], "mesh": meshname})

# rollout -> per-frame world transforms (canonical mesh verts render directly at geom frame)
frames = []
for t in range(HORIZON):
    fr = []
    for g in geoms:
        i = g["idx"]; p = d.geom_xpos[i]; q = mat2quat(d.geom_xmat[i])
        fr.append([round(float(v), 4) for v in (p[0], p[1], p[2], q[0], q[1], q[2], q[3])])
    frames.append(fr)
    obs, _, term, trunc, _ = step_fn(obs)
    if term or trunc: break

json.dump({"geoms": [{k: g[k] for k in ("type", "size", "rgba", "mesh")} for g in geoms]}, open(os.path.join(OUT, "scene.json"), "w"))
json.dump(meshdata, open(os.path.join(OUT, "meshes.json"), "w"))
json.dump({"fps": 20, "frames": frames}, open(os.path.join(OUT, "traj_%s.json" % kind), "w"))
print(f"exported {len(geoms)} geoms, {len(meshdata)} meshes, {len(frames)} frames, robot bodies={len(robot_bodies)} -> {OUT}")
