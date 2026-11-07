"""
Teleoperate a perfect pick to record human BC demos.

Opens the live MuJoCo viewer. You drive the GRIPPER in Cartesian space (the arm follows via IK),
close it on the cube dead-center, lift, and hold. Each demo you save is recorded as (obs, action)
pairs into _human_demos.npz for behavior cloning.

CONTROLS (focus the viewer window, then press keys):
    W / S   : move hand  forward / back   (+y / -y)
    A / D   : move hand  left / right      (-x / +x)
    R / F   : move hand  up / down         (+z / -z)
    G       : toggle gripper  OPEN <-> CLOSE
    ENTER   : SAVE this attempt as a demo, then reset for the next one
    N       : discard + reset (start the attempt over, nothing saved)
    (close the window when you're done)

Tip: a good pick = hand centered over the cube, lower until the pads straddle it deep (not the top
edge), CLOSE (G), confirm it's gripped (it stays when you raise), then R to lift ~10cm and hold a moment.
"""
import sys, os, time
PROJ = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, PROJ); os.chdir(PROJ)
import numpy as np, mujoco, mujoco.viewer

from grasp_fixed_env import GraspFixedEnv

env = GraspFixedEnv(scene="table"); obs, _ = env.reset(seed=0); d, m = env.data, env.model
STEP = 0.01  # metres per key press

state = {"target": env._grasp_point().copy(), "grip": False, "save": False, "reset": False}
cur_obs, cur_act = [], []
SAVE_PATH = os.path.join(PROJ, "_human_demos.npz")
saved_obs, saved_act, n_demos = [], [], 0

def on_key(keycode):
    c = chr(keycode).upper() if 0 < keycode < 256 else ""
    t = state["target"]
    if   c == "W": t[1] += STEP
    elif c == "S": t[1] -= STEP
    elif c == "A": t[0] -= STEP
    elif c == "D": t[0] += STEP
    elif c == "R": t[2] += STEP
    elif c == "F": t[2] -= STEP
    elif c == "G": state["grip"] = not state["grip"]
    elif c == "N": state["reset"] = True
    elif keycode in (257, 335):  # ENTER / numpad-enter
        state["save"] = True

def grasp_jac():
    jl = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jl, None, env.lpad); mujoco.mj_jacBody(m, d, jr, None, env.rpad)
    return 0.5 * (jl + jr)[:, env.arm_dadr]               # Jacobian of the pad midpoint (grasp point)

def action_from_state():
    err = np.clip((state["target"] - env._grasp_point()) * 3.0, -0.06, 0.06)  # move the PADS to your target
    dq = np.clip(np.linalg.pinv(grasp_jac()) @ err, -0.18, 0.18)
    q_des = d.qpos[env.arm_qadr] + dq
    a = np.zeros(8, np.float32)
    a[:7] = np.clip((q_des - env.arm_mid) / env.arm_half, -1, 1)
    a[7] = 1.0 if state["grip"] else -1.0
    return a

_seed = [0]
def reset_attempt():
    global obs, cur_obs, cur_act
    _seed[0] += 1
    obs, _ = env.reset(seed=_seed[0])               # vary box jitter per attempt for demo diversity
    state["target"] = env._grasp_point().copy(); state["grip"] = False
    cur_obs, cur_act = [], []

print(__doc__)
with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
    last = time.time()
    while viewer.is_running():
        a = action_from_state()
        cur_obs.append(obs.copy()); cur_act.append(a.copy())
        obs, r, term, trunc, info = env.step(a)
        if state["save"]:
            state["save"] = False
            saved_obs.append(np.array(cur_obs)); saved_act.append(np.array(cur_act)); n_demos += 1
            np.savez_compressed(SAVE_PATH, obs=np.concatenate(saved_obs), act=np.concatenate(saved_act))
            print(f"[SAVED demo #{n_demos}]  lift={info['lift']*100:.1f}cm gripped={info['gripping']} "
                  f"steps={len(cur_obs)}  -> {SAVE_PATH} ({sum(len(o) for o in saved_obs)} pairs)")
            reset_attempt()
        if state["reset"]:
            state["reset"] = False; reset_attempt(); print("[reset] attempt cleared")
        viewer.sync()
        dt = 0.02 - (time.time() - last)
        if dt > 0: time.sleep(dt)
        last = time.time()
print(f"\nDone. {n_demos} demos saved to {SAVE_PATH}")
