"""
Reliable IK-based "pick" primitive for the fixed-base Unitree-G1 grasp env.

Produces a REAL HELD lift: the box stays pinched (info["gripping"] True) and rides
the hand up. NOT a pop (box flung airborne, detached from the grasp point).

Sequence: reach above box (OPEN) -> descend to box center (OPEN) -> settle ->
close gripper so the force-hold latch engages -> IK-lift the grasp point straight up
(CLOSED) -> hold. Every IK piece here is the validated grasp-point-Jacobian servo.

    from ik_pick import pick
    res = pick(env, lift_target=0.12)
    # res = {success, max_lift_cm, held, centered_cm}
"""
from __future__ import annotations
import numpy as np
import mujoco


def grasp_jac(env):
    """Translational Jacobian of the grasp point (pad midpoint) wrt the 7 arm DOFs."""
    m, d = env.model, env.data
    jl = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
    mujoco.mj_jacBody(m, d, jl, None, env.lpad)
    mujoco.mj_jacBody(m, d, jr, None, env.rpad)
    return 0.5 * (jl + jr)[:, env.arm_dadr]


def servo_action(env, target, grip, damping=0.015):
    """One IK step: action that nudges the grasp point toward Cartesian `target`.
    grip: +1 closed, -1 open.

    Uses damped least squares (Levenberg-Marquardt) instead of a raw pseudo-inverse:
    dq = J^T (J J^T + lambda^2 I)^-1 err. Near-singular arm configs (box at the far
    edge of reach) stall a raw pinv but DLS keeps making progress in the conditioned
    directions, which is what rescued the one seed that previously never converged."""
    d = env.data
    J = grasp_jac(env)                                       # 3 x 7
    err = np.clip((target - env._grasp_point()) * 3.0, -0.06, 0.06)
    JJt = J @ J.T + (damping ** 2) * np.eye(3)
    dq = np.clip(J.T @ np.linalg.solve(JJt, err), -0.18, 0.18)
    q = d.qpos[env.arm_qadr] + dq
    a = np.zeros(8, np.float32)
    a[:7] = np.clip((q - env.arm_mid) / env.arm_half, -1, 1)
    a[7] = grip
    return a


def pick(env, lift_target=0.12, record=None):
    """Run reach -> settle -> close -> lift -> hold on an already-reset env.

    The env must be freshly reset (box at rest on the table). Returns:
        {success, max_lift_cm, held, centered_cm}
      success    : box was HELD (gripping) AND lifted >5cm AND still held at end
      max_lift_cm: peak held lift in cm
      held       : info["gripping"] at the final step
      centered_cm: mean box-to-grasp-point distance (cm) while gripping (<2.5 is good)

    If `record` is a list, (rgba-rendering hook) the caller can pass a callable that is
    invoked as record(env) each step to grab a frame.
    """
    d = env.data
    box = d.xpos[env.tool_bid].copy()      # box rest position
    z0 = float(box[2])

    def run(target_fn, grip, n):
        info = {}
        for _ in range(n):
            a = servo_action(env, target_fn(), grip)
            _, _, _, _, info = env.step(a)
            if record is not None:
                record(env)
        return info

    # 1. reach: grasp point to 10cm above the box, gripper OPEN (converge, don't just count)
    above = box + np.array([0.0, 0.0, 0.10])
    for _ in range(220):
        a = servo_action(env, above, -1.0)
        env.step(a)
        if record is not None:
            record(env)
        if np.linalg.norm(above - env._grasp_point()) < 0.012:
            break
    # 2. descend to box center, gripper OPEN — converge until the pads bracket box center
    for _ in range(220):
        a = servo_action(env, box, -1.0)
        env.step(a)
        if record is not None:
            record(env)
        if np.linalg.norm(box - env._grasp_point()) < 0.013:
            break
    # 3. settle at box center (kill residual motion before the pinch)
    run(lambda: box, -1.0, 15)
    # 4. close gripper so the two-sided-pinch force-hold latch engages
    run(lambda: box, 1.0, 50)

    # centering / grip tracking
    centered_samples = []
    max_lift = 0.0
    peak_gripping = False
    peak_box_to_gp = 0.0

    def track():
        nonlocal max_lift, peak_gripping, peak_box_to_gp
        gp = env._grasp_point()
        bx = d.xpos[env.tool_bid]
        if env._gripping:
            centered_samples.append(float(np.linalg.norm(bx - gp)))
        lift = float(bx[2] - z0)
        if lift > max_lift:
            max_lift = lift
            peak_gripping = bool(env._gripping)
            peak_box_to_gp = float(np.linalg.norm(bx - gp))

    # 5. IK-lift: servo grasp point straight up by +4mm/step, gripper CLOSED
    info = {}
    steps = 0
    max_lift_steps = 200
    while steps < max_lift_steps:
        a = servo_action(env, env._grasp_point() + np.array([0.0, 0.0, 0.004]), 1.0)
        _, _, _, _, info = env.step(a)
        if record is not None:
            record(env)
        track()
        steps += 1
        if (float(d.xpos[env.tool_bid][2]) - z0) >= lift_target and env._gripping:
            break

    # 6. hold at the top, gripper CLOSED
    for _ in range(40):
        gp = env._grasp_point()
        a = servo_action(env, gp, 1.0)
        _, _, _, _, info = env.step(a)
        if record is not None:
            record(env)
        track()

    held = bool(env._gripping)
    max_lift_cm = max_lift * 100.0
    centered_cm = float(np.mean(centered_samples) * 100.0) if centered_samples else float("nan")
    success = bool(held and max_lift_cm > 5.0 and info.get("gripping", False))

    return {
        "success": success,
        "max_lift_cm": max_lift_cm,
        "held": held,
        "centered_cm": centered_cm,
        # extra peak diagnostics (proves HELD not popped)
        "peak_gripping": peak_gripping,
        "peak_box_to_gp_cm": peak_box_to_gp * 100.0,
    }
