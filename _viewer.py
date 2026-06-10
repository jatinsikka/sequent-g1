"""Clean MuJoCo viewer for the factory cell, driven by the real AMO controller.

Physics runs in a background thread paced to REAL-TIME (decoupled from rendering so torch-CUDA
inference and OpenGL don't stall each other). Space plays the AMO-controlled robot at 1x speed.

Controls: L-drag orbit | R-drag/Shift pan | Scroll zoom | Space play/pause | R reset | Esc quit
"""
import threading
import time
import numpy as np
import mujoco
import glfw
from reward_fn import ButtonPressRewardFunction, BUTTON_POSITIONS
from env_wrapper_button import ButtonPressEnv

QPOS0 = 46
reward_fn = ButtonPressRewardFunction(button_position=BUTTON_POSITIONS["button_red"])
env = ButtonPressEnv(button_name="button_red", reward_fn=reward_fn,
                     freeze_arm="left", max_episode_steps=100000, headless=True)
model = env.env.model
data = env.env.data
zero_arm = np.zeros(env.action_space.shape, dtype=np.float32)
CONTROL_DT = float(getattr(env.env, "control_dt", 0.02))

sim_lock = threading.Lock()
playing = [False]
alive = [True]


def _do_reset():
    env.reset()  # actual task spawn (-90deg, facing the console) — now holds station after the action_scale fix
    mujoco.mj_forward(model, data)


def physics_loop():
    while alive[0]:
        if not playing[0]:
            time.sleep(0.005)
            continue
        t0 = time.perf_counter()
        with sim_lock:
            try:
                _, _, term, trunc, _ = env.step(zero_arm)
                if term or trunc:
                    _do_reset()
            except Exception:
                _do_reset()
        # pace one control step to real wall-clock time
        rem = CONTROL_DT - (time.perf_counter() - t0)
        if rem > 0:
            time.sleep(rem)


_do_reset()
threading.Thread(target=physics_loop, daemon=True).start()

btn_left = btn_right = btn_mid = False
lastx = lasty = 0.0


def on_key(window, key, scancode, act, mods):
    if act != glfw.PRESS:
        return
    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    elif key == glfw.KEY_R:
        with sim_lock:
            _do_reset()
    elif key == glfw.KEY_SPACE:
        playing[0] = not playing[0]


def on_mouse_button(window, button, act, mods):
    global btn_left, btn_right, btn_mid, lastx, lasty
    btn_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    btn_right = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    btn_mid = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    lastx, lasty = glfw.get_cursor_pos(window)


def on_mouse_move(window, xpos, ypos):
    global lastx, lasty
    dx, dy = xpos - lastx, ypos - lasty
    lastx, lasty = xpos, ypos
    if not (btn_left or btn_right or btn_mid):
        return
    _, height = glfw.get_window_size(window)
    shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or
             glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)
    if btn_right or btn_mid:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def on_scroll(window, xoffset, yoffset):
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)


glfw.init()
glfw.window_hint(glfw.SAMPLES, 4)
window = glfw.create_window(1366, 850, "Sequent — Factory Cell", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)
cam.lookat[:] = [0.0, -1.5, 0.9]
cam.distance = 4.6
cam.azimuth = 150
cam.elevation = -16

scene = mujoco.MjvScene(model, maxgeom=20000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

glfw.set_key_callback(window, on_key)
glfw.set_cursor_pos_callback(window, on_mouse_move)
glfw.set_mouse_button_callback(window, on_mouse_button)
glfw.set_scroll_callback(window, on_scroll)

HINT = "L-drag orbit   R-drag/Shift pan   Scroll zoom   Space play/pause   R reset   Esc quit"

while not glfw.window_should_close(window):
    w, h = glfw.get_framebuffer_size(window)
    viewport = mujoco.MjrRect(0, 0, w, h)
    with sim_lock:
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
    mujoco.mjr_render(viewport, scene, context)
    mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                       viewport, "Sequent Factory Cell" + ("  [RUNNING — AMO control, real-time]" if playing[0] else "  [PAUSED]"),
                       HINT, context)
    glfw.swap_buffers(window)
    glfw.poll_events()

alive[0] = False
time.sleep(0.05)
glfw.terminate()
