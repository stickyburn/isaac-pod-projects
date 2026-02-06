"""Record teleop demonstrations to HDF5.

Launch (from KASM VNC terminal):
    /opt/IsaacLab/isaaclab.sh -p ./scripts/record_demos.py [--task ...] [--num_demos ...]

The simulation runs headless with offscreen camera rendering (isaaclab.python.rendering.kit).
An OpenCV window is created on the X11 display (:1.0) for visual feedback and keyboard input.

Controls (focus the OpenCV window):
    WASD/QE  - Translate EEF (XY plane / Z axis)
    Arrows   - Rotate EEF (pitch/yaw)
    Z/C      - Roll rotation
    Space    - Toggle gripper open/close
    Shift    - Fast movement (hold Shift + movement key)
    F        - Fine/slow movement
    R        - Reset episode
    P/Enter  - Mark current demo as SUCCESS
    O/Bksp   - Mark current demo as FAILURE
    ESC      - Quit
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record teleop demonstrations to HDF5.")
    parser.add_argument("--task", type=str, default="Shelf-Sim-Recording-Session-A-v0",
                        help="Gym task name.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output HDF5 path (defaults to data/recordings).")
    parser.add_argument("--num_demos", type=int, default=5,
                        help="Number of demonstrations to record.")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max steps per demo (default uses env max).")
    parser.add_argument("--min_steps", type=int, default=10,
                        help="Minimum steps to keep a demo.")
    parser.add_argument("--no_rgb", action="store_true", default=False,
                        help="Disable recording RGB frames.")
    parser.add_argument("--disable_cameras", action="store_true", default=False,
                        help="Disable camera sensors.")
    parser.add_argument("--eef_body", type=str, default="fl_link8",
                        help="EEF body name for teleop.")
    parser.add_argument("--pos_step", type=float, default=0.02,
                        help="EEF translation step (meters).")
    parser.add_argument("--rot_step", type=float, default=0.08,
                        help="EEF rotation step (radians).")
    parser.add_argument("--fast_scale", type=float, default=3.0,
                        help="Speed multiplier when holding Shift.")
    parser.add_argument("--slow_scale", type=float, default=0.3,
                        help="Speed multiplier when holding Ctrl/F.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False,
        help="Disable fabric and use USD I/O operations.",
    )

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    if not args_cli.disable_cameras:
        args_cli.enable_cameras = True

    # DO NOT manually set args_cli.experience here.
    # AppLauncher._config_resolution auto-selects the experience AND sets
    # internal viewport/render flags.  Manual override bypasses those flags.

    return args_cli


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Now safe to import Omni / Isaac Lab modules ────────────────────────────

import cv2
import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import shelf_sim.tasks  # noqa: F401
from shelf_sim.controllers.teleop_ik import IKTeleopController
from shelf_sim.tasks.manager_based.shelf_sim_recording import mdp

# ── Verify OpenCV has GUI support (non-headless build) ─────────────────────

def _check_cv2_gui() -> None:
    """Verify OpenCV was built with GUI (highgui) support.

    The Isaac Sim container ships ``opencv-python-headless`` which strips
    ``cv2.imshow`` / ``cv2.namedWindow``.  Teleop needs a real X11 window
    for display + keyboard input.  If the check fails we print the exact
    pip commands to fix it and exit before wasting time on scene creation.
    """
    try:
        test_win = "__cv2_gui_probe__"
        cv2.namedWindow(test_win, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(test_win)
    except cv2.error:
        print(
            "\n"
            "=" * 64 + "\n"
            "  ERROR: OpenCV has no GUI backend (highgui).\n"
            "=" * 64 + "\n"
            "  The installed opencv-python-headless package cannot create\n"
            "  X11 windows.  Teleop requires a visible window for camera\n"
            "  display and keyboard input.\n"
            "\n"
            "  Fix (run once in your container):\n"
            "\n"
            "    pip uninstall -y opencv-python-headless && \\\n"
            "    pip install opencv-python\n"
            "\n"
            "  Then re-run this script.\n"
            "=" * 64 + "\n"
        )
        raise SystemExit(1)


_check_cv2_gui()


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "projects/shelf_sim/data/recordings"

# ── X11 keysym constants (from /usr/include/X11/keysymdef.h) ───────────────

_XK_UP = 0xFF52
_XK_DOWN = 0xFF54
_XK_LEFT = 0xFF51
_XK_RIGHT = 0xFF53
_XK_SHIFT_L = 0xFFE1
_XK_SHIFT_R = 0xFFE2
_XK_CTRL_L = 0xFFE5
_XK_CTRL_R = 0xFFE6
_XK_SPACE = 32
_XK_ESCAPE = 27
_XK_RETURN = 13
_XK_BACKSPACE = 8


# ── Helpers ─────────────────────────────────────────────────────────────────

def _extract_policy_obs(obs) -> torch.Tensor:
    if isinstance(obs, dict):
        return obs.get("policy", next(iter(obs.values())))
    return obs


def _read_rgb(env) -> np.ndarray | None:
    if "camera" not in env.unwrapped.scene.sensors:
        return None
    camera = env.unwrapped.scene.sensors["camera"]
    frame = camera.data.output["rgb"][0].detach().cpu().numpy()
    if frame.dtype != np.uint8:
        frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _build_hud(frame: np.ndarray, step: int, demo: int, total: int,
               controller: IKTeleopController, success_count: int) -> np.ndarray:
    """Overlay HUD text on camera frame for teleop visualization."""
    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = display.shape[:2]

    # Scale up for easier viewing (camera is 224x224)
    if max(h, w) < 400:
        scale = max(1, 480 // max(h, w))
        display = cv2.resize(display, (w * scale, h * scale),
                             interpolation=cv2.INTER_NEAREST)
        h, w = display.shape[:2]

    # Semi-transparent bar at top
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, display, 0.45, 0, display)

    gripper = "OPEN" if controller._gripper_open else "CLOSED"
    line1 = f"Demo {demo}/{total}  Step {step}"
    line2 = f"Gripper: {gripper}  OK: {success_count}"
    cv2.putText(display, line1, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(display, line2, (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 1, cv2.LINE_AA)
    return display


def _process_cv2_key(key: int, controller: IKTeleopController):
    """Map an OpenCV waitKeyEx code to teleop commands.

    Returns (should_quit, should_reset, mark_success, mark_failure).

    Key state model: every frame we reset all movement keys to False, then
    set only the key that was pressed THIS frame.  OS key-repeat generates
    repeated press events while a key is held, giving continuous movement.
    """
    should_quit = False
    should_reset = False
    mark_success = False
    mark_failure = False

    # Reset all movement state each frame
    for k in list(controller._key_state):
        controller._key_state[k] = False

    if key < 0:
        return should_quit, should_reset, mark_success, mark_failure

    # Decode extended key code
    keysym = key & 0xFFFF

    # ── Arrow keys ──────────────────────────────────────────────────────
    if keysym == _XK_UP:
        controller._key_state["up"] = True
        return should_quit, should_reset, mark_success, mark_failure
    if keysym == _XK_DOWN:
        controller._key_state["down"] = True
        return should_quit, should_reset, mark_success, mark_failure
    if keysym == _XK_LEFT:
        controller._key_state["left"] = True
        return should_quit, should_reset, mark_success, mark_failure
    if keysym == _XK_RIGHT:
        controller._key_state["right"] = True
        return should_quit, should_reset, mark_success, mark_failure

    # ── Modifier-only presses (Shift/Ctrl alone – no movement) ─────────
    if keysym in (_XK_SHIFT_L, _XK_SHIFT_R, _XK_CTRL_L, _XK_CTRL_R):
        return should_quit, should_reset, mark_success, mark_failure

    # ── Regular ASCII keys ──────────────────────────────────────────────
    char = chr(keysym & 0xFF) if keysym < 256 else ""

    # Shift detection: uppercase letter means Shift was held
    if char.isupper():
        controller._key_state["shift"] = True
        char = char.lower()

    # Fine mode: 'f' key (easier than Ctrl through VNC)
    if char == "f":
        controller._key_state["ctrl"] = True
        return should_quit, should_reset, mark_success, mark_failure

    # Movement
    if char in ("w", "a", "s", "d", "q", "e", "z", "c"):
        controller._key_state[char] = True

    # Gripper toggle
    elif keysym == _XK_SPACE:
        controller.process_key("space", True)

    # Reset
    elif char == "r":
        controller.process_key("r", True)
        should_reset = True

    # Mark success
    elif char == "p" or keysym == _XK_RETURN:
        mark_success = True

    # Mark failure
    elif char == "o" or keysym == _XK_BACKSPACE:
        mark_failure = True

    # Quit
    elif keysym == _XK_ESCAPE:
        should_quit = True

    return should_quit, should_reset, mark_success, mark_failure


class DemoRecorder:
    def __init__(self, output_path: Path, metadata: dict, record_rgb: bool):
        self.output_path = output_path
        self.record_rgb = record_rgb
        self.file = h5py.File(str(output_path), "w")
        self.data_group = self.file.create_group("data")
        self.file.attrs["metadata"] = json.dumps(metadata)
        self.demo_index = 0
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self.obs: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.rgb: list[np.ndarray] | None = [] if self.record_rgb else None

    def start_demo(self) -> None:
        self._reset_buffers()

    def record_step(self, obs: np.ndarray, action: np.ndarray,
                    reward: float, done: bool, rgb: np.ndarray | None):
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        if self.record_rgb and rgb is not None:
            self.rgb.append(rgb)

    def end_demo(self, success: bool, min_steps: int) -> bool:
        if len(self.obs) < min_steps:
            return False

        group = self.data_group.create_group(f"demo_{self.demo_index:04d}")
        group.create_dataset("obs", data=np.asarray(self.obs, dtype=np.float32),
                             compression="gzip", compression_opts=4)
        group.create_dataset("actions", data=np.asarray(self.actions, dtype=np.float32),
                             compression="gzip", compression_opts=4)
        group.create_dataset("rewards", data=np.asarray(self.rewards, dtype=np.float32),
                             compression="gzip", compression_opts=4)
        group.create_dataset("dones", data=np.asarray(self.dones, dtype=np.bool_),
                             compression="gzip", compression_opts=4)
        if self.record_rgb and self.rgb:
            group.create_dataset("camera_rgb", data=np.asarray(self.rgb, dtype=np.uint8),
                                 compression="gzip", compression_opts=4)

        group.attrs["success"] = bool(success)
        group.attrs["length"] = len(self.obs)
        self.demo_index += 1
        return True

    def close(self) -> None:
        self.file.flush()
        self.file.close()


# ── Main loop ───────────────────────────────────────────────────────────────

def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make(args_cli.task, cfg=env_cfg)

    teleop_controller = IKTeleopController(
        env.unwrapped,
        eef_body_name=args_cli.eef_body,
        position_step=args_cli.pos_step,
        rotation_step=args_cli.rot_step,
        fast_scale=args_cli.fast_scale,
        slow_scale=args_cli.slow_scale,
    )
    step_dt = getattr(env.unwrapped, "step_dt", 0.0)

    # ── Set up OpenCV window for teleop display + keyboard input ────────
    # The rendering experience (isaaclab.python.rendering.kit) does offscreen
    # rendering for cameras but does NOT create a visible X11 window.
    # We create our own via OpenCV so the user can see the camera feed on
    # VNC and control the robot through keyboard events on that window.
    win_name = "Shelf Sim - Teleop"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 640, 640)
    # Show a placeholder until the first camera frame arrives
    placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Waiting for camera...", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1, cv2.LINE_AA)
    cv2.imshow(win_name, placeholder)
    cv2.waitKey(1)

    # ── Output / recorder setup ─────────────────────────────────────────
    output_path = args_cli.output
    if output_path is None:
        DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUT_DIR / f"{args_cli.task}_{stamp}.hdf5"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task": args_cli.task,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "teleop": {
            "eef_body": args_cli.eef_body,
            "pos_step": args_cli.pos_step,
            "rot_step": args_cli.rot_step,
            "fast_scale": args_cli.fast_scale,
            "slow_scale": args_cli.slow_scale,
        },
    }
    recorder = DemoRecorder(output_path, metadata, record_rgb=not args_cli.no_rgb)

    print(f"\n{'='*60}")
    print("  SHELF SIM TELEOP RECORDING")
    print(f"{'='*60}")
    print(f"  Task:     {args_cli.task}")
    print(f"  Output:   {output_path}")
    print(f"  Demos:    0/{args_cli.num_demos}")
    print(f"  Display:  OpenCV window (DISPLAY={os.environ.get('DISPLAY', '?')})")
    print(f"{'='*60}")
    print("  WASD/QE  translate  |  Arrows  rotate")
    print("  Space    gripper    |  Z/C     roll")
    print("  Shift    fast       |  F       fine")
    print("  R reset | P success | O failure | ESC quit")
    print(f"{'='*60}")
    print("  >>> Click the OpenCV window to give it keyboard focus <<<")
    print(f"{'='*60}\n")

    demos_recorded = 0
    success_count = 0
    obs, _ = env.reset()
    teleop_controller.reset()
    max_steps = args_cli.max_steps or env.unwrapped.max_episode_length
    step_count = 0
    recorder.start_demo()

    while simulation_app.is_running():
        with torch.inference_mode():
            current_obs = _extract_policy_obs(obs).detach().cpu().numpy()
            rgb = _read_rgb(env) if not args_cli.no_rgb else None

            # Compute teleop action from current key state
            arm_action, gripper_action = teleop_controller.advance(step_dt)
            action = torch.cat([arm_action, gripper_action], dim=-1)

            obs, reward, terminated, truncated, _ = env.step(action)

            done = bool(terminated.item() or truncated.item())
            recorder.record_step(
                current_obs,
                action.detach().cpu().numpy()[0],
                float(reward.item()),
                done,
                rgb,
            )
            step_count += 1

            # ── Display camera + capture keyboard via OpenCV ────────────
            if rgb is not None:
                hud = _build_hud(rgb, step_count, demos_recorded,
                                 args_cli.num_demos, teleop_controller, success_count)
                cv2.imshow(win_name, hud)

            key = cv2.waitKeyEx(1)
            should_quit, should_reset, mark_success, mark_failure = _process_cv2_key(
                key, teleop_controller
            )

            if should_quit:
                break

            if should_reset:
                obs, _ = env.reset()
                teleop_controller.reset()
                recorder.start_demo()
                step_count = 0
                continue

            success = bool(mdp.item_placed_in_slot(env.unwrapped)[0].item())
            if mark_success:
                success = True
                done = True
            if mark_failure:
                success = False
                done = True

            if step_count >= max_steps:
                done = True

            if done:
                if recorder.end_demo(success, args_cli.min_steps):
                    demos_recorded += 1
                    success_count += int(success)
                    print(
                        f"[INFO] Demo {demos_recorded}/{args_cli.num_demos} saved "
                        f"(steps={step_count}, success={success})."
                    )
                else:
                    print(f"[INFO] Demo discarded (steps={step_count} < {args_cli.min_steps}).")

                if demos_recorded >= args_cli.num_demos:
                    break

                obs, _ = env.reset()
                teleop_controller.reset()
                recorder.start_demo()
                step_count = 0

    cv2.destroyAllWindows()
    recorder.close()
    env.close()
    print(f"\n[INFO] Finished: {demos_recorded} demos, {success_count} successes.")
    print(f"[INFO] Saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
