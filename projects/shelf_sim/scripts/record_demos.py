from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record teleop demonstrations to HDF5.")
    parser.add_argument("--task", type=str, required=True, help="Gym task name (e.g. Shelf-Sim-Recording-Session-A-v0).")
    parser.add_argument("--output", type=Path, default=None, help="Output HDF5 path (defaults to data/recordings).")
    parser.add_argument("--num_demos", type=int, default=5, help="Number of demonstrations to record.")
    parser.add_argument("--max_steps", type=int, default=None, help="Max steps per demo (default uses env max).")
    parser.add_argument("--min_steps", type=int, default=10, help="Minimum steps to keep a demo.")
    parser.add_argument("--no_rgb", action="store_true", default=False, help="Disable recording RGB frames.")
    parser.add_argument("--disable_cameras", action="store_true", default=False, help="Disable camera sensors.")
    parser.add_argument("--eef_body", type=str, default="fl_link8", help="EEF body name for teleop.")
    parser.add_argument("--pos_step", type=float, default=0.02, help="EEF translation step (meters).")
    parser.add_argument("--rot_step", type=float, default=0.08, help="EEF rotation step (radians).")
    parser.add_argument("--fast_scale", type=float, default=3.0, help="Speed multiplier when holding Shift.")
    parser.add_argument("--slow_scale", type=float, default=0.3, help="Speed multiplier when holding Ctrl.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )

    # Isaac Lab launcher args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if not args_cli.disable_cameras:
        args_cli.enable_cameras = True
    return args_cli


args_cli = _parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import shelf_sim.tasks  # noqa: F401
from shelf_sim.controllers.teleop_ik import IKTeleopController, KeyboardTeleopInterface
from shelf_sim.tasks.manager_based.shelf_sim_recording import mdp


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "projects/shelf_sim/data/recordings"


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
        self.obs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.rgb = [] if self.record_rgb else None

    def start_demo(self) -> None:
        self._reset_buffers()

    def record_step(self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool, rgb: np.ndarray | None):
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
        group.create_dataset("obs", data=np.asarray(self.obs, dtype=np.float32), compression="gzip", compression_opts=4)
        group.create_dataset(
            "actions", data=np.asarray(self.actions, dtype=np.float32), compression="gzip", compression_opts=4
        )
        group.create_dataset(
            "rewards", data=np.asarray(self.rewards, dtype=np.float32), compression="gzip", compression_opts=4
        )
        group.create_dataset("dones", data=np.asarray(self.dones, dtype=np.bool_), compression="gzip", compression_opts=4)
        if self.record_rgb and self.rgb:
            group.create_dataset(
                "camera_rgb", data=np.asarray(self.rgb, dtype=np.uint8), compression="gzip", compression_opts=4
            )

        group.attrs["success"] = bool(success)
        group.attrs["length"] = len(self.obs)
        self.demo_index += 1
        return True

    def close(self) -> None:
        self.file.flush()
        self.file.close()


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
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
    teleop = KeyboardTeleopInterface(teleop_controller)
    teleop.start()

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

    print(f"[INFO] Recording demos to: {output_path}")
    print("[INFO] Controls: WASD/QE translate, arrows rotate, Z/C yaw, Space gripper.")
    print("[INFO] Shift=fast, Ctrl=slow, R=reset, P/Enter=success, O/Backspace=failure, ESC=quit.")

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

            should_quit, should_reset, mark_success, mark_failure = teleop.consume_flags()
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

    teleop.stop()
    recorder.close()
    env.close()
    print(f"[INFO] Finished: {demos_recorded} demos, {success_count} successes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
