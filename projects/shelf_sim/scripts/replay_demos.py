from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay recorded HDF5 demos in the environment.")
    parser.add_argument("--task", type=str, required=True, help="Gym task name.")
    parser.add_argument("--input", type=Path, required=True, help="Input HDF5 path.")
    parser.add_argument("--demo_idx", type=int, default=None, help="Replay a single demo index.")
    parser.add_argument("--video", action="store_true", default=False, help="Record a replay video.")
    parser.add_argument("--video_dir", type=Path, default=None, help="Video output directory.")
    parser.add_argument(
        "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
    )
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    if args_cli.video:
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
from shelf_sim.tasks.manager_based.shelf_sim_recording import mdp


REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_dir = args_cli.video_dir or (REPO_ROOT / "projects/shelf_sim/reports/videos/replay")
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda episode: True,
            video_length=None,
            disable_logger=True,
        )

    with h5py.File(str(args_cli.input), "r") as data:
        demos = sorted(data["data"].keys())
        if args_cli.demo_idx is not None:
            demo_name = f"demo_{args_cli.demo_idx:04d}"
            if demo_name not in data["data"]:
                raise ValueError(f"Demo {demo_name} not found in {args_cli.input}")
            demos = [demo_name]

        for demo_name in demos:
            actions = np.asarray(data["data"][demo_name]["actions"], dtype=np.float32)
            obs, _ = env.reset()
            for action in actions:
                action_tensor = torch.tensor(action, device=env.unwrapped.device).unsqueeze(0)
                obs, reward, terminated, truncated, _ = env.step(action_tensor)
                if bool(terminated.item() or truncated.item()):
                    break

            success = bool(mdp.item_placed_in_slot(env.unwrapped)[0].item())
            print(f"[INFO] Replay {demo_name}: steps={len(actions)}, success={success}")

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
