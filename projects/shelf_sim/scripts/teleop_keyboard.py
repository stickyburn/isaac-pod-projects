"""Keyboard teleoperation script for shelf_sim environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleop for ShelfSim environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Piper-Shelf-v0", help="Name of the task.")
parser.add_argument(
    "--record_hdf5",
    action="store_true",
    default=False,
    help="Record demonstrations to HDF5 using Isaac Lab's recorder manager.",
)
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/shelf_sim_teleop.hdf5",
    help="HDF5 output file path for demonstrations.",
)
parser.add_argument("--step_hz", type=int, default=30, help="Target stepping rate (Hz). Set to 0 to disable.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode

import shelf_sim.tasks  # noqa: F401


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz: int):
        self.hz = hz
        self.period = 1.0 / hz
        self.next_time = time.time() + self.period

    def sleep(self, env: gym.Env) -> None:
        if self.hz <= 0:
            return
        while time.time() < self.next_time:
            time.sleep(min(0.01, self.period))
            env.sim.render()
        self.next_time += self.period
        if self.next_time < time.time():
            self.next_time = time.time() + self.period


def _configure_recording(env_cfg) -> str | None:
    if not args_cli.record_hdf5:
        return None

    output_dir = os.path.dirname(args_cli.dataset_file) or "."
    output_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    env_cfg.env_name = args_cli.task.split(":")[-1]
    env_cfg.terminations.time_out = None
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

    return os.path.join(output_dir, f"{output_name}.hdf5")


def main():
    """Keyboard teleop agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    dataset_path = _configure_recording(env_cfg)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    if dataset_path:
        print(f"[INFO]: HDF5 recording enabled -> {dataset_path}")
        if args_cli.num_envs != 1:
            print("[WARN]: --record_hdf5 expects num_envs=1; only env 0 will be exported.")
    print("\n" + "=" * 60)
    print("KEYBOARD CONTROLS:")
    print("=" * 60)
    print("Movement (Delta EEF Pose):")
    print("  W/S - Move Forward/Backward (X axis)")
    print("  A/D - Move Left/Right (Y axis)")
    print("  Q/E - Move Up/Down (Z axis)")
    print("\nRotation:")
    print("  Z/X - Roll +/-")
    print("  T/G - Pitch +/-")
    print("  C/V - Yaw +/-")
    print("\nGripper:")
    print("  K - Toggle Open/Close")
    if args_cli.record_hdf5:
        print("\nRecording:")
        print("  ENTER - Mark episode as successful and save to HDF5")
    print("\nOther:")
    print("  R - Reset environment")
    print("  ESC - Quit")
    print("=" * 60 + "\n")

    # Initialize keyboard device
    teleop_interface = Se3Keyboard(
        Se3KeyboardCfg(
            pos_sensitivity=0.01,  # 1 cm per keypress
            rot_sensitivity=0.05,  # ~3 degrees per keypress
            sim_device=args_cli.device,
        )
    )

    # Callback flags
    state = {"reset": False, "save": False, "quit": False}

    def request_reset():
        state["reset"] = True

    def request_save():
        state["save"] = True

    def request_quit():
        state["quit"] = True

    teleop_interface.add_callback("R", request_reset)
    teleop_interface.add_callback("ENTER", request_save)
    teleop_interface.add_callback("RETURN", request_save)
    teleop_interface.add_callback("ESCAPE", request_quit)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # rate limiter for stable recordings
    rate_limiter = RateLimiter(args_cli.step_hz) if args_cli.step_hz and args_cli.step_hz > 0 else None

    action_dim = env.action_space.shape[-1]

    # simulate environment
    while simulation_app.is_running() and not state["quit"]:
        # run everything in inference mode
        with torch.inference_mode():
            # Get command from keyboard (delta pose + gripper)
            command = teleop_interface.advance()

            # Create action tensor and broadcast to all envs
            actions = torch.zeros((env.num_envs, action_dim), device=env.unwrapped.device)
            actions[:, : command.numel()] = command

            # Apply the action
            _, _, terminated, truncated, _ = env.step(actions)

            # Handle save to HDF5
            if state["save"]:
                if args_cli.record_hdf5:
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[True]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    count = env.recorder_manager.exported_successful_episode_count
                    print(f"[INFO] Saved successful demo #{count}.")
                else:
                    print("[WARN] Recording disabled. Use --record_hdf5 to save demos.")
                state["save"] = False
                state["reset"] = True

            # Handle automatic resets
            if terminated.any() or truncated.any():
                print("[INFO] Episode ended. Resetting...")
                state["reset"] = True

            # Reset environment if requested
            if state["reset"]:
                env.sim.reset()
                if args_cli.record_hdf5:
                    env.recorder_manager.reset()
                env.reset()
                teleop_interface.reset()
                state["reset"] = False

        if rate_limiter:
            rate_limiter.sleep(env)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
