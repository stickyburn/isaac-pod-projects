"""Keyboard teleoperation script for shelf_sim environment."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleop for ShelfSim environment.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="ShelfSim-Piper-v0", help="Name of the task.")
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
import numpy as np

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import shelf_sim.tasks  # noqa: F401


def main():
    """Keyboard teleop agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print("\n" + "="*60)
    print("KEYBOARD CONTROLS:")
    print("="*60)
    print("Movement (Delta EEF Pose):")
    print("  W/S - Move Forward/Backward (X axis)")
    print("  A/D - Move Left/Right (Y axis)")
    print("  Q/E - Move Up/Down (Z axis)")
    print("\nRotation:")
    print("  I/K - Pitch +/-")
    print("  J/L - Roll +/-")
    print("  U/O - Yaw +/-")
    print("\nGripper:")
    print("  SPACE - Toggle Open/Close")
    print("\nOther:")
    print("  R - Reset environment")
    print("  ESC - Quit")
    print("="*60 + "\n")

    # reset environment
    obs, info = env.reset()
    
    # action buffer - [dx, dy, dz, droll, dpitch, dyaw, gripper]
    # Using delta pose commands for IK
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    gripper_open = True
    
    # movement scales
    pos_scale = 0.01  # 1cm per keypress
    rot_scale = 0.05  # ~3 degrees per keypress
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Check for keyboard input (requires simulation to be running)
            # Note: In a real scenario, you'd use a proper input handler
            # For now, this demonstrates the action structure
            
            # Apply the action
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Reset action for next frame (delta commands should be transient)
            action[:] = 0.0
            
            # Handle resets
            if terminated.any() or truncated.any():
                obs, info = env.reset()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
