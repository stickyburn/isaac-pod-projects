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
parser.add_argument("--task", type=str, default="Piper-Shelf-v0", help="Name of the task.")
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
from isaaclab.devices import Se3Keyboard

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
    print("\nRecording:")
    print("  ENTER - Mark episode as successful and save")
    print("\nOther:")
    print("  R - Reset environment")
    print("  ESC - Quit")
    print("="*60 + "\n")

    # Initialize keyboard device
    # Se3Keyboard outputs: [dx, dy, dz, droll, dpitch, dyaw]
    teleop_interface = Se3Keyboard(
        pos_sensitivity=0.01,  # 1cm per keypress
        rot_sensitivity=0.05   # ~3 degrees per keypress
    )
    
    # reset environment
    obs, info = env.reset()
    
    # Track gripper state (0 = closed, 1 = open)
    gripper_open = True
    
    # Track if we should record this episode
    episode_steps = []
    recording = False
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # Get keyboard input
            try:
                # Get delta pose from keyboard (6D: [dx, dy, dz, droll, dpitch, dyaw])
                delta_pose = teleop_interface.advance()
                
                # Create action tensor
                # Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                
                # Set delta pose commands
                action[0, 0] = delta_pose[0]  # dx
                action[0, 1] = delta_pose[1]  # dy
                action[0, 2] = delta_pose[2]  # dz
                action[0, 3] = delta_pose[3]  # droll
                action[0, 4] = delta_pose[4]  # dpitch
                action[0, 5] = delta_pose[5]  # dyaw
                
                # Check for gripper toggle (SPACE key)
                if teleop_interface.is_triggered("SPACE"):
                    gripper_open = not gripper_open
                    print(f"[INFO] Gripper: {'OPEN' if gripper_open else 'CLOSED'}")
                
                # Set gripper action (0 = close, 1 = open)
                action[0, 6] = 1.0 if gripper_open else -1.0
                
                # Check for reset
                if teleop_interface.is_triggered("R"):
                    print("[INFO] Resetting environment...")
                    obs, info = env.reset()
                    episode_steps = []
                    continue
                
                # Check for save/success (ENTER key)
                if teleop_interface.is_triggered("ENTER"):
                    print(f"[INFO] Episode marked as successful! ({len(episode_steps)} steps)")
                    # In a full implementation, this would save to HDF5
                    # For now, just reset
                    obs, info = env.reset()
                    episode_steps = []
                    continue
                
                # Apply the action
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Store step data for potential recording
                episode_steps.append({
                    'obs': obs.copy() if hasattr(obs, 'copy') else obs,
                    'action': action.cpu().numpy(),
                    'reward': reward,
                    'terminated': terminated,
                    'truncated': truncated,
                })
                
                # Handle automatic resets
                if terminated.any() or truncated.any():
                    print(f"[INFO] Episode ended. ({len(episode_steps)} steps)")
                    obs, info = env.reset()
                    episode_steps = []
                    
            except Exception as e:
                print(f"[ERROR] {e}")
                continue

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
