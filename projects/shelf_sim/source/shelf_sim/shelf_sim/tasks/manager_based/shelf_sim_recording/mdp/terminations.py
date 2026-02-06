# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for manipulation tasks.

Termination conditions:
- item_placed_in_slot: Success - item in target slot
- item_below_threshold: Failure - item dropped
- body_pos_out_of_bounds: Failure - EEF out of safe bounds
"""

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_robot_to_default(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Reset robot joints to default position and velocity.
    
    Args:
        env: The environment instance
        env_ids: Environment indices to reset
    """
    # Get robot asset directly
    asset: Articulation = env.scene["robot"]
    
    # Reset to default joint state
    default_pos = asset.data.default_joint_pos[env_ids]
    default_vel = asset.data.default_joint_vel[env_ids]
    asset.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)


def reset_scene_to_default(env: "ManagerBasedRLEnv", env_ids: torch.Tensor) -> None:
    """Reset scene objects to default state.
    
    Args:
        env: The environment instance
        env_ids: Environment indices to reset
    """
    # Reset all rigid objects to default state
    if env.scene.rigid_objects:
        for obj in env.scene.rigid_objects.values():
            # Write default state for specified environments
            default_state = obj.data.default_root_state[env_ids]
            obj.write_root_state_to_sim(default_state, env_ids=env_ids)


def item_placed_in_slot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if item is successfully placed in target slot.
    
    Returns True if:
    1. Item is within slot bounds (±5cm)
    2. Item velocity is below threshold
    3. Gripper is open
    """
    if "target_item" not in env.scene.rigid_objects:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    target_obj: RigidObject = env.scene.rigid_objects["target_item"]
    target_pos = target_obj.data.root_link_pos_w
    target_vel = target_obj.data.root_link_lin_vel_w

    slot_pos = torch.tensor(
        env.cfg.target_slot_position,
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0)
    tolerance = getattr(env.cfg, "slot_tolerance_m", 0.05)
    stable_vel = getattr(env.cfg, "stable_velocity_threshold_m_s", 0.02)

    pos_err = torch.abs(target_pos - slot_pos)
    within_xy = (pos_err[:, 0] <= tolerance) & (pos_err[:, 1] <= tolerance)
    within_z = pos_err[:, 2] <= tolerance
    stable = torch.linalg.vector_norm(target_vel, dim=-1) <= stable_vel

    gripper_open = _is_gripper_open(env)
    return within_xy & within_z & stable & gripper_open


def item_below_threshold(env: "ManagerBasedRLEnv", threshold: float = 0.5) -> torch.Tensor:
    """Check if item has fallen below threshold height.
    
    Args:
        threshold: Height below which item is considered dropped (meters).
        
    Returns True if item z-position is below threshold.
    """
    if "target_item" not in env.scene.rigid_objects:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    target_obj: RigidObject = env.scene.rigid_objects["target_item"]
    target_pos = target_obj.data.root_link_pos_w
    drop_threshold = getattr(env.cfg, "drop_height_threshold_m", threshold)
    return target_pos[:, 2] < drop_threshold


def body_pos_out_of_bounds(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
) -> torch.Tensor:
    """Check if body position is outside safe bounds.
    
    Args:
        bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        
    Returns True if body is outside bounds.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body position
    if asset_cfg.body_ids:
        pos = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    else:
        pos = asset.data.root_pos_w
    
    # Check bounds
    x_out = (pos[:, 0] < bounds[0][0]) | (pos[:, 0] > bounds[0][1])
    y_out = (pos[:, 1] < bounds[1][0]) | (pos[:, 1] > bounds[1][1])
    z_out = (pos[:, 2] < bounds[2][0]) | (pos[:, 2] > bounds[2][1])
    
    return x_out | y_out | z_out


def _is_gripper_open(env: "ManagerBasedRLEnv") -> torch.Tensor:
    robot: Articulation = env.scene["robot"]

    if not hasattr(env, "_gripper_joint_ids"):
        joint_names = getattr(env.cfg, "gripper_joint_names", ["fl_joint7", "fl_joint8"])
        joint_ids, _ = robot.find_joints(joint_names)
        env._gripper_joint_ids = joint_ids

    joint_pos = robot.data.joint_pos[:, env._gripper_joint_ids]
    open_threshold = getattr(env.cfg, "gripper_open_threshold", 0.02)
    return joint_pos.mean(dim=-1) >= open_threshold


