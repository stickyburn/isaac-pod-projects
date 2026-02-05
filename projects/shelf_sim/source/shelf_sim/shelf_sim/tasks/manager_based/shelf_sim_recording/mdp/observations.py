# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for manipulation tasks.

Observations:
- Camera RGB from wrist or scene camera
- Robot joint positions and velocities
- End-effector pose (position + quaternion)
- Target slot position
- Gripper state
"""

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def camera_rgb(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """RGB image from camera sensor.
    
    Returns a flattened RGB image tensor of shape (N, H*W*3).
    """
    sensor: Camera = env.scene.sensors[sensor_cfg.name]
    # Get RGB data and flatten
    rgb = sensor.data.output["rgb"]
    # Convert from (N, H, W, C) to (N, H*W*C)
    batch_size = rgb.shape[0]
    return rgb.reshape(batch_size, -1)


def body_pos(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body position in world frame.
    
    Returns position of specified body/bodies.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        return asset.data.body_pos_w[:, asset_cfg.body_ids].reshape(env.num_envs, -1)
    return asset.data.body_pos_w.reshape(env.num_envs, -1)


def body_quat(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Body quaternion in world frame (xyzw format).
    
    Returns quaternion of specified body/bodies.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids:
        return asset.data.body_quat_w[:, asset_cfg.body_ids].reshape(env.num_envs, -1)
    return asset.data.body_quat_w.reshape(env.num_envs, -1)


def target_slot_position(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Target slot position in world frame.
    
    This reads the target slot position from the environment configuration.
    """
    # Access target slot position from env config
    target_pos = torch.tensor(
        env.cfg.target_slot_position,
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(env.num_envs, -1)
    return target_pos


def joint_pos(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions.
    
    Returns positions of specified joints.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids:
        return asset.data.joint_pos[:, asset_cfg.joint_ids]
    return asset.data.joint_pos


def joint_pos_rel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Relative joint positions.
    
    Returns joint positions relative to default values.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids:
        return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return asset.data.joint_pos - asset.data.default_joint_pos


def joint_vel_rel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Relative joint velocities.
    
    Returns joint velocities relative to default values.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.joint_ids:
        return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    return asset.data.joint_vel - asset.data.default_joint_vel
