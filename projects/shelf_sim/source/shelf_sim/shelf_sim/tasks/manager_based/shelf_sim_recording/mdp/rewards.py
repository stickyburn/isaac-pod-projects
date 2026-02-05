# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for manipulation tasks.

Reward terms:
- is_success: Item placed in target slot with gripper open
- approach_target_reward: Shaping for approaching target item
- place_target_reward: Shaping for approaching target slot when holding item
- item_dropped: Penalty for dropping item
"""

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_success(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if item is successfully placed in target slot.
    
    Success criteria:
    1. Item is within slot bounds
    2. Item velocity is low (stable)
    3. Gripper is open
    """
    # TODO: Implement based on item and gripper state
    # For now, return zeros
    return torch.zeros(env.num_envs, device=env.device)


def approach_target_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for EEF approaching target item.
    
    Provides negative L2 distance between EEF and target item.
    """
    # Get EEF position
    robot: Articulation = env.scene["robot"]
    eef_pos = robot.data.body_pos_w[:, 0]  # Assuming body 0 is EEF
    
    # Get target item position (first item in scene)
    # TODO: Access item position properly
    # For now, use fixed position from bin
    target_pos = torch.tensor(
        [0.0, 0.0, 0.8],
        device=env.device,
    ).unsqueeze(0).expand(env.num_envs, -1)
    
    # Compute distance
    distance = torch.norm(eef_pos - target_pos, dim=-1)
    
    # Negative distance as reward (closer is better)
    return -distance


def place_target_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for approaching target slot when holding item.
    
    Only active when item is being held (gripper closed).
    """
    # Get EEF position
    robot: Articulation = env.scene["robot"]
    eef_pos = robot.data.body_pos_w[:, 0]
    
    # Get target slot position from config
    target_pos = torch.tensor(
        env.cfg.target_slot_position,
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(env.num_envs, -1)
    
    # Compute distance
    distance = torch.norm(eef_pos - target_pos, dim=-1)
    
    # Negative distance as reward
    return -distance


def item_dropped(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if item has been dropped (fell below threshold).
    
    Returns 1.0 if item is below table level, 0.0 otherwise.
    """
    # TODO: Track item position and check if below threshold
    # For now, return zeros
    return torch.zeros(env.num_envs, device=env.device)
