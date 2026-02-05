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


def item_placed_in_slot(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if item is successfully placed in target slot.
    
    Returns True if:
    1. Item is within slot bounds (±5cm)
    2. Item velocity is below threshold
    3. Gripper is open
    """
    # TODO: Implement proper item tracking
    # For now, always return False (no success)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def item_below_threshold(env: "ManagerBasedRLEnv", threshold: float = 0.5) -> torch.Tensor:
    """Check if item has fallen below threshold height.
    
    Args:
        threshold: Height below which item is considered dropped (meters).
        
    Returns True if item z-position is below threshold.
    """
    # TODO: Track actual item position
    # For now, return False
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


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
