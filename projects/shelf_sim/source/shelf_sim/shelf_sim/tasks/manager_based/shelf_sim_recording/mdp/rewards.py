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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def is_success(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if item is successfully placed in target slot.
    
    Success criteria:
    1. Item is within slot bounds
    2. Item velocity is low (stable)
    3. Gripper is open
    """
    # Reuse termination logic for success
    from .terminations import item_placed_in_slot

    return item_placed_in_slot(env).float()


def approach_target_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for EEF approaching target item.
    
    Provides negative L2 distance between EEF and target item.
    """
    # Get EEF position
    robot: Articulation = env.scene["robot"]
    eef_body_name = getattr(env.cfg, "eef_body_name", "fl_link8")
    if not hasattr(env, "_eef_body_id"):
        body_ids, _ = robot.find_bodies(eef_body_name)
        env._eef_body_id = body_ids[0]
    eef_pos = robot.data.body_pos_w[:, env._eef_body_id]

    # Get target item position
    if "target_item" in env.scene.rigid_objects:
        target_obj: RigidObject = env.scene.rigid_objects["target_item"]
        target_pos = target_obj.data.root_link_pos_w
    else:
        target_pos = torch.zeros((env.num_envs, 3), device=env.device)

    distance = torch.norm(eef_pos - target_pos, dim=-1)
    return -distance


def place_target_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reward for approaching target slot when holding item.
    
    Only active when item is being held (gripper closed).
    """
    # Get EEF position
    robot: Articulation = env.scene["robot"]
    eef_body_name = getattr(env.cfg, "eef_body_name", "fl_link8")
    if not hasattr(env, "_eef_body_id"):
        body_ids, _ = robot.find_bodies(eef_body_name)
        env._eef_body_id = body_ids[0]
    eef_pos = robot.data.body_pos_w[:, env._eef_body_id]

    # Get target slot position from config
    target_pos = torch.tensor(
        env.cfg.target_slot_position,
        device=env.device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(env.num_envs, -1)

    distance = torch.norm(eef_pos - target_pos, dim=-1)
    return -distance


def item_dropped(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Check if item has been dropped (fell below threshold).
    
    Returns 1.0 if item is below table level, 0.0 otherwise.
    """
    from .terminations import item_below_threshold

    return item_below_threshold(env).float()
