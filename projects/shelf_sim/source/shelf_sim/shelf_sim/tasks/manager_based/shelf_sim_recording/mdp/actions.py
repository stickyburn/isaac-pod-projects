# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for manipulation tasks.

Actions:
- EndEffectorPoseDeltaActionCfg: EEF delta pose control using IK
- GripperActionCfg: Binary gripper open/close control
"""

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.controllers.differential_ik import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EndEffectorPoseDeltaAction(ActionTerm):
    """End-effector pose delta action using differential IK.
    
    This action term controls the robot arm by specifying delta changes to the
    end-effector pose (position + orientation). The action is converted to joint
    targets using differential inverse kinematics.
    
    Action space: 7D (3 position delta + 4 quaternion delta)
    - dx, dy, dz: Position delta in meters
    - dqx, dqy, dqz, dqw: Quaternion delta (will be normalized)
    """

    cfg: "EndEffectorPoseDeltaActionCfg"
    _asset: "Articulation"
    _body_id: int
    _controller: DifferentialIKController
    _scale: torch.Tensor

    def __init__(self, cfg: "EndEffectorPoseDeltaActionCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        
        # Resolve asset
        self._asset = self._env.scene[cfg.asset_name]
        
        # Get body ID for the end-effector
        self._body_id = self._asset.find_bodies(cfg.body_name)[0][0]
        
        # Create IK controller
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            ik_method="dls",
            position_offset=None,
        )
        self._controller = DifferentialIKController(
            cfg=ik_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )
        
        # Store scale for position and rotation
        self._scale = torch.tensor(
            [cfg.position_scale, cfg.position_scale, cfg.position_scale,
             cfg.rotation_scale, cfg.rotation_scale, cfg.rotation_scale, cfg.rotation_scale],
            device=self.device,
        ).unsqueeze(0)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._controller.reset(env_ids)

    def process_actions(self, actions: torch.Tensor) -> None:
        # Scale actions
        self._processed_actions = actions * self._scale

    def apply_actions(self) -> None:
        # Get current EEF pose
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_id]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_id]
        
        # Get current joint positions
        joint_pos = self._asset.data.joint_pos
        
        # Compute target EEF pose from delta action
        # Position: current + delta
        target_pos = ee_pos_w + self._processed_actions[:, :3]
        
        # Orientation: apply delta quaternion to current
        delta_quat = self._processed_actions[:, 3:7]
        delta_quat = delta_quat / (torch.norm(delta_quat, dim=-1, keepdim=True) + 1e-8)
        target_quat = self._quat_multiply(ee_quat_w, delta_quat)
        
        # Stack pose command
        target_pose = torch.cat([target_pos, target_quat], dim=-1)
        
        # Compute joint targets using IK
        joint_targets = self._controller.compute(target_pose, joint_pos, self._body_id)
        
        # Apply to robot (only arm joints, not gripper)
        self._asset.set_joint_position_target(joint_targets)

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1[:, 3], q1[:, 0], q1[:, 1], q1[:, 2]
        w2, x2, y2, z2 = q2[:, 3], q2[:, 0], q2[:, 1], q2[:, 2]
        
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return torch.stack([x, y, z, w], dim=-1)


@configclass
class EndEffectorPoseDeltaActionCfg(ActionTermCfg):
    """Configuration for end-effector pose delta action term."""
    
    class_type: type[ActionTerm] = EndEffectorPoseDeltaAction
    
    asset_name: str = MISSING
    """Name of the robot asset."""
    
    body_name: str = MISSING
    """Name of the end-effector body."""
    
    position_scale: float = 0.1
    """Scaling factor for position delta (meters)."""
    
    rotation_scale: float = 0.1
    """Scaling factor for rotation delta (quaternion)."""


class GripperAction(ActionTerm):
    """Gripper open/close action.
    
    Controls the gripper joint positions. Positive values open, negative close.
    
    Action space: 1D
    - > 0: Open gripper
    - < 0: Close gripper
    """

    cfg: "GripperActionCfg"
    _asset: "Articulation"
    _joint_ids: list[int]

    def __init__(self, cfg: "GripperActionCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        
        self._asset = self._env.scene[cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)

    def process_actions(self, actions: torch.Tensor) -> None:
        # Binary gripper control: positive = open, negative = close
        gripper_cmd = actions[:, 0]
        
        # Map to target positions
        target_pos = torch.where(
            gripper_cmd > 0,
            torch.full_like(gripper_cmd, self.cfg.open_value),
            torch.full_like(gripper_cmd, self.cfg.close_value),
        )
        
        self._processed_actions = target_pos.unsqueeze(1).expand(-1, len(self._joint_ids))

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )


@configclass
class GripperActionCfg(ActionTermCfg):
    """Configuration for gripper action term."""
    
    class_type: type[ActionTerm] = GripperAction
    
    asset_name: str = MISSING
    """Name of the robot asset."""
    
    joint_names: list[str] = MISSING
    """Names of the gripper joints."""
    
    open_value: float = 0.04
    """Joint position when gripper is open (meters)."""
    
    close_value: float = 0.0
    """Joint position when gripper is closed (meters)."""
