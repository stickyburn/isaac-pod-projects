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
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

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
    _body_name: str
    _joint_ids: list[int] | slice
    _joint_names: list[str]
    _num_joints: int
    _jacobi_body_idx: int
    _jacobi_joint_ids: list[int] | slice
    _controller: DifferentialIKController
    _scale: torch.Tensor
    _raw_actions: torch.Tensor
    _processed_actions: torch.Tensor

    def __init__(self, cfg: "EndEffectorPoseDeltaActionCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)

        # Resolve asset
        self._asset = self._env.scene[cfg.asset_name]

        # Resolve joints over which IK is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # Get body ID for the end-effector
        body_ids, body_names = self._asset.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for body name '{cfg.body_name}'. Found {len(body_ids)}: {body_names}."
            )
        self._body_id = body_ids[0]
        self._body_name = body_names[0]

        # Resolve jacobian indices
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_id - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_id
            if isinstance(self._joint_ids, slice):
                self._jacobi_joint_ids = slice(6, None)
            else:
                self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # Create IK controller
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            ik_method="dls",
            use_relative_mode=False,
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
        ).repeat(self.num_envs, 1)

        # Create buffers for actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._controller.reset(env_ids)
        if env_ids is None:
            self._raw_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0

    @property
    def action_dim(self) -> int:
        return self._controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        # Store raw actions
        self._raw_actions[:] = actions
        # Scale actions
        self._processed_actions[:] = self._raw_actions * self._scale

    def apply_actions(self) -> None:
        # Get current EEF pose
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_id]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_id]
        
        # Get current joint positions
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        
        # Compute target EEF pose from delta action
        # Position: current + delta
        target_pos = ee_pos_w + self._processed_actions[:, :3]
        
        # Orientation: apply delta quaternion to current
        delta_quat = self._processed_actions[:, 3:7]
        delta_quat = delta_quat / (torch.norm(delta_quat, dim=-1, keepdim=True) + 1e-8)
        target_quat = math_utils.quat_mul(ee_quat_w, delta_quat)
        
        # Stack pose command
        target_pose = torch.cat([target_pos, target_quat], dim=-1)

        # Set desired pose on controller
        self._controller.set_command(target_pose)

        # Compute joint targets using IK
        jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        joint_targets = self._controller.compute(ee_pos_w, ee_quat_w, jacobian, joint_pos)

        # Apply to robot (only arm joints)
        self._asset.set_joint_position_target(joint_targets, joint_ids=self._joint_ids)


@configclass
class EndEffectorPoseDeltaActionCfg(ActionTermCfg):
    """Configuration for end-effector pose delta action term."""
    
    class_type: type[ActionTerm] = EndEffectorPoseDeltaAction
    
    asset_name: str = MISSING
    """Name of the robot asset."""
    
    joint_names: list[str] = MISSING
    """Names of the arm joints used for IK."""

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
    _raw_actions: torch.Tensor
    _processed_actions: torch.Tensor

    def __init__(self, cfg: "GripperActionCfg", env: "ManagerBasedRLEnv") -> None:
        super().__init__(cfg, env)
        
        self._asset = self._env.scene[cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, len(self._joint_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
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
