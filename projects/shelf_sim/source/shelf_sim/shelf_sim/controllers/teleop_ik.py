# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""IK-based teleoperation controller for shelf-sim.

Maps keyboard inputs to EEF delta pose commands using differential IK.

Controls:
- WASD: XY plane movement
- Q/E: Z axis (up/down)
- Arrow keys: Rotation around axes
- Z/C: Yaw rotation
- Shift: Fast movement
- Ctrl: Fine movement
- Space: Toggle gripper (open/close)
- R: Reset to home position
- P/Enter: Mark success (for recording)
- O/Backspace: Mark failure (for recording)
- ESC: Exit
"""

from typing import TYPE_CHECKING

import torch
import numpy as np
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.devices import DeviceBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class IKTeleopController(DeviceBase):
    """Keyboard-based teleoperation with IK control.
    
    Maps keyboard inputs to end-effector delta pose commands.
    Uses differential IK to convert to joint commands.
    
    Args:
        env: The environment instance
        robot_asset_name: Name of the robot asset
        eef_body_name: Name of the end-effector body
        position_step: Step size for position control (meters)
        rotation_step: Step size for rotation control (radians)
    """

    def __init__(
        self,
        env: "ManagerBasedRLEnv",
        robot_asset_name: str = "robot",
        eef_body_name: str = "fl_link8",
        position_step: float = 0.02,
        rotation_step: float = 0.08,
        fast_scale: float = 3.0,
        slow_scale: float = 0.3,
    ):
        super().__init__()
        
        self.env = env
        self.robot_asset_name = robot_asset_name
        self.eef_body_name = eef_body_name
        self.position_step = position_step
        self.rotation_step = rotation_step
        self.fast_scale = fast_scale
        self.slow_scale = slow_scale
        
        # Get robot asset
        self.robot = env.scene[robot_asset_name]
        self.eef_body_id = self.robot.find_bodies(eef_body_name)[0][0]
        
        # Create IK controller
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            ik_method="dls",
        )
        self.ik_controller = DifferentialIKController(
            cfg=ik_cfg,
            num_envs=env.num_envs,
            device=env.device,
        )
        
        # Command state
        self._target_pos = None
        self._target_quat = None
        self._gripper_open = True
        self._gripper_action = 1.0  # Positive = open
        
        # Keyboard state
        self._key_state = {
            'w': False, 'a': False, 's': False, 'd': False,
            'q': False, 'e': False,
            'up': False, 'down': False, 'left': False, 'right': False,
            'z': False, 'c': False,
            'shift': False, 'ctrl': False,
        }
        
        # Initialize target pose
        self._initialize_target_pose()

    def _initialize_target_pose(self):
        """Initialize target pose from current EEF pose."""
        # Get current EEF pose
        ee_pos_w = self.robot.data.body_pos_w[:, self.eef_body_id]
        ee_quat_w = self.robot.data.body_quat_w[:, self.eef_body_id]
        
        self._target_pos = ee_pos_w.clone()
        self._target_quat = ee_quat_w.clone()

    def add_callback(self, key: str, func):
        """Add callback for specific key (required by DeviceBase abstract method)."""
        pass  # Not used for this controller - we use process_key instead

    def reset(self):
        """Reset controller state."""
        self.ik_controller.reset()
        self._initialize_target_pose()
        self._gripper_open = True
        self._gripper_action = 1.0

    def process_key(self, key: str, pressed: bool):
        """Process keyboard input.
        
        Args:
            key: Key identifier ('w', 'a', 's', 'd', 'q', 'e', 'up', 'down', 'left', 'right', 'space', 'r')
            pressed: True if key pressed, False if released
        """
        if key in self._key_state:
            self._key_state[key] = pressed
        elif key == 'space' and pressed:
            # Toggle gripper
            self._gripper_open = not self._gripper_open
            self._gripper_action = 1.0 if self._gripper_open else -1.0
        elif key == 'r' and pressed:
            # Reset to home
            self.reset()

    def advance(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute actions from current key state.
        
        Returns:
            Tuple of (arm_action, gripper_action)
            - arm_action: Tensor of shape (num_envs, 7) with [dx, dy, dz, dqx, dqy, dqz, dqw]
            - gripper_action: Tensor of shape (num_envs, 1) with gripper command
        """
        # Compute speed scaling from modifiers
        if self._key_state['shift']:
            scale = self.fast_scale
        elif self._key_state['ctrl']:
            scale = self.slow_scale
        else:
            scale = 1.0

        # Compute delta position from key state
        dx = 0.0
        dy = 0.0
        dz = 0.0
        
        if self._key_state['w']:
            dy += self.position_step * scale  # Forward
        if self._key_state['s']:
            dy -= self.position_step * scale  # Backward
        if self._key_state['a']:
            dx -= self.position_step * scale  # Left
        if self._key_state['d']:
            dx += self.position_step * scale  # Right
        if self._key_state['q']:
            dz += self.position_step * scale  # Up
        if self._key_state['e']:
            dz -= self.position_step * scale  # Down
        
        # Compute delta rotation from key state (simplified as Euler angles)
        drx = 0.0
        dry = 0.0
        drz = 0.0
        
        if self._key_state['up']:
            drx += self.rotation_step * scale
        if self._key_state['down']:
            drx -= self.rotation_step * scale
        if self._key_state['left']:
            dry += self.rotation_step * scale
        if self._key_state['right']:
            dry -= self.rotation_step * scale
        if self._key_state['z']:
            drz += self.rotation_step * scale
        if self._key_state['c']:
            drz -= self.rotation_step * scale
        
        # Update target pose
        self._target_pos[:, 0] += dx
        self._target_pos[:, 1] += dy
        self._target_pos[:, 2] += dz
        
        # Convert Euler rotation to quaternion delta
        if drx != 0 or dry != 0 or drz != 0:
            delta_quat = self._euler_to_quaternion(drx, dry, drz)
            self._target_quat = self._quat_multiply(self._target_quat, delta_quat)
        
        # Compute arm action as delta from current pose
        current_pos = self.robot.data.body_pos_w[:, self.eef_body_id]
        current_quat = self.robot.data.body_quat_w[:, self.eef_body_id]
        
        delta_pos = self._target_pos - current_pos
        delta_quat = self._quat_delta(current_quat, self._target_quat)
        
        arm_action = torch.cat([delta_pos, delta_quat], dim=-1)
        
        # Gripper action
        gripper_action = torch.full(
            (self.env.num_envs, 1),
            self._gripper_action,
            device=self.env.device,
        )
        
        return arm_action, gripper_action

    def _euler_to_quaternion(self, rx: float, ry: float, rz: float) -> torch.Tensor:
        """Convert Euler angles to quaternion (xyzw format)."""
        cx, sx = np.cos(rx/2), np.sin(rx/2)
        cy, sy = np.cos(ry/2), np.sin(ry/2)
        cz, sz = np.cos(rz/2), np.sin(rz/2)
        
        qw = cx*cy*cz + sx*sy*sz
        qx = sx*cy*cz - cx*sy*sz
        qy = cx*sy*cz + sx*cy*sz
        qz = cx*cy*sz - sx*sy*cz
        
        quat = torch.tensor([qx, qy, qz, qw], device=self.env.device)
        return quat.unsqueeze(0).expand(self.env.num_envs, -1)

    def _quat_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions (xyzw format)."""
        x1, y1, z1, w1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        x2, y2, z2, w2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return torch.stack([x, y, z, w], dim=-1)

    def _quat_delta(self, q_current: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        """Compute quaternion delta from current to target."""
        # Delta = target * conj(current)
        q_current_conj = q_current.clone()
        q_current_conj[:, :3] = -q_current_conj[:, :3]
        return self._quat_multiply(q_target, q_current_conj)


class KeyboardTeleopInterface:
    """Keyboard interface for teleoperation.
    
    Handles keyboard events and routes them to the IK controller.
    """

    def __init__(self, controller: IKTeleopController):
        self.controller = controller
        self._running = False
        self._keyboard_sub = None
        self.should_quit = False
        self.should_reset = False
        self.mark_success = False
        self.mark_failure = False

    def start(self):
        """Start keyboard listening."""
        self._running = True
        try:
            import omni.appwindow
            app_window = omni.appwindow.get_default_app_window()
            input_interface = app_window.get_keyboard()
            
            # Subscribe to keyboard events
            self._keyboard_sub = input_interface.subscribe_to_key_event(self._on_key_event)
        except ImportError:
            print("Warning: Could not setup keyboard interface in headless mode")

    def stop(self):
        """Stop keyboard listening."""
        self._running = False
        self._keyboard_sub = None

    def consume_flags(self) -> tuple[bool, bool, bool, bool]:
        """Consume one-shot control flags (quit/reset/success/failure)."""
        flags = (self.should_quit, self.should_reset, self.mark_success, self.mark_failure)
        self.should_reset = False
        self.mark_success = False
        self.mark_failure = False
        return flags

    def _on_key_event(self, event):
        """Handle keyboard event."""
        if not self._running:
            return

        import omni.appwindow

        key_map = {
            'W': 'w', 'A': 'a', 'S': 's', 'D': 'd',
            'Q': 'q', 'E': 'e',
            'UP': 'up', 'DOWN': 'down', 'LEFT': 'left', 'RIGHT': 'right',
            'Z': 'z', 'C': 'c',
            'LEFT_SHIFT': 'shift', 'RIGHT_SHIFT': 'shift',
            'LEFT_CONTROL': 'ctrl', 'RIGHT_CONTROL': 'ctrl',
        }

        key = event.input.name
        pressed = event.type in (
            omni.appwindow.KeyboardEventType.KEY_PRESS,
            omni.appwindow.KeyboardEventType.KEY_REPEAT,
        )
        if key in key_map:
            key = key_map[key]

        if key in self.controller._key_state:
            self.controller.process_key(key, pressed)
        elif key == 'SPACE' and pressed:
            self.controller.process_key('space', True)
        elif key == 'R' and pressed:
            self.controller.process_key('r', True)
            self.should_reset = True
        elif key in ('ESCAPE', 'ESC') and pressed:
            self.should_quit = True
        elif key in ('P', 'ENTER', 'RETURN') and pressed:
            self.mark_success = True
        elif key in ('O', 'BACKSPACE') and pressed:
            self.mark_failure = True

