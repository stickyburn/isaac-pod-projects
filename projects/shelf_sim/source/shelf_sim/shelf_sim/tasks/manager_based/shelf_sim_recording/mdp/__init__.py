# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for shelf-sim recording environment.

This module provides the specific MDP terms needed for manipulation tasks:
- Actions: EEF delta pose control with IK, gripper control
- Observations: Camera RGB, robot state, EEF pose, object states
- Rewards: Task completion, shaping terms, penalties
- Terminations: Success, failure, timeout conditions
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import *
from .observations import *
from .rewards import *
from .terminations import *
