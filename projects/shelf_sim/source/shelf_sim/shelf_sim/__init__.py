# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Python module serving as a project/extension template.
"""

import os

# Extension paths
SHELF_SIM_EXT_DIR = os.path.dirname(os.path.realpath(__file__))
SHELF_SIM_DATA_DIR = os.path.join(SHELF_SIM_EXT_DIR, "data")

# Asset paths
PROPS_BOXED_DIR = f"{SHELF_SIM_DATA_DIR}/Props/Boxed"
PROPS_CONTAINERS_DIR = f"{SHELF_SIM_DATA_DIR}/Props/Containers"

MUSTARD_JAR_USD_PATH = f"{PROPS_BOXED_DIR}/mustard_jar.usd"
OIL_TIN_USD_PATH = f"{PROPS_BOXED_DIR}/oil_tin.usd"
SALT_BOX_USD_PATH = f"{PROPS_BOXED_DIR}/salt_box.usd"
BLUE_TIN_USD_PATH = f"{PROPS_BOXED_DIR}/blue_tin.usd"
TIN_CAN_USD_PATH = f"{PROPS_CONTAINERS_DIR}/TinCan.usd"

# Register Gym environments.
from .tasks import *

# Register UI extensions.
from .ui_extension_example import *
