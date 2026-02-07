from pathlib import Path
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_PIPER_USD_PATH = str(
    Path(__file__).resolve().parents[5] / "piper_usd" / "piper.usd"
)

##
# Configuration
##

PIPER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_PIPER_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
      joint_pos={
        "fl_joint[1-6]": 0.0,
        "fl_joint[7-8]": 0.0,
      },
    ),
    actuators={
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["fl_joint[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=5.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "gripper_joints": ImplicitActuatorCfg(
            joint_names_expr=["fl_joint[7-8]"],
            effort_limit_sim=10.0,
            velocity_limit_sim=1.0,
            stiffness=500.0,
            damping=50.0,
        ),
    },
)
"""Configuration of Piper arm."""


PIPER_HIGH_PD_CFG = PIPER_CFG.copy()
PIPER_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
PIPER_HIGH_PD_CFG.actuators["arm_joints"].stiffness = 400.0
PIPER_HIGH_PD_CFG.actuators["arm_joints"].damping = 80.0
"""Configuration of Piper arm with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""