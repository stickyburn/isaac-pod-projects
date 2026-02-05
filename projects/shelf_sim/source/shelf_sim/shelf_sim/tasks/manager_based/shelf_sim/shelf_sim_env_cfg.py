# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip

REPO_ROOT = Path(__file__).resolve().parents[8]
DEFAULT_ROBOT_USD = REPO_ROOT / "projects/piper_usd/piper_arm.usd"
ROBOT_BASE_POS = (-0.8, 0.0, 0.0)
ROBOT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)

##
# Scene definition
##


@configclass
class ShelfSimSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # table (kinematic static - work surface at z=0.75m)
    table = AssetBaseCfg(
        prim_path="/World/table",
        spawn=sim_utils.CuboidCfg(
            size=(1.2, 0.8, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.15, 0.15, 0.15)
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, z),
        ),
    )

    # bin (kinematic static - 4 walls + base on table)
    bin_base = AssetBaseCfg(
        prim_path="/World/bin/base",
        spawn=sim_utils.CuboidCfg(
            size=(0.44, 0.34, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.08, 0.12, 0.2)
            ),
            translation=(0.0, 0.0, 0.76),
        ),
    )
    bin_wall_px = AssetBaseCfg(
        prim_path="/World/bin/wall_px",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.34, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.16, 0.28)
            ),
            translation=(0.21, 0.0, 0.87),
        ),
    )
    bin_wall_nx = AssetBaseCfg(
        prim_path="/World/bin/wall_nx",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.34, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.16, 0.28)
            ),
            translation=(-0.21, 0.0, 0.87),
        ),
    )
    bin_wall_py = AssetBaseCfg(
        prim_path="/World/bin/wall_py",
        spawn=sim_utils.CuboidCfg(
            size=(0.44, 0.02, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.16, 0.28)
            ),
            translation=(0.0, 0.16, 0.87),
        ),
    )
    bin_wall_ny = AssetBaseCfg(
        prim_path="/World/bin/wall_ny",
        spawn=sim_utils.CuboidCfg(
            size=(0.44, 0.02, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.16, 0.28)
            ),
            translation=(0.0, -0.16, 0.87),
        ),
    )

    # shelf (kinematic static - frame + shelves at varying heights)
    shelf_frame_left = AssetBaseCfg(
        prim_path="/World/shelf/frame_left",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.6, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2)
            ),
            translation=(0.45, 0.0, 0.45),
        ),
    )
    shelf_frame_right = AssetBaseCfg(
        prim_path="/World/shelf/frame_right",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.6, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2)
            ),
            translation=(0.95, 0.0, 0.45),
        ),
    )
    shelf_back = AssetBaseCfg(
        prim_path="/World/shelf/back",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.02, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.2, 0.2)
            ),
            translation=(0.7, -0.29, 0.45),
        ),
    )
    shelf_bottom = AssetBaseCfg(
        prim_path="/World/shelf/shelf_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
            translation=(0.7, 0.0, 0.02),
        ),
    )
    shelf_middle = AssetBaseCfg(
        prim_path="/World/shelf/shelf_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
            translation=(0.7, 0.0, 0.44),
        ),
    )
    shelf_top = AssetBaseCfg(
        prim_path="/World/shelf/shelf_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18)
            ),
            translation=(0.7, 0.0, 0.86),
        ),
    )

    # robot
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(DEFAULT_ROBOT_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=ROBOT_BASE_POS,
            rot=ROBOT_BASE_ROT,
        ),
        actuators={
            "arm_joints": ImplicitActuatorCfg(
                joint_names_expr=["fl_joint[1-6]"],
                effort_limit_sim=100.0,
                velocity_limit_sim=3.0,
                stiffness=10000.0,
                damping=100.0,
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

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=1500.0,
            enable_color_temperature=True,
            color_temperature=5500,
        ),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=800.0,
            radius=0.25,
            color=(1.0, 0.95, 0.9),
            enable_color_temperature=True,
            color_temperature=5200,
            translation=(1.4, -1.2, 2.0),
        ),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.SphereLightCfg(
            intensity=400.0,
            radius=0.25,
            color=(0.85, 0.9, 1.0),
            enable_color_temperature=True,
            color_temperature=6500,
            translation=(-1.2, 1.2, 1.6),
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


##
# Environment configuration
##


@configclass
class ShelfSimEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: ShelfSimSceneCfg = ShelfSimSceneCfg(num_envs=4096, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation