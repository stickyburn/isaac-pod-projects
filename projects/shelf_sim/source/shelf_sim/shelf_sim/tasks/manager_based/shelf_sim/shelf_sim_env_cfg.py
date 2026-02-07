# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from isaaclab.envs.mdp import DifferentialIKControllerCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidBodyCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import random

from . import mdp
from shelf_sim import (
    MUSTARD_JAR_USD_PATH,
    OIL_TIN_USD_PATH,
    SALT_BOX_USD_PATH,
    BLUE_TIN_USD_PATH,
    TIN_CAN_USD_PATH,
)

##
# Pre-defined configs
##

from shelf_sim.robots import PIPER_HIGH_PD_CFG  # isort:skip

##
# Scene definition
##


@configclass
class ShelfSimSceneCfg(InteractiveSceneCfg):
    """Scene with Piper on a table."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.6, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.375)),
    )

    # robot - mounted on front edge of table, facing the shelf
    robot: ArticulationCfg = PIPER_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.25, 0.40),  # Center X, front edge Y, on top of table
            rot=(0.707, 0.0, 0.0, 0.707),  # 90 deg rotation to face shelf (Y+ direction)
        )
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fl_link1",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                name="ee_frame",
                prim_path="{ENV_REGEX_NS}/Robot/fl_link6",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
            )
        ],
        debug_vis=True,
    )

    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fl_link6/camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.05), rot=(0.5, 0.5, -0.5, -0.5)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=2500.0),
    )

    # Shelf - placed behind the table
    # Back panel
    shelf_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf/Back",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.6, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.8, 0.6)),
    )

    # Shelf levels (3 shelves)
    shelf_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf/Bottom",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.8, 0.1)),
    )

    shelf_middle = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf/Middle",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.8, 0.5)),
    )

    shelf_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf/Top",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.8, 0.9)),
    )

    # Side panels
    shelf_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf/Left",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.02, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 0.51, 0.6)),
    )

    shelf_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Shelf/Right",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.02, 1.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, 1.09, 0.6)),
    )

    # Bucket - open-top container on the table
    # Bucket bottom
    bucket_bottom = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bucket/Bottom",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, -0.1, 0.405)),
    )

    # Bucket sides
    bucket_side1 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bucket/Side1",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.01, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, -0.175, 0.46)),
    )

    bucket_side2 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bucket/Side2",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.01, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, -0.025, 0.46)),
    )

    bucket_side3 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bucket/Side3",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.15, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.125, -0.1, 0.46)),
    )

    bucket_side4 = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bucket/Side4",
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 0.15, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.6, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.275, -0.1, 0.46)),
    )

    graspable_item = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Item",
        spawn=RigidBodyCfg(
            usd_path=MUSTARD_JAR_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.2, -0.1, 0.5)),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP.

    Using joint position control for initial bring-up.
    We will switch to IK control after verifying joint/link names.
    """

    arm_action = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["fl_joint[1-6]"],
        scale=0.5,
        body_name="fl_link6",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls"
        )
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["fl_joint[7-8]"],
        open_command_expr={"fl_joint[7-8]": 0.035},
        close_command_expr={"fl_joint[7-8]": 0.0},
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


ITEM_USD_PATHS = [
    MUSTARD_JAR_USD_PATH,
    OIL_TIN_USD_PATH,
    SALT_BOX_USD_PATH,
    BLUE_TIN_USD_PATH,
    TIN_CAN_USD_PATH,
]


def spawn_random_item(env, scene_entity: SceneEntityCfg = SceneEntityCfg("graspable_item")):
    """Spawn a randomly selected item in the bin on environment reset."""
    import omni.usd
    from pxr import UsdGeom

    selected_path = random.choice(ITEM_USD_PATHS)

    for env_idx in range(env.num_envs):
        prim_path = f"{env.env_ns}/Item"

        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(prim_path):
            stage.RemovePrim(prim_path)

        omni.usd.get_context().get_stage().DefinePrim(prim_path, "Xform").GetReferences().AddReference(selected_path)

        prim = stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            UsdGeom.XformCommonAPI(prim).SetTranslate((0.2, -0.1, 0.5))


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    spawn_item = EventTerm(func=spawn_random_item, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP.

    Placeholder only -- imitation learning does not use rewards.
    The framework requires at least one reward term to run.
    """

    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##


@configclass
class ShelfSimEnvCfg(ManagerBasedRLEnvCfg):
    """Shelf-sim environment: Piper arm on a table."""
    # Scene settings
    scene: ShelfSimSceneCfg = ShelfSimSceneCfg(num_envs=1, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10.0
        # viewer -- camera positioned to see full scene (robot, table, bucket, shelf)
        self.viewer.eye = (1.2, -1.2, 1.5)
        self.viewer.lookat = (0.15, 0.3, 0.5)
        # simulation settings
        self.sim.dt = 0.01  # 100 Hz
        self.sim.render_interval = self.decimation