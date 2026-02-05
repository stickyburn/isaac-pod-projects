# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recording environment configurations for shelf-sim imitation learning.

This module defines the fixed scene configurations for human demonstrations.
Each session type represents a different fixed scene configuration that humans
can reliably demonstrate. During recording, the scene does NOT randomize -
it stays fixed throughout the session to ensure clean demonstrations.

Session Types:
- A: Single item in bin, empty middle shelf target
- B: Two items in bin (target visible on top), empty middle shelf target
- C: Single item in bin, middle shelf has 1 distractor item
- D: Single item in bin, top shelf target (different height)
- E: Different item type (e.g., mustard_jar instead of blue_tin)
- F: Target item offset in bin (not center)
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from . import mdp

##
# Paths and constants
##

REPO_ROOT = Path(__file__).resolve().parents[8]
DEFAULT_MANIFEST = REPO_ROOT / "projects/shelf_sim/assets_manifest.json"
DEFAULT_ROBOT_USD = REPO_ROOT / "projects/piper_usd/piper_arm.usd"

ROBOT_BASE_POS = (-0.8, 0.0, 0.0)
ROBOT_BASE_ROT = (1.0, 0.0, 0.0, 0.0)

# Table and bin dimensions
TABLE_SIZE_M = (1.2, 0.8, 0.05)
TABLE_TOP_Z_M = 0.75
BIN_INNER_SIZE_M = (0.4, 0.3)
BIN_WALL_THICKNESS_M = 0.02
BIN_WALL_HEIGHT_M = 0.2
BIN_BASE_THICKNESS_M = 0.02

# Shelf positions (3 shelves)
SHELF_POSITIONS = {
    "bottom": 0.02,  # z position
    "middle": 0.44,
    "top": 0.86,
}
SHELF_BOUNDS = {
    "x_min": 0.45,
    "x_max": 0.95,
    "y_min": -0.30,
    "y_max": 0.30,
}

# Camera defaults
CAMERA_RESOLUTION = (224, 224)  # For visuomotor policy
DEFAULT_CAMERA_POS = (1.3, -1.2, 1.2)
DEFAULT_CAMERA_TARGET = (0.0, 0.0, TABLE_TOP_Z_M + 0.1)

##
# Scene definition
##


@configclass
class ShelfSimRecordingSceneCfg(InteractiveSceneCfg):
    """Base scene configuration for shelf-sim recording environment."""

    # Number of environments (1 for recording)
    num_envs: int = 1
    env_spacing: float = 4.0

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # Table (kinematic static - work surface)
    table = AssetBaseCfg(
        prim_path="/World/table",
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE_M,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_TOP_Z_M - TABLE_SIZE_M[2] / 2.0),
        ),
    )

    # Bin (kinematic static - 4 walls + base)
    bin_base = AssetBaseCfg(
        prim_path="/World/bin/base",
        spawn=sim_utils.CuboidCfg(
            size=(BIN_INNER_SIZE_M[0] + 2 * BIN_WALL_THICKNESS_M, 
                  BIN_INNER_SIZE_M[1] + 2 * BIN_WALL_THICKNESS_M, 
                  BIN_BASE_THICKNESS_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.12, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M / 2.0),
        ),
    )
    bin_wall_px = AssetBaseCfg(
        prim_path="/World/bin/wall_px",
        spawn=sim_utils.CuboidCfg(
            size=(BIN_WALL_THICKNESS_M, BIN_INNER_SIZE_M[1] + 2 * BIN_WALL_THICKNESS_M, BIN_WALL_HEIGHT_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.16, 0.28)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                BIN_INNER_SIZE_M[0] / 2.0 + BIN_WALL_THICKNESS_M / 2.0,
                0.0,
                TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + BIN_WALL_HEIGHT_M / 2.0,
            ),
        ),
    )
    bin_wall_nx = AssetBaseCfg(
        prim_path="/World/bin/wall_nx",
        spawn=sim_utils.CuboidCfg(
            size=(BIN_WALL_THICKNESS_M, BIN_INNER_SIZE_M[1] + 2 * BIN_WALL_THICKNESS_M, BIN_WALL_HEIGHT_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.16, 0.28)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                -BIN_INNER_SIZE_M[0] / 2.0 - BIN_WALL_THICKNESS_M / 2.0,
                0.0,
                TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + BIN_WALL_HEIGHT_M / 2.0,
            ),
        ),
    )
    bin_wall_py = AssetBaseCfg(
        prim_path="/World/bin/wall_py",
        spawn=sim_utils.CuboidCfg(
            size=(BIN_INNER_SIZE_M[0], BIN_WALL_THICKNESS_M, BIN_WALL_HEIGHT_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.16, 0.28)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                0.0,
                BIN_INNER_SIZE_M[1] / 2.0 + BIN_WALL_THICKNESS_M / 2.0,
                TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + BIN_WALL_HEIGHT_M / 2.0,
            ),
        ),
    )
    bin_wall_ny = AssetBaseCfg(
        prim_path="/World/bin/wall_ny",
        spawn=sim_utils.CuboidCfg(
            size=(BIN_INNER_SIZE_M[0], BIN_WALL_THICKNESS_M, BIN_WALL_HEIGHT_M),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.16, 0.28)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(
                0.0,
                -BIN_INNER_SIZE_M[1] / 2.0 - BIN_WALL_THICKNESS_M / 2.0,
                TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + BIN_WALL_HEIGHT_M / 2.0,
            ),
        ),
    )

    # Shelf (kinematic static - frame + 3 shelves)
    shelf_frame_left = AssetBaseCfg(
        prim_path="/World/shelf/frame_left",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.6, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.45, 0.0, 0.45),
        ),
    )
    shelf_frame_right = AssetBaseCfg(
        prim_path="/World/shelf/frame_right",
        spawn=sim_utils.CuboidCfg(
            size=(0.02, 0.6, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.95, 0.0, 0.45),
        ),
    )
    shelf_back = AssetBaseCfg(
        prim_path="/World/shelf/back",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.02, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.7, -0.29, 0.45),
        ),
    )
    shelf_bottom = AssetBaseCfg(
        prim_path="/World/shelf/shelf_0",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.7, 0.0, SHELF_POSITIONS["bottom"]),
        ),
    )
    shelf_middle = AssetBaseCfg(
        prim_path="/World/shelf/shelf_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.7, 0.0, SHELF_POSITIONS["middle"]),
        ),
    )
    shelf_top = AssetBaseCfg(
        prim_path="/World/shelf/shelf_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.6, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.7, 0.0, SHELF_POSITIONS["top"]),
        ),
    )

    # Robot (AgileX Piper)
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

    # Camera (for visuomotor observations)
    camera = CameraCfg(
        prim_path="/World/camera",
        height=CAMERA_RESOLUTION[0],
        width=CAMERA_RESOLUTION[1],
        update_period=0,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=DEFAULT_CAMERA_POS,
            rot=(0.5, -0.5, 0.5, -0.5),  # Look at scene
            convention="world",
        ),
    )

    # Target indicator (emissive marker for target slot)
    target_indicator = AssetBaseCfg(
        prim_path="/World/target_indicator",
        spawn=sim_utils.CylinderCfg(
            radius=0.03,
            height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0),
                emissive_color=(0.0, 0.5, 0.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.7, 0.0, SHELF_POSITIONS["middle"] + 0.02),
        ),
    )

    # Lights
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
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.4, -1.2, 2.0),
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
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.2, 1.2, 1.6),
        ),
    )


##
# MDP Settings
##


@configclass
class ActionsCfg:
    """Action specifications for manipulation.
    
    Actions: EEF delta pose (SE3) + gripper open/close
    - 3D position delta (dx, dy, dz)
    - 4D quaternion delta (dqx, dqy, dqz, dqw) 
    - 1D gripper action (positive=open, negative=close)
    """

    # EEF delta pose action (will use IK controller)
    arm_action = mdp.EndEffectorPoseDeltaActionCfg(
        asset_name="robot",
        joint_names=["fl_joint[1-6]"],
        body_name="fl_link8",  # EEF link name
        position_scale=0.1,  # Max 10cm per step
        rotation_scale=0.1,  # Max rotation per step
    )
    
    # Gripper action
    gripper_action = mdp.GripperActionCfg(
        asset_name="robot",
        joint_names=["fl_joint7", "fl_joint8"],
        open_value=0.04,  # meters
        close_value=0.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for visuomotor policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Camera RGB observation
        camera_rgb = ObsTerm(
            func=mdp.camera_rgb,
            params={"sensor_cfg": SceneEntityCfg("camera")},
        )
        
        # Robot state
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        
        # EEF pose (position + quaternion)
        eef_pos = ObsTerm(
            func=mdp.body_pos,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["fl_link8"])},
        )
        eef_quat = ObsTerm(
            func=mdp.body_quat,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["fl_link8"])},
        )
        
        # Target slot pose
        target_slot_pos = ObsTerm(
            func=mdp.target_slot_position,
        )
        
        # Gripper state
        gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["fl_joint7", "fl_joint8"])},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events (resets and domain randomization)."""

    # Reset robot to home position
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),  # Fixed home pose
            "velocity_range": (0.0, 0.0),
        },
    )

    # Reset scene objects to default state.
    reset_scene = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Task completion reward
    success = RewTerm(
        func=mdp.is_success,
        weight=10.0,
    )
    
    # Shaping: distance to target item
    approach_reward = RewTerm(
        func=mdp.approach_target_reward,
        weight=1.0,
    )
    
    # Shaping: distance to target slot when holding item
    place_reward = RewTerm(
        func=mdp.place_target_reward,
        weight=1.0,
    )
    
    # Penalty for dropping item
    drop_penalty = RewTerm(
        func=mdp.item_dropped,
        weight=-5.0,
    )
    
    # Penalty for timeout
    timeout_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-1.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # Success: item placed in target slot
    success = DoneTerm(func=mdp.item_placed_in_slot)
    
    # Failure: item dropped (fell below table)
    item_dropped = DoneTerm(func=mdp.item_below_threshold)
    
    # Timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # EEF out of bounds
    eef_out_of_bounds = DoneTerm(
        func=mdp.body_pos_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["fl_link8"]),
            "bounds": ((-1.5, 1.5), (-1.5, 1.5), (0.0, 2.0)),
        },
    )


##
# Session Type Configurations
##


@configclass
class SessionAEnvCfg(ManagerBasedRLEnvCfg):
    """Session A: Single item in bin, empty middle shelf target."""
    
    # Scene settings
    scene: ShelfSimRecordingSceneCfg = ShelfSimRecordingSceneCfg(num_envs=1, env_spacing=4.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Session-specific parameters
    session_type: str = "A"
    target_item: str = "blue_tin"  # Primary item to grasp
    target_slot: str = "middle"  # Target shelf
    target_slot_position: tuple[float, float, float] = (0.7, 0.0, SHELF_POSITIONS["middle"] + 0.05)
    bin_item_count: int = 1
    bin_item_positions: list[tuple[float, float, float]] = [(0.0, 0.0, TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + 0.05)]
    bin_item_types: list[str] = ["blue_tin"]
    distractor_items: list[str] = []  # No distractors
    target_indicator_position: tuple[float, float, float] = (0.7, 0.0, SHELF_POSITIONS["middle"] + 0.02)

    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 30  # 30 seconds max
        self.viewer.eye = (2.0, -1.5, 1.5)
        self.viewer.lookat = (0.0, 0.0, TABLE_TOP_Z_M)
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


@configclass
class SessionBEnvCfg(SessionAEnvCfg):
    """Session B: Two items in bin (target visible on top), empty middle shelf target."""
    
    session_type: str = "B"
    target_item: str = "blue_tin"
    target_slot: str = "middle"
    bin_item_count: int = 2
    # Target on top, distractor below
    bin_item_positions: list[tuple[float, float, float]] = [
        (0.0, 0.0, TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + 0.12),  # Target on top
        (0.05, 0.0, TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + 0.05),  # Distractor below
    ]
    bin_item_types: list[str] = ["blue_tin", "mustard_jar"]
    distractor_items: list[str] = ["mustard_jar"]


@configclass
class SessionCEnvCfg(SessionAEnvCfg):
    """Session C: Single item in bin, middle shelf has 1 distractor item."""
    
    session_type: str = "C"
    target_item: str = "blue_tin"
    target_slot: str = "middle"
    bin_item_count: int = 1
    bin_item_positions: list[tuple[float, float, float]] = [(0.0, 0.0, TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + 0.05)]
    bin_item_types: list[str] = ["blue_tin"]
    # Distractor on shelf
    shelf_distractor: str = "salt_box"
    shelf_distractor_position: tuple[float, float, float] = (0.75, 0.15, SHELF_POSITIONS["middle"] + 0.05)


@configclass
class SessionDEnvCfg(SessionAEnvCfg):
    """Session D: Single item in bin, top shelf target (different height)."""
    
    session_type: str = "D"
    target_item: str = "blue_tin"
    target_slot: str = "top"  # Different height
    target_slot_position: tuple[float, float, float] = (0.7, 0.0, SHELF_POSITIONS["top"] + 0.05)
    bin_item_count: int = 1
    bin_item_positions: list[tuple[float, float, float]] = [(0.0, 0.0, TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + 0.05)]
    bin_item_types: list[str] = ["blue_tin"]
    target_indicator_position: tuple[float, float, float] = (0.7, 0.0, SHELF_POSITIONS["top"] + 0.02)


@configclass
class SessionEEnvCfg(SessionAEnvCfg):
    """Session E: Different item type (mustard_jar instead of blue_tin)."""
    
    session_type: str = "E"
    target_item: str = "mustard_jar"  # Different item type
    target_slot: str = "middle"
    bin_item_count: int = 1
    bin_item_positions: list[tuple[float, float, float]] = [(0.0, 0.0, TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + 0.05)]
    bin_item_types: list[str] = ["mustard_jar"]


@configclass
class SessionFEnvCfg(SessionAEnvCfg):
    """Session F: Target item offset in bin (not center)."""
    
    session_type: str = "F"
    target_item: str = "blue_tin"
    target_slot: str = "middle"
    bin_item_count: int = 1
    # Offset from center
    bin_item_positions: list[tuple[float, float, float]] = [
        (0.1, -0.08, TABLE_TOP_Z_M + BIN_BASE_THICKNESS_M + 0.05)  # Offset position
    ]
    bin_item_types: list[str] = ["blue_tin"]
