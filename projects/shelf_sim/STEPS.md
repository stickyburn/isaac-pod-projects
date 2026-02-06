# Shelf Sim: Imitation Learning with Piper Arm

## Goal

Simulate a shelf-stocking task with an Agilex Piper arm. Record human teleop demonstrations, expand them with Isaac Lab Mimic, and train a visuomotor BC policy that can later transfer to the real robot.

## Pipeline

```
SceneSetup -> IKControl -> Teleop+Record -> Annotate -> MimicGenerate -> TrainBC -> Evaluate
```

---

## Completed

### Milestone 1: Project Scaffolding

- Generated project from `isaaclab.sh --new` (manager-based, single-agent)
- Restructured: moved `.gitignore`, `.gitattributes`, `.flake8`, `.vscode/` to workspace root
- Removed RSL-RL scripts and configs (not needed for imitation learning)
- Cleaned up `/bak` directory (standalone test scripts replaced by proper env)

### Milestone 2: Piper Arm in Simulation

- Created `robots/piper.py` with `PIPER_CFG` and `PIPER_HIGH_PD_CFG` (for future IK use)
- Piper USD at `projects/piper_usd/piper_arm.usd` (converted from URDF, fixed base, convex hull colliders)
- Joint structure: `fl_joint[1-6]` (revolute arm), `fl_joint[7-8]` (prismatic gripper)
- Link structure: `base_link`, `fl_link[1-6]` (arm), `fl_link[7-8]` (gripper fingers)
- Rewrote `shelf_sim_env_cfg.py`: ground + table + Piper arm + dome light
- Actions: joint position control (arm) + binary gripper
- Registered as `ShelfSim-Piper-v0`
- Tested with `random_agent.py --task ShelfSim-Piper-v0 --num_envs 1` via KASM VNC
- **Result**: Isaac Sim launches, arm and table visible, arm moves with random actions

**Known issue**: Arm appears white/untextured. The URDF meshes were removed after USD conversion. The mesh geometry is baked into the USD but the material/texture references may be broken. Options: re-convert from URDF with meshes present, or apply materials in the USD directly. Not blocking -- cosmetic only.

---

## Next Steps

### Step 3: Verify Joint/Link Names from USD

Before adding IK control, we need to confirm the exact link names in the USD (particularly the end-effector link and root link). Write a small introspection script that loads the USD and prints all joint/link names. This tells us:
- Root link name (for `FrameTransformerCfg.prim_path`)
- End-effector body name (for `DifferentialInverseKinematicsActionCfg.body_name`)
- Gripper tip offset (for `body_offset`)

### Step 4: Switch to IK Control

Replace `JointPositionActionCfg` with `DifferentialInverseKinematicsActionCfg`:
- Use `PIPER_HIGH_PD_CFG` (stiffer gains for IK tracking)
- `command_type="pose"`, `use_relative_mode=True` (delta EEF pose)
- Add `FrameTransformerCfg` to track EEF position in observations
- Add `BinaryJointPositionActionCfg` for gripper open/close

### Step 5: Add Graspable Object

Add a rigid object (simple cuboid or cylinder) to the scene on the table:
- `RigidObjectCfg` with collision and mass properties
- Add `object_position_in_robot_root_frame` observation
- Add `reset_root_state_uniform` event to randomize object position
- Add termination if object falls off table

### Step 6: Camera Integration

Attach a camera to the wrist or scene for visuomotor observations:
- `CameraCfg` on the Piper wrist link
- RGB output at target resolution
- Run with `--enable_cameras`

### Step 7: Teleop + Recording (HDF5)

Set up the imitation learning recording pipeline:
- Evolve env to `ManagerBasedRLMimicEnv` with required methods
- Implement: `get_robot_eef_pose`, `target_eef_pose_to_action`, `action_to_target_eef_pose`, `actions_to_gripper_actions`, `get_subtask_term_signals`
- Use `consolidated_demo.py` with keyboard teleop device
- Record 10-30 clean demonstrations to HDF5

### Step 8: Annotate Subtasks

Define subtask boundaries for Mimic:
1. Approach target item
2. Grasp
3. Lift
4. Move to shelf
5. Place and release

### Step 9: Dataset Generation with Mimic

Run `generate_dataset.py` to expand demos via object-centric transformations.

### Step 10: Train BC Policy

Train with `robomimic/train.py` using behavioral cloning on the generated dataset.

### Step 11: Evaluate

Evaluate success rate across randomized scenes.

---

## Environment Info

- **Container**: RunPod with RTX 4090, KASM VNC (xfce4)
- **Isaac Lab**: 0.47.2, Isaac Sim 5.1.0
- **Python**: 3.11 in `/opt/isaaclab-env/`
- **Installed**: `robomimic 0.4.0`, `isaaclab_mimic 1.0.15`
- **Env ID**: `ShelfSim-Piper-v0`
- **Run command**: `/opt/IsaacLab/isaaclab.sh -p scripts/random_agent.py --task ShelfSim-Piper-v0 --num_envs 1`
