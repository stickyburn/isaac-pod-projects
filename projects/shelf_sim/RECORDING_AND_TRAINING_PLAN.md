# Shelf-Sim Imitation Learning Pipeline - Detailed Plan

## Executive Summary

This plan implements a proven two-phase approach for imitation learning:
1. **Recording Phase**: Fixed, consistent scenes for human demonstrations
2. **Training Phase**: Heavily randomized scenes for robust policy learning via Mimic

This approach is used by NVIDIA, Stanford, DeepMind, and is the standard for production-quality policies.

## Current Status (Updated: 2026-02-05)

### ✅ Completed

**Phase 1: Recording Infrastructure**
- ✅ Fixed scene configurations for Session Types A-F implemented and verified
- ✅ All 6 environments registered and loadable via `Shelf-Sim-Recording-Session-{A-F}-v0`
- ✅ MDP actions: EEF delta pose (7D) + gripper control (1D) using differential IK
- ✅ Observations: Camera RGB (224×224×3=150,528 dims) + robot state (8 joints) + EEF pose + target position
- ✅ Environment verified on RTX 4090 (RunPod) with zero_agent test
- ✅ Wrist camera integrated - camera attached to robot gripper (fl_link8)

**Camera Configuration:**
- **Type**: Wrist camera (attached to end-effector, moves with gripper)
- **Prim Path**: `/World/robot/fl_link8/gripper_camera`
- **Position**: 5cm in front of gripper, pointing forward (Z-axis)
- **Resolution**: 224×224 pixels (RGB)
- **FOV**: 24mm focal length, ~20.9mm horizontal aperture
- **USD Update**: Added camera to `piper_arm_sensor.usd`

**Verification Results:**
- Environment loads successfully with all managers active
- Observation space: 150,556 dims (camera + state + EEF + target)
- Action space: 8 dims (7 EEF delta pose + 1 gripper)
- Camera rendering enabled and functional
- Physics simulation stable at 120 Hz

### 🔄 In Progress
- Teleoperation controller integration for keyboard control
- HDF5 recording script with demonstration capture
- Success criteria validation during recording

### ⏳ Next Steps
1. Complete teleoperation interface with IK-based keyboard control
2. Implement HDF5 recording with demonstration metadata
3. Record 15-30 demonstrations across all session types
4. Mimic integration for data generation

### Run with

`/opt/IsaacLab/isaaclab.sh -p scripts/zero_agent.py \
    --task=Shelf-Sim-Recording-Session-A-v0 \
    --headless \
    --enable_cameras \
    --num_envs=1`

---

## Phase 1: Recording Environment (Human Demos)

### Philosophy
Humans need consistency to demonstrate reliable trajectories. Each recording session uses a **fixed scene configuration** that doesn't change during that session. This ensures:
- Clean, consistent demonstrations
- Predictable item positions
- Repeatable success/failure criteria
- Better learning signal for the policy

### Fixed Scene Configuration

#### Scene Layout (Per Session)
```yaml
# Bin configuration
bin_items: 1-2 items total
bin_item_type: FIXED for entire session (e.g., only blue_tin)
bin_item_positions: FIXED positions (e.g., center of bin)
target_item: Clearly marked (emissive highlight or distinct color)

# Shelf configuration
target_slot: FIXED slot for entire session (e.g., middle shelf, center position)
shelf_occupancy: FIXED (empty or with consistent distractor items)
target_indicator: Persistent emissive marker at target location

# Scene elements
table: Static, fixed position
shelf: Static, fixed position
robot: Ground-mounted, fixed base
lighting: Consistent dome + key + fill lights
```

#### Session Variations (Between Sessions)
Create 4-6 different session types to cover variability humans can handle:

**Session Type A**: Single item in bin, empty middle shelf target
**Session Type B**: Two items in bin (target visible on top), empty middle shelf target  
**Session Type C**: Single item in bin, middle shelf has 1 distractor item
**Session Type D**: Single item in bin, top shelf target (different height)
**Session Type E**: Different item type (mustard_jar instead of blue_tin)
**Session Type F**: Target item offset in bin (not center)

### Recording Workflow

```
For each session_type in [A, B, C, D, E, F]:
    1. Launch environment with fixed_scene_cfg[session_type]
    2. Record 3-5 demonstrations via teleop
    3. Validate recordings (replay check)
    4. Mark successful demos in HDF5
    5. Total: 15-30 high-quality demonstrations
```

### Teleoperation Methods

#### Option 1: IK-based Teleop (Recommended)
- **Input**: Keyboard (WASD for XY, Q/E for Z, arrow keys for rotation)
- **Control**: Inverse Kinematics to compute joint targets from EEF delta pose
- **Advantages**: Intuitive, precise, can use external devices (SpaceMouse)
- **Implementation**: Use Isaac Lab's `IkController` with delta pose commands

#### Option 2: Direct Joint Control
- **Input**: Number keys 1-8 for joints, +/- for direction
- **Control**: Direct joint velocity/position commands
- **Advantages**: Simple to implement
- **Disadvantages**: Harder for humans to control EEF precisely

#### Option 3: Hybrid (Best of both)
- Default: IK mode for positioning
- Switch: Joint mode for fine adjustments
- Gripper: Separate button (spacebar to toggle)

### Success Criteria for Recordings

A demonstration is **successful** if:
1. Robot grasps target item (gripper closes around item)
2. Item is lifted clear of bin (no collision with bin walls)
3. Item is transported to target shelf slot
4. Item is released within slot bounds (±5cm)
5. Gripper opens after release
6. No major collisions during transport

A demonstration is **discarded** if:
- Item dropped during transport
- Item placed outside slot bounds
- Excessive collisions with environment
- Timeout (>30 seconds)

---

## Phase 2: Training Environment (Policy Learning)

### Philosophy
During training, we use **heavy domain randomization** to create a robust policy that generalizes. This happens automatically via Mimic data generation, not during human recording.

### Randomization Strategy

#### Per-Episode Randomization (via Mimic)
```yaml
# Bin randomization
bin_item_count: 1-3 items (weighted: 1=50%, 2=30%, 3=20%)
bin_item_types: Random selection from 5 available assets
bin_item_positions: Random XY in bin (±15cm from center), random Z drop
bin_item_orientations: Random yaw (0-360°), small pitch/roll variation

# Shelf randomization
target_slot: Random selection from 6 slots (3 shelves × 2 positions)
shelf_occupancy: 
  - Empty: 30%
  - 1 distractor: 40%
  - 2 distractors: 25%
  - 3+ distractors: 5%
distractor_types: Random from remaining 4 assets
distractor_positions: Random within slot bounds

# Target indicator randomization
indicator_color: Random from palette (green, blue, cyan, yellow)
indicator_shape: Cylinder or box (50/50)
indicator_intensity: 0.8-1.2x base brightness

# Physical randomization
item_mass: 0.4-0.6 kg (base 0.5 ±20%)
friction: 0.5-0.8 (base 0.65 ±20%)
restitution: 0.1-0.3

# Lighting randomization
dome_intensity: 1200-1800 (base 1500 ±20%)
dome_color_temp: 5000-6000K (base 5500 ±10%)
key_light_position: Small offset (±0.2m in XYZ)
key_light_intensity: 640-960 (base 800 ±20%)

# Camera randomization (if using visuomotor)
camera_pose_jitter: ±2cm position, ±3° rotation
exposure: 0.9-1.1x
contrast: 0.9-1.1x
```

#### Hard Negative Mining
Occasionally generate challenging scenarios:
- Target item at bottom of stack (requires removing other items)
- Target slot partially blocked by large distractor
- Items very close together in bin
- Unusual orientations (item on side)

### Mimic Data Generation

Using Isaac Lab Mimic to generate 100-1000x data expansion:

```python
# Mimic generation parameters
generation_config = {
    "num_generated_per_demo": 50,  # 30 demos → 1500 synthetic demos
    "randomization_scope": "full",  # All parameters randomized
    "subtask_segments": [
        "approach_grasp",      # Move to item, close gripper
        "lift",                # Lift clear of bin
        "transport",           # Move to shelf
        "place"                # Lower and release
    ],
    "interpolation": "linear_with_noise",  # Add small noise to trajectories
    "success_criteria": "same_as_recording"  # Use same success check
}
```

---

## Phase 3: Policy Training

### Architecture: Visuomotor Policy

```
Input:
  - Camera RGB (128×128 or 224×224)
  - Robot state (joint positions, velocities)
  - EEF pose (position + quaternion)
  - Target slot pose (position)

Backbone:
  - ResNet-18 or EfficientNet-B0 for image encoding
  - MLP for state encoding
  - Concatenation + 2-3 layer MLP for action prediction

Output:
  - EEF delta pose (dx, dy, dz, dqx, dqy, dqz, dqw)
  - Gripper action (open/close)

Training:
  - Loss: MSE on actions with L2 regularization
  - Optimizer: Adam (lr=1e-4, weight_decay=1e-6)
  - Batch size: 64-128
  - Epochs: 100-200
  - Validation split: 10%
```

### Training Curriculum

1. **Phase 3.1**: Train on original 30 demos (baseline)
2. **Phase 3.2**: Train on 30 demos + 500 Mimic-generated (full randomization)
3. **Phase 3.3**: Fine-tune with harder negatives

### Evaluation Metrics

- **Success Rate**: % of episodes where item placed in target slot
- **Grasp Success**: % of successful grasps
- **Transport Success**: % of items not dropped during transport
- **Placement Precision**: Distance from item center to slot center
- **Episode Length**: Average steps to completion

---

## Phase 4: Sim-to-Real Transfer (Future Work)

### Domain Adaptation
- **Real-world dataset**: 10-50 real demos for fine-tuning
- **Domain randomization**: Add real-world noise patterns
- **System identification**: Tune physics parameters to match real arm

### Validation
- Success rate >80% in sim before real deployment
- Real-world success rate >60% for initial transfer

---

## Implementation Roadmap

### Week 1: Recording Infrastructure
- [x] Implement fixed scene configurations (Session Types A-F) - **COMPLETED**
- [x] Define MDP actions, observations, rewards, terminations - **COMPLETED**
- [x] Environment registration and verification - **COMPLETED**
- [x] Verify on RTX 4090 with zero_agent test - **COMPLETED**
- [x] Add wrist camera to piper_arm_sensor.usd - **COMPLETED**
- [ ] Build IK-based teleoperation controller - **IN PROGRESS**
- [ ] Create recording script with HDF5 output
- [ ] Implement success criteria validation
- [ ] Record 15-30 demonstrations

### Week 2: Mimic Integration
- [ ] Define subtask segmentation logic
- [ ] Implement Mimic environment helpers
- [ ] Create data generation pipeline
- [ ] Generate 1500+ synthetic demonstrations

### Week 3: Policy Training
- [ ] Implement visuomotor policy architecture
- [ ] Set up training pipeline with robomimic
- [ ] Train and evaluate policy
- [ ] Iterate on hyperparameters

### Week 4: Evaluation & Documentation
- [ ] Comprehensive evaluation on held-out test scenes
- [ ] Stress testing with extreme randomization
- [ ] Document results and create demo videos
- [ ] Prepare sim-to-real transfer plan

---

## Technical Implementation Details

### File Structure
```
projects/shelf_sim/
├── PLAN.md                          # This document
├── RECORDING_AND_TRAINING_PLAN.md   # Detailed implementation plan
├── source/shelf_sim/
│   └── shelf_sim/
│       └── tasks/
│           └── manager_based/
│               ├── shelf_sim/              # Base environment (existing)
│               └── shelf_sim_recording/    # NEW: Recording environments
│                   ├── recording_env_cfg.py      # Fixed scene configs for Sessions A-F ✅
│                   ├── mdp/
│                   │   ├── actions.py            # EEF delta pose + gripper actions ✅
│                   │   ├── observations.py       # Camera RGB + state + EEF ✅
│                   │   ├── rewards.py            # Task rewards ✅
│                   │   └── terminations.py       # Success/failure conditions ✅
│                   └── agents/                   # Policy network configs
│               └── shelf_sim_training/     # TODO: Training environments with randomization
├── controllers/
│   └── teleop_ik.py                # IK-based keyboard teleoperation controller
├── scripts/
│   ├── list_envs.py                # List registered environments
│   ├── zero_agent.py               # Test environment with zero actions ✅
│   ├── random_agent.py             # Test environment with random actions
│   ├── record_demos.py             # Teleop recording (TODO)
│   ├── replay_demos.py             # Validation (TODO)
│   ├── generate_mimic_data.py      # Mimic data generation (TODO)
│   ├── train_policy.py             # BC training (TODO)
│   └── eval_policy.py              # Evaluation (TODO)
├── configs/
│   ├── recording_scenes.yaml       # Session type definitions ✅
│   └── training_randomization.yaml # Randomization parameters
└── data/
    ├── recordings/                 # Human demos (HDF5)
    └── mimic_generated/            # Synthetic demos (HDF5)
```

### Key Design Decisions

1. **Separate Recording vs Training Envs**: Explicit separation prevents accidental randomization during recording
2. **Session-based Recording**: Grouping demos by scene type enables stratified training
3. **Subtask Segmentation**: Enables Mimic to mix-and-match trajectory segments
4. **Visuomotor Architecture**: Combines visual and state observations for robustness
5. **Iterative Evaluation**: Test at each phase before proceeding

### Risk Mitigation

- **Demo Quality**: Implement real-time validation during recording
- **Mimic Failures**: Keep original demos as fallback
- **Training Instability**: Start with state-only, add vision gradually
- **Sim2Real Gap**: Document all randomization parameters for real-world tuning

---

## Appendix: References

- NVIDIA Isaac Lab Mimic: [Documentation](https://isaac-sim.github.io/IsaacLab/main/source/api/lab_mimic/isaaclab_mimic.datagen.html)
- Stanford Robomimic: [Best Practices](https://robomimic.github.io/)
- "Learning to Learn from Demonstrations" (Mandlekar et al., 2021)
- "MimicGen: A Data Generation System for Scalable Robot Learning" (Mandlekar et al., 2023)
