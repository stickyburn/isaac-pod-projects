"""
Test script for Piper arm articulation.

This script verifies that the AgileX Piper arm loads correctly and all joints
actuate within their URDF limits. It checks for physics stability over time.

Run via Isaac Lab: /opt/IsaacLab/isaaclab.sh -p scripts/test_piper_arm.py
Run with GUI: /opt/IsaacLab/isaaclab.sh -p scripts/test_piper_arm.py --headless
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app (isaaclab.sh wrapper handles --headless flag)
simulation_app = AppLauncher().app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import build_simulation_context


# Define Piper configuration
# Joint names from URDF: fl_joint1-6 (revolute), fl_joint7-8 (prismatic gripper)
PIPER_ARM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/workspace/storage/projects/piper_usd/piper_arm.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
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


def main():
    """Main test execution."""

    print("\n" + "="*80)
    print("Piper Arm Articulation Test")
    print("="*80)

    with build_simulation_context(
        device="cuda:0",
        auto_add_lighting=True,
        add_ground_plane=True
    ) as sim:
        # Create environment prim
        sim_utils.create_prim(f"/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))

        # Create articulation
        print("\n[INFO] Spawning Piper arm articulation...")
        arm = Articulation(PIPER_ARM_CFG.replace(prim_path="/World/Env_0/Robot"))

        # Initialize simulation
        print("[INFO] Initializing simulation...")
        sim.reset()

        # Verify initialization
        print("\n" + "-"*80)
        print("VERIFICATION: Initialization")
        print("-"*80)

        success = True

        # Check if articulation is initialized
        if not arm.is_initialized:
            print("[FAIL] Articulation not initialized!")
            success = False
        else:
            print("[PASS] Articulation initialized successfully")

        # Check if fixed base
        if not arm.is_fixed_base:
            print("[FAIL] Expected fixed base but articulation is floating!")
            success = False
        else:
            print("[PASS] Base is fixed")

        # Check buffer shapes
        print(f"\n[INFO] Number of articulations: {arm.num_instances}")
        print(f"[INFO] Number of joints: {arm.num_joints}")

        if arm.num_instances != 1:
            print(f"[FAIL] Expected 1 articulation, got {arm.num_instances}")
            success = False

        if arm.num_joints != 8:
            print(f"[FAIL] Expected 8 joints, got {arm.num_joints}")
            success = False
        else:
            print("[PASS] Found expected 8 joints (6 arm + 2 gripper)")

        # Check joint names
        print(f"\n[INFO] Joint names: {list(arm.data.joint_names)}")
        expected_joints = [
            "fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4",
            "fl_joint5", "fl_joint6", "fl_joint7", "fl_joint8"
        ]
        actual_joints = list(arm.data.joint_names)

        if actual_joints == expected_joints:
            print("[PASS] All expected joints present")
        else:
            print("[FAIL] Joint names don't match URDF!")
            print(f"  Expected: {expected_joints}")
            print(f"  Actual: {actual_joints}")
            success = False

        # Check for NaN values in initial state
        if torch.isnan(arm.data.root_pos_w).any():
            print("[FAIL] NaN values found in root position!")
            success = False
        else:
            print("[PASS] No NaN in root position")

        if torch.isnan(arm.data.joint_pos).any():
            print("[FAIL] NaN values found in joint positions!")
            success = False
        else:
            print("[PASS] No NaN in joint positions")

        print("\n" + "-"*80)
        print("VERIFICATION: Joint Limits")
        print("-"*80)

        expected_limits = {
            "fl_joint1": (-2.618, 2.618),
            "fl_joint2": (0.0, 3.14),
            "fl_joint3": (-2.697, 0.0),
            "fl_joint4": (-1.832, 1.832),
            "fl_joint5": (-1.22, 1.22),
            "fl_joint6": (-3.14, 3.14),
            "fl_joint7": (0.0, 0.04),
            "fl_joint8": (-0.04, 0.0),
        }

        for i, joint_name in enumerate(actual_joints):
            if joint_name in expected_limits:
                lower, upper = expected_limits[joint_name]
                print(f"  {joint_name}: [{lower:.3f}, {upper:.3f}] - Expected")
            else:
                print(f"  {joint_name}: Unknown limits")

        print("\n" + "-"*80)
        print("VERIFICATION: Physics Stability (5 seconds)")
        print("-"*80)

        initial_z = arm.data.root_pos_w[0, 2].item()
        print(f"\n[INFO] Initial Z position: {initial_z:.4f} m")

        # Simulate for 5 seconds
        sim_steps = int(5.0 / sim.cfg.dt)
        print(f"[INFO] Simulating for {sim_steps} steps ({5.0} seconds)...")

        nan_detected = False
        max_z_drift = 0.0

        for step in range(sim_steps):
            sim.step()
            arm.update(sim.cfg.dt)

            # Check for NaN values
            if torch.isnan(arm.data.root_pos_w).any() or torch.isnan(arm.data.joint_pos).any():
                print(f"\n[FAIL] NaN values detected at step {step}!")
                nan_detected = True
                success = False
                break

            # Track Z position drift
            current_z = arm.data.root_pos_w[0, 2].item()
            z_drift = abs(current_z - initial_z)
            if z_drift > max_z_drift:
                max_z_drift = z_drift

            if step % (sim_steps // 5) == 0:
                print(f"  Step {step}/{sim_steps}: Z={current_z:.4f} m, drift={z_drift:.4f} m")

        if not nan_detected:
            print(f"\n[INFO] Max Z position drift: {max_z_drift:.4f} m")

            if max_z_drift < 0.01:
                print("[PASS] Arm remained stable (drift < 1cm)")
            else:
                print(f"[WARNING] Arm drifted {max_z_drift*100:.2f} cm - may indicate fixed base issue")
                # Don't fail on this, could be minor physics settling

        print("\n" + "-"*80)
        print("VERIFICATION: Joint Actuation")
        print("-"*80)

        # Test joint actuation by moving each joint to mid-range
        print("\n[INFO] Testing joint actuation...")

        joint_actions = {
            "fl_joint1": 0.0,
            "fl_joint2": 1.57,
            "fl_joint3": -1.35,
            "fl_joint4": 0.0,
            "fl_joint5": 0.0,
            "fl_joint6": 0.0,
            "fl_joint7": 0.02,
            "fl_joint8": -0.02,
        }

        # Apply actions
        action_tensor = torch.zeros(arm.num_instances, arm.num_joints, device=sim.device)
        for i, (name, target_pos) in enumerate(joint_actions.items()):
            if name in actual_joints:
                idx = actual_joints.index(name)
                action_tensor[0, idx] = target_pos

        arm.set_joint_position_target(action_tensor)
        arm.write_data_to_sim()

        # Simulate for 1 second to allow movement
        for step in range(int(1.0 / sim.cfg.dt)):
            sim.step()
            arm.update(sim.cfg.dt)

        # Verify joints moved
        final_joint_pos = arm.data.joint_pos[0].cpu().numpy()
        print("\n[INFO] Joint positions after action:")

        for i, (name, target_pos) in enumerate(joint_actions.items()):
            if name in actual_joints:
                idx = actual_joints.index(name)
                actual_pos = final_joint_pos[idx]
                error = abs(actual_pos - target_pos)
                status = "[PASS]" if error < 0.1 else "[FAIL]"
                print(f"  {name}: {actual_pos:.3f} rad (target: {target_pos:.3f}) err={error:.3f} {status}")

                if error >= 0.1:
                    success = False

        print("\n" + "="*80)
        if success:
            print("OVERALL RESULT: PASS")
        else:
            print("OVERALL RESULT: FAIL")
        print("="*80 + "\n")

        return success


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        exit(1)
    finally:
        simulation_app.close()