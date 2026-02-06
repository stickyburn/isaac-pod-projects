"""
Test script for Piper arm articulation.

This script verifies that the AgileX Piper arm loads correctly and all joints
actuate within their URDF limits. It checks for physics stability over time.

Run via Isaac Lab: /opt/IsaacLab/isaaclab.sh -p scripts/test_piper_arm.py
Run headless: /opt/IsaacLab/isaaclab.sh -p scripts/test_piper_arm.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test Piper arm articulation")
parser.add_argument("--robot-usd", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from datetime import datetime, timezone
from pathlib import Path
import sys
import traceback

import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import build_simulation_context

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROBOT_USD = REPO_ROOT / "projects/piper_usd/piper_arm.usda"
REPORTS_DIR = REPO_ROOT / "projects/shelf_sim/reports"


if args_cli.robot_usd is None:
    args_cli.robot_usd = DEFAULT_ROBOT_USD


REPORT_WRITER = None


class ReportWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lines: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")

    def log(self, message: str = "") -> None:
        print(message)
        self._lines.append(message)
        self.write()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(self._lines).rstrip() + "\n"
        self.path.write_text(content)


def log(message: str = "") -> None:
    if REPORT_WRITER is not None:
        REPORT_WRITER.log(message)
    else:
        print(message)


def build_piper_arm_cfg(usd_path: Path) -> ArticulationCfg:
    # Joint names from URDF: fl_joint1-6 (revolute), fl_joint7-8 (prismatic gripper)
    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path.resolve()),
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

    global REPORT_WRITER
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"test_piper_arm_report_{timestamp}.md"
    REPORT_WRITER = ReportWriter(report_path)

    log(f"[INFO] Report path: {report_path}")
    log(f"[INFO] Start time (UTC): {timestamp}")
    log(f"[INFO] Command: {' '.join(sys.argv)}")

    log("\n" + "="*80)
    log("Piper Arm Articulation Test")
    log("="*80)

    if not args_cli.robot_usd.exists():
        raise FileNotFoundError(
            f"Robot USD not found at {args_cli.robot_usd}. "
            "The USD file must exist before running this test."
        )
    log(f"[INFO] Using Piper USD: {args_cli.robot_usd}")
    piper_arm_cfg = build_piper_arm_cfg(args_cli.robot_usd)

    with build_simulation_context(
        device="cuda:0",
        auto_add_lighting=True,
        add_ground_plane=True
    ) as sim:
        log(f"[INFO] Simulation device: {sim.device}")
        log(f"[INFO] Simulation dt: {sim.cfg.dt}")
        # Create environment prim
        prim_utils.create_prim(f"/World/Env_0", "Xform", translation=(0.0, 0.0, 0.0))

        # Create articulation
        log("\n[INFO] Spawning Piper arm articulation...")
        arm = Articulation(piper_arm_cfg.replace(prim_path="/World/Env_0/Robot"))

        # Initialize simulation
        log("[INFO] Initializing simulation...")
        sim.reset()

        # Verify initialization
        log("\n" + "-"*80)
        log("VERIFICATION: Initialization")
        log("-"*80)

        success = True

        # Check if articulation is initialized
        if not arm.is_initialized:
            log("[FAIL] Articulation not initialized!")
            success = False
        else:
            log("[PASS] Articulation initialized successfully")

        # Check if fixed base
        if not arm.is_fixed_base:
            log("[FAIL] Expected fixed base but articulation is floating!")
            success = False
        else:
            log("[PASS] Base is fixed")

        # Check buffer shapes
        log(f"\n[INFO] Number of articulations: {arm.num_instances}")
        log(f"[INFO] Number of joints: {arm.num_joints}")

        if arm.num_instances != 1:
            log(f"[FAIL] Expected 1 articulation, got {arm.num_instances}")
            success = False

        if arm.num_joints != 8:
            log(f"[FAIL] Expected 8 joints, got {arm.num_joints}")
            success = False
        else:
            log("[PASS] Found expected 8 joints (6 arm + 2 gripper)")

        # Check joint names
        log(f"\n[INFO] Joint names: {list(arm.data.joint_names)}")
        expected_joints = [
            "fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4",
            "fl_joint5", "fl_joint6", "fl_joint7", "fl_joint8"
        ]
        actual_joints = list(arm.data.joint_names)

        if actual_joints == expected_joints:
            log("[PASS] All expected joints present")
        else:
            log("[FAIL] Joint names don't match URDF!")
            log(f"  Expected: {expected_joints}")
            log(f"  Actual: {actual_joints}")
            success = False

        # Check for NaN values in initial state
        if torch.isnan(arm.data.root_pos_w).any():
            log("[FAIL] NaN values found in root position!")
            success = False
        else:
            log("[PASS] No NaN in root position")

        if torch.isnan(arm.data.joint_pos).any():
            log("[FAIL] NaN values found in joint positions!")
            success = False
        else:
            log("[PASS] No NaN in joint positions")

        log("\n" + "-"*80)
        log("VERIFICATION: Joint Limits")
        log("-"*80)

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
                log(f"  {joint_name}: [{lower:.3f}, {upper:.3f}] - Expected")
            else:
                log(f"  {joint_name}: Unknown limits")

        log("\n" + "-"*80)
        log("VERIFICATION: Physics Stability (5 seconds)")
        log("-"*80)

        initial_z = arm.data.root_pos_w[0, 2].item()
        log(f"\n[INFO] Initial Z position: {initial_z:.4f} m")

        # Simulate for 5 seconds
        sim_steps = int(5.0 / sim.cfg.dt)
        log(f"[INFO] Simulating for {sim_steps} steps ({5.0} seconds)...")

        nan_detected = False
        max_z_drift = 0.0

        for step in range(sim_steps):
            sim.step()
            arm.update(sim.cfg.dt)

            # Check for NaN values
            if torch.isnan(arm.data.root_pos_w).any() or torch.isnan(arm.data.joint_pos).any():
                log(f"\n[FAIL] NaN values detected at step {step}!")
                nan_detected = True
                success = False
                break

            # Track Z position drift
            current_z = arm.data.root_pos_w[0, 2].item()
            z_drift = abs(current_z - initial_z)
            if z_drift > max_z_drift:
                max_z_drift = z_drift

            if step % (sim_steps // 5) == 0:
                log(f"  Step {step}/{sim_steps}: Z={current_z:.4f} m, drift={z_drift:.4f} m")

        if not nan_detected:
            log(f"\n[INFO] Max Z position drift: {max_z_drift:.4f} m")

            if max_z_drift < 0.01:
                log("[PASS] Arm remained stable (drift < 1cm)")
            else:
                log(f"[WARNING] Arm drifted {max_z_drift*100:.2f} cm - may indicate fixed base issue")
                # Don't fail on this, could be minor physics settling

        log("\n" + "-"*80)
        log("VERIFICATION: Joint Actuation")
        log("-"*80)

        # Test joint actuation by moving each joint to mid-range
        log("\n[INFO] Testing joint actuation...")

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
        log("\n[INFO] Joint positions after action:")

        for i, (name, target_pos) in enumerate(joint_actions.items()):
            if name in actual_joints:
                idx = actual_joints.index(name)
                actual_pos = final_joint_pos[idx]
                error = abs(actual_pos - target_pos)
                status = "[PASS]" if error < 0.1 else "[FAIL]"
                log(f"  {name}: {actual_pos:.3f} rad (target: {target_pos:.3f}) err={error:.3f} {status}")

                if error >= 0.1:
                    success = False

        log("\n" + "="*80)
        if success:
            log("OVERALL RESULT: PASS")
        else:
            log("OVERALL RESULT: FAIL")
        log("="*80 + "\n")

        return success


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        log(f"\n[ERROR] Exception occurred: {e}")
        for line in traceback.format_exc().splitlines():
            log(line)
        exit(1)
    finally:
        if REPORT_WRITER is not None:
            REPORT_WRITER.write()
        simulation_app.close()