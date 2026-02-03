[INFO] Report path: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/test_piper_arm_report_20260203_031429.md
[INFO] Start time (UTC): 20260203_031429
[INFO] Command: ./projects/shelf_sim/scripts/test_piper_arm.py --headless

================================================================================
Piper Arm Articulation Test
================================================================================
[INFO] Using Piper USD: /workspace/storage/isaac-pod-projects/projects/piper_usd/piper_arm.usd
[INFO] Simulation device: cuda:0
[INFO] Simulation dt: 0.01

[INFO] Spawning Piper arm articulation...
[INFO] Initializing simulation...

--------------------------------------------------------------------------------
VERIFICATION: Initialization
--------------------------------------------------------------------------------
[PASS] Articulation initialized successfully
[PASS] Base is fixed

[INFO] Number of articulations: 1
[INFO] Number of joints: 8
[PASS] Found expected 8 joints (6 arm + 2 gripper)

[INFO] Joint names: ['fl_joint1', 'fl_joint2', 'fl_joint3', 'fl_joint4', 'fl_joint5', 'fl_joint6', 'fl_joint7', 'fl_joint8']
[PASS] All expected joints present
[PASS] No NaN in root position
[PASS] No NaN in joint positions

--------------------------------------------------------------------------------
VERIFICATION: Joint Limits
--------------------------------------------------------------------------------
  fl_joint1: [-2.618, 2.618] - Expected
  fl_joint2: [0.000, 3.140] - Expected
  fl_joint3: [-2.697, 0.000] - Expected
  fl_joint4: [-1.832, 1.832] - Expected
  fl_joint5: [-1.220, 1.220] - Expected
  fl_joint6: [-3.140, 3.140] - Expected
  fl_joint7: [0.000, 0.040] - Expected
  fl_joint8: [-0.040, 0.000] - Expected

--------------------------------------------------------------------------------
VERIFICATION: Physics Stability (5 seconds)
--------------------------------------------------------------------------------

[INFO] Initial Z position: 0.0000 m
[INFO] Simulating for 500 steps (5.0 seconds)...
  Step 0/500: Z=0.0000 m, drift=0.0000 m
  Step 100/500: Z=0.0000 m, drift=0.0000 m
  Step 200/500: Z=0.0000 m, drift=0.0000 m
  Step 300/500: Z=0.0000 m, drift=0.0000 m
  Step 400/500: Z=0.0000 m, drift=0.0000 m

[INFO] Max Z position drift: 0.0000 m
[PASS] Arm remained stable (drift < 1cm)

--------------------------------------------------------------------------------
VERIFICATION: Joint Actuation
--------------------------------------------------------------------------------

[INFO] Testing joint actuation...

[INFO] Joint positions after action:
  fl_joint1: 0.000 rad (target: 0.000) err=0.000 [PASS]
  fl_joint2: 1.571 rad (target: 1.570) err=0.001 [PASS]
  fl_joint3: -1.349 rad (target: -1.350) err=0.001 [PASS]
  fl_joint4: -0.001 rad (target: 0.000) err=0.001 [PASS]
  fl_joint5: 0.000 rad (target: 0.000) err=0.000 [PASS]
  fl_joint6: -0.000 rad (target: 0.000) err=0.000 [PASS]
  fl_joint7: 0.020 rad (target: 0.020) err=0.000 [PASS]
  fl_joint8: -0.020 rad (target: -0.020) err=0.000 [PASS]

================================================================================
OVERALL RESULT: PASS
================================================================================
