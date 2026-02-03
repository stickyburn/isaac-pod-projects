[INFO] Report path: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/test_scene_report_20260203_044027.md
[INFO] Start time (UTC): 20260203_044027
[INFO] Command: ./projects/shelf_sim/scripts/test_scene.py --headless --record-video --camera-mode both
[INFO] Manifest: /workspace/storage/isaac-pod-projects/projects/shelf_sim/assets_manifest.json

================================================================================
Combined Scene Test (Piper Arm + Assets)
================================================================================
[INFO] Using Piper USD: /workspace/storage/isaac-pod-projects/projects/piper_usd/piper_arm.usd
[INFO] Asset count: 8
[INFO] Drop height: 0.50 m
[INFO] Test duration: 3.00 s
[INFO] Forced mass: 0.50 kg
[INFO] Stable vel eps: 0.020 m/s
[INFO] Robot base pos: (-0.8, 0.0, 0.0)
[INFO] Camera mode: both
[INFO] Side camera pos/target: (1.3, -1.2, 1.2) -> (0.0, 0.0, 0.85)
[INFO] Top camera pos/target: (0.0, 0.0, 2.2) -> (0.0, 0.0, 0.75)
[INFO] Simulation device: cuda:0
[INFO] Simulation dt: 0.01
[INFO] Camera recording enabled.
[INFO] Video window: start=0.50s duration=3.00s fps=30

--------------------------------------------------------------------------------
[INFO] Testing asset 1/8: /workspace/assets/Boxed/blue_tin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 5 prim(s).
[INFO] Applied convex-hull mesh collisions to 5 prim(s).
PASS | z=0.806 m | vel=0.000 m/s | fall=0.664 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 2/8: /workspace/assets/Boxed/mustard_jar.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 3 prim(s).
[INFO] Applied convex-hull mesh collisions to 3 prim(s).
PASS | z=0.796 m | vel=0.002 m/s | fall=0.674 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 3/8: /workspace/assets/Boxed/oil_tin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
PASS | z=0.793 m | vel=0.001 m/s | fall=0.677 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 4/8: /workspace/assets/Boxed/salt_box.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 1 prim(s).
[INFO] Applied convex-hull mesh collisions to 1 prim(s).
PASS | z=0.768 m | vel=0.001 m/s | fall=0.702 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 5/8: /workspace/assets/Candles/Medium_Fat.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
PASS | z=0.770 m | vel=0.001 m/s | fall=0.700 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 6/8: /workspace/assets/Candles/TallThin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
PASS | z=0.775 m | vel=0.001 m/s | fall=0.695 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 7/8: /workspace/assets/Containers/MasonJar.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
[INFO] Skipped 1 mesh prim(s) without points.
PASS | z=0.771 m | vel=0.001 m/s | fall=0.699 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 8/8: /workspace/assets/Containers/TinCan.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 1 prim(s).
[INFO] Applied convex-hull mesh collisions to 1 prim(s).
PASS | z=0.809 m | vel=0.001 m/s | fall=0.661 m | inside_bin=True | arm_ok=True

2026-02-03T04:42:42Z [155,187ms] [Warning] [omni.physx.tensors.plugin] prim '/World/Env_0/Assets/asset_007/TinCan' was deleted while being used by a shape in a tensor view class. The physics.tensors simulationView was invalidated.