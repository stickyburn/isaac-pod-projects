[INFO] Report path: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/test_scene_report_20260203_215123.md
[INFO] Start time (UTC): 20260203_215123
[INFO] Command: ./scripts/test_scene.py --headless --record-video
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
[INFO] Camera mode: top
[INFO] Top camera pos/target: (0.0, 0.0, 2.2) -> (0.0, 0.0, 0.75)
[INFO] Simulation device: cuda:0
[INFO] Simulation dt: 0.016666666666666666
[INFO] Camera recording enabled.
[INFO] Video window: start=0.10s duration=3.00s fps=30

--------------------------------------------------------------------------------
[INFO] Testing asset 1/8: /workspace/assets/Boxed/blue_tin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 5 prim(s).
[INFO] Applied convex-hull mesh collisions to 5 prim(s).
[INFO] Recording video: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/videos/scene_000_blue_tin.mp4
PASS | z=0.770 m | vel=0.002 m/s | fall=0.700 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 2/8: /workspace/assets/Boxed/mustard_jar.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 3 prim(s).
[INFO] Applied convex-hull mesh collisions to 3 prim(s).
[INFO] Recording video: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/videos/scene_001_mustard_jar.mp4
PASS | z=0.794 m | vel=0.002 m/s | fall=0.676 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 3/8: /workspace/assets/Boxed/oil_tin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
[INFO] Recording video: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/videos/scene_002_oil_tin.mp4
PASS | z=0.792 m | vel=0.000 m/s | fall=0.678 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 4/8: /workspace/assets/Boxed/salt_box.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 1 prim(s).
[INFO] Applied convex-hull mesh collisions to 1 prim(s).
[INFO] Recording video: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/videos/scene_003_salt_box.mp4
PASS | z=0.793 m | vel=0.001 m/s | fall=0.677 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 5/8: /workspace/assets/Candles/Medium_Fat.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
[INFO] Recording video: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/videos/scene_004_Medium_Fat.mp4
PASS | z=0.770 m | vel=0.004 m/s | fall=0.700 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 6/8: /workspace/assets/Candles/TallThin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
[INFO] Recording video: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/videos/scene_005_TallThin.mp4
PASS | z=0.775 m | vel=0.000 m/s | fall=0.695 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 7/8: /workspace/assets/Containers/MasonJar.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
[INFO] Skipped 1 mesh prim(s) without points.
[INFO] Recording video: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/videos/scene_006_MasonJar.mp4
PASS | z=0.803 m | vel=0.001 m/s | fall=0.667 m | inside_bin=True | arm_ok=True

--------------------------------------------------------------------------------
[INFO] Testing asset 8/8: /workspace/assets/Containers/TinCan.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 1 prim(s).
[INFO] Applied convex-hull mesh collisions to 1 prim(s).
[INFO] Recording video: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/videos/scene_007_TinCan.mp4
PASS | z=0.809 m | vel=0.001 m/s | fall=0.661 m | inside_bin=True | arm_ok=True
