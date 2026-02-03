[INFO] Report path: /workspace/storage/isaac-pod-projects/projects/shelf_sim/reports/test_assets_report_20260203_035617.md
[INFO] Start time (UTC): 20260203_035617
[INFO] Command: ./projects/shelf_sim/scripts/test_assets.py --headless
[INFO] Manifest: /workspace/storage/isaac-pod-projects/projects/shelf_sim/assets_manifest.json

================================================================================
USD Asset Drop Test (Table + Bin)
================================================================================
[INFO] Asset count: 8
[INFO] Drop height: 0.50 m
[INFO] Test duration: 3.00 s
[INFO] Forced mass: 0.50 kg
[INFO] Stable vel eps: 0.020 m/s
[INFO] Table size: 1.20 x 0.80 x 0.05 m
[INFO] Bin inner size: 0.40 x 0.30 m
[INFO] Simulation device: cuda:0
[INFO] Simulation dt: 0.01

--------------------------------------------------------------------------------
[INFO] Testing asset 1/8: /workspace/assets/Boxed/blue_tin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 5 prim(s).
[INFO] Applied convex-hull mesh collisions to 5 prim(s).
PASS | z=0.806 m | vel=0.000 m/s | fall=0.664 m | inside_bin=True

--------------------------------------------------------------------------------
[INFO] Testing asset 2/8: /workspace/assets/Boxed/mustard_jar.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 3 prim(s).
[INFO] Applied convex-hull mesh collisions to 3 prim(s).
PASS | z=0.796 m | vel=0.002 m/s | fall=0.674 m | inside_bin=True

--------------------------------------------------------------------------------
[INFO] Testing asset 3/8: /workspace/assets/Boxed/oil_tin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
PASS | z=0.793 m | vel=0.001 m/s | fall=0.677 m | inside_bin=True

--------------------------------------------------------------------------------
[INFO] Testing asset 4/8: /workspace/assets/Boxed/salt_box.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 1 prim(s).
[INFO] Applied convex-hull mesh collisions to 1 prim(s).
PASS | z=0.768 m | vel=0.001 m/s | fall=0.702 m | inside_bin=True

--------------------------------------------------------------------------------
[INFO] Testing asset 5/8: /workspace/assets/Candles/Medium_Fat.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
PASS | z=0.770 m | vel=0.001 m/s | fall=0.700 m | inside_bin=True

--------------------------------------------------------------------------------
[INFO] Testing asset 6/8: /workspace/assets/Candles/TallThin.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
PASS | z=0.775 m | vel=0.001 m/s | fall=0.695 m | inside_bin=True

--------------------------------------------------------------------------------
[INFO] Testing asset 7/8: /workspace/assets/Containers/MasonJar.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 2 prim(s).
[INFO] Applied convex-hull mesh collisions to 2 prim(s).
[INFO] Skipped 1 mesh prim(s) without points.
PASS | z=0.771 m | vel=0.001 m/s | fall=0.699 m | inside_bin=True

--------------------------------------------------------------------------------
[INFO] Testing asset 8/8: /workspace/assets/Containers/TinCan.usd
--------------------------------------------------------------------------------
[INFO] Applied RigidBodyAPI to asset root.
[INFO] Applied CollisionAPI to 1 prim(s).
[INFO] Applied convex-hull mesh collisions to 1 prim(s).
PASS | z=0.809 m | vel=0.001 m/s | fall=0.661 m | inside_bin=True
