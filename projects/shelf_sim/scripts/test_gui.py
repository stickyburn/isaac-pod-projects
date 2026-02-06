"""Minimal test: does Isaac Sim open a GUI window?"""
import sys
import os

print(f"[DEBUG] Python: {sys.executable}")
print(f"[DEBUG] DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")
print(f"[DEBUG] Args: {sys.argv}")

# Test 1: Try AppLauncher (isaaclab.sh -p way)
try:
    from isaaclab.app import AppLauncher
    print("[DEBUG] AppLauncher imported OK")
    app_launcher = AppLauncher(headless=False)
    simulation_app = app_launcher.app
    print("[DEBUG] AppLauncher created simulation app")
except Exception as e:
    print(f"[DEBUG] AppLauncher failed: {e}")
    # Test 2: Try getting existing app (isaacsim -p way)
    try:
        import omni.timeline
        print("[DEBUG] Running inside Isaac Sim already")
        from isaacsim import SimulationApp
        simulation_app = SimulationApp.instance()
        print(f"[DEBUG] Got existing app: {simulation_app}")
    except Exception as e2:
        print(f"[DEBUG] Also failed: {e2}")
        sys.exit(1)

# Now create something visible
import omni.usd
from pxr import UsdGeom, Gf

print("[INFO] Creating test cube...")
stage = omni.usd.get_context().get_stage()
cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
cube.GetSizeAttr().Set(1.0)
cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))

print("[INFO] Test cube created at /World/TestCube")
print("[INFO] If you see a cube in the viewport, GUI works!")
print("[INFO] Waiting 60 seconds... (Ctrl+C to quit)")

import time
for i in range(60):
    simulation_app.update()
    time.sleep(1)
    if i % 10 == 0:
        print(f"[INFO] Still running... ({i}s)")

simulation_app.close()
