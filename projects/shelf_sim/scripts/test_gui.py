"""Minimal GUI test using SimulationApp directly.

Run with: /opt/IsaacLab/isaaclab.sh -p ./scripts/test_gui.py

This bypasses Isaac Lab's AppLauncher (which forces headless)
and uses SimulationApp directly with headless=False.
"""
import sys
print(f"[TEST] Script running. Python: {sys.executable}", flush=True)

from isaacsim import SimulationApp

# Create SimulationApp with GUI enabled
simulation_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "window_title": "Shelf Sim - GUI Test",
})
print(f"[TEST] SimulationApp created. is_running={simulation_app.is_running()}", flush=True)

# Now we can import omni modules
import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()

# Create a cube
cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
cube.GetSizeAttr().Set(0.5)
cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.25))

# Create a light
light = stage.DefinePrim("/World/DomeLight", "DomeLight")
light.GetAttribute("inputs:intensity").Set(1000.0)

print("[TEST] Cube + light created. Press F in viewport to frame all.", flush=True)
print("[TEST] Running for 60s... Ctrl+C to quit.", flush=True)

import time
for i in range(600):
    simulation_app.update()
    time.sleep(0.1)
    if not simulation_app.is_running():
        break

simulation_app.close()
