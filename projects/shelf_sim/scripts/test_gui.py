"""Minimal test: does Isaac Sim open a GUI window?

Run with: /opt/IsaacLab/isaaclab.sh -p ./scripts/test_gui.py
"""
import sys
import os
import subprocess

print(f"[DEBUG] Python: {sys.executable}")
print(f"[DEBUG] DISPLAY: {os.environ.get('DISPLAY', 'NOT SET')}")

# Check if X11 display actually works
try:
    result = subprocess.run(["xdpyinfo"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        lines = result.stdout.strip().split('\n')
        print(f"[DEBUG] X11 display OK: {lines[0] if lines else 'yes'}")
    else:
        print(f"[DEBUG] X11 display FAILED: {result.stderr.strip()}")
except Exception as e:
    print(f"[DEBUG] xdpyinfo not available: {e}")

# Check Vulkan
try:
    result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        for line in result.stdout.split('\n'):
            if 'GPU' in line or 'deviceName' in line or 'apiVersion' in line:
                print(f"[DEBUG] Vulkan: {line.strip()}")
    else:
        print(f"[DEBUG] Vulkan FAILED: {result.stderr.strip()[:200]}")
except Exception as e:
    print(f"[DEBUG] vulkaninfo not available: {e}")

print("[DEBUG] Creating AppLauncher with headless=False, window title visible...")

from isaaclab.app import AppLauncher

# Explicitly request GUI with rendering
launcher_cfg = {"headless": False, "enable_cameras": True}
app_launcher = AppLauncher(launcher_cfg)
simulation_app = app_launcher.app

print(f"[DEBUG] simulation_app created: {type(simulation_app)}")
print(f"[DEBUG] simulation_app.is_running: {simulation_app.is_running()}")

# Check if it's actually headless
try:
    is_headless = simulation_app.get_setting("/app/window/enabled") is False
    print(f"[DEBUG] Window enabled setting: {simulation_app.get_setting('/app/window/enabled')}")
    print(f"[DEBUG] Renderer: {simulation_app.get_setting('/app/renderer/active')}")
    print(f"[DEBUG] Experience: {simulation_app.get_setting('/app/experience')}")
except Exception as e:
    print(f"[DEBUG] Could not read settings: {e}")

# Create a visible cube
import omni.usd
from pxr import UsdGeom, Gf

stage = omni.usd.get_context().get_stage()
cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
cube.GetSizeAttr().Set(1.0)
cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.5))

print("[INFO] Test cube at /World/TestCube. Waiting 30s...")

import time
for i in range(30):
    simulation_app.update()
    time.sleep(1)

simulation_app.close()
