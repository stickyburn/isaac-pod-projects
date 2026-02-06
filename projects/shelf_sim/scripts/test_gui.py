"""Minimal test for isaacsim -p GUI.

Run with: isaacsim -p ./scripts/test_gui.py
Do NOT run with isaaclab.sh (that forces headless).
"""
import sys
print(f"[TEST_GUI] Script is executing! Python: {sys.executable}", flush=True)

try:
    import omni.usd
    from pxr import UsdGeom, Gf, Sdf

    print("[TEST_GUI] Creating test scene...", flush=True)
    stage = omni.usd.get_context().get_stage()

    # Ground plane
    ground = UsdGeom.Mesh.Define(stage, "/World/Ground")
    ground.CreatePointsAttr([(-50, -50, 0), (50, -50, 0), (50, 50, 0), (-50, 50, 0)])
    ground.CreateFaceVertexCountsAttr([4])
    ground.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

    # Colored cube
    cube = UsdGeom.Cube.Define(stage, "/World/TestCube")
    cube.GetSizeAttr().Set(0.5)
    cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.25))

    # Light
    light = stage.DefinePrim("/World/Light", "DomeLight")
    light.GetAttribute("inputs:intensity").Set(1000.0)

    print("[TEST_GUI] Scene created! You should see a cube in the viewport.", flush=True)
    print("[TEST_GUI] If viewport is empty, press F to frame all objects.", flush=True)
    print("[TEST_GUI] Script done - Isaac Sim will keep running.", flush=True)

except Exception as e:
    print(f"[TEST_GUI] ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
