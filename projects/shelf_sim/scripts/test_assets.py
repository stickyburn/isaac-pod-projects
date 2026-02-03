from __future__ import annotations

"""
Test script for shelf_sim USD assets.

This script drop-tests each asset into a kinematic bin placed on a table.
It verifies each asset spawns, falls, settles, and remains inside bin bounds.

Run via Isaac Lab: /opt/IsaacLab/isaaclab.sh -p scripts/test_assets.py
Run headless: /opt/IsaacLab/isaaclab.sh -p scripts/test_assets.py --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = REPO_ROOT / "projects/shelf_sim/assets_manifest.json"
REPORTS_DIR = REPO_ROOT / "projects/shelf_sim/reports"

SIM_DEVICE = "cuda:0"
DROP_HEIGHT_M = 0.5
TEST_SECONDS = 3.0
STABLE_LIN_VEL_EPS_M_S = 0.02
MIN_FALL_DISTANCE_M = 0.05
FORCED_MASS_KG = 0.5

TABLE_SIZE_M = (1.2, 0.8, 0.05)
TABLE_TOP_Z_M = 0.75

BIN_CENTER_XY = (0.0, 0.0)
BIN_INNER_SIZE_M = (0.4, 0.3)
BIN_WALL_THICKNESS_M = 0.02
BIN_WALL_HEIGHT_M = 0.2
BIN_BASE_THICKNESS_M = 0.02
BIN_INNER_MARGIN_M = 0.02

parser = argparse.ArgumentParser(description="Drop-test USD assets into a bin on a table.")
parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
parser.add_argument("--drop-height-m", type=float, default=DROP_HEIGHT_M)
parser.add_argument("--test-seconds", type=float, default=TEST_SECONDS)
parser.add_argument("--mass-kg", type=float, default=FORCED_MASS_KG)
parser.add_argument("--stable-vel-eps", type=float, default=STABLE_LIN_VEL_EPS_M_S)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import build_simulation_context
from isaaclab.sim import schemas as schema_utils
from pxr import PhysxSchema, UsdGeom, UsdPhysics


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
        content = "\n".join(self._lines).rstrip() + "\n"
        self.path.write_text(content)


def log(message: str = "") -> None:
    if REPORT_WRITER is not None:
        REPORT_WRITER.log(message)
    else:
        print(message)


def load_manifest(path: Path) -> list[Path]:
    data = json.loads(path.read_text())
    assets = data.get("assets", [])
    return [Path(p) for p in assets]


def spawn_table(root_path: str) -> None:
    table_cfg = sim_utils.CuboidCfg(
        size=TABLE_SIZE_M,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15)),
    )
    translation = (0.0, 0.0, TABLE_TOP_Z_M - TABLE_SIZE_M[2] / 2.0)
    table_cfg.func(f"{root_path}/Table", table_cfg, translation=translation)


def spawn_bin(root_path: str) -> None:
    bin_root = f"{root_path}/Bin"
    prim_utils.create_prim(bin_root, "Xform")

    inner_x, inner_y = BIN_INNER_SIZE_M
    wall_t = BIN_WALL_THICKNESS_M
    wall_h = BIN_WALL_HEIGHT_M
    base_t = BIN_BASE_THICKNESS_M
    center_x, center_y = BIN_CENTER_XY

    base_size = (inner_x + 2 * wall_t, inner_y + 2 * wall_t, base_t)
    base_cfg = sim_utils.CuboidCfg(
        size=base_size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.08, 0.12, 0.2)),
    )
    base_z = TABLE_TOP_Z_M + base_t / 2.0
    base_cfg.func(f"{bin_root}/Base", base_cfg, translation=(center_x, center_y, base_z))

    wall_cfg = sim_utils.CuboidCfg(
        size=(wall_t, inner_y + 2 * wall_t, wall_h),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.16, 0.28)),
    )
    wall_z = TABLE_TOP_Z_M + base_t + wall_h / 2.0
    wall_offset_x = inner_x / 2.0 + wall_t / 2.0
    wall_cfg.func(f"{bin_root}/Wall_PosX", wall_cfg, translation=(center_x + wall_offset_x, center_y, wall_z))
    wall_cfg.func(f"{bin_root}/Wall_NegX", wall_cfg, translation=(center_x - wall_offset_x, center_y, wall_z))

    wall_cfg_y = sim_utils.CuboidCfg(
        size=(inner_x, wall_t, wall_h),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.16, 0.28)),
    )
    wall_offset_y = inner_y / 2.0 + wall_t / 2.0
    wall_cfg_y.func(f"{bin_root}/Wall_PosY", wall_cfg_y, translation=(center_x, center_y + wall_offset_y, wall_z))
    wall_cfg_y.func(f"{bin_root}/Wall_NegY", wall_cfg_y, translation=(center_x, center_y - wall_offset_y, wall_z))


def _mesh_has_points(prim) -> bool:
    if not prim.IsA(UsdGeom.Mesh):
        return False
    mesh = UsdGeom.Mesh(prim)
    points_attr = mesh.GetPointsAttr()
    if not points_attr:
        return False
    points = points_attr.Get()
    if points is None:
        return False
    return len(points) > 0


def _mesh_needs_convex_approx(prim) -> bool:
    if PhysxSchema.PhysxConvexHullCollisionAPI(prim):
        return False
    if PhysxSchema.PhysxConvexDecompositionCollisionAPI(prim):
        return False
    if PhysxSchema.PhysxSDFMeshCollisionAPI(prim):
        return False
    if PhysxSchema.PhysxTriangleMeshCollisionAPI(prim):
        return True
    if PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI(prim):
        return True

    mesh_collision_api = UsdPhysics.MeshCollisionAPI(prim)
    if mesh_collision_api:
        approximation = mesh_collision_api.GetApproximationAttr().Get()
        if approximation in (None, "", "none", "meshSimplification", "triangleMesh"):
            return True
        return False

    return True


def _ensure_asset_physics(root_prim_path: str) -> dict:
    rigid_bodies = sim_utils.get_all_matching_child_prims(
        root_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    collisions = sim_utils.get_all_matching_child_prims(
        root_prim_path, predicate=lambda prim: prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    mesh_prims = sim_utils.get_all_matching_child_prims(
        root_prim_path, predicate=lambda prim: prim.IsA(UsdGeom.Mesh)
    )
    mesh_prims_with_points: list = []
    mesh_prims_without_points = 0
    for prim in mesh_prims:
        if _mesh_has_points(prim):
            mesh_prims_with_points.append(prim)
        else:
            mesh_prims_without_points += 1

    applied_rb = False
    if len(rigid_bodies) == 0:
        schema_utils.define_rigid_body_properties(
            root_prim_path, sim_utils.RigidBodyPropertiesCfg()
        )
        rigid_bodies = [prim_utils.get_prim_at_path(root_prim_path)]
        applied_rb = True

    if len(rigid_bodies) != 1:
        return {
            "ok": False,
            "reason": f"expected 1 rigid body, found {len(rigid_bodies)}",
            "rigid_body_count": len(rigid_bodies),
            "collision_count": len(collisions),
            "mesh_count": len(mesh_prims),
            "applied_rb": applied_rb,
            "applied_collision": 0,
            "mesh_collision_applied": 0,
            "skipped_mesh_without_points": mesh_prims_without_points,
            "rigid_body_root": None,
        }

    rigid_body_root = rigid_bodies[0].GetPath().pathString
    schema_utils.define_mass_properties(
        rigid_body_root, sim_utils.MassPropertiesCfg(mass=args_cli.mass_kg)
    )

    applied_collision = 0
    if len(collisions) == 0:
        targets = mesh_prims_with_points if mesh_prims_with_points else [rigid_bodies[0]]
        for prim in targets:
            schema_utils.define_collision_properties(
                prim.GetPath().pathString, sim_utils.CollisionPropertiesCfg()
            )
        applied_collision = len(targets)

    mesh_collision_applied = 0
    convex_cfg = schema_utils.ConvexHullPropertiesCfg()
    for prim in mesh_prims_with_points:
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if not _mesh_needs_convex_approx(prim):
            continue
        schema_utils.define_mesh_collision_properties(prim.GetPath().pathString, convex_cfg)
        mesh_collision_applied += 1

    return {
        "ok": True,
        "rigid_body_count": len(rigid_bodies),
        "collision_count": len(collisions),
        "mesh_count": len(mesh_prims),
        "applied_rb": applied_rb,
        "applied_collision": applied_collision,
        "mesh_collision_applied": mesh_collision_applied,
        "skipped_mesh_without_points": mesh_prims_without_points,
        "rigid_body_root": rigid_body_root,
    }


def spawn_asset(usd_path: Path, prim_path: str, drop_height_m: float) -> tuple[RigidObject, float, dict]:
    base_t = BIN_BASE_THICKNESS_M
    wall_h = BIN_WALL_HEIGHT_M
    spawn_z = TABLE_TOP_Z_M + base_t + wall_h + drop_height_m
    center_x, center_y = BIN_CENTER_XY

    if prim_utils.is_prim_path_valid(prim_path):
        prim_utils.delete_prim(prim_path)

    prim_utils.create_prim(
        prim_path,
        prim_type="Xform",
        usd_path=str(usd_path),
        translation=(center_x, center_y, spawn_z),
        orientation=(1.0, 0.0, 0.0, 0.0),
    )

    physics_info = _ensure_asset_physics(prim_path)
    if not physics_info["ok"]:
        raise RuntimeError(f"Physics setup failed: {physics_info['reason']}")

    obj_cfg = RigidObjectCfg(
        prim_path=prim_path,
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(center_x, center_y, spawn_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    return RigidObject(cfg=obj_cfg), spawn_z, physics_info


def is_inside_bin(pos_x: float, pos_y: float) -> bool:
    inner_x, inner_y = BIN_INNER_SIZE_M
    margin = BIN_INNER_MARGIN_M
    half_x = max((inner_x / 2.0) - margin, 0.0)
    half_y = max((inner_y / 2.0) - margin, 0.0)
    center_x, center_y = BIN_CENTER_XY
    return abs(pos_x - center_x) <= half_x and abs(pos_y - center_y) <= half_y


def main() -> int:
    global REPORT_WRITER
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"test_assets_report_{timestamp}.md"
    REPORT_WRITER = ReportWriter(report_path)

    log(f"[INFO] Report path: {report_path}")
    log(f"[INFO] Start time (UTC): {timestamp}")
    log(f"[INFO] Command: {' '.join(sys.argv)}")
    log(f"[INFO] Manifest: {args_cli.manifest}")

    log("\n" + "=" * 80)
    log("USD Asset Drop Test (Table + Bin)")
    log("=" * 80)

    if not args_cli.manifest.exists():
        log(f"[ERROR] Manifest not found: {args_cli.manifest}")
        return 1

    asset_paths = load_manifest(args_cli.manifest)
    if not asset_paths:
        log("[ERROR] No assets found in manifest.")
        return 1

    log(f"[INFO] Asset count: {len(asset_paths)}")
    log(f"[INFO] Drop height: {args_cli.drop_height_m:.2f} m")
    log(f"[INFO] Test duration: {args_cli.test_seconds:.2f} s")
    log(f"[INFO] Forced mass: {args_cli.mass_kg:.2f} kg")
    log(f"[INFO] Stable vel eps: {args_cli.stable_vel_eps:.3f} m/s")
    log(
        "[INFO] Table size: "
        f"{TABLE_SIZE_M[0]:.2f} x {TABLE_SIZE_M[1]:.2f} x {TABLE_SIZE_M[2]:.2f} m"
    )
    log(
        "[INFO] Bin inner size: "
        f"{BIN_INNER_SIZE_M[0]:.2f} x {BIN_INNER_SIZE_M[1]:.2f} m"
    )

    success = True
    missing_assets = 0
    results: list[tuple[str, str]] = []

    with build_simulation_context(
        device=SIM_DEVICE,
        auto_add_lighting=True,
        add_ground_plane=True,
    ) as sim:
        log(f"[INFO] Simulation device: {sim.device}")
        log(f"[INFO] Simulation dt: {sim.cfg.dt}")

        prim_utils.create_prim("/World/Env_0", "Xform")
        prim_utils.create_prim("/World/Env_0/Props", "Xform")
        prim_utils.create_prim("/World/Env_0/Assets", "Xform")

        spawn_table("/World/Env_0/Props")
        spawn_bin("/World/Env_0/Props")

        sim_steps = int(math.ceil(args_cli.test_seconds / sim.cfg.dt))

        for idx, usd_path in enumerate(asset_paths):
            log("\n" + "-" * 80)
            log(f"[INFO] Testing asset {idx + 1}/{len(asset_paths)}: {usd_path}")
            log("-" * 80)

            if not usd_path.exists():
                log(f"[FAIL] Missing asset file: {usd_path}")
                success = False
                missing_assets += 1
                results.append(("MISSING", str(usd_path)))
                continue

            prim_path = f"/World/Env_0/Assets/asset_{idx:03d}"
            if prim_utils.is_prim_path_valid(prim_path):
                prim_utils.delete_prim(prim_path)

            try:
                obj, spawn_z, physics_info = spawn_asset(
                    usd_path, prim_path, args_cli.drop_height_m
                )
                if physics_info["applied_rb"]:
                    log("[INFO] Applied RigidBodyAPI to asset root.")
                if physics_info["applied_collision"] > 0:
                    log(f"[INFO] Applied CollisionAPI to {physics_info['applied_collision']} prim(s).")
                if physics_info["mesh_collision_applied"] > 0:
                    log(
                        "[INFO] Applied convex-hull mesh collisions to "
                        f"{physics_info['mesh_collision_applied']} prim(s)."
                    )
                if physics_info["skipped_mesh_without_points"] > 0:
                    log(
                        "[INFO] Skipped "
                        f"{physics_info['skipped_mesh_without_points']} mesh prim(s) without points."
                    )
                sim.reset()
            except Exception as exc:
                log(f"[FAIL] Physics setup failed for {usd_path}: {exc}")
                success = False
                results.append(("FAIL", str(usd_path)))
                if prim_utils.is_prim_path_valid(prim_path):
                    prim_utils.delete_prim(prim_path)
                continue

            for _ in range(sim_steps):
                sim.step()
                obj.update(sim.cfg.dt)

            pos = obj.data.root_link_pos_w[0]
            vel = obj.data.root_link_lin_vel_w[0]
            vel_norm = float(torch.linalg.vector_norm(vel))
            final_z = float(pos[2])
            fall_distance = spawn_z - final_z

            fell = fall_distance >= MIN_FALL_DISTANCE_M
            stable = vel_norm <= args_cli.stable_vel_eps
            inside = is_inside_bin(float(pos[0]), float(pos[1]))

            status = "PASS" if (fell and stable and inside) else "FAIL"
            if status != "PASS":
                success = False

            results.append((status, str(usd_path)))

            log(
                f"{status} | z={final_z:.3f} m | "
                f"vel={vel_norm:.3f} m/s | "
                f"fall={fall_distance:.3f} m | "
                f"inside_bin={inside}"
            )

            prim_utils.delete_prim(prim_path)

    log("\n" + "=" * 80)
    log("ASSET TEST SUMMARY")
    log("=" * 80)
    for status, usd_path in results:
        log(f"{status} | {usd_path}")
    log(f"[INFO] Missing assets: {missing_assets}")
    log(f"[INFO] Overall result: {'PASS' if success else 'FAIL'}")

    return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"\n[ERROR] Exception occurred: {exc}")
        for line in traceback.format_exc().splitlines():
            log(line)
        raise SystemExit(1)
    finally:
        if REPORT_WRITER is not None:
            REPORT_WRITER.write()
        simulation_app.close()
