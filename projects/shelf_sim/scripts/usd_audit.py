from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from pxr import Usd, UsdGeom, UsdPhysics

PURPOSES = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]


def select_root_prim(stage: Usd.Stage) -> Usd.Prim | None:
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return default_prim

    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            return prim

    return None


def audit_usd(usd_path: Path) -> dict[str, object]:
    result: dict[str, object] = {"usd_path": str(usd_path)}

    if not usd_path.exists():
        result["status"] = "missing_file"
        return result

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        result["status"] = "open_failed"
        return result

    result["status"] = "ok"
    result["up_axis"] = str(UsdGeom.GetStageUpAxis(stage))
    result["meters_per_unit"] = float(UsdGeom.GetStageMetersPerUnit(stage))

    default_prim = stage.GetDefaultPrim()
    result["default_prim"] = default_prim.GetPath().pathString if default_prim else None

    root_prim = select_root_prim(stage)
    result["root_prim"] = root_prim.GetPath().pathString if root_prim else None

    if root_prim:
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES)
        world_bound = bbox_cache.ComputeWorldBound(root_prim)
        size = world_bound.ComputeAlignedBox().GetSize()
        result["bbox_size_m"] = [float(size[0]), float(size[1]), float(size[2])]

    rigid_bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    collisions = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    masses = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.MassAPI)]
    articulations = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]

    result["rigid_body_count"] = len(rigid_bodies)
    result["collision_count"] = len(collisions)
    result["mass_api_count"] = len(masses)
    result["articulation_root_count"] = len(articulations)

    return result


def load_paths_from_manifest(path: Path) -> list[Path]:
    data = json.loads(path.read_text())
    assets = data.get("assets", [])
    return [Path(p) for p in assets]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit USD assets for scale and physics schemas.")
    parser.add_argument("--usd", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("usd_audit.json"))
    args = parser.parse_args()

    if args.usd is None and args.manifest is None:
        raise SystemExit("Provide --usd or --manifest")

    if args.usd is not None:
        usd_paths = [args.usd]
    else:
        usd_paths = load_paths_from_manifest(args.manifest)

    results = [audit_usd(p) for p in usd_paths]
    args.output.write_text(json.dumps(results, indent=2))
    print(f"Wrote audit results to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())