from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from isaaclab.app import AppLauncher


SIM_DT = 1.0 / 120.0
RENDER_INTERVAL = 1
DROP_HEIGHT_M = 0.5
GRID_COLS = 6
GRID_SPACING_M = 0.6
TEST_SECONDS = 3.0
STABLE_LIN_VEL_EPS_M_S = 0.02
MIN_FALL_DISTANCE_M = 0.05
FALL_THROUGH_Z_M = -0.1
FORCED_MASS_KG = 0.5
FLOOR_SIZE_M = 10.0
FLOOR_THICKNESS_M = 0.1


def slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()


def load_manifest(path: Path) -> list[Path]:
    data = json.loads(path.read_text())
    assets = data.get("assets", [])
    return [Path(p) for p in assets]


def main() -> int:
    parser = argparse.ArgumentParser(description="Drop-test USD assets in Isaac Lab.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--drop-height-m", type=float, default=DROP_HEIGHT_M)
    parser.add_argument("--test-seconds", type=float, default=TEST_SECONDS)
    parser.add_argument("--mass-kg", type=float, default=FORCED_MASS_KG)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch
    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.assets import RigidObject, RigidObjectCfg

    asset_paths = load_manifest(args.manifest)

    sim_cfg = SimulationCfg(dt=SIM_DT, render_interval=RENDER_INTERVAL)
    sim = SimulationContext(sim_cfg)

    prim_utils.create_prim("/World/Assets", "Xform")

    floor_cfg = sim_utils.CuboidCfg(
        size=(FLOOR_SIZE_M, FLOOR_SIZE_M, FLOOR_THICKNESS_M),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2)),
    )
    floor_cfg.func(
        "/World/Floor",
        floor_cfg,
        translation=(0.0, 0.0, -FLOOR_THICKNESS_M / 2.0),
    )

    objects: list[tuple[str, RigidObject]] = []
    for idx, usd_path in enumerate(asset_paths):
        name = f"asset_{idx:03d}_{slugify(usd_path.stem) or 'asset'}"
        col = idx % GRID_COLS
        row = idx // GRID_COLS
        pos_x = col * GRID_SPACING_M
        pos_y = row * GRID_SPACING_M

        spawn_cfg = sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=args.mass_kg),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )

        obj_cfg = RigidObjectCfg(
            prim_path=f"/World/Assets/{name}",
            spawn=spawn_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(pos_x, pos_y, args.drop_height_m),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        obj = RigidObject(cfg=obj_cfg)
        objects.append((str(usd_path), obj))

    sim.reset()

    num_steps = int(math.ceil(args.test_seconds / SIM_DT))
    for _ in range(num_steps):
        sim.step()
        for _, obj in objects:
            obj.update(SIM_DT)

    print("Drop test results:")
    for usd_path, obj in objects:
        pos = obj.data.root_link_pos_w[0]
        vel = obj.data.root_link_lin_vel_w[0]
        vel_norm = float(torch.linalg.vector_norm(vel))
        fall_distance = args.drop_height_m - float(pos[2])

        fell = fall_distance >= MIN_FALL_DISTANCE_M
        fell_through = float(pos[2]) < FALL_THROUGH_Z_M
        stable = vel_norm <= STABLE_LIN_VEL_EPS_M_S

        status = "PASS" if (fell and stable and not fell_through) else "FAIL"

        print(
            f"{status} | {usd_path} | z={float(pos[2]):.3f} "
            f"vel={vel_norm:.3f} fall={fall_distance:.3f} "
            f"fell_through={fell_through}"
        )

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())