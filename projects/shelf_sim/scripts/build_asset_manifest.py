from __future__ import annotations

import argparse
import json
from pathlib import Path

ASSET_EXTENSIONS = (".usd", ".usda", ".usdc", ".usdz")
IGNORE_DIR_NAMES = {".thumbs"}

def discover_usd_files(root: Path) -> list[Path]:
  if not root.exists():
    raise SystemExit(f"Assets not found at {root}")

  results: list[Path] = []
  for ext in ASSET_EXTENSIONS:
    for path in root.rglob(f"*{ext}"):
      if any(part in IGNORE_DIR_NAMES for part in path.parts):
        continue
      results.append(path)

  return sorted(results)

def main() -> int:
  parser = argparse.ArgumentParser(description="Build an asset manifest.")
  parser.add_argument("--assets-root", type=Path, default=Path("/workspace/assets"))
  parser.add_argument("--output", type=Path, default=Path("assets_manifest.json"))
  args = parser.parse_args()

  usd_paths = [str(p.resolve()) for p in discover_usd_files(args.assets_root)]
  payload = {"assets": usd_paths}
  args.output.write_text(json.dumps(payload, indent=2))
  print(f"Wrote {len(usd_paths)} assets to {args.output}")
  return 0

if __name__ == "__main__":
    raise SystemExit(main())