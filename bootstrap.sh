#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SOURCE="${REPO_ROOT}/projects/shelf_sim/source/shelf_sim"

# Configurable constants (edit here when needed).
# Assets are stored in the extension's data directory (gitignored)
ASSETS_DIR="${PROJECT_SOURCE}/shelf_sim/data/Props"
DOWNLOADS_DIR="/tmp/shelf_sim_downloads"
ASSETS_ZIP_URL="https://drive.google.com/uc?id=13GMQuDB87-cP5kB_MRCS5-M5Nnid1Tkq"
ASSETS_ZIP_NAME="shelf_sim_assets.zip"

echo "Repo root: ${REPO_ROOT}"

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("isaaclab") else 1)
PY
then
  if [[ -f "/opt/isaaclab-env/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "/opt/isaaclab-env/bin/activate"
    echo "Activated Isaac Lab venv."
  else
    echo "Isaac Lab venv not active and activation script not found." >&2
    exit 1
  fi
fi

echo "Installing shelf_sim as an Isaac Lab extension (editable)."
python -m pip install -e "${PROJECT_SOURCE}"

# Create data directory for assets
mkdir -p "${ASSETS_DIR}"
mkdir -p "${DOWNLOADS_DIR}"

download_and_extract() {
  local url="$1"
  local zip_path="${DOWNLOADS_DIR}/${ASSETS_ZIP_NAME}"

  echo "Downloading assets from ${url}"
  python -m gdown --fuzzy "${url}" -O "${zip_path}"

  echo "Extracting ${zip_path} -> ${ASSETS_DIR}"
  unzip -q "${zip_path}" -d "${ASSETS_DIR}"

  rm -f "${zip_path}"
}

python -m pip install gdown

download_and_extract "${ASSETS_ZIP_URL}"

# Remove temporary downloads directory
rm -rf "${DOWNLOADS_DIR}" 2>/dev/null || true

# Remove macOS metadata files
find "${ASSETS_DIR}" -name "__MACOSX" -type d -exec rm -rf {} + 2>/dev/null || true

echo "Assets installed to: ${ASSETS_DIR}"
echo "You can now reference them using:"
echo "  from shelf_sim import MUSTARD_JAR_USD_PATH, OIL_TIN_USD_PATH, etc."
