#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_SOURCE="${REPO_ROOT}/projects/shelf_sim/source/shelf_sim"

# Configurable constants (edit here when needed).
ASSETS_DIR="/workspace/assets"
DOWNLOADS_DIR="${ASSETS_DIR}/downloads"
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
