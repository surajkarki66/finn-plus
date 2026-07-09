#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Load .env if present
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# UID/GID are readonly in bash; use HOST_UID/HOST_GID for docker compose
export HOST_UID="${HOST_UID:-$(id -u)}"
export HOST_GID="${HOST_GID:-$(id -g)}"

FINN_XILINX_PATH="${FINN_XILINX_PATH:-/tools/Xilinx}"
FINN_XILINX_VERSION="${FINN_XILINX_VERSION:-2024.2}"

if [[ ! -f .env && -f docker/.env.example ]]; then
    echo "Tip: cp docker/.env.example .env and set FINN_XILINX_PATH to your host Vivado install."
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not installed on the host."
    exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
    echo "ERROR: docker compose plugin is not available."
    exit 1
fi

if [[ ! -d "${FINN_XILINX_PATH}" ]]; then
    echo "ERROR: Xilinx tools not found at ${FINN_XILINX_PATH}"
    echo ""
    echo "Install Vivado/Vitis on your Ubuntu 24.04 host first, then set FINN_XILINX_PATH in .env"
    echo "Example: FINN_XILINX_PATH=/tools/Xilinx"
    exit 1
fi

if [[ ! -f "${FINN_XILINX_PATH}/Vivado/${FINN_XILINX_VERSION}/settings64.sh" ]]; then
    echo "WARNING: ${FINN_XILINX_PATH}/Vivado/${FINN_XILINX_VERSION}/settings64.sh not found."
    echo "         Check FINN_XILINX_VERSION in .env matches your installation."
fi

echo "Host: $(. /etc/os-release 2>/dev/null && echo "${PRETTY_NAME:-unknown}" || echo "unknown")"
echo "Container: Ubuntu 22.04 (finn-plus:22.04) — Poetry-only dev flow (wiki Option B)"
echo "Xilinx (from host): ${FINN_XILINX_PATH} (same path inside container)"
echo "User inside container: ${HOST_UID}:${HOST_GID}"

docker compose build finn
docker compose run --rm finn "$@"
