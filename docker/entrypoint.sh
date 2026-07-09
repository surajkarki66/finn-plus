#!/usr/bin/env bash
set -euo pipefail

HOST_UID="${HOST_UID:-1000}"
HOST_GID="${HOST_GID:-1000}"
CONTAINER_USER="${CONTAINER_USER:-finn}"

# Run once as root: create a passwd entry matching the host user, then drop privileges.
if [[ "$(id -u)" -eq 0 ]]; then
    if ! getent group "${HOST_GID}" >/dev/null; then
        groupadd -g "${HOST_GID}" "${CONTAINER_USER}"
    fi
    GROUP_NAME="$(getent group "${HOST_GID}" | cut -d: -f1)"

    if ! getent passwd "${HOST_UID}" >/dev/null; then
        useradd -u "${HOST_UID}" -g "${GROUP_NAME}" -d "/home/${CONTAINER_USER}" -s /bin/bash -M "${CONTAINER_USER}"
    fi

    mkdir -p "/home/${CONTAINER_USER}"
    chown "${HOST_UID}:${HOST_GID}" "/home/${CONTAINER_USER}"
    chown "${HOST_UID}:${HOST_GID}" /finn_build /finn_deps 2>/dev/null || true

    exec gosu "${HOST_UID}:${HOST_GID}" /usr/local/bin/finn-entrypoint.sh "$@"
fi

export HOME="${HOME:-/home/${CONTAINER_USER}}"

XILINX_BASE="${FINN_XILINX_PATH:-/tools/Xilinx}"
XILINX_VER="${FINN_XILINX_VERSION:-2024.2}"

source_xilinx_settings() {
    local settings="$1"
    if [[ -f "${settings}" ]]; then
        # shellcheck disable=SC1090
        source "${settings}"
    fi
}

if [[ -f "${XILINX_BASE}/Vivado/${XILINX_VER}/settings64.sh" ]]; then
    source_xilinx_settings "${XILINX_BASE}/Vivado/${XILINX_VER}/settings64.sh"
elif [[ -d "${XILINX_BASE}/Vivado" ]]; then
    echo "WARNING: Vivado ${XILINX_VER} not found under ${XILINX_BASE}."
    echo "         Install Xilinx tools inside the container, or set FINN_XILINX_VERSION."
fi

if [[ -f "${XILINX_BASE}/Vitis/${XILINX_VER}/settings64.sh" ]]; then
    source_xilinx_settings "${XILINX_BASE}/Vitis/${XILINX_VER}/settings64.sh"
fi

if [[ -n "${XILINX_VIVADO:-}" && -n "${XILINX_VITIS:-}" ]]; then
    export LD_LIBRARY_PATH="${XILINX_VIVADO}/lib/lnx64.o:${XILINX_VITIS}/lnx64/tools/fpo_v7_1:${LD_LIBRARY_PATH:-}"
fi

# AMD AR 000034450: Vivado/Flexera license code crashes in Docker during udev device scan
# (realloc abort in libudev → libXil_lmgr11.so). Preload the system libudev as workaround.
if [[ -f /lib/x86_64-linux-gnu/libudev.so.1 ]]; then
    export LD_PRELOAD="/lib/x86_64-linux-gnu/libudev.so.1${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

export FINN_BUILD_DIR="${FINN_BUILD_DIR:-/finn_build}"
export FINN_DEPS="${FINN_DEPS:-/finn_deps}"
export NUM_DEFAULT_WORKERS="${NUM_DEFAULT_WORKERS:-4}"

mkdir -p "${FINN_BUILD_DIR}" "${FINN_DEPS}"

ensure_settings() {
    if [[ ! -f /workspace/settings.yaml && -f /workspace/docker/settings.yaml.example ]]; then
        echo "Creating /workspace/settings.yaml from docker/settings.yaml.example..."
        cp /workspace/docker/settings.yaml.example /workspace/settings.yaml
    fi
}

needs_poetry_install() {
    if [[ "${FINN_FORCE_POETRY_INSTALL:-0}" == "1" ]]; then
        return 0
    fi
    if ! poetry -C /workspace env info --path >/dev/null 2>&1; then
        return 0
    fi
    if ! poetry -C /workspace run python -c "import finn, torch" >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

activate_poetry_venv() {
    local venv_path
    venv_path="$(poetry -C /workspace env info --path)"
    # shellcheck disable=SC1091
    source "${venv_path}/bin/activate"
    export VIRTUAL_ENV="${venv_path}"
    export PATH="${venv_path}/bin:${PATH}"
}

setup_finn_development() {
    if [[ ! -f /workspace/pyproject.toml ]]; then
        return 0
    fi

    ensure_settings

    if needs_poetry_install; then
        echo "Installing FINN+ development environment (poetry install)..."
        poetry -C /workspace install --no-interaction
    fi

    activate_poetry_venv

    # Not in tests/pyproject.toml; required by end2end/test_end2end_bnn_pynq.py
    poetry -C /workspace run pip install finn-dataset-loading

    if [[ ! -f "${VIRTUAL_ENV}/.finn-deps-updated" || "${FINN_FORCE_DEPS_UPDATE:-0}" == "1" ]]; then
        echo "Updating FINN+ git/download dependencies (finn deps update)..."
        if finn deps update --accept-defaults; then
            touch "${VIRTUAL_ENV}/.finn-deps-updated"
        else
            echo "WARNING: finn deps update failed; continuing with existing dependencies."
        fi
    fi

    if [[ ! -f "${VIRTUAL_ENV}/.finn-check-ok" || "${FINN_FORCE_CHECK:-0}" == "1" ]]; then
        echo "Verifying FINN+ setup (finn check)..."
        if finn check; then
            touch "${VIRTUAL_ENV}/.finn-check-ok"
        else
            echo "WARNING: finn check failed; review settings and dependencies."
        fi
    fi
}

setup_finn_development

if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec "$@"
