#!/usr/bin/env bash
# Download PYNQ board files on the HOST (where SSL usually works) if finn deps update fails
# for pynq-z1 / pynq-z2 inside the container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BOARD_DIR="${REPO_ROOT}/finn_deps/board_files"
mkdir -p "${BOARD_DIR}"
cd "${BOARD_DIR}"

download() {
    local name="$1"
    local url="$2"
    if [[ -d "${name}" ]]; then
        echo "${name} already present, skipping."
        return
    fi
    echo "Downloading ${name}..."
    wget -q -O "${name}.zip" "${url}"
    unzip -qo "${name}.zip"
    rm -f "${name}.zip"
    echo "${name} installed."
}

download "pynq-z1" "https://github.com/cathalmccabe/pynq-z1_board_files/raw/master/pynq-z1.zip"
download "pynq-z2" "https://dpoauwgwqsy2x.cloudfront.net/Download/pynq-z2.zip"

echo "Done. Board files are in ${BOARD_DIR}"
