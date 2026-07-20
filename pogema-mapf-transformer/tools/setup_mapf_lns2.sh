#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${ROOT_DIR}/external/MAPF-LNS2"

if ! dpkg-query -W libboost-all-dev libeigen3-dev >/dev/null 2>&1; then
  echo "Missing build dependencies: libboost-all-dev and/or libeigen3-dev" >&2
  echo "Install them first: sudo apt install libboost-all-dev libeigen3-dev" >&2
  exit 2
fi

if [[ ! -d "${SOURCE_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/Jiaoyang-Li/MAPF-LNS2.git "${SOURCE_DIR}"
fi

cmake -S "${SOURCE_DIR}" -B "${SOURCE_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${SOURCE_DIR}/build" --parallel

if [[ -x "${SOURCE_DIR}/build/lns" ]]; then
  ln -sfn build/lns "${SOURCE_DIR}/lns"
elif [[ ! -x "${SOURCE_DIR}/lns" ]]; then
  echo "Build completed but the lns binary was not found." >&2
  exit 3
fi

"${SOURCE_DIR}/lns" --help >/dev/null || true
echo "MAPF-LNS2 ready: ${SOURCE_DIR}/lns"
