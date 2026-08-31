#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SOURCE=${LACAM_SOURCE:-"$ROOT/external/MAPF-GPT"}
BUILD=${LACAM_BUILD:-"$ROOT/external/lacam-build"}

if [[ ! -d "$SOURCE/.git" ]]; then
  git clone --depth 1 https://github.com/CognitiveAISystems/MAPF-GPT.git "$SOURCE"
fi

cmake -S "$SOURCE/dataset/lacam" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" --parallel "${BUILD_JOBS:-8}"
test -s "$BUILD/liblacam.so"
echo "$BUILD/liblacam.so"
