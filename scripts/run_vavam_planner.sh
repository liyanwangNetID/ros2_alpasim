#!/usr/bin/env bash

set -eo pipefail

SCRIPT_DIRECTORY="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${SCRIPT_DIRECTORY}/load_local_paths.sh"

WORKSPACE="${ALPASIM_ROS2_WS}"
ALPASIM="${ALPASIM_ROOT}"

exec uv run \
  --project "${ALPASIM}/src/driver" \
  python \
  "${WORKSPACE}/src/alpasim_planning/alpasim_planning/vavam_trajectory_planner.py" \
  --ros-args \
  --params-file \
  "${WORKSPACE}/install/alpasim_planning/share/alpasim_planning/config/vavam_planner.yaml"
