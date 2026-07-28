#!/usr/bin/env bash

set -eo pipefail

WORKSPACE="/home/lab/alpasim_ros2_ws"
ALPASIM="/home/lab/alpasim"

exec uv run \
  --project "${ALPASIM}/src/driver" \
  python \
  "${WORKSPACE}/src/alpasim_planning/alpasim_planning/vavam_trajectory_planner.py" \
  --ros-args \
  --params-file \
  "${WORKSPACE}/install/alpasim_planning/share/alpasim_planning/config/vavam_planner.yaml"
