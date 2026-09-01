#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIRECTORY="$(
    cd "$(dirname "${BASH_SOURCE[0]}")" &&
    pwd
)"

source "${SCRIPT_DIRECTORY}/load_local_paths.sh"

REPO_DIR="${ALPASIM_ROOT}"
SUITE_CSV="${REPO_DIR}/data/scenes/my_gt_sim_suites.csv"
SUITE_ID="public_2601"

cd "${REPO_DIR}" || exit 1

cleanup_scene() {
    if [[ -n "${SCENE_WORK_DIR:-}" ]] &&
       [[ -d "${SCENE_WORK_DIR}" ]]; then
        rm -rf "${SCENE_WORK_DIR}"
    fi
}

trap cleanup_scene EXIT INT TERM

mapfile -t SCENE_IDS < <(
    python3 - "${SUITE_CSV}" "${SUITE_ID}" <<'PY'
import csv
import sys

csv_path = sys.argv[1]
suite_id = sys.argv[2]

with open(csv_path, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        if row["test_suite_id"] == suite_id:
            scene_id = row["scene_id"].strip()

            if scene_id:
                print(scene_id)
PY
)

if [[ "${#SCENE_IDS[@]}" -eq 0 ]]; then
    echo "No scenes found for suite: ${SUITE_ID}" >&2
    exit 1
fi

echo "Found ${#SCENE_IDS[@]} scenes"

scene_index=0
failed_scenes=()

for scene_id in "${SCENE_IDS[@]}"; do
    scene_index=$((scene_index + 1))

    cleanup_scene

    SCENE_WORK_DIR="$(
        mktemp -d \
            "/tmp/alpasim-${scene_index}-XXXXXX"
    )"

    SCENE_CACHE="${SCENE_WORK_DIR}/nre-artifacts"
    LOG_DIR="${SCENE_WORK_DIR}/wizard-log"
    HF_HOME_DIR="${SCENE_WORK_DIR}/huggingface"

    mkdir -p \
        "${SCENE_CACHE}" \
        "${LOG_DIR}" \
        "${HF_HOME_DIR}"

    echo
    echo "============================================================"
    echo "Running scene ${scene_index}/${#SCENE_IDS[@]}"
    echo "Scene ID: ${scene_id}"
    echo "Temporary directory: ${SCENE_WORK_DIR}"
    echo "============================================================"

    # Keep Hugging Face's download cache inside the temporary directory.
    # Removing SCENE_WORK_DIR after the run therefore removes both the
    # AlpaSim scene cache and the associated Hugging Face cache.
    export HF_HOME="${HF_HOME_DIR}"

    uv run --project src/wizard alpasim_wizard \
        deploy=local \
        driver=manual \
        driver_source=external_static \
        topology=1gpu \
        wizard.log_dir="${LOG_DIR}" \
        "scenes.scene_ids=[\"${scene_id}\"]" \
        scenes.scene_cache="${SCENE_CACHE}" \
        wizard.external_services.driver='["172.23.0.1:6789"]' \
        runtime.endpoints.renderer.n_concurrent_rollouts=1 \
        runtime.endpoints.driver.n_concurrent_rollouts=1 \
        runtime.endpoints.physics.n_concurrent_rollouts=1 \
        runtime.endpoints.controller.n_concurrent_rollouts=1 \
        runtime.simulation_config.n_rollouts=1 \
        runtime.simulation_config.n_sim_steps=200 \
        runtime.simulation_config.control_timestep_us=100000 \
        runtime.simulation_config.pose_reporting_interval_us=100000 \
        +runtime.simulation_config.realtime_factor=1.0 \
        'runtime.simulation_config.cameras=[{height:480,width:854,logical_id:camera_cross_left_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_wide_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_tele_30fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_cross_right_120fov,frame_interval_us:100000,shutter_duration_us:30000}]'

    status=$?

    if [[ "${status}" -eq 0 ]]; then
        echo "Scene completed successfully: ${scene_id}"
    else
        echo "Scene failed with status ${status}: ${scene_id}" >&2
        failed_scenes+=("${scene_id}")
    fi

    echo "Deleting temporary USDZ and logs"
    cleanup_scene
    SCENE_WORK_DIR=""
done

echo
echo "============================================================"
echo "Batch finished"
echo "Total scenes: ${#SCENE_IDS[@]}"
echo "Failed scenes: ${#failed_scenes[@]}"
echo "============================================================"

if [[ "${#failed_scenes[@]}" -gt 0 ]]; then
    printf '  %s\n' "${failed_scenes[@]}"
    exit 1
fi