#!/usr/bin/env bash

# Run globally deduplicated AlpaSim scenes one at a time.
#
# Usage:
#   ./run_gt_dataset_streaming_all_unique.sh
#   ./run_gt_dataset_streaming_all_unique.sh 1 100
#   ./run_gt_dataset_streaming_all_unique.sh 101 200
#
# Arguments are inclusive manifest indices. With no arguments, all scenes run.

set -u
set -o pipefail

REPO_DIR="/home/lab/alpasim"
MANIFEST_CSV="${REPO_DIR}/data/scenes/all_unique_scenes.csv"

START_INDEX="${1:-1}"
END_INDEX="${2:-999999999}"

cd "${REPO_DIR}" || exit 1

SCENE_WORK_DIR=""

cleanup_scene() {
    if [[ -n "${SCENE_WORK_DIR}" ]] &&
       [[ -d "${SCENE_WORK_DIR}" ]]; then
        rm -rf -- "${SCENE_WORK_DIR}"
    fi
    SCENE_WORK_DIR=""
}

handle_signal() {
    echo
    echo "Interrupted. Cleaning the current temporary scene directory." >&2
    cleanup_scene
    exit 130
}

trap cleanup_scene EXIT
trap handle_signal INT TERM

if [[ ! -f "${MANIFEST_CSV}" ]]; then
    echo "Manifest does not exist: ${MANIFEST_CSV}" >&2
    exit 1
fi

if ! [[ "${START_INDEX}" =~ ^[0-9]+$ ]] ||
   ! [[ "${END_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "START_INDEX and END_INDEX must be positive integers." >&2
    exit 1
fi

if (( START_INDEX < 1 )); then
    echo "START_INDEX must be at least 1." >&2
    exit 1
fi

if (( END_INDEX < START_INDEX )); then
    echo "END_INDEX must be greater than or equal to START_INDEX." >&2
    exit 1
fi

# Each output line is tab-separated:
# index, scene_id, uuid, release, source_name, scenes_csv
mapfile -t SCENE_ROWS < <(
    python3 - \
        "${MANIFEST_CSV}" \
        "${START_INDEX}" \
        "${END_INDEX}" <<'PY'
import csv
import sys

manifest_path = sys.argv[1]
start_index = int(sys.argv[2])
end_index = int(sys.argv[3])

required_columns = {
    "index",
    "scene_id",
    "uuid",
    "release",
    "source_name",
    "scenes_csv",
}

with open(manifest_path, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    if reader.fieldnames is None:
        raise SystemExit("Manifest has no CSV header")

    missing = required_columns - set(reader.fieldnames)
    if missing:
        raise SystemExit(
            f"Manifest is missing required columns: {sorted(missing)}"
        )

    selected_count = 0

    for row in reader:
        index = int(row["index"])

        if index < start_index or index > end_index:
            continue

        values = [
            str(index),
            row["scene_id"].strip(),
            row["uuid"].strip(),
            row["release"].strip(),
            row["source_name"].strip(),
            row["scenes_csv"].strip(),
        ]

        if any("\t" in value or "\n" in value for value in values):
            raise SystemExit(
                f"Manifest row {index} contains unsupported whitespace"
            )

        if not values[1] or not values[2] or not values[5]:
            raise SystemExit(
                f"Manifest row {index} contains an empty required value"
            )

        print("\t".join(values))
        selected_count += 1

    if selected_count == 0:
        raise SystemExit(
            f"No manifest rows found in range {start_index}..{end_index}"
        )
PY
)

if [[ "${#SCENE_ROWS[@]}" -eq 0 ]]; then
    echo "No scenes selected from ${MANIFEST_CSV}." >&2
    exit 1
fi

TOTAL_SELECTED="${#SCENE_ROWS[@]}"

echo "Manifest: ${MANIFEST_CSV}"
echo "Requested index range: ${START_INDEX}..${END_INDEX}"
echo "Selected unique scenes: ${TOTAL_SELECTED}"

batch_position=0
successful_scenes=()
failed_scenes=()

for scene_row in "${SCENE_ROWS[@]}"; do
    batch_position=$((batch_position + 1))

    IFS=$'\t' read -r \
        manifest_index \
        scene_id \
        scene_uuid \
        release \
        source_name \
        scenes_csv \
        <<< "${scene_row}"

    if [[ ! -f "${scenes_csv}" ]]; then
        echo "Artifact catalog does not exist: ${scenes_csv}" >&2
        failed_scenes+=("${manifest_index}:${scene_id}:missing_catalog")
        continue
    fi

    cleanup_scene

    SCENE_WORK_DIR="$(
        mktemp -d "/tmp/alpasim-${manifest_index}-XXXXXX"
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
    echo "Batch item: ${batch_position}/${TOTAL_SELECTED}"
    echo "Manifest index: ${manifest_index}"
    echo "Scene ID: ${scene_id}"
    echo "UUID: ${scene_uuid}"
    echo "Release: ${release}"
    echo "Source: ${source_name}"
    echo "Artifact catalog: ${scenes_csv}"
    echo "Temporary directory: ${SCENE_WORK_DIR}"
    echo "============================================================"

    # Keep the Hugging Face cache inside the scene-specific temporary
    # directory. Deleting SCENE_WORK_DIR removes the USDZ, Wizard logs,
    # AlpaSim scene cache, and Hugging Face download cache together.
    export HF_HOME="${HF_HOME_DIR}"

    # The manifest has already selected one unique scene_id and the correct
    # artifact catalog. For current catalogs, direct scene selection resolves
    # the available artifact from that catalog. The UUID is printed above and
    # retained in the manifest for traceability.
    uv run --project src/wizard alpasim_wizard \
        deploy=local \
        driver=manual \
        driver_source=external_static \
        topology=1gpu \
        wizard.log_dir="${LOG_DIR}" \
        "scenes.scene_ids=[\"${scene_id}\"]" \
        "scenes.scenes_csv=[\"${scenes_csv}\"]" \
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
        successful_scenes+=("${manifest_index}:${scene_id}")
    else
        echo \
            "Scene failed with status ${status}: ${scene_id}" \
            >&2
        failed_scenes+=("${manifest_index}:${scene_id}:status_${status}")
    fi

    echo "Deleting temporary USDZ, AlpaSim logs, and HF cache"
    cleanup_scene
done

echo
echo "============================================================"
echo "Batch finished"
echo "Selected scenes: ${TOTAL_SELECTED}"
echo "Successful scenes: ${#successful_scenes[@]}"
echo "Failed scenes: ${#failed_scenes[@]}"
echo "============================================================"

if [[ "${#failed_scenes[@]}" -gt 0 ]]; then
    echo "Failed scene records:" >&2
    printf '  %s\n' "${failed_scenes[@]}" >&2
    exit 1
fi
