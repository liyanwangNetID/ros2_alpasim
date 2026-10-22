#!/usr/bin/env bash
# Stream scenes with download-timeout fallback: 2601, 2604, then 2507.
# Static occupancy is optional and never gates simulation.
set -u
set -o pipefail

SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIRECTORY}/load_local_paths.sh"
REPO_DIR="${ALPASIM_ROOT}"
CANDIDATE_MANIFEST="${REPO_DIR}/data/scenes/static_occupancy_scene_candidates.csv"
PREPARE_SCRIPT="${SCRIPT_DIRECTORY}/prepare_static_occupancy_candidate.py"
START_INDEX="${1:-1}"
END_INDEX="${2:-999999999}"
REPORT_CSV="${STATIC_OCCUPANCY_STREAMING_REPORT:-/home/lab/Downloads/static_occupancy_streaming_report.csv}"
DOWNLOAD_TIMEOUT_S="${ALPASIM_USDZ_DOWNLOAD_TIMEOUT_S:-180}"

cd "${REPO_DIR}" || exit 1

if [[ ! -f "${CANDIDATE_MANIFEST}" ]]; then
    echo "Candidate manifest missing: ${CANDIDATE_MANIFEST}" >&2
    echo "Run scripts/generate_static_occupancy_scene_candidates.py first." >&2
    exit 1
fi
if [[ ! -f "${PREPARE_SCRIPT}" ]]; then
    echo "Candidate preparation helper missing: ${PREPARE_SCRIPT}" >&2
    exit 1
fi
if ! [[ "${START_INDEX}" =~ ^[0-9]+$ ]] || ! [[ "${END_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "START_INDEX and END_INDEX must be positive integers." >&2
    exit 1
fi
if (( START_INDEX < 1 || END_INDEX < START_INDEX )); then
    echo "Invalid inclusive index range: ${START_INDEX}..${END_INDEX}" >&2
    exit 1
fi

mapfile -t SCENE_IDS < <(
python3 - "${CANDIDATE_MANIFEST}" "${START_INDEX}" "${END_INDEX}" <<'PY'
import csv, sys
path, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
seen = set()
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        index = int(row["scene_index"])
        scene = row["scene_id"].strip()
        if start <= index <= end and scene not in seen:
            print(f"{index}\t{scene}")
            seen.add(scene)
PY
)
if [[ "${#SCENE_IDS[@]}" -eq 0 ]]; then
    echo "No logical scenes selected." >&2
    exit 1
fi

mkdir -p "$(dirname "${REPORT_CSV}")"
if [[ ! -f "${REPORT_CSV}" ]]; then
    printf 'timestamp,scene_index,scene_id,status,selected_uuid,release,source_name,candidate_rank,attempted_candidates,details\n' > "${REPORT_CSV}"
fi

SCENE_WORK_DIR=""
cleanup_scene() {
    if [[ -n "${SCENE_WORK_DIR}" && -d "${SCENE_WORK_DIR}" ]]; then
        rm -rf -- "${SCENE_WORK_DIR}"
    fi
    SCENE_WORK_DIR=""
}
trap cleanup_scene EXIT
trap 'cleanup_scene; exit 130' INT TERM

append_report() {
    python3 - "${REPORT_CSV}" "$@" <<'PY'
import csv, datetime, sys
path = sys.argv[1]
values = sys.argv[2:]
with open(path, "a", newline="", encoding="utf-8") as f:
    csv.writer(f).writerow([datetime.datetime.now().isoformat(timespec="seconds"), *values])
PY
}

selected_total="${#SCENE_IDS[@]}"
successful=0
unsupported=0
failed=0
position=0

for scene_entry in "${SCENE_IDS[@]}"; do
    position=$((position + 1))
    IFS=$'\t' read -r scene_index scene_id <<< "${scene_entry}"
    mapfile -t CANDIDATES < <(
    python3 - "${CANDIDATE_MANIFEST}" "${scene_id}" <<'PY'
import csv, sys
path, scene_id = sys.argv[1], sys.argv[2]
with open(path, newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["scene_id"].strip() == scene_id]
priority = {"2601": 1, "2601_local": 1, "2604": 2, "2507": 3}
rows.sort(key=lambda r: (priority.get(r["release"].strip(), 99), int(r["candidate_rank"]), r["uuid"]))
for r in rows:
    values = [r[k].strip() for k in (
        "candidate_rank", "candidate_count", "uuid", "release", "source_name", "scenes_csv"
    )]
    if any("\t" in v or "\n" in v for v in values):
        raise SystemExit("Unsupported whitespace in candidate manifest")
    print("\t".join(values))
PY
    )

    echo
    echo "============================================================"
    echo "Logical scene: ${position}/${selected_total}"
    echo "Scene index: ${scene_index}"
    echo "Scene ID: ${scene_id}"
    echo "Historical candidates: ${#CANDIDATES[@]}"
    echo "Selection rule: try 2601, then 2604, then 2507; download timeout=${DOWNLOAD_TIMEOUT_S}s"
    echo "============================================================"

    scene_completed=0
    scene_hard_failed=0
    attempted=0
    download_failures=()

    for candidate in "${CANDIDATES[@]}"; do
        attempted=$((attempted + 1))
        IFS=$'\t' read -r rank candidate_count uuid release source_name scenes_csv <<< "${candidate}"
        cleanup_scene
        SCENE_WORK_DIR="$(mktemp -d "/tmp/alpasim-static-${scene_index}-${rank}-XXXXXX")"
        SCENE_CACHE="${SCENE_WORK_DIR}/nre-artifacts"
        LOG_DIR="${SCENE_WORK_DIR}/wizard-log"
        HF_HOME_DIR="${SCENE_WORK_DIR}/huggingface"
        CANDIDATE_CSV="${SCENE_WORK_DIR}/candidate-scenes.csv"
        PREFLIGHT_JSON="${SCENE_WORK_DIR}/preflight.json"
        HYDRA_DIR="${SCENE_WORK_DIR}/hydra"
        mkdir -p "${SCENE_CACHE}" "${LOG_DIR}" "${HF_HOME_DIR}" "${HYDRA_DIR}"

        echo
        echo "Candidate ${rank}/${candidate_count}: release=${release} source=${source_name} uuid=${uuid}"
        echo "Downloading USDZ with timeout=${DOWNLOAD_TIMEOUT_S}s"
        export HF_HOME="${HF_HOME_DIR}"
        timeout --signal=TERM --kill-after=15s "${DOWNLOAD_TIMEOUT_S}s" \
            uv run --project src/wizard python "${PREPARE_SCRIPT}" \
            --catalog "${scenes_csv}" \
            --uuid "${uuid}" \
            --scene-cache "${SCENE_CACHE}" \
            --candidate-catalog "${CANDIDATE_CSV}" \
            --result-json "${PREFLIGHT_JSON}"
        prepare_code=$?
        if [[ "${prepare_code}" -ne 0 ]]; then
            if [[ "${prepare_code}" -eq 124 || "${prepare_code}" -eq 137 ]]; then
                reason="download_timeout_${DOWNLOAD_TIMEOUT_S}s"
            else
                reason="download_status_${prepare_code}"
            fi
            echo "Candidate download failed; trying next version: ${reason}" >&2
            download_failures+=("${release}:${uuid}:${reason}")
            cleanup_scene
            continue
        fi
        echo "USDZ downloaded; starting exactly one simulation for this logical scene"
        ALPASIM_STATIC_OCCUPANCY_ENABLED=1 \
        uv run --project src/wizard alpasim_wizard \
            deploy=local \
            driver=manual \
            driver_source=external_static \
            topology=1gpu \
            wizard.log_dir="${LOG_DIR}" \
            hydra.run.dir="${HYDRA_DIR}" \
            "scenes.scene_ids=[\"${scene_id}\"]" \
            "scenes.scenes_csv=[\"${CANDIDATE_CSV}\"]" \
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
        run_code=$?
        if [[ "${run_code}" -eq 0 ]]; then
            echo "Scene completed with selected artifact ${uuid} (${release})"
            append_report "${scene_index}" "${scene_id}" "success" "${uuid}" "${release}" "${source_name}" "${rank}" "${attempted}" "usdz_downloaded_static_optional"
            successful=$((successful + 1))
            scene_completed=1
        else
            echo "Complete artifact simulation failed with status ${run_code}; not falling back" >&2
            append_report "${scene_index}" "${scene_id}" "simulation_failed" "${uuid}" "${release}" "${source_name}" "${rank}" "${attempted}" "run_status_${run_code}"
            failed=$((failed + 1))
            scene_hard_failed=1
        fi
        cleanup_scene
        break
    done

    if [[ "${scene_completed}" -eq 0 && "${scene_hard_failed}" -eq 0 ]]; then
        detail="$(IFS=';'; echo "${download_failures[*]}")"
        echo "No artifact version downloaded within the allowed time: ${scene_id}" >&2
        append_report "${scene_index}" "${scene_id}" "download_failed" "" "" "" "" "${attempted}" "${detail}"
        unsupported=$((unsupported + 1))
    fi
done

echo
echo "============================================================"
echo "Static occupancy streaming batch finished"
echo "Selected logical scenes: ${selected_total}"
echo "Successful scenes: ${successful}"
echo "Download-failed scenes: ${unsupported}"
echo "Failed scenes: ${failed}"
echo "Report: ${REPORT_CSV}"
echo "============================================================"

if (( failed > 0 )); then
    exit 1
fi
exit 0
