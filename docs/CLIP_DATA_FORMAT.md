# AlpaSim Recorded Clip Data Format

**Document status:** Updated for the frozen Step 7 codebase on 2026-09-24  
**Scope:** Raw recorded Clip contract consumed by the dataset annotation tools.  
**Recorder:** `src/alpasim_dataset_tools/alpasim_dataset_tools/data_batch_recorder.py`  
**ROS 2 entry point:** `ros2 run alpasim_dataset_tools data_batch_recorder`  
**Current format:** `dataset_format_version = "0.2-batch"`

## 1. Clip identity and lifecycle

A completed rollout is stored as `test_clip_NNN`, where `NNN` is a zero-padded positive integer. A `test_clip_NNN.tmp` directory is incomplete or invalid and must not be treated as production-ready.

```text
clip_active false to true
  allocate number
  create test_clip_NNN.tmp
  start writers

clip_active true to false
  stop accepting messages
  drain writers
  write final JSON artifacts
  validate

validation passes
  rename to test_clip_NNN

validation fails
  retain test_clip_NNN.tmp
```

## 2. Directory layout

```text
test_clip_NNN/
├── metadata.json
├── validation.json
├── calibration/
│   ├── front_wide.json
│   ├── front_tele.json
│   ├── cross_left.json
│   └── cross_right.json
├── cameras/
│   ├── front_wide/
│   ├── front_tele/
│   ├── cross_left/
│   └── cross_right/
├── ego/
│   ├── ego_state.jsonl
│   ├── ground_truth_future.jsonl
│   ├── planner_output.jsonl
│   ├── executed_path_points.jsonl
│   ├── executed_path_final.json
│   └── complete_recording_ground_truth.json
├── actors/
│   ├── current.jsonl
│   ├── history.jsonl
│   └── future.jsonl
├── route/
│   ├── map_route.jsonl
│   └── navigation_route_local.jsonl
└── map/
    └── vector_map.json
```

Each camera directory contains JPEG frames and `timestamps.jsonl`. Camera streams are independent and may have different counts. Synchronize by timestamp, never by cross-camera frame index.

## 3. Time representation

Canonical dataset time is integer nanoseconds:

```text
stamp_ns = sec * 1,000,000,000 + nanosec
```

ROS message payloads preserve the original `sec` and `nanosec` structure. Wall-clock metadata timestamps are diagnostic only and must not be treated as simulation time.

Dynamic Topic JSONL records are generally in callback-arrival order. Readers must validate monotonicity or explicitly handle duplicates according to the consuming task.

## 4. Coordinates and units

- Distance: metres
- Speed: metres per second
- Acceleration: metres per second squared
- Angles: radians
- Ego-local planar axes: `x` forward, `y` left
- Positive planar yaw: counter-clockwise or left under current utilities

Consumers must read declared frame fields. Do not mix map and Ego-local coordinates directly. Use the shared Step 2 coordinate utilities.

Observed frames include:

- Ego pose: `map`, child `base_link`
- Actor current pose: `map`
- Navigation model-input Route: `base_link`
- VectorMap: frame stored in `map/vector_map.json`

## 5. Metadata and validation

`metadata.json` records Clip number, name, format, status, camera names, recorder settings, and lifecycle wall times.

`validation.json` includes:

- `valid`
- `checks`
- `required_checks`
- camera statistics
- Topic counts
- calibration status
- first and last simulation times
- simulation duration
- map and GT service errors

Current rename-gate requirements include all four calibrations, all four cameras with saved images, no queue drops, no encoding failures, Clock, Ego state, executed path, current Actors, model-input Navigation Route, and VectorMap.

`has_complete_gt_service` is recorded but is not part of the recorder rename gate. Consumers requiring complete-recording GT must check it explicitly.

## 6. Camera data

Selected cameras:

```text
front_wide
front_tele
cross_left
cross_right
```

Each JPEG filename is a per-camera frame index. Each `timestamps.jsonl` row records frame index, `stamp_ns`, frame ID, source width and height, encoding, row stride, relative image path, and JPEG byte count.

Each camera uses an independent bounded writer queue. Queue drops or encoding failures invalidate the Clip under current recorder rules.

## 7. Camera calibration

Calibration files contain recorded F-theta intrinsics and `rig_to_camera` extrinsics. Consumers must not assume a pinhole model.

Observed calibration structure includes:

- logical camera IDs
- source resolution
- F-theta parameters
- principal point and polynomial coefficients
- rig-to-camera quaternion and translation

Calibration may have been cached before the Clip begins and written when the recording session starts.

Review-only image alignment corrections, when required for human-review overlays, belong to annotation tooling. Review offsets do not change raw Clip calibration or the raw format contract.

## 8. Dynamic Topic envelope

Most dynamic Topic JSONL files use:

```json
{
  "topic": "/alpasim/example",
  "message": {}
}
```

Camera timestamps and executed-path point rows use their own direct structures.

## 9. Ego data

### Ego state

`ego/ego_state.jsonl` contains current pose, velocity, acceleration, yaw-related quantities, frames, and speed.

### Future ground truth

`ego/ground_truth_future.jsonl` is future supervision data. It may support Step 3 eligibility and Step 4 labels. It must not be used for Step 6 Navigation or Step 7 current Scene Facts.

### Planner output

`ego/planner_output.jsonl` is not valid evidence for current Scene Facts.

### Executed path

`ego/executed_path_points.jsonl` stores deduplicated points in increasing timestamp order. `ego/executed_path_final.json` stores the final complete executed path message.

### Complete recording ground truth

`ego/complete_recording_ground_truth.json` is optional under the recorder rename gate. Consumers must explicitly check availability.

## 10. Actor data

### Current Actor snapshots

`actors/current.jsonl` contains a sequence of current Actor snapshots. Each Actor may include:

- `track_id`
- `label_class`
- `is_static`
- pose and orientation
- dimensions
- velocity and acceleration
- yaw, yaw rate, and yaw acceleration
- speed

Simulator Actor truth includes Actors that may not be camera-observable. Step 7 applies an observability gate before using Actor truth in final supervision.

A camera-visible object may occasionally lack a usable Actor identity or valid Actor projection at the exact Anchor. Downstream annotation must not fabricate missing Actor state.

### Actor history and future

`actors/history.jsonl` is optional for the frozen Step 7 design because the formal short-history stage derives a past window from `actors/current.jsonl` snapshots at or before the Anchor.

`actors/future.jsonl` contains future simulator truth and must not be used to decide current Scene Facts or current observability.

## 11. Route data

- `route/map_route.jsonl`: map Route stream
- `route/navigation_route_local.jsonl`: model-input Route in Ego-local coordinates

Step 6 queries the latest Route available at or before the Anchor. The final model input exposes only coarse Navigation semantics and must not reveal precise future coordinates, timing, speed, or controls.

## 12. VectorMap

`map/vector_map.json` stores frame, map and scene IDs, bounds, lanes, road edges, traffic signs, and wait lines. VectorMap is required by recorder validation and is used by Steps 4, 6, and 7.

Step 7 uses VectorMap conservatively for Road Context. Wait-line type is not fully propagated into final Stop-line and Yield-line semantics. A visible traffic sign is not automatically a camera-confirmed road fact.

## 13. Annotation-step usage and leakage policy

### Step 1

Scans Clip structure and validates required raw artifacts, readability, timestamp consistency, time ranges, Route, Actors, GT availability, and VectorMap.

### Step 2

Provides unified reading, temporal indexing, interpolation, coordinate transforms, calibration loading, and VectorMap caching.

### Step 3

Uses camera and Ego history, Route availability, and future GT availability to select eligible Candidate Anchors.

### Step 4

Generates supervision labels and may use future Ego trajectory, executed motion, and lane topology.

### Step 5

Uses Candidate Anchors, Meta-actions, and Step 4 features to select Keyframes. Step 5 produces `manifests/keyframe_contract_v0.1.json`, linking the selected Keyframe file to record count and SHA-256.

### Step 6

Uses only Anchor-time-or-earlier Route, Anchor-time Ego state, and static VectorMap. Step 6 validates the Keyframe contract and exact Anchor coverage.

### Step 7 frozen implementation

Step 7 uses:

- selected Keyframes
- exact current Ego and Actor snapshots for formal current geometry
- current and past snapshots at or before the Anchor
- four camera calibrations and synchronized images
- F-theta projection
- Actor oriented 3D boxes
- projected box and surface geometry
- low-resolution Actor surface depth rasters
- Actor-to-Actor Z-buffer competition
- reviewed Actor observability
- short-history relative motion
- static VectorMap Road Context
- deterministic bounded Actor-list selection

Step 7 must not use Actor future, Ego future, complete-recording future trajectory, planner output, Meta-action, or future Ego behavior to decide current Actor presence, observability, motion, roles, or road facts.

Final Actor lists:

```text
lead_actors: maximum 4
left_nearby_actors: maximum 6
right_nearby_actors: maximum 6
```

Final output:

```text
annotations/v0.1-draft/scene_facts.jsonl
```

Frozen result:

```text
rows: 3500
Schema validation errors: 0
Role conflicts: 0
SHA-256: 735203f9ddf3b9f49e892edbb185936caa9db1cd46cbfcdd1b9e0f685958e2b5
```

The earlier standalone Step 7E geometric evidence products are historical development artifacts. Current development must consume the unified frozen Step 7 outputs rather than resume old standalone exporters.

### Step 8

Step 8 may join frozen Keyframes, Meta-actions, Navigation, and Scene Facts using exact Anchor identity. Step 8 must distinguish model-input evidence from supervision-only evidence and must never fabricate missing raw data.

## 14. Reader compatibility requirements

A compatible reader must:

- validate `metadata.json` and format version
- reject or separately classify `.tmp` Clips
- use camera timestamp indexes
- not assume equal camera counts
- support exact-time preference and bounded-tolerance fallback
- validate timestamps
- distinguish Topic envelopes from direct row formats
- respect frame declarations
- cache large files when appropriate
- never fabricate missing data

## 15. Format evolution

Do not silently change the raw Clip schema. If directory names, file names, row envelopes, meanings, time semantics, frame semantics, or validation rules change:

- increment `dataset_format_version`
- update this document
- update Step 2 reader compatibility
- update Step 1 Manifest validation
- add synthetic and real-data tests
- state whether old Clips remain supported

Changes to annotation code, artifact output roots, or review overlays alone do not change the raw Clip format.

## 16. Current related code

Recorder:

```text
src/alpasim_dataset_tools/alpasim_dataset_tools/data_batch_recorder.py
```

Shared reader and path code:

```text
scripts/dataset_tools/project_paths.py
scripts/dataset_tools/step2/clip_reader.py
scripts/dataset_tools/step2/temporal_index.py
scripts/dataset_tools/step2/coordinate_utils.py
scripts/dataset_tools/step2/vector_map_reader.py
```

Frozen Step 7 raw-data consumers and geometry:

```text
scripts/dataset_tools/step7/
scripts/dataset_tools/tests/step7/
```

Primary project handoff:

```text
docs/AI_DATASET_DEVELOPMENT_HANDOFF.md
```
