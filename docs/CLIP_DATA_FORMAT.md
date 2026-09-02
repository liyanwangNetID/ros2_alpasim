# AlpaSim Recorded Clip Data Format

> **Scope:** This document defines the raw recorded Clip contract consumed by the dataset annotation tools.
>
> **Recorder:** `src/alpasim_dataset_tools/alpasim_dataset_tools/data_batch_recorder.py`
>
> **ROS 2 entry point:** `ros2 run alpasim_dataset_tools data_batch_recorder`
>
> **Current format:** `dataset_format_version = "0.2-batch"`

## 1. Clip identity and naming

Each completed rollout is stored in one directory:

```text
test_clip_NNN
```

Examples:

```text
test_clip_001
test_clip_002
test_clip_343
```

Rules:

- `NNN` is a zero-padded positive integer with a minimum width of three digits.
- `clip_name` in `metadata.json` must match the directory name.
- `clip_number` is the numeric suffix.
- The recorder allocates the smallest unused positive integer.
- Both completed `test_clip_NNN` directories and retained `test_clip_NNN.tmp` directories reserve their number.
- Gaps in numbering are allowed.

## 2. Recording lifecycle

The batch recorder is controlled by `/alpasim/simulation/clip_active` edges.

```text
false -> true
    allocate the smallest unused Clip number
    create test_clip_NNN.tmp
    create writers and begin accepting messages

true -> false
    stop accepting messages
    drain and close image and JSONL writers
    write final JSON artifacts
    write validation.json
    replace metadata.json with final status

validation passes
    rename test_clip_NNN.tmp to test_clip_NNN

validation fails
    retain test_clip_NNN.tmp for diagnosis
```

A directory without the `.tmp` suffix represents a finalized Clip. A `.tmp` directory must not be treated as a production-ready Clip.

## 3. Directory layout

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
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   └── timestamps.jsonl
│   ├── front_tele/
│   │   ├── *.jpg
│   │   └── timestamps.jsonl
│   ├── cross_left/
│   │   ├── *.jpg
│   │   └── timestamps.jsonl
│   └── cross_right/
│       ├── *.jpg
│       └── timestamps.jsonl
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

The four camera streams are independent. Their frame counts do not have to be equal. Downstream synchronization must use timestamps rather than frame indices across cameras.

## 4. Time representation

### 4.1 Canonical integer time

Dataset tools convert ROS time to integer nanoseconds:

```text
stamp_ns = sec * 1,000,000,000 + nanosec
```

Anchor identifiers and downstream temporal indexing use nanoseconds.

### 4.2 ROS time objects

ROS message payloads retain their original structure:

```json
{
  "sec": 9299,
  "nanosec": 912661000
}
```

### 4.3 Ordering expectations

- Camera `timestamps.jsonl` records correspond to the JPEG frame sequence.
- Dynamic Topic JSONL records represent callback arrival order.
- `executed_path_points.jsonl` appends only points whose timestamp is greater than the last appended executed-path timestamp.
- Downstream readers must validate monotonicity and reject or explicitly handle duplicate timestamps.

### 4.4 Wall time versus simulation time

`metadata.json` contains wall-clock Unix timestamps for recorder lifecycle diagnostics:

- `created_wall_time_unix`
- `closed_wall_time_unix`

Simulation time is recorded separately in `validation.json`:

- `first_sim_time_ns`
- `last_sim_time_ns`
- `sim_duration_sec`

Do not use wall time as simulation time.

## 5. Coordinate and unit conventions

The individual ROS messages declare their frames. Consumers must not assume that all files share one frame.

Observed examples:

- Ego pose:
  - `pose_frame_id = "map"`
  - `child_frame_id = "base_link"`
  - `dynamics_frame_id = "base_link"`
- Actor current state:
  - `pose_frame_id = "map"`
  - `dynamics_frame_id = "map"`
- Navigation model-input Route:
  - `frame_id = "base_link"`
  - `source_frame_id = "base_link"`
- VectorMap:
  - `frame_id` is stored in `map/vector_map.json`.

Conventions used by the current annotation tools:

- position and distance: metres;
- speed: metres per second;
- acceleration: metres per second squared;
- yaw and angular quantities: radians;
- Ego-local planar axes: x forward, y left;
- positive planar yaw: counter-clockwise/left under the current geometry utilities.

Consumers must read frame fields and use the shared coordinate utilities rather than mixing map and Ego-local coordinates directly.

## 6. Metadata

### File

```text
metadata.json
```

### Initial recording state

At session start, the recorder writes metadata with:

```json
{
  "dataset_format_version": "0.2-batch",
  "clip_number": 1,
  "clip_name": "test_clip_001",
  "status": "recording",
  "image_storage": "JPEG_PER_FRAME",
  "jpeg_quality": 90,
  "save_all_received_frames": true,
  "camera_names": [
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right"
  ],
  "created_wall_time_unix": 0.0
}
```

### Final state

At finalization, the recorder replaces the file with:

```json
{
  "dataset_format_version": "0.2-batch",
  "clip_number": 1,
  "clip_name": "test_clip_001",
  "status": "complete",
  "image_storage": "JPEG_PER_FRAME",
  "jpeg_quality": 90,
  "save_all_received_frames": true,
  "camera_names": [
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right"
  ],
  "created_wall_time_unix": 0.0,
  "closed_wall_time_unix": 0.0,
  "validation_file": "validation.json"
}
```

For a failed validation, `status` is `invalid` and the directory retains its `.tmp` suffix.

## 7. Validation

### File

```text
validation.json
```

### Required checks

A Clip is renamed to its final directory only when all current required checks pass:

```text
all_four_calibrations
all_four_cameras_saved_images
no_camera_queue_drops
no_camera_encode_failures
has_clock
has_ego_state
has_executed_path
has_actor_current
has_navigation_route
has_vector_map
```

### Reported but currently non-required check

```text
has_complete_gt_service
```

The complete-recording GT service result is recorded in validation, but it is not currently in `required_checks`. Therefore:

- a Clip may pass recorder validation without it;
- a downstream task that requires `ego/complete_recording_ground_truth.json` must check the file explicitly;
- Step 1 Manifest validation may impose stricter requirements than the recorder's rename gate.

### Other validation fields

```text
valid
checks
required_checks
camera_statistics
topic_counts
calibrations_saved
first_sim_time_ns
last_sim_time_ns
sim_duration_sec
map_service_error
gt_service_error
```

Each camera statistics record contains:

```text
received_count
saved_count
dropped_queue_full
encode_failures
```

## 8. Camera data

### Cameras

```text
front_wide
front_tele
cross_left
cross_right
```

### Image files

Each accepted frame is encoded independently as JPEG:

```text
000000.jpg
000001.jpg
...
```

The filename is based on the per-camera `frame_index`, not simulation time.

### Timestamp index

Each camera directory contains `timestamps.jsonl`. Each row has:

```json
{
  "frame_index": 0,
  "stamp_ns": 9299912661000,
  "frame_id": "camera_frame",
  "width": 1920,
  "height": 1080,
  "encoding": "rgb8",
  "step": 5760,
  "image_path": "000000.jpg",
  "jpeg_bytes": 123456
}
```

Field meanings:

- `frame_index`: zero-based index within that camera stream;
- `stamp_ns`: source image timestamp in nanoseconds;
- `frame_id`: ROS image-header frame;
- `width`, `height`: source image dimensions;
- `encoding`: source ROS image encoding;
- `step`: source row stride in bytes;
- `image_path`: JPEG filename relative to the camera directory;
- `jpeg_bytes`: encoded JPEG size.

### Queue behavior

Each camera has an independent bounded queue and JPEG writer thread.

- Queue overflow increments `dropped_queue_full`.
- Encoding errors increment `encode_failures`.
- Either condition causes current recorder validation to fail.
- The recorder attempts to save every received frame.

## 9. Camera calibration

### Files

```text
calibration/front_wide.json
calibration/front_tele.json
calibration/cross_left.json
calibration/cross_right.json
```

Each file is the parsed JSON payload from the corresponding transient-local calibration Topic.

Observed structure:

```text
logical_id
available_camera
  logical_id
  intrinsics
    logical_id
    resolution_h
    resolution_w
    shutter_type
    ftheta_param
  rig_to_camera
    quat
      x, y, z, w
    vec
      x, y, z
```

The actual intrinsics are F-theta model parameters, including principal point and polynomial coefficients. Consumers must not assume a pinhole camera model.

The calibration cache belongs to the recorder, so calibration received before a Clip begins may be written into the new Clip at session start.

## 10. Dynamic Topic JSONL envelope

Except for executed-path points and camera timestamps, recorded dynamic ROS Topics use this envelope:

```json
{
  "topic": "/alpasim/example",
  "message": {
    "...": "ROS message fields"
  }
}
```

The `message` value comes from `message_to_ordereddict()` and preserves the ROS message field hierarchy.

## 11. Ego data

### 11.1 Ego state

```text
ego/ego_state.jsonl
```

Source Topic:

```text
/alpasim/ego_state
```

Envelope: dynamic Topic JSONL.

Observed message fields:

```text
stamp
pose_frame_id
child_frame_id
dynamics_frame_id
position
orientation
linear_velocity
angular_velocity
linear_acceleration
angular_acceleration
speed
```

### 11.2 Future ground-truth trajectory stream

```text
ego/ground_truth_future.jsonl
```

Source Topic:

```text
/alpasim/ground_truth/ego/future_trajectory
```

Envelope: dynamic Topic JSONL.

Observed message fields:

```text
reference_stamp
pose_frame_id
dynamics_frame_id
producer
source
requested_duration
actual_duration
is_model_generated
force_gt_active
points
```

This is future supervision data. It may be used for Step 3 eligibility checks and Step 4 supervision labels. It must not be used to generate Step 6 Navigation model input.

### 11.3 Planner output

```text
ego/planner_output.jsonl
```

Source Topic:

```text
/alpasim/planning/ego/trajectory
```

Envelope: dynamic Topic JSONL.

The message shape is the EgoTrajectory structure used by the future trajectory stream.

### 11.4 Executed path points

```text
ego/executed_path_points.jsonl
```

Source Topic:

```text
/alpasim/ego/executed_path
```

This file does not use the Topic envelope. Each row is one deduplicated trajectory point. Observed fields:

```text
stamp
time_from_reference
pose
linear_velocity
linear_acceleration
speed
yaw
yaw_rate
yaw_acceleration
```

Only points with a timestamp greater than the last appended executed-path timestamp are written.

### 11.5 Final executed-path message

```text
ego/executed_path_final.json
```

At finalization, the recorder writes the last complete executed-path EgoTrajectory message. Observed top-level fields:

```text
reference_stamp
pose_frame_id
dynamics_frame_id
producer
source
requested_duration
actual_duration
is_model_generated
force_gt_active
points
```

### 11.6 Complete recording ground truth

```text
ego/complete_recording_ground_truth.json
```

Source service:

```text
/alpasim/navigation/get_ground_truth_ego_trajectory
```

Stored structure:

```json
{
  "revision": 1,
  "recording_start_stamp": {
    "sec": 0,
    "nanosec": 0
  },
  "recording_end_stamp": {
    "sec": 0,
    "nanosec": 0
  },
  "trajectory": {}
}
```

The request uses the first observed simulation time as its reference and requests the complete available trajectory using zero sampling interval and zero maximum-point limit.

## 12. Actor data

### 12.1 Current actor states

```text
actors/current.jsonl
```

Source Topic:

```text
/alpasim/actors/current
```

Envelope: dynamic Topic JSONL.

Observed message fields:

```text
stamp
pose_frame_id
dynamics_frame_id
actors
```

Each actor contains fields including:

```text
track_id
label_class
is_static
pose
  position
  orientation
dimensions
linear_velocity
linear_acceleration
yaw
yaw_rate
yaw_acceleration
speed
```

Actor truth may include actors that are not visible in the selected cameras. Future Scene-Fact generation must apply observability filtering before writing facts or reasoning supervision.

### 12.2 Actor history

```text
actors/history.jsonl
```

Source Topic:

```text
/alpasim/actors/history
```

Observed message fields:

```text
reference_stamp
pose_frame_id
dynamics_frame_id
producer
source
requested_duration
actual_duration
sampling_interval
is_model_generated
trajectories
```

### 12.3 Actor future trajectories

```text
actors/future.jsonl
```

Source Topic:

```text
/alpasim/ground_truth/actors/future
```

The message shape matches the actor trajectory-array structure used for history.

This file contains future simulator truth and must not be used as model input. It may support supervision generation only when observability and task-specific leakage constraints are satisfied.

## 13. Route data

### 13.1 Map Route

```text
route/map_route.jsonl
```

Source Topic:

```text
/alpasim/route/map
```

Envelope: dynamic Topic JSONL.

### 13.2 Navigation model-input Route

```text
route/navigation_route_local.jsonl
```

Source Topic:

```text
/alpasim/route/model_input
```

Envelope: dynamic Topic JSONL.

Observed message fields:

```text
reference_stamp
frame_id
source_frame_id
generator_type
producer
sequence
lookahead_distance
expected_point_count
points
```

Each Route point contains:

```text
valid
position
  x, y, z
longitudinal_distance
```

In the observed model-input Route, both `frame_id` and `source_frame_id` are `base_link`.

### Step 6 time policy

Step 6 must query the latest Route available at or before the Anchor. It must not select a Route published after the Anchor.

### Step 6 leakage policy

The Route may describe planned road geometry ahead, but final model input is compressed to coarse semantics such as:

```text
Continue along the road.
Continue straight through the upcoming intersection.
Turn left at the upcoming intersection.
Turn right at the upcoming intersection.
```

Do not expose exact future coordinates, exact turn time, future speed, or future controls.

## 14. VectorMap

### File

```text
map/vector_map.json
```

Source service:

```text
/alpasim/map/get_vector_map
```

Observed top-level fields:

```text
frame_id
map_id
scene_id
revision
minimum
maximum
lanes
road_edges
traffic_signs
wait_lines
```

The service response must report success and must not return `not_modified` because the recorder does not maintain a local map cache across the request.

`vector_map.json` is a required recorder validation artifact.

## 15. File requirement matrix

### Required by the current recorder finalization gate

```text
metadata.json
validation.json
all four calibration JSON files
at least one saved image for every camera
all four camera timestamps.jsonl files
at least one /clock message
ego/ego_state.jsonl with data
ego/executed_path_points.jsonl with data
actors/current.jsonl with data
route/navigation_route_local.jsonl with data
map/vector_map.json
no camera queue drops
no camera encoding failures
```

### Written when corresponding data or service succeeds

```text
ego/ground_truth_future.jsonl
ego/planner_output.jsonl
actors/history.jsonl
actors/future.jsonl
route/map_route.jsonl
ego/executed_path_final.json
ego/complete_recording_ground_truth.json
```

Some of these files are expected by the current Step 1 Manifest or later annotation tools even though they are not all part of the recorder's rename gate. Downstream tools must perform their own explicit checks.

## 16. Raw-data use by annotation steps

### Step 1

Scans Clip structure and validates required raw artifacts, readability, frame/timestamp consistency, time ranges, Route, GT, actors, and VectorMap.

### Step 2

Provides shared readers, temporal indexing, interpolation, coordinate transforms, and caching.

### Step 3

Uses camera history, Ego history, Route availability, and future GT availability to select valid Candidate Anchors.

### Step 4

May use future Ego trajectory, executed path, motion, and lane topology because Meta-action is a supervision label.

### Step 5

Uses Candidate Anchors, Meta-actions, and Step 4 features to select valuable Keyframes.

### Step 6

Must use only Anchor-time-or-earlier Navigation Route, Anchor-time Ego state, and static VectorMap. It must not read future Ego motion or Meta-action as Navigation evidence.

## 17. Reader compatibility requirements

A reader compatible with format `0.2-batch` must:

- validate `metadata.json` and its format version;
- reject or separately classify `.tmp` Clips;
- use `timestamps.jsonl` to locate camera images;
- not assume equal frame counts across cameras;
- support exact-time preference with bounded tolerance fallback;
- validate monotonic timestamps;
- distinguish dynamic Topic envelopes from direct executed-path point rows;
- respect each message's frame fields;
- avoid reparsing large JSONL files repeatedly where caching is available;
- treat optional/missing data according to the consuming task rather than silently fabricating values.

## 18. Format evolution policy

Do not silently change the existing raw Clip schema.

When a recorder change affects directory names, file names, row envelopes, field meaning, time semantics, frame semantics, or validation requirements:

1. increment `dataset_format_version`;
2. update this document;
3. update `DrivingClipReader` compatibility logic;
4. update Step 1 Manifest checks;
5. add or update synthetic and real-data tests;
6. state whether old Clips remain supported;
7. avoid mixing incompatible raw formats in one annotation run unless compatibility is explicitly implemented.

Changes to the output root alone do not change the Clip format.

## 19. Verified real-Clip example

The format was checked against `test_clip_001`.

Observed characteristics:

```text
metadata status: complete
format version: 0.2-batch
simulation duration: 19.5 s
front_wide: 122 JPEGs and 122 timestamp rows
front_tele: 120 JPEGs and 120 timestamp rows
cross_left: 127 JPEGs and 127 timestamp rows
cross_right: 114 JPEGs and 114 timestamp rows
complete recording GT: present
VectorMap: present
validation valid: true
```

This example demonstrates that camera streams can have different counts while remaining valid.

## 20. Related code and documentation

Recorder:

```text
src/alpasim_dataset_tools/alpasim_dataset_tools/data_batch_recorder.py
```

Legacy/single-Clip recorder:

```text
src/alpasim_dataset_tools/alpasim_dataset_tools/dataset_recorder.py
```

The verified production batch data in this document uses `data_batch_recorder.py` and format `0.2-batch`.

Annotation reader and validation:

```text
scripts/dataset_tools/build_clip_manifest.py
scripts/dataset_tools/clip_reader.py
scripts/dataset_tools/temporal_index.py
scripts/dataset_tools/coordinate_utils.py
```

Project handoff:

```text
docs/AI_DATASET_DEVELOPMENT_HANDOFF.md
```
