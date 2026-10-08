# AlpaSim ROS 2 Bridge and Dataset Collection Platform

## 1. Project purpose and current status

This workspace turns AlpaSim into a ROS 2 based closed-loop simulation and data collection platform. It provides:

- Strongly typed ROS 2 interfaces for important AlpaSim observations.
- RViz2 visualization of the ego vehicle, cameras, actors, vector map, routes, ground-truth trajectories, planned trajectories, and executed path.
- A unified external-planner interface that accepts future ego trajectories from ROS 2.
- A ground-truth replay planner for oracle driving and dataset collection.
- Sequential multi-scene execution with per-scene USDZ download and cleanup.
- A clip activity signal for starting and stopping external recorders.
- A globally deduplicated public-scene manifest for running each logical scene exactly once.

The main system path is:

```text
AlpaSim Runtime
  -> UDP/TCP bridge channels
  -> ROS 2 topics, services, TF, and RViz2
  -> external ROS 2 planner
  -> /alpasim/planning/ego/trajectory
  -> External Trajectory Driver
  -> AlpaSim Controller and Physics
  -> closed-loop simulation
```

The current data-generation path is:

```text
all_unique_scenes.csv
  -> download one USDZ
  -> run one closed-loop clip
  -> publish ROS 2 observations
  -> Ground Truth Replay Planner
  -> data_batch_recorder
  -> delete the USDZ, Wizard logs, and Hugging Face cache
  -> continue to the next unique scene
```

## 2. Important paths

```text
AlpaSim repository:
/home/lab/alpasim

ROS 2 workspace:
/home/lab/alpasim_ros2_ws

Bridge package:
/home/lab/alpasim_ros2_ws/src/alpasim_bridge

Planning package:
/home/lab/alpasim_ros2_ws/src/alpasim_planning

Messages and services:
/home/lab/alpasim_ros2_ws/src/alpasim_msgs

Dataset recorder package:
/home/lab/alpasim_ros2_ws/src/alpasim_dataset_tools

Project scripts:
/home/lab/alpasim_ros2_ws/scripts

AlpaSim scene catalogs:
/home/lab/alpasim/data/scenes
```

## 3. Architecture and design principles

AlpaSim Runtime exports data through non-blocking background transports. ROS 2 bridge nodes publish dynamic observations, cache static data, expose services, and create RViz markers.

```text
AlpaSim Runtime
├── ego state, clock, and cameras
├── actor states and trajectories
├── VectorMap
├── full recorded ground-truth ego trajectory
└── dynamic route updates
        |
        v
ROS 2 Bridge
├── dynamic publishers
├── static-data servers
├── query clients
├── TF publishers
└── visualization publishers
        |
        v
ROS topics, services, TF, RViz2, planners, and recorders
```

Key principles:

- Data transport and visualization are separate. Algorithm nodes consume typed messages rather than RViz markers.
- High-frequency dynamic exporters use small queues and prefer the newest state.
- Static critical data, including VectorMap and complete recorded ego trajectory, are retried and cached.
- Ground truth, model output, controller references, and executed motion remain distinguishable.
- Marker publishers use stable namespaces and IDs. They must not periodically publish global `Marker.DELETEALL`.
- Long-running ROS nodes must reset per-rollout state when simulation time or Runtime connections restart.
- Current external planning uses a unified 2 Hz waypoint contract.

## 4. Coordinate frames and timing

Primary frames:

```text
map
└── base_link
```

- `map` is the AlpaSim true local simulation frame.
- `base_link` is the current ego rig frame.
- `map -> base_link` is generated from ego state.
- Map, actors, complete GT trajectory, rolling GT future, and executed path are normally expressed in `map`.
- Driver model-input route is expressed in `base_link`.
- External planner output is expected in `base_link` for the current External Trajectory Driver workflow.

Messages preserve explicit frame metadata such as:

```text
frame_id
pose_frame_id
child_frame_id
dynamics_frame_id
source_frame_id
```

Do not infer coordinate semantics only from topic names.

### Unified external-planner timing contract

All external planners should produce a 2 Hz waypoint trajectory:

```text
waypoint interval: 0.5 s
default horizon: 3.0 s
default points: 6
future times: 0.5, 1.0, 1.5, 2.0, 2.5, 3.0 s
```

External Driver configuration:

```yaml
inference:
  output_frequency_hz: 2
```

The planned speed is implicit in waypoint displacement over time. The MPC controller uses the timestamped spatial trajectory and the current vehicle state to compute steering and longitudinal actuation.

## 5. Runtime data semantics

### 5.1 Actors

`TrafficObject.trajectory` is a simulation trajectory and may include historical states and already-computed short-term future states. It is not a probabilistic predictor output.

Rules:

- Current actor state must be sampled at the current simulation timestamp.
- `/alpasim/ground_truth/actors/future` is privileged ground truth.
- `/alpasim/prediction/actors` remains a placeholder unless a real predictor is integrated.
- Actor ground truth must not be used as an online model input unless the experiment explicitly allows privileged information.

### 5.2 VectorMap

Runtime map source:

```text
state.unbound.vector_map
```

Type:

```text
trajdata.maps.VectorMap
```

Currently exported map information includes:

```text
RoadLane
├── centerline: x, y, z, heading
├── left and right boundaries
├── predecessor and successor IDs
├── left and right adjacent IDs
├── traffic sign IDs
├── wait line IDs
└── road area IDs

RoadEdge
└── polyline: x, y, z, heading

TrafficSign
├── ID
├── position
└── native sign_type

WaitLine
├── ID
├── polyline
├── STOP, YIELD, or UNKNOWN
└── is_implicit
```

Not yet reliably available:

- Traffic-light state
- Lane speed limit
- Lane type
- Intersection flag
- Crosswalk and walkway geometry
- Road-area polygons

Invalid lane reference `-1` is filtered during Runtime export.

### 5.3 Route

The active route generator is normally `RouteGeneratorMap`:

```text
recorded ego trajectory
  -> candidate lane matching
  -> continuous lane sequence
  -> projection to centerlines
  -> extension through next_lanes
```

Driver model-input route characteristics:

```text
20 waypoints
80 m nominal lookahead
approximately 4.21 m spacing
frame: base_link
z: 0
```

`prepare_for_policy()` produces fixed-length input. Missing tail points may use NaN padding internally. ROS represents validity through `RoutePoint.valid`; invalid floating-point values must not be forwarded to JSON, ROS geometry, or RViz.

### 5.4 Recorded ground-truth ego trajectory

Runtime source:

```text
state.unbound.gt_ego_trajectory
```

Typical clip:

```text
202 points
20.0 s duration
approximately 10 Hz
```

Available values include timestamps, positions, quaternions, velocities, accelerations, yaw, yaw rate, and yaw acceleration. The complete trajectory is cached by a server and queried as a rolling future segment.

### 5.5 LiDAR status

SensorSim protobuf contains LiDAR request and return interfaces, including PANDAR128 and AT128. However, current AlpaSim Runtime integration still lacks the complete LiDAR request, response, export, and ROS `PointCloud2` path. Current platform development therefore uses camera input and does not depend on simulated LiDAR.

## 6. ROS 2 nodes

### `ego_state_publisher`

Publishes ego state, `/clock`, `map -> base_link` TF, camera images, camera calibration, and camera TF.

### `actor_state_publisher`

Publishes current actors, actor history, actor ground-truth future, and the prediction placeholder.

### `actor_marker_publisher`

Publishes actor boxes, labels, speed text, and velocity arrows.

### `map_server`

Receives and caches VectorMap data and provides `/alpasim/map/get_vector_map`.

### `map_marker_publisher`

Queries the map service and publishes cached static `MarkerArray` visualization.

### `ground_truth_trajectory_server`

Caches the full recorded ego trajectory and serves clipped and resampled future windows.

### `ground_truth_future_publisher`

Uses `/clock` and the GT trajectory service to publish a rolling future trajectory.

### `navigation_state_publisher`

Receives dynamic navigation packets and publishes:

```text
/alpasim/route/map
/alpasim/route/model_input
```

Each Runtime connection gets a local `connection_id`. A new connection is treated as a new rollout, and per-rollout sequence tracking is reset. This fixes the previous issue where the second clip restarted at sequence 1 but was incorrectly rejected against the previous clip's final sequence.

### `executed_path_publisher`

Maintains and publishes the physics-corrected ego path. It clears history when simulation time rewinds.

### `navigation_marker_publisher`

Visualizes route, GT future, planned trajectory, and executed path.

### `clip_state_publisher`

Publishes:

```text
/alpasim/simulation/clip_active
std_msgs/msg/Bool
```

At 10 Hz:

```text
true: a clip is active
false: no clip has started, a clip has ended, or the system is waiting
```

Activity is inferred from `/alpasim/ground_truth/ego/future_trajectory` using wall-clock monotonic time. Default inactivity timeout is 2 seconds.

### `ground_truth_replay_planner`

Subscribes to the rolling 10 Hz GT future in `map`, resamples it to six 2 Hz points, transforms the points into the current `base_link`, and publishes to:

```text
/alpasim/planning/ego/trajectory
```

The planner clears its TF buffer when simulation time rewinds, which allows repeated and sequential rollouts without restarting the planner.

## 7. Main ROS interfaces

### Topics

```text
/alpasim/ego_state
/clock
/tf
/tf_static

/alpasim/camera/cross_left/image
/alpasim/camera/front_wide/image
/alpasim/camera/front_tele/image
/alpasim/camera/cross_right/image
/alpasim/camera/rear/image

/alpasim/actors/current
/alpasim/actors/history
/alpasim/ground_truth/actors/future
/alpasim/prediction/actors
/alpasim/actors/markers

/alpasim/route/map
/alpasim/route/model_input
/alpasim/ground_truth/ego/future_trajectory
/alpasim/planning/ego/trajectory
/alpasim/ego/executed_path
/alpasim/navigation/markers

/alpasim/map/markers
/alpasim/simulation/clip_active
```

### Services

```text
/alpasim/map/get_vector_map
  alpasim_msgs/srv/GetVectorMap

/alpasim/navigation/get_ground_truth_ego_trajectory
  alpasim_msgs/srv/GetGroundTruthEgoTrajectory
```

The GT service accepts a reference timestamp, future duration, sampling interval, maximum point count, and known revision. It returns only the available recorded interval and does not extrapolate.

### Custom message families

```text
Ego:
EgoState.msg
EgoTrajectory.msg
TrajectoryPoint.msg

Actors:
ActorState.msg
ActorStateArray.msg
ActorTrajectory.msg
ActorTrajectoryArray.msg

Map:
MapPolyline.msg
MapLane.msg
MapRoadEdge.msg
MapTrafficSign.msg
MapWaitLine.msg
VectorMap.msg

Route:
RoutePoint.msg
Route.msg
```

## 8. RViz2 visualization

Recommended fixed frame:

```text
map
```

Recommended displays:

```text
/alpasim/actors/markers
/alpasim/map/markers
/alpasim/navigation/markers
camera image topics
robot model
TF
```

Colors:

```text
Lane centerlines: cyan or teal
Lane boundaries: white or light gray
Road edges: red
STOP lines: yellow
YIELD lines: orange
UNKNOWN wait lines: gray
Traffic signs: magenta

AlpaSim route: bright pink
GT future: green
Planned path: electric blue
Executed path: white
```

Actor categories:

```text
automobile: blue
heavy truck: orange
trailer: purple
pedestrian: red
bicycle: green
motorcycle: yellow
unknown: gray
```

## 9. Network ports

```text
Ego state: existing dedicated transport
Camera and sensor TCP: 15001
Actor state TCP: 15002
VectorMap TCP: 15003
Complete GT ego TCP: 15004
Dynamic navigation TCP: 15005
External trajectory driver: 6789
```

Check listeners:

```bash
ss -ltnp | grep -E ':1500[1-5]|:6789'
```

The bridge servers used from Docker must listen on `0.0.0.0`, not only `127.0.0.1`.

## 10. Build and environment

```bash
source /home/lab/alpasim_ros2_ws/scripts/load_local_paths.sh
cd /home/lab/alpasim_ros2_ws
source /opt/ros/jazzy/setup.zsh

colcon build \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

source install/setup.zsh
```

Build selected packages:

```bash
colcon build \
  --symlink-install \
  --packages-select \
    alpasim_msgs \
    alpasim_bridge \
    alpasim_planning \
    alpasim_dataset_tools \
    ego_vehicle_description

source install/setup.zsh
```

## 11. Standard startup order

Use separate terminals and keep the long-running ROS components alive across multiple clips.

### Terminal 1: ROS bridge and RViz2

```bash
source /opt/ros/jazzy/setup.zsh
source /home/lab/alpasim_ros2_ws/install/setup.zsh

ros2 launch \
  ego_vehicle_description \
  display.launch.py
```

The launch includes the clip-state publisher and navigation-state publisher.

### Terminal 2: External Trajectory Driver

```bash
cd /home/lab/alpasim

uv run --project src/driver python -m alpasim_driver.main \
  --config-path=/home/lab/alpasim/src/driver/configs \
  --config-name=external_trajectory
```

### Terminal 3: Planner

For oracle replay and dataset collection:

```bash
source /opt/ros/jazzy/setup.zsh
source /home/lab/alpasim_ros2_ws/install/setup.zsh

ros2 run \
  alpasim_planning \
  ground_truth_replay_planner
```

Other planner options:

```bash
ros2 run alpasim_planning synthetic_trajectory_planner
/home/lab/alpasim_ros2_ws/scripts/run_vavam_planner.sh
```

### Terminal 4: Dataset recorder

```bash
source /opt/ros/jazzy/setup.zsh
source /home/lab/alpasim_ros2_ws/install/setup.zsh

ros2 run \
  alpasim_dataset_tools \
  data_batch_recorder
```

The recorder uses `/alpasim/simulation/clip_active` to delimit clips.

### Terminal 5: AlpaSim scene execution

Run one scene or use the streaming batch script described below.

## 12. Run one scene

```bash
cd /home/lab/alpasim

uv run --project src/wizard alpasim_wizard \
  deploy=local \
  driver=manual \
  driver_source=external_static \
  topology=1gpu \
  wizard.log_dir="$PWD/runs/external_control" \
  scenes.scene_ids='["clipgt-01d503d4-449b-46fc-8d78-9085e70d3554"]' \
  wizard.external_services.driver='["172.23.0.1:6789"]' \
  runtime.simulation_config.n_sim_steps=200 \
  runtime.simulation_config.control_timestep_us=100000 \
  runtime.simulation_config.pose_reporting_interval_us=100000 \
  +runtime.simulation_config.realtime_factor=1.0 \
  'runtime.simulation_config.cameras=[{height:480,width:854,logical_id:camera_cross_left_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_wide_120fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_front_tele_30fov,frame_interval_us:100000,shutter_duration_us:30000},{height:480,width:854,logical_id:camera_cross_right_120fov,frame_interval_us:100000,shutter_duration_us:30000}]'
```

Current standard camera set:

```text
camera_cross_left_120fov
camera_front_wide_120fov
camera_front_tele_30fov
camera_cross_right_120fov
```

A rear camera ID exists in some assets:

```text
camera_rear_left_70fov
```

Not every public scene contains it, so the standard batch collection remains four-camera.

## 13. Public scene catalogs and global deduplication

Scene catalog files currently present:

```text
/home/lab/alpasim/data/scenes/
├── sim_scenes_2505.csv
├── sim_suites_2505.csv
├── sim_scenes.csv
├── sim_suites.csv
├── sim_scenes_latest.csv
├── sim_suites_latest.csv
├── all_unique_scenes.csv
└── my_gt_sim_suites.csv
```

Catalog sources:

- `sim_suites_2505.csv`: legacy public 2507 release.
- local `sim_suites.csv`: older local public 2601 catalog.
- `sim_suites_latest.csv`: latest downloaded public 2601, public 2601 video-model subset, and public 2604.

The latest catalog counts observed during development were:

```text
public_2601: 913
public_2601_video_model: 729
public_2604: 1606
```

`public_2601_video_model` is fully contained in `public_2601`, including identical UUIDs.

Global deduplication priority:

```text
public_2604
  > latest public_2601
  > local older public_2601
  > legacy public_2507
```

Generate the global unique manifest:

```bash
/home/lab/alpasim_ros2_ws/scripts/generate_all_unique_scenes.py
```

Current result:

```text
Unique logical scene IDs: 2369
Unique selected UUIDs: 2369
Duplicate scene IDs: 0

Selected artifacts:
public_2604: 1606
latest public_2601: 754
local public_2601 unique scenes: 3
legacy public_2507 unique scenes: 6
```

Manifest:

```text
/home/lab/alpasim/data/scenes/all_unique_scenes.csv
```

Important fields:

```text
index
scene_id
uuid
release
source_name
source_suite_id
scenes_csv
suites_csv
```

`scene_id` identifies a logical driving scene. `uuid` identifies one exact USDZ artifact. The manifest selects one UUID per logical scene.

## 14. Streaming all unique scenes without filling the disk

Script:

```text
/home/lab/alpasim_ros2_ws/scripts/run_gt_dataset_streaming_all_unique.sh
```

The script reads `all_unique_scenes.csv` and processes one scene at a time:

```text
create a scene-specific /tmp directory
  -> configure scene cache, Wizard log directory, and HF_HOME inside it
  -> download the current USDZ
  -> run one rollout
  -> collect data through ROS
  -> delete the USDZ, AlpaSim caches, logs, and Hugging Face cache
  -> continue
```

Run one scene:

```bash
/home/lab/alpasim_ros2_ws/scripts/run_gt_dataset_streaming_all_unique.sh \
  1 1
```

Run a small smoke test:

```bash
/home/lab/alpasim_ros2_ws/scripts/run_gt_dataset_streaming_all_unique.sh \
  1 5
```

Run an inclusive range:

```bash
/home/lab/alpasim_ros2_ws/scripts/run_gt_dataset_streaming_all_unique.sh \
  101 200
```

Previously used continuation example:

```bash
/home/lab/alpasim_ros2_ws/scripts/run_gt_dataset_streaming_all_unique.sh \
  2054 2369
```

Run all scenes only when intended:

```bash
/home/lab/alpasim_ros2_ws/scripts/run_gt_dataset_streaming_all_unique.sh
```

The script forces each service's concurrent rollouts to 1, because the current ROS topics are not namespaced by rollout and the recorder expects sequential clips.

## 15. Dataset recorder and future training dataset

The ROS bridge is used to create data for later VLM fine-tuning. Current collection can include:

- Four synchronized camera streams
- Ego state and TF
- Actor current state, history, and ground-truth future
- VectorMap
- Route in `map`
- Model-input route in `base_link`
- Complete and rolling ground-truth ego future
- Ground-truth replay planner output
- Executed ego path
- Camera calibration
- Clip activity state

Future additions to storage may be needed. Those additions have not yet been finalized and should be handled as a separate development task.

### Dataset development roadmap

```text
Step 0: freeze schema and versions
Step 1: clip manifest
Step 2: unified reader and temporal API
Step 3: candidate anchors
Step 4: meta-actions
Step 5: keyframes
Step 6: navigation labels
Step 7: scene facts
Step 8: structured CoC
Step 9: human-readable reasoning
Step 10: sample manifest
Step 11: train, validation, and test split
Step 12: audit and statistics
```

Important dataset rules:

- Split by clip, never randomly by neighboring anchors.
- Do not leak exact GT futures into online model inputs.
- Navigation labels may use processed lane-level route guidance.
- Scene facts generated from simulator truth require observability filtering.
- Structured CoC is the structured fact source.
- Default training does not require free-form reasoning. Reasoning is primarily for human review, debugging, and optional comparison experiments.

## 16. Ground-truth leakage constraints

Privileged evaluation data:

```text
/alpasim/ground_truth/actors/future
/alpasim/ground_truth/ego/future_trajectory
```

These topics must not be used as normal online inputs for the trained driving model.

Allowed navigation guidance:

```text
/alpasim/route/model_input
```

This is a processed lane-level navigation route and does not directly expose the exact recorded future trajectory, speed, timing, or future control values.

## 17. CATK traffic behavior

`trafficsim=catk` enables responsive background traffic. It was validated to produce response updates, but observed behavior was not consistently natural enough to make CATK the default data-generation dependency.

Use only for explicit experiments:

```bash
trafficsim=catk
```

The default dataset workflow can use recorded non-responsive background traffic, with evaluation accounting for non-responsive rear-end interactions where relevant.

## 18. Validation commands

```bash
ros2 node list
ros2 topic list | grep alpasim
ros2 service list | grep alpasim

ros2 topic echo /alpasim/route/map --once
ros2 topic echo /alpasim/route/model_input --once
ros2 topic echo /alpasim/planning/ego/trajectory --once
ros2 topic echo /alpasim/ground_truth/ego/future_trajectory --once
ros2 topic echo /alpasim/ego/executed_path --once
ros2 topic echo /alpasim/simulation/clip_active

ros2 topic hz /alpasim/actors/current
ros2 topic hz /alpasim/route/map
ros2 topic hz /alpasim/route/model_input
ros2 topic hz /alpasim/planning/ego/trajectory
ros2 topic hz /alpasim/ground_truth/ego/future_trajectory
ros2 topic hz /alpasim/simulation/clip_active
```

## 19. Important fixes already completed

### Ground Truth Replay Planner TF reset

Problem:

```text
same clip rerun
  -> simulation timestamp repeated
  -> old TF data stayed in the buffer
  -> TF_OLD_DATA and past extrapolation
  -> vehicle stopped
```

Fix:

```python
self.tf_buffer.clear()
```

when rollout time rewinds.

### Navigation state sequence reset

Problem:

```text
clip 1 ended at sequence 197
clip 2 opened a new Runtime connection and restarted at sequence 1
long-running ROS node retained last_sequence=197
clip 2 navigation packets were all rejected as stale
```

Fix:

```text
assign a local connection_id to each TCP connection
  -> detect new rollout in the ROS executor thread
  -> reset packet_count, last_sequence, and last_reference_timestamp_us
```

### Sequential rollout restriction

Current topics are shared and not rollout-namespaced. Parallel scenes would mix cameras, actors, routes, TF, and planning messages. Batch scripts therefore force all service concurrency to 1.

## 20. Known issues and cautions

- Some nodes print `rcl_shutdown already called` when a launch is stopped with Ctrl+C. This is shutdown cleanup noise and is separate from normal runtime operation.
- `actor_marker_publisher` may attempt to publish marker cleanup after the ROS context is invalid during shutdown.
- Rear camera availability is scene-dependent.
- LiDAR is not currently integrated end-to-end.
- CATK is optional and not the default dataset traffic source.
- `wizard.log_dir` is also a runtime working directory. To avoid permanent logs, place it in a temporary directory and remove that directory after the run.
- Route topics use dynamic navigation updates. Keep `navigation_state_publisher` in the main launch file so it receives the correct host configuration.

## 21. Common troubleshooting

### Route topics work for the first clip but disappear for the second

Confirm logs show a new `connection_id` and `Published first navigation update` for each clip. The connection-based sequence reset must remain in `navigation_state_publisher.py`.

### Replay planner works once but the same clip does not move on rerun

Confirm the planner detects simulation time rewind and clears its TF buffer.

### Map appears in the first rollout but disappears later

Check whether another marker publisher sends global `Marker.DELETEALL`. Map markers should be cached and republished.

### Route contains invalid points

Inspect `RoutePoint.valid`. Invalid tail points can result from legal fixed-length padding.

### `Address already in use`

```bash
ss -ltnp | grep -E ':1500[1-5]|:6789'
```

Stop old launches or temporary receivers.

### `Polyline.h` raises an error

Some polylines contain only `x, y, z`. Check the final array dimension before accessing heading:

```python
points.shape[-1] >= 4
```

### Batch fills the disk

Use `run_gt_dataset_streaming_all_unique.sh`. It sets `HF_HOME`, AlpaSim scene cache, and Wizard logs inside a scene-specific temporary directory and removes it after each scene.

## 22. Research direction

The primary scientific baseline is the original Alpamayo model. The proposed method is Geo-enhanced Alpamayo. Autoware Universe was considered but rejected as a core baseline because it introduces a different modular stack, different information privileges, map conversion requirements, and perception assumptions.

Suggested experiment structure:

```text
Original Alpamayo
Alpamayo + Route
Alpamayo + local lane geometry
Alpamayo + lane topology
Full Geo-enhanced Alpamayo
```

Keep camera configuration, simulation scenes, controller, traffic configuration, rollout count, and trajectory timing fixed so improvements can be attributed to Geo information.

## 23. Completion state

```text
Ego state, clock, and TF: complete
Multi-camera images and calibration: complete
Actors, history, and GT future: complete
VectorMap service and RViz markers: complete
Route, GT future, and executed path: complete
External trajectory control: complete
Ground Truth Replay Planner: complete
Clip activity state: complete
Sequential multi-scene execution: complete
Per-scene download and cleanup: complete
Global unique-scene manifest: complete
Cross-rollout TF reset: complete
Cross-rollout navigation sequence reset: complete
LiDAR export: not implemented
Future dataset storage additions: pending separate design
```
