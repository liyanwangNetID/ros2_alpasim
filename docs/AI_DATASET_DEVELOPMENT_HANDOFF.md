# AlpaSim VLM Dataset Development Handoff

**Document status:** Updated to the frozen Step 7 codebase on 2026-09-24  
**Audience:** A new AI assistant or developer continuing this repository.  
**Purpose:** This is the primary, self-contained handoff. A new conversation should be able to continue directly with Step 8 after reading this file, without requiring additional project history from the user.

## 1. Project goal

Build a supervised dataset for VLM backbone training from recorded AlpaSim driving Clips.

### Model input

- Four cameras: `front_wide`, `front_tele`, `cross_left`, `cross_right`
- Two frames per camera: approximately `t0 - 0.5 s` and `t0`
- Ego history
- Coarse Navigation text

### Supervision target

- Structured Scene Facts
- Driving decision
- Structured chain of causality
- Short causal reasoning

Step 7 produces structured Scene Facts only. Step 8 produces structured chain of causality. Natural-language reasoning begins in Step 9.

## 2. Repositories, paths, and artifact policy

There are two repositories and one external data root:

- `ALPASIM_ROOT`: external AlpaSim simulator repository
- `ALPASIM_ROS2_WS`: this ROS 2 workspace
- `ALPASIM_DATA_ROOT`: raw `test_clip_NNN` directories and generated dataset artifacts

Python tools use `scripts/dataset_tools/project_paths.py`. Shell tools use `scripts/load_local_paths.sh`. Environment variables and supported CLI path arguments override local defaults.

On the current development machine:

```text
ALPASIM_ROS2_WS=/home/lab/alpasim_ros2_ws
ALPASIM_DATA_ROOT=/home/lab/data_from_alpasim
```

Formal artifacts belong under:

```text
annotations/
manifests/
reports/
schemas/
backups/
```

Do not store formal generated artifacts in the code directory. The data artifact root is configured per machine through shell configuration.

## 3. Development rules for the AI assistant

- The user manages Git. Do not inspect, stage, commit, or modify Git state unless explicitly requested.
- Work by Step and close each Step before moving on.
- Use larger closed-loop batches: implementation, full tests, artifact comparison, and cleanup. Split into focused diagnostics only after a failure.
- Prefer the complete test suite directly unless a high-risk change requires a focused diagnostic.
- After a verified substep, continue to the next development action. Stop only when a user decision is genuinely required.
- When multiple versions exist, inspect the actual imports, execution chain, version constants, and product source before selecting the active implementation.
- Preserve deterministic outputs and compare SHA-256 after production changes.
- Do not mix temporary diagnostics, Shadow outputs, review products, and production artifacts.
- Non-ROS dataset tools belong under `scripts/dataset_tools/`.
- Do not ask the user to create scripts with `cat`, heredoc, or manual copy-and-paste. Create downloadable files directly.
- When multiple downloadable files are supplied, also provide a directly executable batch move or installation command.
- Before sending Shell or Python commands, verify that no characters are corrupted or replaced by star characters.
- Do not require `tar.gz` uploads. If source must be uploaded together, use an uploadable plain-text bundle.

## 4. Data leakage boundaries

### Step 4

Step 4 generates supervision labels and may use future Ego trajectory within its defined horizon.

### Step 6

Step 6 generates model input and must not use future execution, future speed, future controls, or Meta-action labels. Step 6 uses only the Route available at or before the Anchor, Anchor-time Ego state, and static VectorMap.

### Step 7

Step 7 describes the scene at the Anchor. Step 7 may use current and past Actor snapshots at or before the Anchor, current Ego state, camera calibration and images, and static VectorMap.

Step 7 must not use:

- Actor future
- Ego future
- planner output
- complete-recording future trajectory
- future executed behavior
- Meta-action labels as current-scene evidence

### Step 8

Step 8 may combine frozen current-scene evidence, coarse Navigation, and supervision labels to produce a structured chain of causality. Step 8 must keep model inputs and supervision sources explicit so future supervision does not leak into the model-input side of the sample.

## 5. Dataset development route

```text
Step 0   Freeze schemas and versions
Step 1   Clip Manifest
Step 2   Unified reading API
Step 3   Candidate Anchors
Step 4   Meta-actions
Step 5   Keyframes
Step 6   Navigation
Step 7   Scene Facts
Step 8   Structured chain of causality
Step 9   Reasoning
Step 10  Sample Manifest
Step 11  Dataset split
Step 12  Audit and statistics
```

Steps 1 through 7 are implemented. Step 7 is frozen. Steps 8 through 12 have not started.

## 6. Current code organization

Production and tests are organized by Step:

```text
scripts/dataset_tools/step1/
scripts/dataset_tools/step2/
scripts/dataset_tools/step3/
scripts/dataset_tools/step4/
scripts/dataset_tools/step5/
scripts/dataset_tools/step6/
scripts/dataset_tools/step7/
scripts/dataset_tools/tests/step1/
...
scripts/dataset_tools/tests/step7/
```

The frozen Step 7 package contains:

```text
actor_roles.py
build_actor_roles.py
build_history.py
build_observability.py
build_occlusion.py
build_projection_evidence.py
build_road_context.py
build_scene_facts.py
build_scene_features.py
build_step7.py
geometry.py
history.py
observability.py
occlusion.py
projection.py
raster.py
review_scene_fact.py
road_context.py
scene_facts.py
```

A reusable Step 7 diagnostic remains outside the package:

```text
scripts/render_step7_all_actor_debug.py
```

## 7. Steps 1 through 4

Steps 1 through 4 are implemented. Their formal behavior and frozen outputs were not changed during the Step 5 through Step 7 reorganization and freeze work.

### Step 1: Clip Manifest

Step 1 scans recorded Clip structure and validates required raw artifacts, readability, timestamp consistency, temporal ranges, Route, Actors, ground-truth availability, and VectorMap availability.

### Step 2: Unified reading API

Step 2 provides unified Clip reading, temporal indexes, bounded lookup, interpolation, coordinate conversion, calibration loading, and VectorMap caching. Shared components include:

```text
scripts/dataset_tools/step2/clip_reader.py
scripts/dataset_tools/step2/temporal_index.py
scripts/dataset_tools/step2/coordinate_utils.py
scripts/dataset_tools/step2/vector_map_reader.py
```

### Step 3: Candidate Anchors

Step 3 uses camera and Ego history, Route availability, and future ground-truth availability to select eligible Candidate Anchors.

### Step 4: Meta-actions

Step 4 generates supervision labels using future Ego trajectory, executed motion, and lane topology within the defined label horizon.

Frozen Step 4 result:

```text
Candidate Anchors: 10231
Meta-action format: 0.2-draft
Generator: 0.2.1
Rule version: meta_action_rules_v0.2.1
Formal output: annotations/v0.1-draft/meta_actions_v0.2.jsonl
SHA-256: a07aacf417829e11d2fe437f01318d509d2d5a007a196440f3d9be95110f5973
```

A conservative direction-consistency guard changes contradictory branch-relative turn labels to `unknown` instead of emitting an opposite-direction turn.

## 8. Step 5: Keyframes

### Status

Implemented and reorganized. Selection quotas scale with Candidate Anchor count rather than relying on a fixed dataset size.

### Production entry point

```bash
python3 -m step5.build_keyframes_v01 --force
```

### Production stages

```text
step5.detect_keyframe_events_v01
step5.deduplicate_keyframe_events_v01
step5.select_keyframes_v01
```

### Formal outputs

```text
annotations/v0.1-draft/keyframes.jsonl
reports/keyframe_selection_summary_v0.1.json
manifests/keyframe_contract_v0.1.json
```

### Frozen result

```text
Candidate Anchors: 10231
Event Anchors retained: 2571
Selected Keyframes: 3500
Keyframe SHA-256: bb3cc755d537c0b8fa0c68aff457106ee00d583bff448bf737c2d286e0ccabf7
```

Selection sources:

```text
normal_driving_baseline: 500
event_candidate: 2571
balanced_stable_longitudinal: 300
balanced_stable_lateral: 129
```

The Keyframe contract records format, selector and rule versions, upstream Meta-action contract linkage, Candidate count, Event count, Keyframe count, Keyframe SHA-256, and the proportional quota policy.

## 9. Step 6: Navigation

### Status

Complete and frozen.

### Production entry point

```bash
python3 -m step6.build_navigation_v01 --force
```

### Production stages

```text
step6.profile_navigation_branch_context_v01
step6.profile_road_level_navigation_features_v01
step6.profile_navigation_route_features_v01
step6.generate_navigation_v01
```

The old Candidate-to-Final double stage was removed because there was no independent manual-review input. The formal generator writes `navigation.jsonl` directly while preserving prior final bytes.

### Current production modules

```text
step6/__init__.py
step6/build_navigation_v01.py
step6/generate_navigation_v01.py
step6/navigation_route_features_v01.py
step6/profile_navigation_branch_context_v01.py
step6/profile_navigation_route_features_v01.py
step6/profile_road_level_navigation_features_v01.py
```

`navigation_route_features_v01.py` remains independent because it is shared by three production modules. Old one-use Map Context and Road-level helpers were merged into their active profilers.

### Contract validation

`step6.generate_navigation_v01` validates `manifests/keyframe_contract_v0.1.json`, including Keyframe SHA-256, record count, unique Anchor IDs, and exact feature Anchor coverage. The Summary records the contract path, Keyframe SHA-256, Keyframe count, and coverage validity.

### Frozen result

```text
Records: 3500
straight: 2921
unknown: 419
right: 103
left: 57
usable: 3081
unknown quality: 419
```

### Frozen hashes

```text
navigation_branch_context_v0.1.jsonl
  afcaf3ce2222c383fd457432ed1eb6ecc800c7a3d1c318f0ebd028ff93541aad
road_level_navigation_features_v0.1.jsonl
  f4d1e5d6fab6c047ddb9c20acd7e669842aab5513b87cf35a718ce80d9853629
navigation_route_features_v0.1.jsonl
  140c2613ec44e59c865d45dd6b99bada5f955ec0909f967f090c319e167c0475
navigation.jsonl
  d025699fcfff677e7929c9df13eb72023d8acd6b815d0fa80c604044c6b7bf90
```

## 10. Step 7: Scene Facts

### Status

```text
PASS / FROZEN
```

Step 7 produces exactly one validated Scene-Fact row for each selected Keyframe. The old single-role fields were removed. The formal interface uses bounded Actor lists.

### Unified production entry point

```bash
python3 -u -m step7.build_step7
```

Selective rebuild from Actor Role Selection:

```bash
python3 -u -m step7.build_step7 \
  --from-stage actor_roles
```

### Production stages

```text
projection_evidence
occlusion
observability_policy
history
road_context
actor_roles
scene_features
scene_facts
```

### Implemented geometry and observability stack

The frozen implementation contains:

- current Actor geometry
- F-theta camera calibration and projection
- Actor oriented 3D boxes
- adaptive projected edge sampling
- angular-FOV and near-plane clipping
- projected hull and surface geometry
- box surface triangulation
- front-facing and near-plane-clipped triangles
- projected triangle raster cells
- barycentric and perspective depth interpolation
- Actor surface depth rasters
- per-camera Actor depth rasters
- multi-Actor Z-buffer competition
- per-camera and multi-camera occlusion evidence
- reviewed Actor observability policy
- projection context for geometric candidates without sampled surfaces
- short-history relative motion
- VectorMap Road Context
- deterministic Actor-list selection
- final feature and Scene-Fact export
- Draft 2020-12 JSON Schema validation

Simulator Actor truth is not automatically visual supervision. Final selected Actors must pass the frozen observability policy. Static-scene occlusion is not evaluated, so final records state:

```json
"static_occlusion_evaluated": false
```

### Actor eligibility and lists

Eligible classes include:

```text
automobile
bus
heavy_truck
other_vehicle
trailer
train_or_tram_car
rider
person
```

Non-independent labels, including `protruding_object`, are excluded.

Final lists:

```text
lead_actors: maximum 4
left_nearby_actors: maximum 6
right_nearby_actors: maximum 6
```

One Actor may occur in only one list per Anchor. Each list has deterministic `role_rank` values beginning at 1.

### Selection ranges

```text
Lead forward horizon:
  max(20 m, reference Ego speed * 10 s)
  no fixed maximum

Side forward horizon:
  clamp(reference Ego speed * 5 s, 20 m, 120 m)

Rear horizon:
  clamp(reference Ego speed * 2 s, 15 m, 50 m)

Lead lateral corridor:
  absolute lateral offset <= 2.5 m

Side lateral range:
  0.5 m < absolute lateral offset <= 12 m

Side person forward range:
  maximum 30 m
```

Lead-corridor `person` and `rider` Actors retain the full Lead horizon. Lead rank follows longitudinal path order. Lane relation is supporting ranking evidence rather than a hard exclusion, so a valid `unrelated` Actor inside the strict Lead corridor may still be selected.

### Reference Ego speed

Actor selection uses the maximum of:

- recorded Ego speed
- executed Ego-state speed when available
- recent pose-derived speed

Pose-derived speed uses the longest available past-only history window, approximately 3.0 seconds down to two samples near Clip boundaries.

### Relative semantics

Relative positions:

```text
front
front_left
front_right
left
right
rear_left
rear
rear_right
overlapping
unknown
```

Relative distance:

```text
near: distance <= 10 m
medium: 10 m < distance <= 30 m
far: distance > 30 m
unknown
```

Distance trend:

```text
approaching
receding
stable_distance
uncertain
```

Relative speed:

```text
slower_than_ego
similar_to_ego
faster_than_ego
stationary
uncertain
```

Current thresholds:

```text
stable distance-rate magnitude <= 0.5 m/s
stationary Actor speed <= 0.5 m/s
similar speed difference <= 1.0 m/s
```

### Road Context

```text
lane_following
intersection_approach
intersection
unknown
```

Final fields include `intersection_proximity`, `stop_line_proximity`, and `yield_line_proximity`. These are conservative map-based structural facts. Wait-line type is not fully propagated. A visible STOP sign is not automatically a camera-confirmed Stop-line fact.

### Final record structure

Top-level fields:

```text
scene_fact_format_version
generator_version
rule_version
anchor_id
clip_id
anchor_ns
road_context
actor_context
quality
```

`actor_context` contains:

```text
reference_ego_speed_mps
forward_horizon_m
side_forward_horizon_m
rear_horizon_m
lists
lead_actors
left_nearby_actors
right_nearby_actors
```

Each selected Actor contains rank, Track ID, class, relative geometry, distance and motion categories, Actor and Ego speeds, observability, visible cameras, quality, and reasons.

### Formal outputs

```text
annotations/v0.1-draft/intermediate/actor_role_selection_v0.1.jsonl
annotations/v0.1-draft/intermediate/scene_fact_features_v0.1.jsonl
annotations/v0.1-draft/scene_facts.jsonl
annotations/v0.1-draft/step7h_actor_role_selection_summary_v01.json
annotations/v0.1-draft/step7k_scene_fact_features_summary_v01.json
annotations/v0.1-draft/step7l_scene_facts_summary_v01.json
schemas/scene_fact_schema_v0.1-draft.json
```

The earlier Step 7E products remain historical and implementation evidence, but Step 7 continuation must use the unified current build rather than old standalone exporters.

### Final verified result

```text
Scene-Fact rows: 3500
Schema validation errors: 0
Actor role conflicts: 0
Road contexts:
  intersection: 257
  intersection_approach: 1016
  lane_following: 2180
  unknown: 47
Quality statuses:
  usable: 2870
  unknown: 630
Selected Actors:
  lead_actors: 3284
  left_nearby_actors: 4722
  right_nearby_actors: 5181
Truncated Anchors:
  lead_actors: 38
  left_nearby_actors: 75
  right_nearby_actors: 127
```

Final Scene-Fact SHA-256:

```text
735203f9ddf3b9f49e892edbb185936caa9db1cd46cbfcdd1b9e0f685958e2b5
```

The final hash remained unchanged after Step 7 code consolidation, compatibility-layer removal, and deterministic rebuild.

### Human review

```bash
python3 -u -m step7.review_scene_fact \
  --force
```

Overlay colors:

```text
Lead: red
Left: yellow
Right: green
```

Validated review-only offsets for `test_clip_894`:

```text
front_wide: -16 source-image pixels
front_tele: -35 source-image pixels
```

Unconfigured Clips use zero offset. These corrections affect review visualization only, not formal evidence or Scene-Fact bytes.

### Known limitations

- static-scene occlusion is not evaluated
- no rendered depth or pixel-accurate instance masks
- some camera-visible objects lack a usable Actor identity or projection at the exact Anchor
- Step 7 does not fabricate missing Actor state
- STOP-sign presence is not a dedicated visual fact
- wait-line type is not fully propagated
- opposing-traffic direction is not a dedicated final Actor field
- review alignment offsets are Clip-specific visual corrections
- final Scene Facts are structured labels, not causality or reasoning text

## 11. Test baseline and deterministic validation

Run the complete suite with:

```bash
cd /home/lab/alpasim_ros2_ws/scripts/dataset_tools

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q
```

The previous fixed count of `1038 passed, 7 subtests passed` belongs to the earlier active-development snapshot. Obsolete compatibility tests were later removed during Step 7 freeze cleanup. The current acceptance condition is that the complete current suite passes.

Frozen Step 7 production acceptance:

```text
3500 final rows
0 schema validation errors
0 role conflicts
final SHA-256 equals 735203f9ddf3b9f49e892edbb185936caa9db1cd46cbfcdd1b9e0f685958e2b5
```

## 12. Immediate next actions: Step 8

Do not repeat Step 7 migration, policy selection, or freeze work. Do not reopen Step 7 unless Step 8 exposes a concrete dependency defect.

The next closed-loop batch should:

1. Inventory Step 8 inputs from the frozen Keyframes, Meta-actions, Navigation, and Scene Facts.
2. Freeze Step 8 schema, vocabulary, generator version, and rule version.
3. Define the structured chain-of-causality contract.
4. Keep model-input evidence and supervision-only evidence explicitly separated.
5. Define conservative unknown states and explicit evidence references.
6. Generate exactly one Step 8 row per Keyframe.
7. Validate exact Anchor closure against `keyframe_contract_v0.1.json`.
8. Add a unified production entry point, Summary, JSON Schema validation, deterministic hashes, and human-review tooling.
9. Run the complete test suite, rebuild, compare artifacts, and clean temporary development files.
10. Preserve frozen Step 7 bytes unless an explicitly approved upstream contract change is required.

## 13. Quick start for a new AI conversation

This document is the only mandatory handoff file. After reading this file, a new assistant should be able to begin Step 8 without asking the user to repeat project history.

Optional specialist references:

- `CLIP_DATA_FORMAT.md`: authoritative raw Clip contract
- `SCENE_FACT_DESIGN.md`: detailed frozen Step 7 technical contract

Initial read-only checks:

```bash
cd /home/lab/alpasim_ros2_ws/scripts/dataset_tools

find step7 -maxdepth 1 -type f -print | sort
find tests/step7 -maxdepth 1 -type f -print | sort

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q

sha256sum \
  "$ALPASIM_DATA_ROOT/annotations/v0.1-draft/scene_facts.jsonl"
```

Expected final Step 7 SHA-256:

```text
735203f9ddf3b9f49e892edbb185936caa9db1cd46cbfcdd1b9e0f685958e2b5
```

After these checks, continue directly with Step 8. Do not rediscover Steps 1 through 7 unless a new failure demonstrates a specific dependency problem.
