# AlpaSim VLM Dataset Development Handoff

> **Audience:** A new AI assistant or developer continuing this repository.
>
> **Purpose:** Read this document before changing dataset code. It records the project goal, repository layout, machine-local configuration, implemented pipeline, current frozen outputs, safety constraints, verified entry points, tests, known limitations, and development workflow.

## 1. Project goal

Build a supervised dataset for VLM backbone training from recorded AlpaSim driving clips.

### Model input

- Four cameras, two frames per camera:
  - `front_wide`
  - `front_tele`
  - `cross_left`
  - `cross_right`
- Frame times:
  - `t0 - 0.5 s`
  - `t0`
- Ego history:
  - past `M` waypoints
- Navigation text, for example:
  - `Turn right at the upcoming intersection.`

### Supervision target

```json
{
  "scene_facts": {
    "road_context": "intersection",
    "traffic_control": "red_light",
    "critical_actor": "vehicle_ahead"
  },
  "decision": {
    "lateral": "keep_lane",
    "longitudinal": "stop"
  },
  "reasoning": "The route continues straight, but the traffic light ahead is red. The ego vehicle should remain in its lane and stop before entering the intersection."
}
```

## 2. Repositories and machine-local paths

There are two separate repositories and one data root:

- `ALPASIM_ROOT`
  - External AlpaSim simulator repository.
  - Contains simulator code and scene catalogs under `data/scenes`.
  - This repository has additional input/output modules used for data collection.
- `ALPASIM_ROS2_WS`
  - This repository.
  - Contains ROS 2 code, data-collection launch scripts, and `scripts/dataset_tools`.
- `ALPASIM_DATA_ROOT`
  - Recorded `test_clip_NNN` directories and generated dataset artifacts.

### Local configuration

Copy the tracked template:

```bash
cp config/local_paths.env.example config/local_paths.env
```

Edit only `config/local_paths.env` on each machine:

```bash
ALPASIM_DATA_ROOT=/path/to/data_from_alpasim
ALPASIM_ROOT=/path/to/alpasim
ALPASIM_ROS2_WS=/path/to/alpasim_ros2_ws
```

`config/local_paths.env` is intentionally ignored by Git.

### Configuration implementation

- Python tools read paths through:
  - `scripts/dataset_tools/project_paths.py`
- Shell tools load paths through:
  - `scripts/load_local_paths.sh`
- Environment variables override values in `config/local_paths.env`.
- CLI path arguments override script defaults where supported.
- `/tmp` and Python `tempfile` are allowed for temporary runtime files and are not machine-specific dataset paths.

### Path audit status

At the time this handoff was prepared:

- Machine-bound absolute paths in code were reduced to zero.
- `config/local_paths.env` was verified as ignored by Git.
- `config/local_paths.env.example` remains tracked as the portable template.

## 3. Development rules for the AI assistant

Follow these rules before changing code or regenerating artifacts:

1. Work in small increments.
2. Do not provide many unverified steps at once.
3. After each small change, run syntax checks and focused tests before continuing.
4. When multiple script versions exist, verify:
   - the actual production entry point;
   - import and subprocess call chains;
   - version constants;
   - Git status;
   - artifact provenance.
5. Do not assume a script with the largest version-looking name is the active one.
6. Do not combine temporary diagnosis scripts, backups, Shadow outputs, and production fixes in one commit.
7. Prefer conservative `unknown` labels over incorrect directional labels.
8. Before changing a frozen rule or distribution:
   - run a Shadow Evaluation;
   - count affected Anchors and independent Clips;
   - manually review representative cases;
   - add regression tests;
   - update frozen distributions only after verification.
9. Preserve deterministic outputs and record SHA-256 values.
10. Never use future execution information when generating Navigation input text.
11. Future video or trajectory may be used for offline human label auditing, but must not enter Step 6 production features.
12. If a heredoc prompt becomes stuck or pasted code contains unexpected characters, stop with `Ctrl+C` and inspect the file before proceeding.

## 4. Data leakage boundary

### Step 4

Step 4 is a supervision-label generator. It is allowed to use future ego trajectory and future motion within its defined supervision horizon.

### Step 6

Step 6 generates model input, so it must not leak:

- future precise positions;
- future speed;
- exact turn time;
- future control values;
- future executed ego trajectory;
- future Meta-action labels.

Step 6 production features use only:

- the navigation Route available at or before the Anchor;
- Anchor-time Ego pose and speed;
- static VectorMap;
- derived coarse navigation semantics.

Human review videos may show future behavior, but are only for auditing.

## 5. Upstream data collection flow

This is upstream of Annotation Step 1.

```text
External AlpaSim scene catalogs
    -> scripts/generate_all_unique_scenes.py
    -> $ALPASIM_ROOT/data/scenes/all_unique_scenes.csv
    -> scripts/run_gt_dataset_streaming_all_unique.sh
    -> recorded test_clip_NNN directories
    -> $ALPASIM_DATA_ROOT
```

Related scripts:

- `scripts/generate_all_unique_scenes.py`
- `scripts/run_vavam_planner.sh`
- `scripts/run_gt_dataset_streaming.sh`
- `scripts/run_gt_dataset_streaming_all_unique.sh`

These scripts now use the shared local path configuration.

## 6. Pipeline overview

```text
Step 0  Freeze schemas and versions
Step 1  Build Clip Manifest
Step 2  Unified time axis and data-reading API
Step 3  Build Candidate Anchors
Step 4  Generate Meta-actions
Step 5  Select Keyframes
Step 6  Generate Navigation text
Step 7  Generate Scene Facts
Step 8  Build structured chain of causality
Step 9  Generate short reasoning text
Step 10 Build sample manifest
Step 11 Split train/validation/test by Clip
Step 12 Audit and report dataset statistics
```

Steps 1 through 6 are implemented and were inspected in detail. Steps 7 through 12 are planned future work.

## 7. Step 0: schemas and versions

Planned responsibility:

- freeze raw dataset schema;
- freeze annotation schema;
- freeze sample schema;
- define Clip structure;
- use nanoseconds consistently;
- document coordinate frames and camera names;
- define JSON/JSONL fields and enumerations;
- record annotation and generator versions.

Planned artifacts:

```text
schemas/
  clip_schema.md
  meta_action_schema.json
  scene_fact_schema.json
  sample_schema.json
```

Before continuing future steps, check what Step 0 artifacts currently exist. Do not assume the planned files were completed merely because later code exists.

## 8. Step 1: Build Clip Manifest

### Status

Implemented. One execution stage.

### Production entry point

```text
scripts/dataset_tools/build_clip_manifest.py
```

### Versions

```text
SCRIPT_VERSION = 0.1.0
MANIFEST_VERSION = 0.1
```

### Responsibilities

- scan finalized `test_clip_NNN` directories;
- collect Clip metadata and validation state;
- count camera frames and time ranges;
- inspect Routes, actor data, GT trajectory, executed path, and VectorMap;
- verify required files;
- check JSON/JSONL readability;
- check camera timestamps and frame counts;
- mark `manifest_usable`;
- never modify raw Clip directories.

### Outputs

```text
$ALPASIM_DATA_ROOT/manifests/clips_v0.1.jsonl
$ALPASIM_DATA_ROOT/reports/clip_manifest_summary_v0.1.json
```

### Rebuild

```bash
cd "$ALPASIM_ROS2_WS/scripts/dataset_tools"
python3 -u build_clip_manifest.py --force
```

### Tests verified

```text
tests/test_build_clip_manifest.py
tests/test_clip_reader.py
tests/test_temporal_index.py
tests/test_coordinate_utils.py
```

The path-default tests verify both shared configuration defaults and CLI overrides.

## 9. Step 2: unified data-reading API

### Status

Implemented. No standalone dataset artifact.

### Core modules

```text
scripts/dataset_tools/clip_reader.py
scripts/dataset_tools/temporal_index.py
scripts/dataset_tools/coordinate_utils.py
```

### Responsibilities

- nearest timestamp lookup;
- exact synchronization preference;
- tolerance-based lookup;
- multi-camera frame sequence selection;
- Ego history;
- future trajectory for supervision pipelines;
- Route and actor lookup;
- pose and trajectory interpolation;
- local/map coordinate conversion;
- VectorMap caching;
- missing-data handling.

### Tests verified

```text
test_clip_reader.py: 10 tests
test_temporal_index.py: 23 tests
test_coordinate_utils.py: 24 tests
```

Total verified in the portability pass: 57 tests.

## 10. Step 3: Build Candidate Anchors

### Status

Implemented. One execution stage.

### Production entry point

```text
scripts/dataset_tools/build_candidate_anchors.py
```

### Versions

```text
SCRIPT_VERSION = 0.1.0
ANCHOR_FORMAT_VERSION = 0.1-draft
```

### Main implementation

```text
scripts/dataset_tools/anchor_selector.py
```

### Input

```text
$ALPASIM_DATA_ROOT/manifests/clips_v0.1.jsonl
```

### Outputs

```text
$ALPASIM_DATA_ROOT/annotations/v0.1-draft/candidate_anchors.jsonl
$ALPASIM_DATA_ROOT/reports/candidate_anchor_per_clip_v0.1.jsonl
$ALPASIM_DATA_ROOT/reports/candidate_anchor_summary_v0.1.json
```

### Rebuild

```bash
cd "$ALPASIM_ROS2_WS/scripts/dataset_tools"
python3 -u build_candidate_anchors.py --force
```

Use `--limit-clips N` for a smoke test.

### Tests verified

```text
tests/test_anchor_selector.py: 13 tests
tests/test_build_candidate_anchors.py: 2 path tests
```

## 11. Step 4: Meta-action Generator

### Status

Implemented and frozen for the current 10,231 Candidate Anchors.

### Only production entry point

```text
scripts/dataset_tools/build_meta_actions_v02.py
```

Do not use the legacy chain as the production workflow.

### Production call chain

```text
build_meta_actions_v02.py
  1. profile_lane_matching_features.py
  2. refine_lane_matching_features.py
  3. profile_lateral_action_features.py
  4. profile_meta_action_features.py
  5. profile_lane_change_geometry_features.py --all --force
  6. apply meta_action_rules_v02.py
  7. write meta_actions_v0.2.jsonl
```

### Current versions

```text
Meta-action generator: 0.2.1
Rule version: meta_action_rules_v0.2.1
Output format: 0.2-draft
```

### Inputs

```text
$ALPASIM_DATA_ROOT/annotations/v0.1-draft/candidate_anchors.jsonl
raw Clip data under $ALPASIM_DATA_ROOT/test_clip_NNN
```

### Important intermediate outputs

```text
annotations/v0.1-draft/intermediate/lane_matching_features_v0.1.jsonl
annotations/v0.1-draft/intermediate/lane_matching_features_v0.2.jsonl
annotations/v0.1-draft/intermediate/lateral_action_features_v0.3.jsonl
annotations/v0.1-draft/intermediate/meta_action_features_v0.2.jsonl
annotations/v0.1-draft/intermediate/lane_change_geometry_features_v0.1.jsonl
```

### Final outputs

```text
annotations/v0.1-draft/meta_actions_v0.2.jsonl
reports/meta_action_generation_summary_v0.2.json
```

### Current frozen distribution

```text
Anchor count: 10231

Lateral:
  keep_direction: 9472
  unknown: 387
  turn_left: 16
  turn_right: 57
  change_lane_left: 155
  change_lane_right: 144

Longitudinal:
  maintain_speed: 5589
  unknown: 401
  accelerate: 1532
  decelerate: 1810
  stop: 899

Overall quality:
  usable: 9465
  unknown: 766
```

### Current output hash

```text
meta_actions_v0.2.jsonl SHA-256:
a07aacf417829e11d2fe437f01318d509d2d5a007a196440f3d9be95110f5973
```

### Direction-consistency guard

A conservative guard was added for `branch_relative_turn` labels.

If the proposed branch-relative turn direction simultaneously conflicts with:

1. Ego total yaw direction;
2. filtered future path signed heading direction;
3. final relative lateral offset;

then the lateral label becomes `unknown` rather than an incorrect opposite turn.

Affected reviewed Anchors:

```text
test_clip_343_4799457112155000: turn_right -> unknown
test_clip_369_2107945209046000: turn_right -> unknown
```

The Shadow Evaluation found only these two affected Anchors among 62 branch-relative turn labels. The other 60 were direction-consistent.

### Rebuild

Full feature and label rebuild:

```bash
cd "$ALPASIM_ROS2_WS/scripts/dataset_tools"
python3 -u build_meta_actions_v02.py --force
```

Reuse already rebuilt feature files and regenerate only labels:

```bash
python3 -u build_meta_actions_v02.py \
  --reuse-existing-features \
  --force
```

Keep strict distribution checking enabled for production. Do not use `--no-strict-distribution-check` except for controlled Shadow/diagnostic work.

### Legacy/non-production tools

```text
generate_meta_actions.py
finalize_meta_actions_v02.py
evaluate_lateral_shadow_rules.py
```

These remain portable but must not replace `build_meta_actions_v02.py` as the production entry point.

### Tests verified during the last rule pass

```text
test_meta_action_rules_v02.py: 4 tests
test_build_meta_actions_v02.py: 2 tests
test_build_meta_actions_v02_full_step4.py: 2 tests
test_lane_matcher.py: 10 tests
test_lane_change_geometry.py: 6 tests
```

## 12. Step 5: Keyframe Selector

### Status

Implemented with a unified entry point.

### Production entry point

```text
scripts/dataset_tools/build_keyframes_v01.py
```

### Production stages

```text
1. detect_keyframe_events_v01.py
2. deduplicate_keyframe_events_v01.py
3. select_keyframes_v01.py
```

### Inputs

```text
candidate_anchors.jsonl
meta_actions_v0.2.jsonl
lateral_action_features_v0.3.jsonl
meta_action_features_v0.2.jsonl
lane_change_geometry_features_v0.1.jsonl
```

### Outputs

```text
annotations/v0.1-draft/intermediate/keyframe_event_candidates_v0.1.jsonl
annotations/v0.1-draft/intermediate/keyframe_event_candidates_deduplicated_v0.1.jsonl
annotations/v0.1-draft/keyframes.jsonl
reports/keyframe_event_candidate_summary_v0.1.json
reports/keyframe_event_deduplication_summary_v0.1.json
reports/keyframe_selection_summary_v0.1.json
```

### Current verified result

```text
Candidate Anchors: 10231
Event Anchors retained: 2571
Selected Keyframes: 3500

Selection sources:
  normal_driving_baseline: 500
  event_candidate: 2571
  balanced_stable_longitudinal: 300
  balanced_stable_lateral: 129

Lateral:
  keep_direction: 3048
  unknown: 127
  turn_left: 16
  turn_right: 57
  change_lane_left: 123
  change_lane_right: 129

Longitudinal:
  maintain_speed: 1699
  unknown: 304
  accelerate: 596
  decelerate: 643
  stop: 258

Quality:
  usable: 3087
  unknown: 413
```

### Current Keyframe hash

```text
bb3cc755d537c0b8fa0c68aff457106ee00d583bff448bf737c2d286e0ccabf7
```

### Rebuild

```bash
cd "$ALPASIM_ROS2_WS/scripts/dataset_tools"
python3 -u build_keyframes_v01.py --force
```

If event candidates already exist and only final selection must be rebuilt:

```bash
python3 -u build_keyframes_v01.py \
  --reuse-existing-events \
  --force
```

### Tests verified

```text
test_keyframe_event_rules_v01.py: 4 tests
test_keyframe_event_dedup_rules_v01.py: 3 tests
test_keyframe_selection_rules_v01.py: 3 tests
test_build_keyframes_v01.py: 4 tests
```

## 13. Step 6: Navigation Text Generator

### Status

Implemented with a unified entry point.

### Production entry point

```text
scripts/dataset_tools/build_navigation_v01.py
```

### Production stages

```text
1. profile_navigation_branch_context_v01.py
2. profile_road_level_navigation_features_v01.py
3. profile_navigation_route_features_v01.py
4. generate_navigation_candidates_v01.py
5. finalize_navigation_v01.py
```

`profile_navigation_map_context_v01.py` is a diagnostic profiler, not a required production stage. Branch Context directly reuses the necessary map-context functions.

### Inputs

```text
annotations/v0.1-draft/keyframes.jsonl
Anchor-time-or-earlier Navigation Route
Anchor-time Ego pose and speed
VectorMap
```

### Important intermediate outputs

```text
annotations/v0.1-draft/intermediate/navigation_branch_context_v0.1.jsonl
annotations/v0.1-draft/intermediate/road_level_navigation_features_v0.1.jsonl
annotations/v0.1-draft/intermediate/navigation_route_features_v0.1.jsonl
annotations/v0.1-draft/intermediate/navigation_candidates_v0.1.jsonl
```

### Final outputs

```text
annotations/v0.1-draft/navigation.jsonl
reports/navigation_generation_summary_v0.1.json
```

### Versions

```text
Generator version: 0.1.4
Candidate rule version: navigation_rules_v0.1.4-candidate
Final rule version: navigation_rules_v0.1.4
Branch profiler version: 0.1.1
Family Guard version: natural_corridor_family_guard_v0.3
```

### Dynamic upcoming distance

Navigation prompt timing uses a dynamic distance based on current speed, a configured time horizon, minimum and maximum bounds, and Route lookahead:

```text
min(route lookahead, maximum bound, max(minimum bound, speed * time horizon))
```

This dynamic distance controls whether an intersection or branch is considered upcoming. It is distinct from Step 4's fixed branch geometry evaluation distance.

### Natural Corridor Family Guard

The Guard evaluates Route and Natural-successor direction families over 40 m, 60 m, and 80 m horizons.

- Stable and different families:
  - preserve `left_of_natural` or `right_of_natural`.
- Same family or unstable/unavailable family:
  - output `family_guard_unresolved`.
- It never changes successor IDs.
- It prevents contradictory records such as `natural_continuation` with different Natural and Route successor IDs.
- `family_guard_unresolved` is handled conservatively as Navigation `unknown`.

### Current verified output

```text
Records: 3500

Actions:
  straight: 2921
  unknown: 419
  right: 103
  left: 57

Quality:
  usable: 3081
  unknown: 419
```

### Current hashes

```text
navigation_branch_context_v0.1.jsonl:
afcaf3ce2222c383fd457432ed1eb6ecc800c7a3d1c318f0ebd028ff93541aad

road_level_navigation_features_v0.1.jsonl:
f4d1e5d6fab6c047ddb9c20acd7e669842aab5513b87cf35a718ce80d9853629

navigation_route_features_v0.1.jsonl:
140c2613ec44e59c865d45dd6b99bada5f955ec0909f967f090c319e167c0475

navigation_candidates_v0.1.jsonl:
9b8aaaffb86839522ee5d564534ee21cf74d198e7533febaac9a496a41ad92d7

navigation.jsonl:
d025699fcfff677e7929c9df13eb72023d8acd6b815d0fa80c604044c6b7bf90
```

### Rebuild

```bash
cd "$ALPASIM_ROS2_WS/scripts/dataset_tools"
python3 -u build_navigation_v01.py --force
```

### Tests verified

```text
test_natural_corridor_family_guard_v01.py: 6 tests
test_navigation_rules_v014.py: 9 tests
test_road_level_navigation_features_v01.py: 3 tests
test_build_navigation_v01.py: 3 tests
```

## 14. Manual review findings

A joint Step 4 and Step 6 review sampled 15 independent Clips covering:

- matching turn behavior and Navigation;
- differences between current Meta-action and upcoming Route instruction;
- lane changes with straight Navigation;
- stopped vehicles with pending turn instructions;
- `family_guard_unresolved` cases;
- road-level Natural-continuation left/right/straight cases;
- quality-unknown cases.

Important findings:

### Confirmed Step 4 issue, now guarded

```text
test_clip_343_4799457112155000
```

The original Step 4 result was `turn_right`, but video showed the vehicle entered the leftmost left-turn-only lane. Diagnostics showed the 30 m branch-relative net-heading comparison was weak and window-sensitive, while multiple absolute trajectory signals pointed left. The new conservative consistency guard changes this label to `unknown`.

```text
test_clip_369_2107945209046000
```

The original result was `turn_right`. Video ended before a future turn could be confirmed, while all available absolute direction signals conflicted with right. The new Guard changes it to `unknown`.

### Step 6 evidence-association limitation, deferred

```text
test_clip_032_2056242717255000
```

The complete Route and local map support a future left turn at a controlled multi-successor branch around 42.1 m ahead. However, the current rule may use an unrelated first-intersection evidence item near Route distance 0 m when describing the branch as an upcoming intersection. The final direction appears supported, but intersection-evidence-to-branch pairing is not fully rigorous.

This was consciously deferred as a smaller issue under a "抓大放小" policy. Do not silently treat it as resolved.

### Step 6 conservative coverage limitation

Some `family_guard_unresolved` cases are visually straight but become Navigation `unknown`. This reduces coverage but avoids incorrect direction labels. Improve only through a controlled Shadow Evaluation and manual review.

## 15. Step 7: Scene-Fact Generator

### Status

Planned, not yet implemented in the verified Step 1 through Step 6 workflow.

### Initial scope

- road context;
- lead-vehicle presence;
- nearby left/right vehicles;
- relative motion category;
- intersection proximity;
- stop/yield-line proximity;
- decision-relevant critical actor.

### Observability requirement

Do not write invisible simulator truth into supervision text.

Initial observability gate may use:

- actor inside at least one selected camera frustum;
- projected box inside image bounds;
- projected area above a threshold;
- reasonable distance;
- relevance to the driving decision.

Strict occlusion handling may be deferred, but the limitation must be recorded.

### Planned output

```text
annotations/scene_facts.jsonl
```

## 16. Step 8: structured chain of causality

Planned combination of:

- Navigation;
- Scene Facts;
- Meta-action.

Example structure:

```json
{
  "navigation": "turn_right",
  "critical_components": [
    "approaching intersection",
    "slower lead vehicle"
  ],
  "decision": {
    "lateral": "prepare_right_turn",
    "longitudinal": "decelerate"
  }
}
```

## 17. Step 9: Reasoning Generator

Convert structured causal data into short text. Start with deterministic templates where possible. A teacher model may be considered later, but generated reasoning must remain consistent with structured labels and observability constraints.

## 18. Step 10: Sample Manifest Builder

Planned sample assembly:

- `sample_id`;
- `clip_id`;
- `anchor_ns`;
- 4 cameras x 2 frames;
- Ego-history index range;
- Navigation text;
- Scene Facts;
- driving decision;
- reasoning text.

Planned output:

```text
manifests/samples_v0.jsonl
```

## 19. Step 11: train/validation/test split

Split by Clip, never randomly by Anchor.

Planned outputs:

```text
manifests/train_v0.jsonl
manifests/validation_v0.jsonl
manifests/test_v0.jsonl
```

Check action, road-context, actor-interaction, and quality distributions across splits. Adjacent Anchors from one Clip must not leak across splits.

## 20. Step 12: audit and statistics

### Automated checks

- image paths exist;
- timestamps satisfy tolerance;
- Ego history is complete;
- Route exists;
- target JSON parses;
- reasoning agrees with Meta-action and Navigation;
- invisible facts do not enter reasoning;
- distributions are acceptable.

### Manual review

Plan to review 100 to 200 Anchors, covering:

- straight;
- left turn;
- right turn;
- deceleration;
- stop;
- lead vehicle;
- intersection;
- traffic light;
- all camera views.

### Planned outputs

```text
reports/dataset_statistics.json
reports/manual_review.csv
```

## 21. Rebuild order and dependency rules

The implemented dependency chain is:

```text
Step 1 Clip Manifest
  -> Step 3 Candidate Anchors
  -> Step 4 Meta-actions
  -> Step 5 Keyframes
  -> Step 6 Navigation
```

Step 2 is shared code used throughout.

If Step 4 labels change, rebuild Step 5. If the Keyframe set or Keyframe records change, rebuild Step 6.

Recommended production rebuild sequence:

```bash
cd "$ALPASIM_ROS2_WS/scripts/dataset_tools"

python3 -u build_clip_manifest.py --force
python3 -u build_candidate_anchors.py --force
python3 -u build_meta_actions_v02.py --force
python3 -u build_keyframes_v01.py --force
python3 -u build_navigation_v01.py --force
```

Do not run the complete chain blindly during development. Validate each Step's summary, record count, versions, distribution, and SHA-256 before continuing.

## 22. Portable path migration status

The portability pass covered:

- Step 1 through Step 6 production scripts;
- Step 2 real-data tests;
- Step 4 diagnostic and review tools;
- Step 5 and Step 6 scan tools;
- legacy Meta-action tools;
- VectorMap real-data tests;
- scene-manifest generation;
- Planner and GT streaming Shell scripts.

Key files added:

```text
config/local_paths.env.example
scripts/dataset_tools/project_paths.py
scripts/load_local_paths.sh
scripts/dataset_tools/tests/test_build_clip_manifest.py
scripts/dataset_tools/tests/test_build_candidate_anchors.py
```

## 23. Test commands before committing

At minimum, run focused tests for modified steps. For the current portability work, use:

```bash
cd "$ALPASIM_ROS2_WS/scripts/dataset_tools"

python3 -m py_compile \
  project_paths.py \
  build_clip_manifest.py \
  build_candidate_anchors.py \
  build_meta_actions_v02.py \
  build_keyframes_v01.py \
  build_navigation_v01.py

python3 tests/test_build_clip_manifest.py
python3 tests/test_clip_reader.py
python3 tests/test_temporal_index.py
python3 tests/test_coordinate_utils.py
python3 tests/test_anchor_selector.py
python3 tests/test_build_candidate_anchors.py
python3 tests/test_meta_action_rules_v02.py
python3 tests/test_build_meta_actions_v02.py
python3 tests/test_build_meta_actions_v02_full_step4.py
python3 tests/test_lane_matcher.py
python3 tests/test_lane_change_geometry.py
python3 tests/test_keyframe_event_rules_v01.py
python3 tests/test_keyframe_event_dedup_rules_v01.py
python3 tests/test_keyframe_selection_rules_v01.py
python3 tests/test_build_keyframes_v01.py
python3 tests/test_natural_corridor_family_guard_v01.py
python3 tests/test_navigation_rules_v014.py
python3 tests/test_road_level_navigation_features_v01.py
python3 tests/test_build_navigation_v01.py
python3 tests/test_vector_map_reader.py
```

Shell syntax checks:

```bash
cd "$ALPASIM_ROS2_WS"

bash -n \
  scripts/load_local_paths.sh \
  scripts/run_vavam_planner.sh \
  scripts/run_gt_dataset_streaming.sh \
  scripts/run_gt_dataset_streaming_all_unique.sh
```

Absolute-path audit:

```bash
cd "$ALPASIM_ROS2_WS"

grep -RInE \
  --include='*.py' \
  --include='*.sh' \
  --include='*.yaml' \
  --include='*.yml' \
  --include='*.json' \
  '(/home/lab|/home/[^/]+|/mnt/data|file://)' \
  scripts config \
  2>/dev/null \
  | grep -v '/__pycache__/' \
  | grep -v '\.pyc:' \
  | grep -v 'config/local_paths.env:'
```

Expected result: no output.

## 24. Current Git context at handoff preparation

```text
Branch: get_training_data
Latest observed commit: c228f42 fix step 6 4
Previous key commits:
  1da2bc1 fix step 6 navigation natural continuation handling
  4ed677b step 6
  d4e3910 step 5 and record for all alpasim unique scenes
  91779b3 dataset step 4 finish
```

The portability changes and this handoff document were not yet represented in the commit list above at the time the notes were assembled. Always verify current Git status and log before continuing.

## 25. Immediate next actions

Before starting Step 7:

1. Run a final complete portability regression suite.
2. Run `git diff --check`.
3. Review `git diff --stat` and `git status --short`.
4. Update the ordinary repository `README.md` to remove old machine-specific examples.
5. Commit and push the portability changes and this handoff document.
6. Begin Step 7 only after the workspace is clean and the production entry points are documented.

## 26. Quick start for a new AI conversation

When this document is provided to a new AI assistant, the assistant should first ask for or inspect only:

```bash
cd "$ALPASIM_ROS2_WS"
git status --short
git log -5 --oneline
```

Then verify local configuration:

```bash
source scripts/load_local_paths.sh
printf '%s\n' "$ALPASIM_ROOT" "$ALPASIM_ROS2_WS" "$ALPASIM_DATA_ROOT"
```

Do not repeat discovery of the Step 1 through Step 6 production call chains unless Git history shows they changed after this handoff.
