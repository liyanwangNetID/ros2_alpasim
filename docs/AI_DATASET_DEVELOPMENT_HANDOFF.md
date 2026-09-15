# AlpaSim VLM Dataset Development Handoff

**Audience:** A new AI assistant or developer continuing this repository.  
**Purpose:** Read this document before changing dataset code. It records the current implementation, frozen outputs, development constraints, active Step 7 state, and the exact next work.

## 1. Project goal

Build a supervised dataset for VLM backbone training from recorded AlpaSim driving clips.

### Model input

- Four cameras: `front_wide`, `front_tele`, `cross_left`, `cross_right`
- Two frames per camera: approximately `t0 - 0.5 s` and `t0`
- Ego history
- Coarse Navigation text

### Supervision target

Structured Scene Facts, driving decision, and short causal reasoning. Step 7 produces structured Scene Facts only. Natural-language reasoning begins later.

## 2. Repositories and paths

There are two repositories and one external data root:

- `ALPASIM_ROOT`: external AlpaSim simulator repository
- `ALPASIM_ROS2_WS`: this ROS 2 workspace
- `ALPASIM_DATA_ROOT`: raw `test_clip_NNN` directories and generated dataset artifacts

Python tools use `scripts/dataset_tools/project_paths.py`. Shell tools use `scripts/load_local_paths.sh`. Environment variables and supported CLI path arguments override local defaults.

On the current development machine, the artifact root is `/home/lab/data_from_alpasim`, with formal artifacts under `annotations`, `manifests`, `reports`, `schemas`, and `backups`. Do not store formal generated artifacts in the code directory.

## 3. Development rules for the AI assistant

- The user manages Git. Do not inspect, stage, commit, or modify Git state unless explicitly requested.
- Work by Step and close each Step before moving on.
- Use larger closed-loop batches: implementation, full tests, artifact comparison, cleanup. Split into smaller diagnostics only after a failure.
- Prefer the complete test suite directly unless a high-risk change needs a focused diagnostic.
- Do not ask for confirmation between routine substeps. Continue until a user decision is genuinely required.
- When multiple versions exist, inspect the actual import and execution chain before deciding which version is active.
- Preserve deterministic outputs and compare SHA-256 after production changes.
- Do not mix temporary diagnostics, Shadow outputs, and production artifacts.
- If pasted Shell or Python text is corrupted, especially if a star character replaces letters, stop immediately. Never provide commands containing corrupted identifiers.
- Do not use heredoc or `cat` instructions for the user to create new scripts. The assistant should create files directly when file creation is requested.

## 4. Data leakage boundaries

### Step 4

Step 4 generates supervision labels and may use future Ego trajectory within its defined horizon.

### Step 6

Step 6 generates model input and must not use future execution, future speed, future controls, or Meta-action labels. It uses only the Route available at or before the Anchor, Anchor-time Ego state, and static VectorMap.

### Step 7

Step 7 describes the scene at the Anchor. It may use current and past Actor snapshots at or before the Anchor, current Ego state, camera calibration and images, and static VectorMap. It must not use Actor future, Ego future, planner output, future executed behavior, or Meta-action to decide current facts or observability.

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

Steps 1 through 6 are implemented and reorganized. Step 7 is actively under development. Steps 8 through 12 have not started.

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

Non-ROS dataset utility scripts belong under `scripts/dataset_tools/` and the corresponding Step package. Formal artifacts belong under `ALPASIM_DATA_ROOT`.

## 7. Steps 1 through 4

Steps 1 through 4 are implemented. Their formal behavior and frozen outputs were not changed during the latest Step 5 through Step 7 reorganization.

### Step 4 frozen result

- Candidate Anchors: 10,231
- Meta-action format: `0.2-draft`
- Generator: `0.2.1`
- Rule version: `meta_action_rules_v0.2.1`
- Formal output: `annotations/v0.1-draft/meta_actions_v0.2.jsonl`
- SHA-256: `a07aacf417829e11d2fe437f01318d509d2d5a007a196440f3d9be95110f5973`

A conservative direction-consistency guard changes contradictory branch-relative turn labels to `unknown` rather than emitting an opposite-direction turn.

## 8. Step 5: Keyframes

### Status

Implemented and reorganized. Selection quotas scale with Candidate Anchor count rather than relying on a fixed dataset size.

### Production entry point

`python3 -m step5.build_keyframes_v01 --force`

### Production stages

1. `step5.detect_keyframe_events_v01`
2. `step5.deduplicate_keyframe_events_v01`
3. `step5.select_keyframes_v01`

### Formal outputs

- `annotations/v0.1-draft/keyframes.jsonl`
- `reports/keyframe_selection_summary_v0.1.json`
- `manifests/keyframe_contract_v0.1.json`

### Current verified result

- Candidate Anchors: 10,231
- Event Anchors retained: 2,571
- Selected Keyframes: 3,500
- Keyframe SHA-256: `bb3cc755d537c0b8fa0c68aff457106ee00d583bff448bf737c2d286e0ccabf7`

Selection sources:

```text
normal_driving_baseline: 500
event_candidate: 2571
balanced_stable_longitudinal: 300
balanced_stable_lateral: 129
```

The Keyframe contract records the format, selector and rule versions, upstream Meta-action contract linkage, Candidate count, Event count, Keyframe count, Keyframe SHA-256, and proportional quota policy.

## 9. Step 6: Navigation

### Status

Complete and closed.

### Production entry point

`python3 -m step6.build_navigation_v01 --force`

### Four production stages

1. `step6.profile_navigation_branch_context_v01`
2. `step6.profile_road_level_navigation_features_v01`
3. `step6.profile_navigation_route_features_v01`
4. `step6.generate_navigation_v01`

The old Candidate-to-Final double stage was removed because there was no independent manual-review input. The formal generator now writes `navigation.jsonl` directly while preserving the prior final bytes.

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

`navigation_route_features_v01.py` remains independent because it is shared by three production modules. The old Map Context helper and Road-level helper were merged into their only active profilers.

### Contract validation

`step6.generate_navigation_v01` validates `manifests/keyframe_contract_v0.1.json`, including Keyframe SHA-256, record count, unique Anchor IDs, and exact feature Anchor coverage. The production Summary records the contract path, Keyframe SHA-256, Keyframe count, and coverage validity.

### Current result

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

## 10. Step 7: current development state

### Overall status

Step 7 is not complete. Existing code migration is complete, and the active Step 7E geometric evidence chain has been formalized. Final observability policy, Actor-role rules, road facts, final Scene Facts, and unified Step 7 production entry point remain unfinished.

### Current development position

```text
7A  Schema and vocabulary                     implemented baseline
7B  Current Actor snapshot access             available through Step 2 reader
7C  Ego-relative Actor geometry               implemented
7D  F-theta camera projection                 implemented and tested
7E  Observability and Actor-to-Actor occlusion
    - projection foundation                   implemented
    - low-resolution surface depth and Z-buffer implemented
    - Actor observability intermediate        generated
    - geometric occlusion evidence v02        generated and formalized
    - Shadow scans and policy candidate       available
    - final observability policy              not frozen
7F  Relative-motion features                  not completed as production stage
7G  Road and wait-line features               not completed
7H  Actor role selection                      not completed
7I through 7M                                 not completed
```

### Directory migration

All identified Step 7 Python files were moved from the `dataset_tools` root into `step7/`. Corresponding tests are under `tests/step7/`. After migration, the root scan found zero Step 7 files, the complete test suite passed, and both formal Step 7E products retained their hashes.

### Packaged v02 dependency closure

The v02 geometric occlusion export dependency closure contains 50 packaged Step 7 modules. Its only local external dependencies are `project_paths` and `step2.clip_reader`.

### Actor Observability product

- Path: `annotations/v0.1-draft/intermediate/actor_observability_v0.1.jsonl`
- Structure: one Actor row per `anchor_id` and `track_id`
- Row count: 151,908
- Anchor count with Actor rows: 3,474
- Status counts: `candidate_visible` 86,598; `not_visible` 65,310
- SHA-256: `ed7204a75dadc0194720cf6512083e8997fe8aeb60afde555c5ebed37b1c4b70`

### Geometric Occlusion evidence v02

- Export entry point: `python3 -m step7.export_step7e_geometric_occlusion_evidence_v02`
- Path: `annotations/v0.1-draft/step7e_geometric_occlusion_evidence_v02.jsonl`
- Summary: `annotations/v0.1-draft/step7e_geometric_occlusion_evidence_v02.summary.json`
- Schema: `step7e-geometric-occlusion-evidence-v02`
- Actor rows: 151,908
- Keyframes: 3,500
- Anchors with Actor rows: 3,474
- Rowless Anchors: 26
- Rowless reason: all 26 have one exact Actor snapshot with an empty Actor list
- Duplicate `anchor_id` and `track_id` identities: 0
- Evidence statuses: `candidate_without_sampled_surface` 56; `combined_evidence_available` 86,542; `no_geometric_candidate` 65,310
- Missing-surface projection contexts: 209
- SHA-256: `635a96ead83d301173461304ee1c946e9370e829404ba5a0e1518a8ce75e7724`

The v02 Writer records `output_sha256`. The v02 Exporter validates `keyframe_contract_v0.1.json` and writes Keyframe contract metadata plus complete rowless-Anchor snapshot coverage into the normal Summary. The previous second-pass Summary regeneration and duplicate coverage-check scripts were deleted.

### Step 7E limitations

- No static-scene occlusion evaluation
- No building, wall, vegetation, or arbitrary mesh occlusion
- No rendered depth or instance masks
- No final visibility thresholds or final Scene-Fact labels yet
- Geometric evidence is not strict pixel-level ground truth

## 11. Test baseline

Latest complete regression result:

```text
1038 passed, 7 subtests passed
```

Run the complete suite with:

```bash
cd "$ALPASIM_ROS2_WS/scripts/dataset_tools"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

Prefer this complete suite over repeating focused and complete suites unless failure localization is needed.

## 12. Immediate next actions

Do not repeat Step 7 directory migration. It is complete.

The next development batch should remain inside `step7/` and should:

1. Inventory the packaged Step 7 files by role: production library, production entry point, active analysis, Shadow or review tooling, and obsolete one-off diagnostics.
2. Determine the actual active Observability policy candidate from the generated scan and review artifacts.
3. Preserve the v02 evidence product and hash while final thresholds are evaluated.
4. Freeze the final observability policy only after checking affected Actor rows, independent Clips, per-class impact, and reviewed visual cases.
5. Convert accepted observability states into a formal Actor-observability product without exposing hidden Actor truth.
6. Develop Step 7F short-history relative-motion features.
7. Develop Step 7G road and wait-line features.
8. Develop Step 7H Actor-role selection.
9. Build `scene_fact_features_v0.1.jsonl`, then generate exactly one `scene_facts.jsonl` row per Keyframe.
10. Add a unified `step7.build_scene_facts_v01` entry point and complete deterministic rebuild validation.

The next session should not delete Shadow, review, check, diagnose, or specialized Profile files merely from their names. First inspect imports, outputs, and whether they support the unfinished policy decision. Cleanup should follow the active-chain decision.

## 13. Quick start for a new AI conversation

Provide the new assistant these three files:

- `AI_DATASET_DEVELOPMENT_HANDOFF.md`
- `CLIP_DATA_FORMAT.md`
- `SCENE_FACT_DESIGN.md`

Then state that development should continue from the Step 7 state recorded here.

The assistant should begin with a small read-only status request, without inspecting Git:

```bash
cd /home/lab/alpasim_ros2_ws/scripts/dataset_tools
find step7 -maxdepth 1 -type f -print | sort
find tests/step7 -maxdepth 1 -type f -print | sort
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q
```

It should also verify the two active Step 7E data baselines:

```bash
sha256sum \
  "$ALPASIM_DATA_ROOT/annotations/v0.1-draft/intermediate/actor_observability_v0.1.jsonl" \
  "$ALPASIM_DATA_ROOT/annotations/v0.1-draft/step7e_geometric_occlusion_evidence_v02.jsonl"
```

Expected hashes:

```text
ed7204a75dadc0194720cf6512083e8997fe8aeb60afde555c5ebed37b1c4b70
635a96ead83d301173461304ee1c946e9370e829404ba5a0e1518a8ce75e7724
```

After those checks, continue directly with the Immediate next actions above. Do not rediscover Steps 1 through 6 unless a later code change requires it.
