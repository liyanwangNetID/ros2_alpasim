# Step 7 Scene-Fact Generator Design and Development State

**Status:** Active development through Step 7E; final Scene Facts are not yet generated  
**Target dataset:** Current AlpaSim recorded Clip dataset  
**Formal schema:** `schemas/scene_fact_schema_v0.1-draft.json`  
**Planned final artifact:** `annotations/v0.1-draft/scene_facts.jsonl`

## 1. Purpose

Step 7 generates structured Scene Facts for every selected Keyframe. It describes the scene at the Anchor without using future execution information.

Planned facts include:

- road context
- intersection and wait-line proximity
- observable lead vehicle
- observable nearby left and right vehicles
- relative position and distance
- short-history relative motion
- component quality and explicit reasons

Step 7 does not generate natural-language reasoning.

## 2. Anchor contract

Step 7 operates on `annotations/v0.1-draft/keyframes.jsonl` and validates `manifests/keyframe_contract_v0.1.json` in active formal exporters.

Final requirement:

```text
Scene-Fact Anchor IDs equal Keyframe Anchor IDs
```

Every Keyframe must eventually produce exactly one final Scene-Fact record. Missing evidence produces conservative `unknown` values with reasons, not a missing row.

## 3. Allowed and forbidden inputs

### Allowed

- Keyframes and Keyframe contract
- camera calibration and timestamp indexes
- selected camera images for review
- Anchor-time Ego state
- current and past Actor snapshots at or before the Anchor
- static VectorMap

### Forbidden for current Scene Facts

- `actors/future.jsonl`
- `ego/ground_truth_future.jsonl`
- `ego/complete_recording_ground_truth.json`
- `ego/planner_output.jsonl`
- future Ego execution
- Meta-action labels as scene evidence

## 4. Time and history policy

Final facts describe `anchor_ns`. Current Ego and Actor data use exact-time preference or bounded lookup, and accepted timing error must be explicit.

A short Actor history window may use only snapshots with timestamps at or before the Anchor. It supports persistence and relative-motion stability but does not change the time represented by the final record.

Actor data should be loaded once per Clip and indexed rather than rescanned for every Keyframe.

## 5. Coordinates and cameras

Actor geometry is evaluated in Anchor-time Ego-local coordinates:

```text
x positive: forward
y positive: left
```

The four cameras are:

```text
front_wide
front_tele
cross_left
cross_right
```

Camera streams are synchronized by timestamps. Recorded F-theta intrinsics and `rig_to_camera` extrinsics must be used. A pinhole approximation is not permitted.

## 6. Current development state

```text
7A  Schema, vocabulary, design boundary       implemented baseline
7B  Current Actor snapshot access             available through Step 2 reader
7C  Ego-relative Actor geometry               implemented
7D  F-theta projection                        implemented and tested
7E  Observability and Actor occlusion         active, advanced but unfinished
7F  Short-history relative motion             not production-complete
7G  Road and wait-line facts                  not production-complete
7H  Actor role selection                      not production-complete
7I  Full feature profiling                    not complete
7J  Threshold scan and manual review          partially represented by Step 7E tooling
7K  Frozen Scene-Fact rules                   not complete
7L  Unified production entry point            not complete
7M  Full deterministic validation             not complete
```

All identified Step 7 Python code has been migrated into `scripts/dataset_tools/step7/`. Tests are under `scripts/dataset_tools/tests/step7/`. The migration preserved the complete test baseline and formal product hashes.

## 7. Implemented geometry stack

The packaged Step 7 implementation includes:

- Scene-Fact schema vocabulary
- current Actor geometry
- F-theta camera calibration and projection
- Actor oriented 3D box construction
- adaptive projected edge sampling
- FOV clipping
- projected hull geometry
- box surface triangulation
- front-facing and near-plane-clipped triangles
- angular FOV diagnostics and subdivision
- projected triangle raster cells
- barycentric and perspective depth interpolation
- Actor surface depth rasters
- per-camera Actor depth rasters
- multi-Actor Z-buffer competition
- per-camera and multi-camera occlusion evidence
- geometric observability aggregation
- projection context for geometric candidates without sampled surfaces

The active v02 production dependency closure contains 50 Step 7 modules and depends externally only on shared path configuration and `step2.clip_reader`.

## 8. Actor Observability intermediate

Formal intermediate product:

```text
annotations/v0.1-draft/intermediate/actor_observability_v0.1.jsonl
```

Current verified baseline:

```text
Actor rows: 151908
Anchors with Actor rows: 3474
candidate_visible: 86598
not_visible: 65310
SHA-256: ed7204a75dadc0194720cf6512083e8997fe8aeb60afde555c5ebed37b1c4b70
```

This product is one row per Actor identity, not one row per Keyframe. Repeated Anchor IDs are expected because one Anchor can contain many Actors.

## 9. Geometric Occlusion evidence v02

Production entry point:

```bash
python3 -m step7.export_step7e_geometric_occlusion_evidence_v02
```

Formal products:

```text
annotations/v0.1-draft/step7e_geometric_occlusion_evidence_v02.jsonl
annotations/v0.1-draft/step7e_geometric_occlusion_evidence_v02.summary.json
```

Current verified baseline:

```text
Schema: step7e-geometric-occlusion-evidence-v02
Actor rows: 151908
Source Keyframes: 3500
Anchors with Actor rows: 3474
Anchors without Actor rows: 26
Duplicate Anchor and Actor identities: 0
candidate_without_sampled_surface: 56
combined_evidence_available: 86542
no_geometric_candidate: 65310
Missing-surface projection contexts: 209
SHA-256: 635a96ead83d301173461304ee1c946e9370e829404ba5a0e1518a8ce75e7724
```

All 26 rowless Anchors were explained by one exact current Actor snapshot with an empty Actor list.

The v02 Writer records the Evidence output SHA-256. The v02 Exporter validates the Step 5 Keyframe contract and writes:

- contract path and version
- Keyframe SHA-256 and count
- source Keyframe count
- Anchors with and without Actor rows
- rowless Anchor IDs
- rowless snapshot evidence and reason counts
- Actor evidence distributions
- projection-context distributions

The previous second-pass Summary regenerator and duplicate coverage-check scripts were removed because the normal exporter now generates the complete Summary.

## 10. Observability policy boundary

Simulator truth is not automatically visual supervision. An Actor may enter final Scene Facts only after a reviewed observability policy accepts it.

Evidence considered includes:

- camera-frustum compatibility
- valid F-theta projection
- image intersection
- projected extent and height
- inside-image ratio
- distance
- sampled Actor surface support
- Actor-to-Actor occlusion
- per-camera and multi-camera consistency

Current geometric evidence statuses are not yet the final Scene-Fact observability vocabulary. Shadow scans, class summaries, failure attribution, baseline-visible impact, fragment guards, and visual review tools exist under `step7/`, but the final policy is not frozen.

## 11. Occlusion boundary

Step 7 v0.1 evaluates Actor-to-Actor occlusion using projected Actor box surfaces and a low-resolution software depth buffer.

It does not evaluate:

- buildings
- walls
- guardrails
- vegetation
- arbitrary static-scene meshes
- pixel-accurate instance masks
- rendered depth

Final quality metadata must continue to state that static occlusion is not evaluated. The output must not be described as strict pixel-level visibility ground truth.

## 12. Hidden-truth protection

A hidden or rejected Actor must not appear in final supervision.

Final Actor-role presence values:

```text
present
not_present
unknown
```

- `present`: an accepted observable Actor was selected
- `not_present`: evidence was sufficient and no accepted candidate was selected
- `unknown`: observability, timing, map, projection, motion, or role evidence was insufficient

A non-present or unknown role must not expose a rejected Actor's Track ID, class, distance, position, or motion.

## 13. Planned relative position, distance, and motion

Intermediate position categories:

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

Final relative distance:

```text
near
medium
far
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

Thresholds remain unfrozen. Conflicting instantaneous and past-window evidence should become conservative `uncertain` unless a reviewed rule resolves the conflict.

## 14. Planned Actor roles

Step 7 v0.1 plans three roles:

```text
lead_vehicle
left_nearby_vehicle
right_nearby_vehicle
```

Selection must consider observability, Ego-local geometry, heading and road compatibility, distance, and relevance. Do not select a Lead Vehicle merely because it is the nearest Actor in front. Opposing and crossing traffic must not be mislabeled as lead traffic.

## 15. Planned road facts

Road context values:

```text
lane_following
intersection_approach
intersection
unknown
```

Planned proximity facts:

```text
intersection_proximity
stop_line_proximity
yield_line_proximity
```

Allowed proximity values:

```text
at
near
approaching
far
none
unknown
```

These are map-based structural facts unless visual projection of the corresponding map element is separately implemented and reviewed. Do not call them visually confirmed wait lines.

## 16. Quality policy

Quality states are `usable` and `unknown`. Each component records `quality_status` and `reasons`. Missing evidence produces explicit unknown fields, never fabricated facts.

Every final record also records that static occlusion is not evaluated.

## 17. Planned final record

Each final JSONL row will contain:

```text
scene_fact_format_version
generator_version
rule_version
anchor_id
clip_id
anchor_ns
road_context
lead_vehicle
left_nearby_vehicle
right_nearby_vehicle
quality
```

A present Actor role must include at least one accepted camera. Non-present roles must not contain hidden Actor identity or state.

## 18. Production outputs still missing

The following are not yet available as formal production outputs:

```text
annotations/v0.1-draft/intermediate/scene_fact_features_v0.1.jsonl
annotations/v0.1-draft/scene_facts.jsonl
reports/scene_fact_feature_summary_v0.1.json
reports/scene_fact_generation_summary_v0.1.json
```

There is no unified `build_scene_facts_v01` production entry point yet.

## 19. Test baseline

Latest complete regression:

```text
1038 passed, 7 subtests passed
```

Key formal hashes:

```text
Keyframes
bb3cc755d537c0b8fa0c68aff457106ee00d583bff448bf737c2d286e0ccabf7

Actor Observability
ed7204a75dadc0194720cf6512083e8997fe8aeb60afde555c5ebed37b1c4b70

Geometric Occlusion v02
635a96ead83d301173461304ee1c946e9370e829404ba5a0e1518a8ce75e7724
```

## 20. Immediate next development work

Directory migration is complete. Do not repeat it.

The next closed-loop batch should:

1. Inventory files inside `step7/` by actual role and imports.
2. Identify production libraries and entry points versus active Shadow, review, specialized Profile, and obsolete one-off tools.
3. Determine the current intended Observability policy candidate from existing scan and visual-review artifacts.
4. Evaluate affected Actor rows, Clips, classes, and baseline-visible impact.
5. Freeze the final Observability policy with regression tests and a versioned rule identifier.
6. Generate a formal accepted-observability intermediate that cannot expose hidden Actor truth.
7. Continue Step 7F relative-motion features, Step 7G road facts, and Step 7H Actor-role selection.
8. Build complete Scene-Fact features and one final Scene-Fact row per Keyframe.
9. Add a unified Step 7 production entry point.
10. Rebuild deterministically and document final hashes and distributions.

Use larger closed-loop batches: implementation, one complete test suite, artifact comparison, cleanup, and final complete test. Split only if a failure requires diagnosis.

Do not delete a Step 7 tool solely because its filename begins with `check`, `diagnose`, `inspect`, `profile`, `scan`, or `generate`. First determine whether it supports the still-unfinished Observability policy decision.

## 21. Known v0.1 limitations

- static-scene occlusion is not evaluated
- no rendered depth or instance masks
- no external segmentation or monocular-depth model
- STOP and YIELD initially mean map proximity, not camera confirmation
- geometry-based visibility remains conservative
- final thresholds are not frozen
- final Scene Facts remain structured labels, not reasoning text
- other datasets are outside the current v0.1 scope
