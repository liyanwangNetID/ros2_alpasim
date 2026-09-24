# Step 7 Scene-Fact Generator Design and Frozen Contract

**Document status:** Updated to the frozen Step 7 codebase on 2026-09-24  
**Status:** `PASS / FROZEN`  
**Target dataset:** Current AlpaSim recorded Clip dataset  
**Formal schema:** `schemas/scene_fact_schema_v0.1-draft.json`  
**Final artifact:** `annotations/v0.1-draft/scene_facts.jsonl`  
**Final SHA-256:** `735203f9ddf3b9f49e892edbb185936caa9db1cd46cbfcdd1b9e0f685958e2b5`

## 1. Purpose

Step 7 generates exactly one structured Scene-Fact record for every selected Keyframe. Step 7 describes the scene at the Anchor without using future execution information.

Final facts include:

- Road Context
- intersection, Stop-line, and Yield-line proximity
- bounded observable Lead, Left, and Right Actor lists
- relative position and distance
- short-history relative motion
- per-Actor and record-level quality states and reasons

Step 7 does not generate structured causality or natural-language reasoning. Structured causality begins in Step 8. Natural-language reasoning begins in Step 9.

## 2. Anchor contract

Step 7 operates on `annotations/v0.1-draft/keyframes.jsonl` and validates the Keyframe contract.

Final closure:

```text
Scene-Fact Anchor IDs = Keyframe Anchor IDs
Scene-Fact row count = Keyframe row count = 3500
```

Every Keyframe produces exactly one final Scene-Fact row. Missing or incomplete evidence produces conservative quality states and explicit reasons, not a missing row.

## 3. Allowed and forbidden inputs

### Allowed

- Keyframes and Keyframe contract
- camera calibration and timestamp indexes
- synchronized camera images
- Anchor-time Ego state
- current Actor snapshots
- current and past Actor snapshots at or before the Anchor
- static VectorMap

### Forbidden for current Scene Facts

- `actors/future.jsonl`
- `ego/ground_truth_future.jsonl`
- `ego/complete_recording_ground_truth.json`
- `ego/planner_output.jsonl`
- future Ego execution
- Meta-action labels as scene evidence

## 4. Time, history, and reference Ego speed

Final facts describe `anchor_ns`.

Formal current geometry requires exact Anchor-time Ego and Actor snapshots. Short-history features use only snapshots at or before the Anchor.

Actor selection uses a reference Ego speed equal to the maximum of:

- recorded Ego speed
- executed Ego-state speed when available
- recent pose-derived speed

Pose-derived speed uses the longest usable past-only history window, approximately 3.0 seconds down to two samples near Clip boundaries.

## 5. Coordinates and cameras

Actor geometry is evaluated in Anchor-time Ego-local coordinates:

```text
x positive: forward
y positive: left
```

Cameras:

```text
front_wide
front_tele
cross_left
cross_right
```

Camera streams are synchronized by timestamps. Recorded F-theta intrinsics and `rig_to_camera` extrinsics are used. A pinhole approximation is not permitted.

## 6. Production stages and entry points

Unified production entry point:

```bash
python3 -u -m step7.build_step7
```

Ordered stages:

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

Selective rebuild of the final three stages:

```bash
python3 -u -m step7.build_step7 \
  --from-stage actor_roles
```

The historical 7A through 7M development plan is complete. Current development must use the unified build rather than resume old standalone Step 7E exporters.

## 7. Implemented geometry stack

The frozen implementation includes:

- current Actor geometry
- F-theta camera calibration and projection
- Actor oriented 3D boxes
- adaptive projected edge sampling
- angular-FOV and near-plane clipping
- projected hull geometry
- box surface triangulation
- front-facing and near-plane-clipped triangles
- projected triangle raster cells
- barycentric and perspective depth interpolation
- Actor surface depth rasters
- per-camera Actor depth rasters
- multi-Actor Z-buffer competition
- per-camera and multi-camera occlusion evidence
- projection context for geometric candidates without sampled surfaces
- reviewed geometric observability aggregation
- short-history relative motion
- VectorMap Road Context
- deterministic Actor-list selection
- final feature and Scene-Fact export
- Draft 2020-12 JSON Schema validation

## 8. Observability and occlusion boundary

Simulator Actor truth is not automatically valid visual supervision. An Actor may enter final Scene Facts only after the frozen observability policy accepts the Actor.

Evidence includes:

- camera-frustum compatibility
- valid F-theta projection
- image intersection
- projected extent and height
- inside-image ratio
- distance
- sampled Actor surface support
- Actor-to-Actor occlusion
- per-camera and multi-camera consistency

Step 7 evaluates Actor-to-Actor occlusion. Step 7 does not evaluate:

- buildings
- walls
- guardrails
- vegetation
- arbitrary static-scene meshes
- pixel-accurate instance masks
- rendered depth

Final records state:

```json
"static_occlusion_evaluated": false
```

The output must not be described as strict pixel-level visibility ground truth.

A camera-visible object may lack a usable Actor identity or valid Actor projection at the exact Anchor. Step 7 does not fabricate a Track ID or Actor state to compensate.

## 9. Actor eligibility

Eligible classes:

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

An Actor must satisfy the frozen observability gate. History quality and lane matching contribute to quality and ranking evidence but do not substitute for observability.

## 10. Final Actor lists

```text
lead_actors: maximum 4
left_nearby_actors: maximum 6
right_nearby_actors: maximum 6
```

One Actor may occur in only one list per Anchor. Each selected Actor has a deterministic `role_rank` beginning at 1.

### Lead Actors

A Lead candidate must be:

- observable and eligible
- ahead of Ego by more than 0.5 m
- within the strict 2.5 m Lead lateral corridor
- within the Lead forward horizon

Lead forward horizon:

```text
max(20 m, reference Ego speed * 10 s)
```

There is no fixed maximum Lead distance.

Lead rank follows longitudinal path order. The nearest path-relevant Actor is rank 1. Lane relation is supporting ranking evidence rather than a hard gate. A valid `unrelated` Actor in the strict Lead corridor may still be selected.

`person` and `rider` may enter the Lead list when they occupy the strict forward corridor.

### Left and Right Actors

Side forward horizon:

```text
clamp(reference Ego speed * 5 s, 20 m, 120 m)
```

Rear horizon:

```text
clamp(reference Ego speed * 2 s, 15 m, 50 m)
```

Side lateral range:

```text
0.5 m < absolute lateral offset <= 12 m
```

For `person`, side forward range is additionally capped at 30 m. Lead-corridor persons retain the full Lead horizon.

## 11. Relative position, distance, and motion

Relative position:

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

Frozen thresholds:

```text
stable distance-rate magnitude <= 0.5 m/s
stationary Actor speed <= 0.5 m/s
similar speed difference <= 1.0 m/s
```

Conflicting or insufficient evidence becomes conservative `uncertain` or produces an `unknown` quality status with reasons.

## 12. Road facts

Road Context values:

```text
lane_following
intersection_approach
intersection
unknown
```

Final proximity fields:

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

These are conservative map-based structural facts. Wait-line type is not fully propagated into the final feature layer. A visible STOP sign is not automatically a camera-confirmed Stop-line fact.

## 13. Quality policy

Quality states:

```text
usable
unknown
```

Reasons record insufficient Actor history, unmatched lanes, unknown Ego Road Context, and related evidence limitations.

Unknown or missing evidence must never be replaced with fabricated facts.

## 14. Final record structure

Each final JSONL row contains:

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

Each selected Actor contains:

```text
role_rank
track_id
actor_class
relative_position
relative_distance
relative_x_m
relative_y_m
distance_m
distance_trend
relative_speed_category
actor_speed_mps
ego_speed_mps
observability_status
visible_in_cameras
quality_status
reasons
```

List metadata contains candidate count, selected count, maximum count, and truncation status.

The old single-role presence structure is obsolete and must not be reintroduced.

## 15. Formal outputs

```text
annotations/v0.1-draft/intermediate/actor_role_selection_v0.1.jsonl
annotations/v0.1-draft/intermediate/scene_fact_features_v0.1.jsonl
annotations/v0.1-draft/scene_facts.jsonl
annotations/v0.1-draft/step7h_actor_role_selection_summary_v01.json
annotations/v0.1-draft/step7k_scene_fact_features_summary_v01.json
annotations/v0.1-draft/step7l_scene_facts_summary_v01.json
schemas/scene_fact_schema_v0.1-draft.json
```

The historical Actor Observability and geometric occlusion products remain useful provenance, but the unified build and final artifacts are authoritative.

## 16. Frozen result

```text
Scene-Fact rows: 3500
Schema validation errors: 0
Role conflicts: 0
Road contexts:
  intersection: 257
  intersection_approach: 1016
  lane_following: 2180
  unknown: 47
Quality:
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

The final SHA-256 remained unchanged after code consolidation, compatibility-layer removal, and deterministic rebuild.

## 17. Human review

Random review:

```bash
python3 -u -m step7.review_scene_fact \
  --force
```

Specific Anchor:

```bash
python3 -u -m step7.review_scene_fact \
  --anchor-id test_clip_894_2057127617252000 \
  --force
```

Overlay colors:

```text
Lead: red
Left: yellow
Right: green
```

Validated review-only alignment for `test_clip_894`:

```text
front_wide: -16 source-image pixels
front_tele: -35 source-image pixels
```

All unconfigured Clips use zero offset. Review offsets do not alter formal evidence or final Scene-Fact outputs.

Reusable all-Actor diagnostic:

```text
scripts/render_step7_all_actor_debug.py
```

## 18. Validation and deterministic rebuild

Complete tests:

```bash
cd /home/lab/alpasim_ros2_ws/scripts/dataset_tools

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python3 -m pytest -q
```

Selective deterministic rebuild:

```bash
python3 -u -m step7.build_step7 \
  --from-stage actor_roles
```

Acceptance conditions:

```text
3500 final rows
0 schema validation errors
0 role conflicts
final SHA-256 equals the frozen hash
```

The earlier `1038 passed, 7 subtests passed` count belongs to an active-development snapshot. Obsolete compatibility tests were removed during freeze cleanup, so the authoritative test condition is that the complete current suite passes.

## 19. Known limitations

- static-scene occlusion is not evaluated
- no rendered depth or pixel-accurate instance masks
- some camera-visible objects lack usable Actor coverage at the exact Anchor
- Stop-sign presence is not a dedicated visual fact
- wait-line type is not fully propagated
- opposing-traffic direction is not a dedicated final Actor field
- review offsets are Clip-specific visualization corrections
- final Scene Facts are structured labels, not causality or reasoning text

## 20. Freeze rule and Step 8 interface

Step 7 is frozen. Do not change the formal schema, selection rules, thresholds, final bytes, or production modules while beginning Step 8 unless a concrete Step 8 dependency defect is identified and the change is explicitly approved.

Step 8 may rely on:

- exactly 3500 Scene-Fact rows
- exact Anchor identity matching with Keyframes
- stable Road Context vocabulary
- stable Actor-list keys and ranks
- explicit quality and reasons
- the frozen final SHA-256

Step 8 must not infer hidden Actors or repair upstream coverage by fabricating Scene Facts.
