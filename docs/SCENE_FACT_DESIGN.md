# Step 7 Scene-Fact Generator Design

> **Status:** Step 7A design baseline
>
> **Target dataset:** Current AlpaSim recorded Clip dataset
>
> **Formal schema:** `schemas/scene_fact_schema_v0.1-draft.json`
>
> **Final artifact:** `annotations/v0.1-draft/scene_facts.jsonl`

## 1. Purpose

Step 7 generates structured scene facts for every selected Keyframe.

The facts describe the current driving situation at the Keyframe Anchor:

- road context;
- intersection proximity;
- STOP and YIELD wait-line proximity;
- observable lead vehicle;
- observable nearby vehicle on the left;
- observable nearby vehicle on the right;
- relative position;
- relative distance;
- relative motion.

The final output is structured supervision for later causal reasoning and training-sample construction.

Step 7 does not generate natural-language reasoning.

## 2. Production scope

Step 7 v0.1 is developed for the current AlpaSim recorded Clip format:

```text
test_clip_NNN
```

The current implementation does not support additional datasets. Support for other datasets may be considered only after the current AlpaSim implementation is stable and verified.

## 3. Anchor set

Step 7 operates on the complete Anchor set in:

```text
annotations/v0.1-draft/keyframes.jsonl
```

The final output must satisfy:

```text
Scene-Fact Anchor IDs == Keyframe Anchor IDs
```

Every Keyframe must produce exactly one final Scene-Fact record.

When reliable facts cannot be generated, the record must still exist and use conservative `unknown` values with explicit reasons.

## 4. Allowed inputs

Step 7 v0.1 may read:

```text
annotations/v0.1-draft/keyframes.jsonl

test_clip_NNN/
├── calibration/*.json
├── cameras/*/timestamps.jsonl
├── cameras/*/*.jpg
├── ego/ego_state.jsonl
├── actors/current.jsonl
└── map/vector_map.json
```

Camera images may be opened for projection review and manual validation.

## 5. Forbidden future inputs

Step 7 describes the situation at the Anchor. It must not use future scene or future execution information to determine current Scene Facts.

The following sources are forbidden:

```text
actors/future.jsonl
ego/ground_truth_future.jsonl
ego/complete_recording_ground_truth.json
ego/planner_output.jsonl
```

Step 7 must not use Meta-action or future Ego behavior to decide whether an Actor is present, visible, relevant, approaching, or associated with the current road context.

## 6. Current-time policy

The final Scene Facts describe the world at:

```text
anchor_ns
```

Current Ego and Actor messages are selected using bounded nearest-time lookup.

The maximum accepted time error must be recorded in intermediate features. A message outside the accepted tolerance must not be silently reused.

## 7. Past Actor window

Step 7 v0.1 uses a short history window from:

```text
actors/current.jsonl
```

`actors/current.jsonl` contains a sequence of current Actor snapshots throughout the Clip. It is not a single static record.

Only snapshots satisfying:

```text
snapshot_time <= anchor_ns
```

may be used.

The past window is used only to improve:

- distance trend;
- relative-motion stability;
- Actor persistence;
- consistency between instantaneous velocity and recent motion.

It does not change the time represented by the final Scene Facts.

Step 7 v0.1 does not require:

```text
actors/history.jsonl
```

## 8. Actor-window implementation policy

Actor snapshots must be loaded once per Clip and reused by all Keyframes from that Clip.

The implementation must not reopen and rescan `actors/current.jsonl` from the beginning for every Anchor.

The planned flow is:

```text
load actors/current.jsonl once
→ build a timestamp index
→ query snapshots in [anchor_ns - history_window_ns, anchor_ns]
→ group Actor states by track_id
→ calculate current and historical features
```

History-window duration, minimum history samples, and timestamp tolerance will remain configurable until profiling and review are complete.

## 9. Coordinate policy

Actor geometry is evaluated in the Anchor-time Ego-local frame:

```text
x > 0: forward
x < 0: rear
y > 0: left
y < 0: right
```

The implementation must use the shared coordinate-conversion utilities.

World-coordinate x or y signs must not be used directly to classify relative Actor position. Each source message's declared coordinate frames must be checked.

## 10. Selected cameras

Step 7 v0.1 uses:

```text
front_wide
front_tele
cross_left
cross_right
```

The four camera streams are independent. They may have different frame counts and timestamps.

Camera selection must use timestamp lookup rather than matching frame indices across cameras.

## 11. Camera projection

The recorded cameras use F-theta calibration.

Step 7 must not assume a standard pinhole camera model.

The projection implementation must use:

- recorded F-theta polynomial parameters;
- principal point;
- recorded source resolution;
- `rig_to_camera` extrinsics;
- Actor 3D position, orientation, and dimensions;
- Anchor-time Ego pose.

For every Actor and camera, intermediate features should record:

```text
in_front_of_camera
projection_valid
projected_bbox
projected_area_px
projected_height_px
inside_image_area_px
inside_image_ratio
minimum_depth_m
maximum_depth_m
truncated
failure_reason
```

Projection thresholds are not frozen in Step 7A.

## 12. Projection verification requirement

Before projection results are used for final Scene Facts, the implementation must be visually checked on representative real Keyframes.

The review should overlay projected Actor boxes or projected geometry on the selected camera images and cover:

- an Actor near the image center;
- an Actor near each image edge;
- a partially truncated Actor;
- an Actor behind the camera;
- a distant Actor;
- different camera extrinsics;
- vehicles with different dimensions and orientations.

Projection must not be promoted to production use solely because synthetic geometry tests pass.

## 13. Observability policy

Simulator Actor truth is not automatically valid visual supervision.

An Actor may enter a final Scene Fact only after passing the Step 7 observability filter.

The filter will consider:

- camera-frustum compatibility;
- valid F-theta projection;
- image intersection;
- projected image area;
- projected image height;
- inside-image ratio;
- distance;
- Actor-to-Actor occlusion.

Observable Actor states are:

```text
candidate_visible
partially_occluded
heavily_occluded
not_visible
unknown
```

Only Actors with an accepted observable state may appear as `present` in final Scene Facts.

## 14. Occlusion boundary

Step 7 v0.1 will model Actor-to-Actor occlusion using projected 3D boxes and software depth comparison.

Step 7 v0.1 will not model:

- building occlusion;
- wall occlusion;
- guardrail occlusion;
- vegetation occlusion;
- arbitrary static-scene mesh occlusion;
- pixel-accurate instance masks;
- rendered depth images.

Every final record must state:

```json
{
  "static_occlusion_evaluated": false
}
```

The first version must not describe its output as strict pixel-level visibility ground truth.

## 15. Actor-to-Actor occlusion

The planned Actor-to-Actor occlusion method is a low-resolution software depth buffer:

```text
construct Actor 3D boxes
→ triangulate visible box surfaces
→ project surfaces into each camera
→ compare depth in a low-resolution image grid
→ estimate visible pixels and visible ratio
→ record likely occluding Actor IDs
```

Planned intermediate fields include:

```text
projected_pixel_count
visible_pixel_count
visible_ratio
occluded_by_actor_ids
occlusion_method
```

Exact grid resolution and acceptance thresholds remain unfrozen until profiling and manual review.

## 16. Hidden-truth protection

The final Scene-Fact record must not reveal the identity or state of an Actor that failed observability filtering.

Final Actor-role presence states are:

```text
present
not_present
unknown
```

Meanings:

- `present`: an accepted observable Actor was selected for the role;
- `not_present`: observation and role-selection conditions were sufficient, and no accepted Actor was selected;
- `unknown`: input, projection, observability, map, or role evidence was insufficient.

The internal value:

```text
not_observed
```

may be stored in intermediate diagnostic features, but it must not appear in a final Actor-role record.

A hidden Actor's `track_id`, class, motion, or position must not be copied into final supervision.

## 17. Relative position

Intermediate Actor geometry may classify:

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

Final Actor roles are not selected from the region alone.

Role selection must also consider road compatibility, Actor heading, distance, observability, and other relevant evidence.

## 18. Relative distance

Final relative-distance categories are:

```text
near
medium
far
unknown
```

Intermediate features must retain continuous metric values before classification, including:

```text
relative_x_m
relative_y_m
planar_distance_m
longitudinal_distance_m
lateral_distance_m
```

Distance thresholds are not frozen in Step 7A.

## 19. Relative motion

Relative motion uses two evidence sources.

### 19.1 Instantaneous evidence

Derived from current Ego and Actor state:

```text
instantaneous relative speed
current longitudinal separation
current lateral separation
current Actor heading
current Ego heading
```

### 19.2 Past-window evidence

Derived from recent Actor snapshots at or before the Anchor:

```text
distance change rate
mean relative longitudinal speed
Actor persistence
history sample count
history duration
```

Final distance-trend categories are:

```text
approaching
receding
stable_distance
uncertain
```

Final relative-speed categories are:

```text
slower_than_ego
similar_to_ego
faster_than_ego
stationary
uncertain
```

When instantaneous and historical evidence conflict, Step 7 must prefer a conservative `uncertain` result unless a later reviewed rule explicitly resolves the conflict.

Insufficient history does not remove a currently observable Actor. It only reduces confidence in the relative-motion facts.

## 20. Actor roles

Step 7 v0.1 produces three primary Actor roles:

```text
lead_vehicle
left_nearby_vehicle
right_nearby_vehicle
```

The final record contains at most one selected Actor for each role.

Intermediate features should preserve all role candidates, scores, and rejection reasons.

## 21. Lead-vehicle selection

A Lead Vehicle must not be selected only because it is the closest Actor in front of the Ego.

Selection must consider:

- positive Ego-local longitudinal position;
- compatibility with the Ego driving corridor;
- compatible heading or road direction;
- accepted observability;
- valid distance;
- nearest relevant longitudinal candidate.

Opposing-traffic and crossing-traffic Actors must not be mislabeled as Lead Vehicles.

When lane or road compatibility is unreliable, the result should become `unknown` rather than selecting a geometrically nearby but unrelated Actor.

## 22. Left and right nearby-vehicle selection

Selection must consider:

- Ego-local left or right relation;
- longitudinal neighborhood;
- adjacent lane or nearby driving corridor;
- Actor heading compatibility;
- accepted observability;
- distance;
- decision relevance.

A vehicle far to one side but unrelated to the Ego driving corridor must not be selected merely because its Ego-local y-coordinate has the correct sign.

## 23. Road context

Step 7 v0.1 produces:

```text
lane_following
intersection_approach
intersection
unknown
```

Road context is derived from Anchor-time Ego position and VectorMap evidence.

It must not be inferred from future Ego execution.

Intersection evidence may include lane topology, branching, wait-line associations, and other current map relationships already represented by the VectorMap reader.

## 24. Proximity facts

Step 7 v0.1 produces:

```text
intersection_proximity
stop_line_proximity
yield_line_proximity
```

Allowed proximity values are:

```text
at
near
approaching
far
none
unknown
```

Intermediate features must preserve continuous metric distances before they are converted into categories.

Step 7A does not freeze the metric thresholds.

The first version describes map-based proximity. It must not call these fields:

```text
visible_stop_line
visible_yield_line
```

unless visual projection of the corresponding map element is implemented and validated separately.

## 25. Road facts versus visual facts

Step 7 must distinguish:

### Map-based structural facts

Examples:

```text
intersection proximity
STOP wait-line proximity
YIELD wait-line proximity
current-lane branch structure
```

### Camera-observable Actor facts

Examples:

```text
observable lead vehicle
observable left nearby vehicle
observable right nearby vehicle
```

A map-based fact must not be described as visually confirmed unless it is separately projected into a camera and validated.

## 26. Quality policy

Quality states are:

```text
usable
unknown
```

Every road-context and Actor-role component records:

```text
quality_status
reasons
```

The complete record also contains:

```text
quality.status
quality.static_occlusion_evaluated
quality.reasons
```

Missing or unreliable evidence must result in explicit `unknown` output rather than fabricated facts.

## 27. Final output fields

Each final record contains:

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

The final production artifact is:

```text
annotations/v0.1-draft/scene_facts.jsonl
```

The formal JSON Schema is stored in the current dataset under:

```text
schemas/scene_fact_schema_v0.1-draft.json
```

Each JSONL line represents one Keyframe Anchor.

## 28. Present Actor-role structure

A selected Actor role follows this structure:

```json
{
  "presence_status": "present",
  "track_id": "18",
  "actor_class": "trailer",
  "relative_position": "front",
  "relative_distance": "near",
  "distance_trend": "approaching",
  "relative_speed_category": "slower_than_ego",
  "observability_status": "candidate_visible",
  "visible_in_cameras": [
    "front_wide",
    "front_tele"
  ],
  "quality_status": "usable",
  "reasons": []
}
```

A `present` role must include at least one camera in `visible_in_cameras`.

## 29. Non-present Actor-role structure

A role without a selected Actor follows one of these forms:

```json
{
  "presence_status": "not_present",
  "quality_status": "usable",
  "reasons": []
}
```

or:

```json
{
  "presence_status": "unknown",
  "quality_status": "unknown",
  "reasons": [
    "actor_observability_insufficient"
  ]
}
```

A non-present final role must not contain a hidden Actor's `track_id`, class, distance, or motion.

## 30. Example final record

```json
{
  "scene_fact_format_version": "0.1-draft",
  "generator_version": "0.1.0",
  "rule_version": "scene_fact_rules_v0.1-draft",
  "anchor_id": "test_clip_001_9305000000000",
  "clip_id": "test_clip_001",
  "anchor_ns": 9305000000000,
  "road_context": {
    "type": "intersection_approach",
    "intersection_proximity": "near",
    "stop_line_proximity": "near",
    "yield_line_proximity": "none",
    "quality_status": "usable",
    "reasons": []
  },
  "lead_vehicle": {
    "presence_status": "present",
    "track_id": "18",
    "actor_class": "trailer",
    "relative_position": "front",
    "relative_distance": "near",
    "distance_trend": "approaching",
    "relative_speed_category": "slower_than_ego",
    "observability_status": "candidate_visible",
    "visible_in_cameras": [
      "front_wide",
      "front_tele"
    ],
    "quality_status": "usable",
    "reasons": []
  },
  "left_nearby_vehicle": {
    "presence_status": "not_present",
    "quality_status": "usable",
    "reasons": []
  },
  "right_nearby_vehicle": {
    "presence_status": "unknown",
    "quality_status": "unknown",
    "reasons": [
      "actor_observability_insufficient"
    ]
  },
  "quality": {
    "status": "usable",
    "static_occlusion_evaluated": false,
    "reasons": []
  }
}
```

The example illustrates the record shape only. Its values and thresholds are not frozen Step 7 rules.

## 31. Intermediate outputs

Step 7 will preserve diagnostic information separately from final supervision.

Planned intermediate outputs:

```text
annotations/v0.1-draft/intermediate/
├── actor_observability_v0.1.jsonl
└── scene_fact_features_v0.1.jsonl
```

Intermediate features may contain:

- complete current Actor truth;
- Actors rejected by observability filtering;
- per-camera projection details;
- Actor-to-Actor occlusion diagnostics;
- continuous distances;
- historical motion evidence;
- all Actor-role candidates;
- selection scores;
- failure and downgrade reasons.

Hidden Actor truth in intermediate files must not be copied into final Scene-Fact supervision.

## 32. Planned summary outputs

Planned reports are:

```text
reports/actor_observability_summary_v0.1.json
reports/scene_fact_feature_summary_v0.1.json
reports/scene_fact_generation_summary_v0.1.json
```

Expected summary categories include:

- input Keyframe count;
- output record count;
- Actor count;
- observability status counts;
- projection failure counts;
- rejection-reason counts;
- per-camera visible-Actor counts;
- Actor-to-Actor occlusion statistics;
- insufficient-history counts;
- road-context counts;
- Actor-role presence counts;
- quality-status counts;
- output SHA-256.

Exact report fields will be frozen with the corresponding production stages.

## 33. Version constants

Step 7 v0.1 begins with:

```text
SCENE_FACT_FORMAT_VERSION = 0.1-draft
OBSERVABILITY_FORMAT_VERSION = 0.1-draft
FEATURE_FORMAT_VERSION = 0.1-draft
GENERATOR_VERSION = 0.1.0
RULE_VERSION = scene_fact_rules_v0.1-draft
```

The Rule version remains a draft until thresholds and Actor-role selection rules have been profiled, reviewed, tested, and frozen.

## 34. Planned implementation modules

```text
scene_fact_schema_v01.py
scene_fact_geometry_v01.py
camera_projection_v01.py
actor_observability_v01.py
scene_fact_rules_v01.py

profile_actor_observability_v01.py
profile_scene_fact_features_v01.py
generate_scene_facts_v01.py
build_scene_facts_v01.py
```

The final production call chain is planned as:

```text
build_scene_facts_v01.py
├── profile_actor_observability_v01.py
├── profile_scene_fact_features_v01.py
└── generate_scene_facts_v01.py
```

The exact call sequence may be adjusted during implementation, but `build_scene_facts_v01.py` will be the only final production entry point.

## 35. Development order

```text
7A  Schema, vocabulary, and design boundary
7B  Actor current-snapshot history-window API
7C  Ego-relative Actor geometry
7D  F-theta camera projection
7E  Actor observability and Actor-to-Actor occlusion
7F  Short-history relative-motion features
7G  Road and wait-line features
7H  Actor role selection
7I  Full Scene-Fact feature profiling
7J  Threshold scan and manual review
7K  Frozen Scene-Fact rules
7L  Unified production entry point
7M  Full validation, deterministic rebuild, and documentation
```

## 36. Step 7A completion criteria

Step 7A is complete when:

- the Python vocabulary module exists;
- the Python vocabulary tests pass;
- `SCHEMA_ROOT` is defined in the shared path module;
- the formal JSON Schema exists under the dataset `schemas/` directory;
- Python and JSON Schema vocabularies have a consistency test;
- this design document is committed;
- no thresholds or production labels have been silently frozen.

## 37. Step 7B entry criteria

Before Step 7B begins:

- Step 7A tests must pass;
- the formal Schema must remain unchanged unless a reviewed defect is found;
- Git status and active files must be checked;
- no incomplete heredoc output or temporary document may remain.

Step 7B will add a current-Actor snapshot window API that only reads snapshots at or before the Anchor.

## 38. Schema backfill policy

The current Scene-Fact Schema is the first standalone JSON Schema stored under the dataset `schemas/` directory.

After Step 7 is complete, a separate Schema Backfill stage will document the existing outputs from Steps 1 through 6.

Historical Schemas must be derived from existing production artifacts and active code. They must not redesign or reinterpret previously frozen records.

## 39. Known v0.1 limitations

Step 7 v0.1 intentionally accepts these limitations:

- static-scene occlusion is not evaluated;
- no rendered depth or instance mask is recorded;
- no USDZ ray casting is used;
- no external monocular depth or segmentation model is used;
- STOP and YIELD facts initially represent map proximity, not confirmed image visibility;
- Actor visibility is geometry-based and conservative;
- thresholds remain subject to profiling and manual review;
- final Scene Facts are structured labels, not natural-language reasoning.

These limitations do not permit hidden Actor truth to enter final supervision. Ambiguous cases must become `unknown`.

## 40. Non-goals for Step 7 v0.1

The following are outside the first version:

```text
traffic-light color classification
strict static-environment occlusion
pixel-perfect visible instance masks
natural-language reasoning
full Chain-of-Causation generation
counterfactual scene generation
support for additional public datasets
```

They may be addressed in later versions only after the current Step 7 output is stable and audited.
