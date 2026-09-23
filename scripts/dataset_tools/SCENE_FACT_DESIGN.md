# Step 7 Scene-Fact Freeze Contract

Frozen date: 2026-09-23

## Formal Actor lists

- `lead_actors`: maximum 4, strict forward corridor of 2.5 m.
- `left_nearby_actors`: maximum 6.
- `right_nearby_actors`: maximum 6.
- One Actor may occur in only one list per Anchor.
- Lead rank follows longitudinal path order.

## Selection ranges

- Lead forward horizon: `max(20 m, reference Ego speed * 10 s)`, no fixed maximum.
- Side forward horizon: `clamp(reference Ego speed * 5 s, 20 m, 120 m)`.
- Rear horizon: `clamp(reference Ego speed * 2 s, 15 m, 50 m)`.
- Side lateral range: 12 m.
- Side `person` forward range: maximum 30 m.

## Eligible classes

Vehicles, `rider`, and `person`. Non-independent labels such as `protruding_object` are excluded.

## Ego reference speed

Maximum of recorded speed, executed-path speed, and recent pose-derived speed. History uses the longest available window from 3.0 s down to 0.1 s.

## Evidence policy

Only current and past evidence is used. Future Actors, future Ego state, planner output, and meta-action are forbidden.

## Human review

Lead boxes are red, Left boxes yellow, and Right boxes green. Review-only offsets are Clip-specific and do not alter formal projection evidence or Scene-Fact products.
