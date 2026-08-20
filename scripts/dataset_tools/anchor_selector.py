#!/usr/bin/env python3
"""Candidate-anchor selection for the offline AlpaSim dataset pipeline.

Step 3 responsibility:
- derive anchor candidates from synchronized current camera timestamps;
- validate the configured visual history and non-visual context;
- retain detailed failure reasons;
- greedily enforce a minimum time spacing between selected anchors.

The selector never writes files and never modifies raw clips. Batch traversal
and JSONL output belong to build_candidate_anchors.py.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Protocol, Sequence


DEFAULT_CAMERA_NAMES = (
    "front_wide",
    "front_tele",
    "cross_left",
    "cross_right",
)
DEFAULT_FRAME_OFFSETS_NS = (-400_000_000, -200_000_000, 0)
DEFAULT_CAMERA_TOLERANCE_NS = 50_000_000
DEFAULT_EGO_HISTORY_POINTS = 16
DEFAULT_EGO_INTERVAL_NS = 100_000_000
DEFAULT_ROUTE_MAX_AGE_NS = 200_000_000
DEFAULT_ACTOR_TOLERANCE_NS = 100_000_000
DEFAULT_FUTURE_HORIZON_NS = 3_000_000_000
DEFAULT_MINIMUM_SPACING_NS = 500_000_000


class TimeIndexLike(Protocol):
    @property
    def timestamps_ns(self) -> tuple[int, ...]: ...


class ClipReaderLike(Protocol):
    clip_id: str
    start_ns: int
    end_ns: int

    @property
    def camera_indexes(self) -> Mapping[str, TimeIndexLike]: ...

    def get_multicamera_frames(
        self,
        anchor_ns: int,
        *,
        offsets_ns: Sequence[int],
        tolerance_ns: int,
        prefer_exact: bool,
    ) -> Any | None: ...

    def get_ego_history(
        self,
        anchor_ns: int,
        *,
        num_waypoints: int,
        interval_ns: int,
    ) -> Any | None: ...

    def get_navigation_route_at(
        self,
        anchor_ns: int,
        *,
        maximum_age_ns: int,
    ) -> Any | None: ...

    def get_actors_at(
        self,
        anchor_ns: int,
        *,
        tolerance_ns: int,
    ) -> Any | None: ...

    def get_future_ego_trajectory(
        self,
        anchor_ns: int,
        *,
        horizon_ns: int,
    ) -> Any | None: ...


@dataclass(frozen=True, slots=True)
class AnchorSelectorConfig:
    camera_names: tuple[str, ...] = DEFAULT_CAMERA_NAMES
    frame_offsets_ns: tuple[int, ...] = DEFAULT_FRAME_OFFSETS_NS
    camera_tolerance_ns: int = DEFAULT_CAMERA_TOLERANCE_NS
    prefer_exact_camera_sync: bool = True
    ego_history_points: int = DEFAULT_EGO_HISTORY_POINTS
    ego_history_interval_ns: int = DEFAULT_EGO_INTERVAL_NS
    route_maximum_age_ns: int = DEFAULT_ROUTE_MAX_AGE_NS
    actor_tolerance_ns: int = DEFAULT_ACTOR_TOLERANCE_NS
    future_horizon_ns: int = DEFAULT_FUTURE_HORIZON_NS
    minimum_spacing_ns: int = DEFAULT_MINIMUM_SPACING_NS

    def __post_init__(self) -> None:
        if not self.camera_names:
            raise ValueError("camera_names must not be empty")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be unique")
        if not self.frame_offsets_ns:
            raise ValueError("frame_offsets_ns must not be empty")
        if tuple(sorted(self.frame_offsets_ns)) != self.frame_offsets_ns:
            raise ValueError("frame_offsets_ns must be sorted")
        if self.frame_offsets_ns[-1] != 0:
            raise ValueError("the last frame offset must be zero")
        if len(set(self.frame_offsets_ns)) != len(self.frame_offsets_ns):
            raise ValueError("frame_offsets_ns must be unique")
        integer_fields = {
            "camera_tolerance_ns": self.camera_tolerance_ns,
            "ego_history_points": self.ego_history_points,
            "ego_history_interval_ns": self.ego_history_interval_ns,
            "route_maximum_age_ns": self.route_maximum_age_ns,
            "actor_tolerance_ns": self.actor_tolerance_ns,
            "future_horizon_ns": self.future_horizon_ns,
            "minimum_spacing_ns": self.minimum_spacing_ns,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def ego_history_span_ns(self) -> int:
        return (self.ego_history_points - 1) * self.ego_history_interval_ns

    @property
    def visual_history_span_ns(self) -> int:
        return -min(self.frame_offsets_ns)


@dataclass(frozen=True, slots=True)
class AnchorEvaluation:
    clip_id: str
    anchor_ns: int
    checks: dict[str, bool]
    failure_reasons: tuple[str, ...]
    visual_sequence_mode: str | None
    visual_group_modes: tuple[str, ...]
    maximum_camera_skew_ns: int | None
    maximum_camera_target_error_ns: int | None
    route_time_error_ns: int | None
    actor_time_error_ns: int | None

    @property
    def fully_valid(self) -> bool:
        return not self.failure_reasons

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["fully_valid"] = self.fully_valid
        return result


@dataclass(frozen=True, slots=True)
class AnchorSelectionResult:
    clip_id: str
    source_candidate_count: int
    boundary_eligible_count: int
    fully_valid_count: int
    selected_count: int
    evaluations: tuple[AnchorEvaluation, ...]
    selected_anchors: tuple[AnchorEvaluation, ...]

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "source_candidate_count": self.source_candidate_count,
            "boundary_eligible_count": self.boundary_eligible_count,
            "fully_valid_count": self.fully_valid_count,
            "selected_count": self.selected_count,
            "selected_anchor_ns": [
                anchor.anchor_ns for anchor in self.selected_anchors
            ],
        }


class AnchorSelector:
    """Select camera-driven candidate anchors for one clip."""

    def __init__(self, config: AnchorSelectorConfig | None = None) -> None:
        self.config = config or AnchorSelectorConfig()

    def shared_current_timestamps(
        self,
        reader: ClipReaderLike,
    ) -> tuple[int, ...]:
        missing = [
            name
            for name in self.config.camera_names
            if name not in reader.camera_indexes
        ]
        if missing:
            raise ValueError(
                "reader is missing required camera indexes: "
                + ", ".join(missing)
            )

        shared: set[int] | None = None
        for camera_name in self.config.camera_names:
            timestamps = set(
                reader.camera_indexes[camera_name].timestamps_ns
            )
            shared = timestamps if shared is None else shared.intersection(
                timestamps
            )
            if not shared:
                return tuple()
        return tuple(sorted(shared or ()))

    def boundary_eligible_timestamps(
        self,
        reader: ClipReaderLike,
        timestamps_ns: Sequence[int],
    ) -> tuple[int, ...]:
        earliest = reader.start_ns + max(
            self.config.ego_history_span_ns,
            self.config.visual_history_span_ns,
        )
        latest = reader.end_ns - self.config.future_horizon_ns
        if earliest > latest:
            return tuple()
        return tuple(
            timestamp
            for timestamp in timestamps_ns
            if earliest <= timestamp <= latest
        )

    @staticmethod
    def _visual_metadata(sequence: Any | None) -> tuple[
        str | None,
        tuple[str, ...],
        int | None,
        int | None,
    ]:
        if sequence is None:
            return None, tuple(), None, None
        groups = tuple(sequence.groups)
        modes = tuple(str(group.mode) for group in groups)
        if modes and all(mode == "exact" for mode in modes):
            sequence_mode = "all_exact"
        elif modes and all(mode == "approximate" for mode in modes):
            sequence_mode = "all_approximate"
        else:
            sequence_mode = "mixed"
        maximum_skew = max(
            (int(group.maximum_skew_ns) for group in groups),
            default=0,
        )
        maximum_target_error = max(
            (int(group.maximum_target_error_ns) for group in groups),
            default=0,
        )
        return (
            sequence_mode,
            modes,
            maximum_skew,
            maximum_target_error,
        )

    def evaluate_anchor(
        self,
        reader: ClipReaderLike,
        anchor_ns: int,
    ) -> AnchorEvaluation:
        sequence = reader.get_multicamera_frames(
            anchor_ns,
            offsets_ns=self.config.frame_offsets_ns,
            tolerance_ns=self.config.camera_tolerance_ns,
            prefer_exact=self.config.prefer_exact_camera_sync,
        )
        history = reader.get_ego_history(
            anchor_ns,
            num_waypoints=self.config.ego_history_points,
            interval_ns=self.config.ego_history_interval_ns,
        )
        route = reader.get_navigation_route_at(
            anchor_ns,
            maximum_age_ns=self.config.route_maximum_age_ns,
        )
        actors = reader.get_actors_at(
            anchor_ns,
            tolerance_ns=self.config.actor_tolerance_ns,
        )
        future = reader.get_future_ego_trajectory(
            anchor_ns,
            horizon_ns=self.config.future_horizon_ns,
        )

        checks = {
            "multicamera_frames": sequence is not None,
            "ego_history": history is not None,
            "navigation_route": route is not None,
            "actors_current": actors is not None,
            "future_ego_trajectory": future is not None,
        }
        failures = tuple(
            name for name, passed in checks.items() if not passed
        )
        (
            sequence_mode,
            group_modes,
            maximum_skew,
            maximum_target_error,
        ) = self._visual_metadata(sequence)
        return AnchorEvaluation(
            clip_id=reader.clip_id,
            anchor_ns=anchor_ns,
            checks=checks,
            failure_reasons=failures,
            visual_sequence_mode=sequence_mode,
            visual_group_modes=group_modes,
            maximum_camera_skew_ns=maximum_skew,
            maximum_camera_target_error_ns=maximum_target_error,
            route_time_error_ns=(
                int(route.time_error_ns) if route is not None else None
            ),
            actor_time_error_ns=(
                int(actors.time_error_ns) if actors is not None else None
            ),
        )

    def evaluate_clip(
        self,
        reader: ClipReaderLike,
    ) -> tuple[int, tuple[AnchorEvaluation, ...]]:
        source = self.shared_current_timestamps(reader)
        eligible = self.boundary_eligible_timestamps(reader, source)
        evaluations = tuple(
            self.evaluate_anchor(reader, timestamp)
            for timestamp in eligible
        )
        return len(source), evaluations

    def apply_minimum_spacing(
        self,
        anchors: Sequence[AnchorEvaluation],
    ) -> tuple[AnchorEvaluation, ...]:
        valid = sorted(
            (anchor for anchor in anchors if anchor.fully_valid),
            key=lambda anchor: anchor.anchor_ns,
        )
        selected: list[AnchorEvaluation] = []
        last_selected_ns: int | None = None
        for anchor in valid:
            if (
                last_selected_ns is None
                or anchor.anchor_ns - last_selected_ns
                >= self.config.minimum_spacing_ns
            ):
                selected.append(anchor)
                last_selected_ns = anchor.anchor_ns
        return tuple(selected)

    def select(self, reader: ClipReaderLike) -> AnchorSelectionResult:
        source_count, evaluations = self.evaluate_clip(reader)
        selected = self.apply_minimum_spacing(evaluations)
        fully_valid_count = sum(
            evaluation.fully_valid for evaluation in evaluations
        )
        return AnchorSelectionResult(
            clip_id=reader.clip_id,
            source_candidate_count=source_count,
            boundary_eligible_count=len(evaluations),
            fully_valid_count=fully_valid_count,
            selected_count=len(selected),
            evaluations=evaluations,
            selected_anchors=selected,
        )
