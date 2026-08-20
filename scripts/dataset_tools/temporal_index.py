#!/usr/bin/env python3
"""Nanosecond time indexing and multi-camera synchronization utilities.

This module is independent of ROS 2. All timestamps and tolerances are integer
nanoseconds. It supplies the shared temporal behavior used by the offline
AlpaSim dataset pipeline.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Generic, Iterable, Mapping, Sequence, TypeVar


T = TypeVar("T")


class TemporalIndexError(ValueError):
    """Base exception for invalid temporal indexes or queries."""


class EmptyTemporalIndexError(TemporalIndexError):
    """Raised when an operation requires data but the index is empty."""


class TimestampOrderError(TemporalIndexError):
    """Raised when input timestamps are not strictly increasing."""


class TemporalMatchError(LookupError):
    """Raised when no timestamp satisfies a requested temporal constraint."""


@dataclass(frozen=True, slots=True)
class TemporalMatch(Generic[T]):
    """One matched item from a temporal index."""

    index: int
    timestamp_ns: int
    target_ns: int
    error_ns: int
    value: T


@dataclass(frozen=True, slots=True)
class MultiCameraMatch(Generic[T]):
    """One synchronized multi-camera group for a target time."""

    target_ns: int
    mode: str
    matches: dict[str, TemporalMatch[T]]
    maximum_skew_ns: int
    maximum_target_error_ns: int


@dataclass(frozen=True, slots=True)
class MultiCameraSequence(Generic[T]):
    """Synchronized camera groups for several target offsets."""

    anchor_ns: int
    offsets_ns: tuple[int, ...]
    groups: tuple[MultiCameraMatch[T], ...]


class TemporalIndex(Generic[T]):
    """Binary-search index over strictly increasing integer timestamps.

    Args:
        timestamps_ns: Timestamps in strictly increasing order.
        values: Optional values aligned one-to-one with timestamps. If omitted,
            each value is the corresponding timestamp.
        name: Human-readable identifier used in error messages.
    """

    def __init__(
        self,
        timestamps_ns: Iterable[int],
        values: Iterable[T] | None = None,
        *,
        name: str = "temporal_index",
    ) -> None:
        timestamps = tuple(timestamps_ns)
        self.name = name

        for position, timestamp in enumerate(timestamps):
            if isinstance(timestamp, bool) or not isinstance(timestamp, int):
                raise TypeError(
                    f"{name}: timestamp at position {position} must be an "
                    f"integer nanosecond value, got {type(timestamp).__name__}"
                )

        for position in range(1, len(timestamps)):
            previous = timestamps[position - 1]
            current = timestamps[position]
            if current <= previous:
                relation = "duplicate" if current == previous else "out of order"
                raise TimestampOrderError(
                    f"{name}: {relation} timestamp at position {position}: "
                    f"previous={previous}, current={current}"
                )

        if values is None:
            aligned_values = tuple(timestamps)  # type: ignore[assignment]
        else:
            aligned_values = tuple(values)
            if len(aligned_values) != len(timestamps):
                raise TemporalIndexError(
                    f"{name}: timestamp/value length mismatch: "
                    f"{len(timestamps)} != {len(aligned_values)}"
                )

        self._timestamps_ns = timestamps
        self._values = aligned_values

    def __len__(self) -> int:
        return len(self._timestamps_ns)

    def __bool__(self) -> bool:
        return bool(self._timestamps_ns)

    @property
    def timestamps_ns(self) -> tuple[int, ...]:
        return self._timestamps_ns

    @property
    def first_timestamp_ns(self) -> int | None:
        return self._timestamps_ns[0] if self._timestamps_ns else None

    @property
    def last_timestamp_ns(self) -> int | None:
        return self._timestamps_ns[-1] if self._timestamps_ns else None

    def _match(self, index: int, target_ns: int) -> TemporalMatch[T]:
        timestamp = self._timestamps_ns[index]
        return TemporalMatch(
            index=index,
            timestamp_ns=timestamp,
            target_ns=target_ns,
            error_ns=timestamp - target_ns,
            value=self._values[index],
        )

    @staticmethod
    def _validate_target(target_ns: int) -> None:
        if isinstance(target_ns, bool) or not isinstance(target_ns, int):
            raise TypeError("target_ns must be an integer nanosecond value")

    @staticmethod
    def _validate_tolerance(tolerance_ns: int | None) -> None:
        if tolerance_ns is None:
            return
        if isinstance(tolerance_ns, bool) or not isinstance(tolerance_ns, int):
            raise TypeError("tolerance_ns must be an integer or None")
        if tolerance_ns < 0:
            raise ValueError("tolerance_ns must be non-negative")

    @staticmethod
    def _within_tolerance(error_ns: int, tolerance_ns: int | None) -> bool:
        return tolerance_ns is None or abs(error_ns) <= tolerance_ns

    def exact(self, target_ns: int) -> TemporalMatch[T] | None:
        """Return the exact match, or None when the timestamp is absent."""
        self._validate_target(target_ns)
        index = bisect_left(self._timestamps_ns, target_ns)
        if index < len(self) and self._timestamps_ns[index] == target_ns:
            return self._match(index, target_ns)
        return None

    def nearest(
        self,
        target_ns: int,
        *,
        tolerance_ns: int | None = None,
    ) -> TemporalMatch[T] | None:
        """Return the closest timestamp, preferring the earlier on a tie."""
        self._validate_target(target_ns)
        self._validate_tolerance(tolerance_ns)
        if not self:
            return None

        insertion = bisect_left(self._timestamps_ns, target_ns)
        candidates: list[int] = []
        if insertion > 0:
            candidates.append(insertion - 1)
        if insertion < len(self):
            candidates.append(insertion)

        index = min(
            candidates,
            key=lambda item: (
                abs(self._timestamps_ns[item] - target_ns),
                self._timestamps_ns[item],
            ),
        )
        match = self._match(index, target_ns)
        if not self._within_tolerance(match.error_ns, tolerance_ns):
            return None
        return match

    def at_or_before(
        self,
        target_ns: int,
        *,
        tolerance_ns: int | None = None,
    ) -> TemporalMatch[T] | None:
        """Return the latest timestamp at or before target_ns."""
        self._validate_target(target_ns)
        self._validate_tolerance(tolerance_ns)
        index = bisect_right(self._timestamps_ns, target_ns) - 1
        if index < 0:
            return None
        match = self._match(index, target_ns)
        if not self._within_tolerance(match.error_ns, tolerance_ns):
            return None
        return match

    def at_or_after(
        self,
        target_ns: int,
        *,
        tolerance_ns: int | None = None,
    ) -> TemporalMatch[T] | None:
        """Return the earliest timestamp at or after target_ns."""
        self._validate_target(target_ns)
        self._validate_tolerance(tolerance_ns)
        index = bisect_left(self._timestamps_ns, target_ns)
        if index >= len(self):
            return None
        match = self._match(index, target_ns)
        if not self._within_tolerance(match.error_ns, tolerance_ns):
            return None
        return match

    def require_nearest(
        self,
        target_ns: int,
        *,
        tolerance_ns: int | None = None,
    ) -> TemporalMatch[T]:
        """Nearest lookup that raises TemporalMatchError on failure."""
        match = self.nearest(target_ns, tolerance_ns=tolerance_ns)
        if match is None:
            raise TemporalMatchError(
                f"{self.name}: no match for target {target_ns} within "
                f"tolerance {tolerance_ns}"
            )
        return match

    def query_offsets(
        self,
        anchor_ns: int,
        offsets_ns: Sequence[int],
        *,
        tolerance_ns: int | None = None,
        require_unique: bool = True,
    ) -> tuple[TemporalMatch[T], ...] | None:
        """Match nearest entries at anchor_ns plus each requested offset."""
        self._validate_target(anchor_ns)
        matches: list[TemporalMatch[T]] = []
        for offset_ns in offsets_ns:
            if isinstance(offset_ns, bool) or not isinstance(offset_ns, int):
                raise TypeError("every offset must be an integer nanosecond value")
            match = self.nearest(
                anchor_ns + offset_ns,
                tolerance_ns=tolerance_ns,
            )
            if match is None:
                return None
            matches.append(match)

        if require_unique:
            selected = [match.index for match in matches]
            if len(selected) != len(set(selected)):
                return None
        return tuple(matches)


def _validate_camera_indexes(
    indexes: Mapping[str, TemporalIndex[T]],
) -> None:
    if not indexes:
        raise ValueError("at least one camera index is required")
    for camera_name, index in indexes.items():
        if not camera_name:
            raise ValueError("camera names must be non-empty")
        if not isinstance(index, TemporalIndex):
            raise TypeError(
                f"camera {camera_name!r} does not contain a TemporalIndex"
            )


def synchronize_cameras_at(
    indexes: Mapping[str, TemporalIndex[T]],
    target_ns: int,
    *,
    tolerance_ns: int,
    prefer_exact: bool = True,
) -> MultiCameraMatch[T] | None:
    """Synchronize all cameras for one target time.

    Exact mode searches for a timestamp shared by every camera and selects the
    shared timestamp nearest to target_ns within tolerance_ns. If no shared
    timestamp is available and fallback is permitted, approximate mode selects
    each camera's nearest frame. Approximate matching requires both every
    target error and the overall camera skew to be within tolerance_ns.
    """
    _validate_camera_indexes(indexes)
    TemporalIndex._validate_target(target_ns)
    TemporalIndex._validate_tolerance(tolerance_ns)

    if prefer_exact:
        common: set[int] | None = None
        for index in indexes.values():
            timestamps = set(index.timestamps_ns)
            common = timestamps if common is None else common.intersection(timestamps)
            if not common:
                break

        if common:
            shared_timestamp = min(
                common,
                key=lambda timestamp: (
                    abs(timestamp - target_ns),
                    timestamp,
                ),
            )
            if abs(shared_timestamp - target_ns) <= tolerance_ns:
                matches: dict[str, TemporalMatch[T]] = {}
                for camera_name, index in indexes.items():
                    match = index.exact(shared_timestamp)
                    if match is None:
                        raise RuntimeError("shared timestamp lookup became inconsistent")
                    matches[camera_name] = TemporalMatch(
                        index=match.index,
                        timestamp_ns=match.timestamp_ns,
                        target_ns=target_ns,
                        error_ns=match.timestamp_ns - target_ns,
                        value=match.value,
                    )
                return MultiCameraMatch(
                    target_ns=target_ns,
                    mode="exact",
                    matches=matches,
                    maximum_skew_ns=0,
                    maximum_target_error_ns=abs(shared_timestamp - target_ns),
                )

    approximate: dict[str, TemporalMatch[T]] = {}
    for camera_name, index in indexes.items():
        match = index.nearest(target_ns, tolerance_ns=tolerance_ns)
        if match is None:
            return None
        approximate[camera_name] = match

    timestamps = [match.timestamp_ns for match in approximate.values()]
    maximum_skew = max(timestamps) - min(timestamps)
    maximum_target_error = max(
        abs(match.error_ns) for match in approximate.values()
    )
    if maximum_skew > tolerance_ns or maximum_target_error > tolerance_ns:
        return None

    return MultiCameraMatch(
        target_ns=target_ns,
        mode="approximate",
        matches=approximate,
        maximum_skew_ns=maximum_skew,
        maximum_target_error_ns=maximum_target_error,
    )


def synchronize_camera_sequence(
    indexes: Mapping[str, TemporalIndex[T]],
    anchor_ns: int,
    offsets_ns: Sequence[int],
    *,
    tolerance_ns: int,
    prefer_exact: bool = True,
    require_unique_per_camera: bool = True,
) -> MultiCameraSequence[T] | None:
    """Build synchronized groups at anchor_ns plus the supplied offsets."""
    groups: list[MultiCameraMatch[T]] = []
    used_indexes: dict[str, set[int]] = {
        camera_name: set() for camera_name in indexes
    }

    for offset_ns in offsets_ns:
        if isinstance(offset_ns, bool) or not isinstance(offset_ns, int):
            raise TypeError("every offset must be an integer nanosecond value")
        group = synchronize_cameras_at(
            indexes,
            anchor_ns + offset_ns,
            tolerance_ns=tolerance_ns,
            prefer_exact=prefer_exact,
        )
        if group is None:
            return None

        if require_unique_per_camera:
            for camera_name, match in group.matches.items():
                if match.index in used_indexes[camera_name]:
                    return None
            for camera_name, match in group.matches.items():
                used_indexes[camera_name].add(match.index)

        groups.append(group)

    return MultiCameraSequence(
        anchor_ns=anchor_ns,
        offsets_ns=tuple(offsets_ns),
        groups=tuple(groups),
    )
