"""Fixed Actor-box face and triangle topology for Step 7E occlusion.

The corner order must match actor_box_projection_v01.local_box_corners():
0=(-,-,-), 1=(+,-,-), 2=(-,+,-), 3=(+,+,-),
4=(-,-,+), 5=(+,-,+), 6=(-,+,+), 7=(+,+,+).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Sequence

from camera_projection_v01 import Vector3


@dataclass(frozen=True, slots=True)
class BoxTriangle3D:
    face_name: str
    corner_indices: tuple[int, int, int]
    vertices: tuple[Vector3, Vector3, Vector3]


# Each quad is wound counter-clockwise when viewed from outside the box.
BOX_FACE_INDEX_QUADS: Final[tuple[tuple[str, tuple[int, int, int, int]], ...]] = (
    ("negative_x", (0, 4, 6, 2)),
    ("positive_x", (1, 3, 7, 5)),
    ("negative_y", (0, 1, 5, 4)),
    ("positive_y", (2, 6, 7, 3)),
    ("negative_z", (0, 2, 3, 1)),
    ("positive_z", (4, 5, 7, 6)),
)

BOX_TRIANGLE_INDEX_TRIPLES: Final[
    tuple[tuple[str, tuple[int, int, int]], ...]
] = tuple(
    triangle
    for face_name, (first, second, third, fourth) in BOX_FACE_INDEX_QUADS
    for triangle in (
        (face_name, (first, second, third)),
        (face_name, (first, third, fourth)),
    )
)


def triangulate_box_surfaces(
    corners: Sequence[Vector3],
) -> tuple[BoxTriangle3D, ...]:
    """Return the twelve outward-wound triangles of an eight-corner box."""
    if len(corners) != 8:
        raise ValueError("an Actor box must contain exactly eight corners")
    return tuple(
        BoxTriangle3D(
            face_name=face_name,
            corner_indices=indices,
            vertices=tuple(corners[index] for index in indices),
        )
        for face_name, indices in BOX_TRIANGLE_INDEX_TRIPLES
    )
