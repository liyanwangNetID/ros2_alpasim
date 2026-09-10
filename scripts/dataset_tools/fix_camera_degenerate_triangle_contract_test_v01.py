#!/usr/bin/env python3
from __future__ import annotations
import ast
from pathlib import Path

PATH = Path("test_camera_triangle_depth_samples_v01.py")
FUNCTION = "test_degenerate_projected_triangle_is_rejected_by_sampler"
REPLACEMENT = """def test_degenerate_projected_triangle_is_rejected_by_sampler():
    class DegenerateProjectionCalibration:
        def project_camera_point(self, point):
            class Projection:
                pass

            result = Projection()
            result.u = point.x * 10.0
            result.v = point.x * 10.0
            result.positive_z = True
            result.within_fov = True
            result.valid = True
            return result

    result = sample_camera_triangle_depths(
        (
            Vector3(1.0, 0.0, 10.0),
            Vector3(2.0, 0.0, 10.0),
            Vector3(3.0, 0.0, 10.0),
        ),
        DegenerateProjectionCalibration(),
        image_width_px=100,
        image_height_px=100,
        raster_width=50,
        raster_height=50,
    )

    assert result.center_sampled_depths == ()
    assert result.center_sampled_cell_count == 0
"""

text = PATH.read_text(encoding="utf-8")
tree = ast.parse(text)
node = next(
    item for item in tree.body
    if isinstance(item, ast.FunctionDef) and item.name == FUNCTION
)
lines = text.splitlines(keepends=True)
updated = (
    "".join(lines[:node.lineno - 1])
    + REPLACEMENT.rstrip()
    + "\n"
    + "".join(lines[node.end_lineno:])
)
compile(updated, str(PATH), "exec")
PATH.write_text(updated, encoding="utf-8")
print("updated:", PATH)
