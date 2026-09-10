#!/usr/bin/env python3
from __future__ import annotations
import ast
from pathlib import Path

REPLACEMENTS = {
    "test_projected_triangle_depth_samples_v01.py": """def test_degenerate_projected_triangle_is_rejected():
    result = sample_projected_triangle_depths(
        (
            PixelPoint(10.0, 10.0),
            PixelPoint(20.0, 20.0),
            PixelPoint(30.0, 30.0),
        ),
        (10.0, 11.0, 12.0),
        image_width_px=100,
        image_height_px=100,
        raster_width=50,
        raster_height=50,
    )

    assert result.center_sampled_depths == ()
    assert result.center_sampled_cell_count == 0
""",
    "test_camera_triangle_depth_samples_v01.py": """def test_degenerate_projected_triangle_is_rejected_by_sampler():
    triangle = (
        Vector3(1.0, 0.0, 10.0),
        Vector3(2.0, 0.0, 10.0),
        Vector3(3.0, 0.0, 10.0),
    )
    calibration = StubCalibration(
        {
            triangle[0]: PixelProjection(10.0, 10.0, True, True, True),
            triangle[1]: PixelProjection(20.0, 20.0, True, True, True),
            triangle[2]: PixelProjection(30.0, 30.0, True, True, True),
        }
    )
    result = sample_camera_triangle_depths(
        triangle,
        calibration,
        image_width_px=100,
        image_height_px=100,
        raster_width=50,
        raster_height=50,
    )

    assert result.center_sampled_depths == ()
    assert result.center_sampled_cell_count == 0
""",
}

FUNCTIONS = {
    "test_projected_triangle_depth_samples_v01.py": "test_degenerate_projected_triangle_is_rejected",
    "test_camera_triangle_depth_samples_v01.py": "test_degenerate_projected_triangle_is_rejected_by_sampler",
}

for filename, replacement in REPLACEMENTS.items():
    path = Path(filename)
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == FUNCTIONS[filename]
    )
    lines = text.splitlines(keepends=True)
    updated = "".join(lines[:node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])
    compile(updated, str(path), "exec")
    path.write_text(updated, encoding="utf-8")
    print("updated:", path)
