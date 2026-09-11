from __future__ import annotations

import math

import pytest

from triangle_barycentric_coordinates_v01 import (
    Point2D,
    triangle_barycentric_coordinates,
)


def triangle():
    return (
        Point2D(0.0, 0.0),
        Point2D(4.0, 0.0),
        Point2D(0.0, 4.0),
    )


def test_vertices_have_unit_basis_weights():
    source = triangle()
    expected = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    for point, weights in zip(source, expected):
        result = triangle_barycentric_coordinates(source, point)
        assert (
            result.first_weight,
            result.second_weight,
            result.third_weight,
        ) == pytest.approx(weights)
        assert result.inside_closed_triangle


def test_centroid_has_equal_weights():
    result = triangle_barycentric_coordinates(
        triangle(), Point2D(4.0 / 3.0, 4.0 / 3.0)
    )
    assert result.first_weight == pytest.approx(1.0 / 3.0)
    assert result.second_weight == pytest.approx(1.0 / 3.0)
    assert result.third_weight == pytest.approx(1.0 / 3.0)
    assert result.weight_sum == pytest.approx(1.0)
    assert result.inside_closed_triangle


def test_edge_point_is_inside_closed_triangle():
    result = triangle_barycentric_coordinates(
        triangle(), Point2D(2.0, 2.0)
    )
    assert result.first_weight == pytest.approx(0.0)
    assert result.inside_closed_triangle


def test_outside_point_has_negative_weight():
    result = triangle_barycentric_coordinates(
        triangle(), Point2D(3.0, 3.0)
    )
    assert min(
        result.first_weight,
        result.second_weight,
        result.third_weight,
    ) < 0.0
    assert not result.inside_closed_triangle
    assert result.weight_sum == pytest.approx(1.0)


def test_reversed_winding_preserves_geometric_weights_for_matching_vertices():
    source = triangle()
    point = Point2D(1.0, 1.0)
    forward = triangle_barycentric_coordinates(source, point)
    reverse = triangle_barycentric_coordinates(
        (source[0], source[2], source[1]), point
    )
    assert reverse.first_weight == pytest.approx(forward.first_weight)
    assert reverse.second_weight == pytest.approx(forward.third_weight)
    assert reverse.third_weight == pytest.approx(forward.second_weight)
    assert reverse.inside_closed_triangle == forward.inside_closed_triangle


def test_affine_reconstruction_matches_sample_point():
    source = triangle()
    point = Point2D(0.75, 1.25)
    result = triangle_barycentric_coordinates(source, point)
    reconstructed_x = sum(
        weight * vertex.x
        for weight, vertex in zip(
            (result.first_weight, result.second_weight, result.third_weight),
            source,
        )
    )
    reconstructed_y = sum(
        weight * vertex.y
        for weight, vertex in zip(
            (result.first_weight, result.second_weight, result.third_weight),
            source,
        )
    )
    assert reconstructed_x == pytest.approx(point.x)
    assert reconstructed_y == pytest.approx(point.y)


def test_collinear_triangle_is_rejected():
    with pytest.raises(ValueError, match="non-degenerate"):
        triangle_barycentric_coordinates(
            (
                Point2D(0.0, 0.0),
                Point2D(1.0, 1.0),
                Point2D(2.0, 2.0),
            ),
            Point2D(1.0, 1.0),
        )


def test_coincident_triangle_is_rejected():
    point = Point2D(1.0, 1.0)
    with pytest.raises(ValueError, match="non-degenerate"):
        triangle_barycentric_coordinates((point, point, point), point)


@pytest.mark.parametrize("count", (0, 2, 4))
def test_wrong_triangle_size_is_rejected(count):
    source = tuple(Point2D(float(index), 0.0) for index in range(count))
    with pytest.raises(ValueError, match="exactly three"):
        triangle_barycentric_coordinates(source, Point2D(0.0, 0.0))


@pytest.mark.parametrize("epsilon", (0.0, -1.0, math.nan, math.inf))
def test_invalid_degeneracy_epsilon_is_rejected(epsilon):
    with pytest.raises(ValueError, match="degeneracy_epsilon"):
        triangle_barycentric_coordinates(
            triangle(),
            Point2D(1.0, 1.0),
            degeneracy_epsilon=epsilon,
        )


def test_nonfinite_triangle_coordinate_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        triangle_barycentric_coordinates(
            (
                Point2D(math.nan, 0.0),
                Point2D(1.0, 0.0),
                Point2D(0.0, 1.0),
            ),
            Point2D(0.0, 0.0),
        )


def test_nonfinite_sample_coordinate_is_rejected():
    with pytest.raises(ValueError, match="finite"):
        triangle_barycentric_coordinates(
            triangle(), Point2D(math.inf, 0.0)
        )


def test_large_translated_triangle_remains_valid():
    source = (
        Point2D(1_000_000.0, 1_000_000.0),
        Point2D(1_000_004.0, 1_000_000.0),
        Point2D(1_000_000.0, 1_000_004.0),
    )
    result = triangle_barycentric_coordinates(
        source, Point2D(1_000_001.0, 1_000_001.0)
    )
    assert result.inside_closed_triangle
    assert result.weight_sum == pytest.approx(1.0)
