from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest

from alpasim_bridge.static_occupancy_server import decode_static_occupancy


def packet():
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        grid_i=np.asarray([0, 1], dtype=np.int32),
        grid_j=np.asarray([2, 3], dtype=np.int32),
        min_z=np.asarray([-1.0, 0.0], dtype=np.float32),
        max_z=np.asarray([1.0, 2.0], dtype=np.float32),
        point_count=np.asarray([4, 5], dtype=np.uint32),
        flags=np.asarray([1, 2], dtype=np.uint16),
    )
    payload = stream.getvalue()
    header = {
        "message_type": "static_occupancy",
        "schema_version": "0.1",
        "scene_id": "scene",
        "scene_uuid": "uuid",
        "frame_id": "map",
        "representation": "single_interval_2p5d_static_occupancy",
        "semantic_labels": False,
        "grid_size_xy_m": 0.4,
        "column_count": 2,
        "payload_size": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return json.dumps(header).encode(), payload


def test_decode_valid_static_occupancy():
    header, payload = packet()
    result = decode_static_occupancy(header, payload)
    assert result.header["column_count"] == 2
    assert result.arrays["grid_i"].tolist() == [0, 1]


def test_decode_rejects_checksum_mismatch():
    header, payload = packet()
    source = json.loads(header)
    source["payload_sha256"] = "bad"
    with pytest.raises(ValueError, match="checksum"):
        decode_static_occupancy(json.dumps(source).encode(), payload)
