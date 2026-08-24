#!/usr/bin/env python3
from __future__ import annotations
import tempfile
import sys
import unittest
from pathlib import Path

MODULE_DIRECTORY = Path(__file__).resolve().parent.parent
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))

from review_lateral_action import (
    TimestampedImage,
    discover_front_wide_images,
    extract_timestamp_ns,
    frame_duration_seconds,
    select_window,
    write_concat_file,
)


class ReviewUtilityTests(unittest.TestCase):
    def test_extract_timestamp(self):
        self.assertEqual(extract_timestamp_ns(Path("frame_1234567890123.jpg")), 1234567890123)
        self.assertIsNone(extract_timestamp_ns(Path("frame.jpg")))

    def test_discovery_and_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = root / "camera" / "front_wide"
            camera.mkdir(parents=True)
            for stamp in (100, 200, 300):
                (camera / f"{stamp:010d}.jpg").write_bytes(b"x")
            (root / "camera" / "other").mkdir()
            (root / "camera" / "other" / "0000000250.jpg").write_bytes(b"x")
            images = discover_front_wide_images(root)
            self.assertEqual([item.stamp_ns for item in images], [100, 200, 300])
            selected = select_window(images, 150, 300)
            self.assertEqual([item.stamp_ns for item in selected], [200, 300])

    def test_duration_bounds(self):
        self.assertAlmostEqual(frame_duration_seconds(0, 100_000_000, 10.0), 0.1)
        self.assertAlmostEqual(frame_duration_seconds(0, None, 10.0), 0.1)
        self.assertEqual(frame_duration_seconds(0, 5_000_000_000, 10.0), 1.0)

    def test_concat_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "a.jpg"
            second = root / "b.jpg"
            first.write_bytes(b"x")
            second.write_bytes(b"x")
            output = root / "frames.ffconcat"
            write_concat_file(
                (
                    TimestampedImage(0, first),
                    TimestampedImage(100_000_000, second),
                ),
                output,
                10.0,
            )
            text = output.read_text(encoding="utf-8")
            self.assertIn("ffconcat version 1.0", text)
            self.assertEqual(text.count("file '"), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
