#!/usr/bin/env python3
import unittest


def threshold(level: str) -> float:
    return 8.0 if level in {"A", "B"} else 12.0


class JunctionResidualThresholdTests(unittest.TestCase):
    def test_level_a_uses_eight_degrees(self):
        self.assertEqual(threshold("A"), 8.0)

    def test_level_b_uses_eight_degrees(self):
        self.assertEqual(threshold("B"), 8.0)

    def test_level_c_keeps_twelve_degrees(self):
        self.assertEqual(threshold("C"), 12.0)

    def test_known_case_separation(self):
        self.assertGreaterEqual(8.98, threshold("A"))
        self.assertLess(2.14, threshold("C"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
