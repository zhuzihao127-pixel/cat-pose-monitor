import math
import unittest
from types import SimpleNamespace

import numpy as np

from cat_pose_monitor.temporal import TemporalMonitor


def cat(x=100, visible=True, posture="low_posture", limb_offset=0):
    points = np.column_stack([np.linspace(x, x + 80, 17), np.linspace(50, 100, 17)])
    points[[6, 7, 9, 10, 12, 13, 15, 16], 1] += limb_offset
    return SimpleNamespace(box=np.array([x, 30, x + 100, 150]), points=points,
                           visible=np.full(17, visible), posture=posture)


def feed(monitor, start, end, fps=10, pose=None):
    result = None
    for index in range(round(start * fps), round(end * fps)):
        result = monitor.update([cat() if pose is None else pose], index / fps, (360, 640))
    return result


class TemporalTests(unittest.TestCase):
    def test_stationary_duration_uses_seconds_across_frame_rates(self):
        durations = []
        for fps in (10, 30):
            monitor = TemporalMonitor()
            result = feed(monitor, 0, 4, fps)
            self.assertEqual(result["state"], "stationary")
            durations.append(result["stationary_s"])
        self.assertLess(abs(durations[0] - durations[1]), 0.15)

    def test_missing_detection_breaks_stationary_timer(self):
        monitor = TemporalMonitor()
        feed(monitor, 0, 3)
        missing = monitor.update([], 3.0, (360, 640))
        self.assertEqual(missing["state"], "no_detection")
        resumed = monitor.update([cat()], 3.1, (360, 640))
        self.assertEqual(resumed["stationary_s"], 0)
        self.assertEqual(resumed["low_posture_s"], 0)
        self.assertEqual(resumed["state"], "unknown")
        self.assertAlmostEqual(monitor.active.observed_seconds, 2.9)

    def test_long_gap_closes_at_last_seen_and_opens_new_episode(self):
        monitor = TemporalMonitor()
        feed(monitor, 0, 2)
        monitor.update([], 3.0, (360, 640))
        result = monitor.update([cat()], 3.1, (360, 640))
        self.assertEqual(len(monitor.episodes), 1)
        self.assertEqual(monitor.episodes[0]["last_seen_s"], 1.9)
        self.assertEqual(monitor.episodes[0]["end_reason"], "visibility_lost")
        self.assertEqual(result["episode_id"], 2)

    def test_clip_boundary_and_outside_roi_are_distinct(self):
        monitor = TemporalMonitor(roi=(0, 0, 0.5, 1))
        feed(monitor, 0, 2)
        result = monitor.update([cat(x=400)], 2.0, (360, 640))
        self.assertEqual(result["state"], "outside_roi")
        self.assertIsNotNone(monitor.active)
        monitor.update([cat(x=400)], 2.5, (360, 640))
        self.assertEqual(monitor.episodes[0]["end_reason"], "left_roi")
        self.assertTrue(monitor.episodes[0]["starts_at_recording_boundary"])
        feed(monitor, 3, 5)
        monitor.finish()
        self.assertEqual(monitor.episodes[-1]["end_reason"], "recording_end")

    def test_brief_roi_boundary_crossing_retains_episode(self):
        monitor = TemporalMonitor(roi=(0, 0, 0.5, 1))
        feed(monitor, 0, 2)
        monitor.update([cat(x=400)], 2.0, (360, 640))
        result = monitor.update([cat()], 2.1, (360, 640))
        self.assertEqual(result["episode_id"], 1)
        self.assertEqual(len(monitor.episodes), 0)

    def test_multiple_cats_interrupt_single_cat_episode(self):
        monitor = TemporalMonitor()
        feed(monitor, 0, 2)
        result = monitor.update([cat(), cat(300)], 2.0, (360, 640))
        self.assertEqual(result["state"], "multiple_cats")
        self.assertIsNone(monitor.active)
        self.assertEqual(monitor.episodes[0]["end_reason"], "multiple_cats")

    def test_low_visibility_is_unknown(self):
        monitor = TemporalMonitor()
        result = feed(monitor, 0, 3, pose=cat(visible=False))
        self.assertEqual(result["state"], "unknown")
        self.assertEqual(result["stationary_s"], 0)

    def test_cyclic_motion_is_detected_when_endpoints_match(self):
        monitor = TemporalMonitor()
        for frame in range(11):
            pose = cat(limb_offset=40 * math.sin(2 * math.pi * frame / 10))
            result = monitor.update([pose], frame / 10, (360, 640))
        self.assertEqual(result["state"], "body_movement")

    def test_time_reversal_is_rejected(self):
        monitor = TemporalMonitor()
        monitor.update([cat()], 1.0, (360, 640))
        with self.assertRaises(ValueError):
            monitor.update([cat()], 0.9, (360, 640))


if __name__ == "__main__":
    unittest.main()
