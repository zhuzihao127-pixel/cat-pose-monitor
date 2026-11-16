"""Time-based, single-cat visual monitoring for a fixed camera."""

from collections import Counter, deque
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Episode:
    number: int
    start: float
    last_seen: float
    starts_at_recording_boundary: bool
    observed_seconds: float = 0.0
    states: Counter = field(default_factory=Counter)
    longest_stationary_seconds: float = 0.0
    longest_low_posture_seconds: float = 0.0

    def as_dict(self, reason):
        review_reasons = []
        if self.longest_stationary_seconds >= 3.0:
            review_reasons.append("sustained_stationary")
        if self.longest_low_posture_seconds >= 2.0:
            review_reasons.append("sustained_low_posture")
        if self.states["unknown"] > self.observed_seconds * 0.5:
            review_reasons.append("limited_motion_evidence")
        return {
            "episode_id": self.number,
            "start_s": round(self.start, 3),
            "last_seen_s": round(self.last_seen, 3),
            "span_s": round(self.last_seen - self.start, 3),
            "observed_s": round(self.observed_seconds, 3),
            "starts_at_recording_boundary": self.starts_at_recording_boundary,
            "end_reason": reason,
            "state_durations_s": {key: round(value, 3) for key, value in self.states.items()},
            "longest_stationary_s": round(self.longest_stationary_seconds, 3),
            "longest_low_posture_s": round(self.longest_low_posture_seconds, 3),
            "review_reasons": review_reasons,
        }


class TemporalMonitor:
    def __init__(self, roi=(0.0, 0.0, 1.0, 1.0), window_seconds=1.0,
                 lost_seconds=1.0, min_episode_seconds=0.5):
        self.roi = tuple(roi)
        self.window_seconds = window_seconds
        self.lost_seconds = lost_seconds
        self.min_episode_seconds = min_episode_seconds
        self.history = deque()
        self.episodes = []
        self.active = None
        self.previous_time = None
        self.outside_since = None
        self.previous_status = "unknown"
        self.stationary_seconds = 0.0
        self.low_seconds = 0.0
        self.previous_low = False

    def _clear_motion(self):
        self.history.clear()
        self.previous_status = "unknown"
        self.stationary_seconds = 0.0
        self.low_seconds = 0.0
        self.previous_low = False

    def finish(self, reason="recording_end"):
        if self.active and self.active.observed_seconds >= self.min_episode_seconds:
            self.episodes.append(self.active.as_dict(reason))
        self.active = None
        self.outside_since = None
        self._clear_motion()

    def _motion_state(self, pose, timestamp):
        self.history.append((timestamp, pose))
        while self.history and timestamp - self.history[0][0] > self.window_seconds:
            self.history.popleft()
        old_time, old_pose = self.history[0]
        duration = timestamp - old_time
        common = pose.visible & old_pose.visible
        core = np.array([3, 4, 5, 8, 11, 14])
        common_core = core[common[core]]
        if duration < self.window_seconds * 0.6 or common.sum() < 6 or len(common_core) < 2:
            return "unknown", "uncertain", None

        scale = max(float(np.linalg.norm(pose.box[2:] - pose.box[:2])), 1.0)
        body_displacements, joint_displacements = [], []
        for _, sample in self.history:
            valid = common & sample.visible
            valid_core = core[valid[core]]
            if valid.sum() < 6 or len(valid_core) < 2:
                continue
            displacement = sample.points - old_pose.points
            body_shift = np.median(displacement[valid_core], axis=0)
            body_displacements.append(np.linalg.norm(body_shift))
            joint_displacements.append(np.percentile(
                np.linalg.norm(displacement[valid] - body_shift, axis=1), 75
            ))
        if len(body_displacements) < max(3, len(self.history) // 2):
            return "unknown", "uncertain", None
        # Include interior frames so a repeated motion that returns to its
        # starting position is still measured as activity.
        body_speed = float(np.percentile(body_displacements, 75) / scale / duration)
        articulation = float(np.percentile(joint_displacements, 75) / scale / duration)
        # Scores describe image motion in box diagonals per second. They depend
        # on the fixed camera, pose noise, and view direction.
        if body_speed > 0.08:
            state = "moving"
        elif articulation > 0.05:
            state = "body_movement"
        else:
            state = "stationary"

        votes = Counter(sample.posture for _, sample in self.history)
        posture, count = votes.most_common(1)[0]
        if count / len(self.history) < 0.7 or pose.posture == "uncertain":
            posture = "uncertain"
        return state, posture, {"body_speed": round(body_speed, 4),
                                "articulation_speed": round(articulation, 4)}

    def update(self, poses, timestamp, shape):
        if self.previous_time is not None and timestamp <= self.previous_time:
            raise ValueError("Monitoring timestamps must strictly increase")
        delta = 0.0 if self.previous_time is None else timestamp - self.previous_time
        first_frame = self.previous_time is None
        self.previous_time = timestamp
        height, width = shape[:2]
        left, top, right, bottom = np.array(self.roi) * [width, height, width, height]
        inside = []
        for pose in poses:
            x, y = (pose.box[:2] + pose.box[2:]) / 2
            if left <= x <= right and top <= y <= bottom:
                inside.append(pose)

        state = "no_detection" if not poses else "outside_roi"
        result = {"state": state, "posture": "uncertain", "episode_id": None,
                  "episode_elapsed_s": None,
                  "stationary_s": 0.0, "low_posture_s": 0.0, "motion": None}
        if len(poses) > 1:
            self.finish("multiple_cats")
            result["state"] = "multiple_cats"
            return result
        if not inside:
            self._clear_motion()
            if self.active and poses:
                if self.outside_since is None:
                    self.outside_since = timestamp
                if timestamp - self.outside_since >= 0.4:
                    self.finish("left_roi")
            else:
                self.outside_since = None
                if self.active and timestamp - self.active.last_seen >= self.lost_seconds:
                    self.finish("visibility_lost")
            return result

        self.outside_since = None
        pose = inside[0]
        consecutive = bool(self.history) and delta <= 0.5
        if self.active and timestamp - self.active.last_seen >= self.lost_seconds:
            self.finish("visibility_lost")
            consecutive = False
        if consecutive:
            previous_pose = self.history[-1][1]
            old_center = (previous_pose.box[:2] + previous_pose.box[2:]) / 2
            center = (pose.box[:2] + pose.box[2:]) / 2
            scale = max(float(np.linalg.norm(previous_pose.box[2:] - previous_pose.box[:2])), 1)
            if np.linalg.norm(center - old_center) > scale * 0.75:
                self.finish("tracking_discontinuity")
                consecutive = False
        if not consecutive:
            self._clear_motion()
        if self.active is None:
            self.active = Episode(len(self.episodes) + 1, timestamp, timestamp, first_frame)
        self.active.last_seen = timestamp

        state, posture, motion = self._motion_state(pose, timestamp)
        continuous_stationary = consecutive and state == self.previous_status == "stationary"
        self.stationary_seconds = self.stationary_seconds + delta if continuous_stationary else 0.0
        low_posture = state == "stationary" and posture == "low_posture"
        self.low_seconds = self.low_seconds + delta if consecutive and low_posture and self.previous_low else 0.0
        self.previous_low = low_posture
        if consecutive:
            self.active.observed_seconds += delta
            # A transition interval has uncertain state until both endpoints agree.
            interval_state = state if state == self.previous_status else "unknown"
            self.active.states[interval_state] += delta
        self.previous_status = state
        self.active.longest_stationary_seconds = max(
            self.active.longest_stationary_seconds, self.stationary_seconds
        )
        self.active.longest_low_posture_seconds = max(
            self.active.longest_low_posture_seconds, self.low_seconds
        )
        result.update(state=state, posture=posture, episode_id=self.active.number,
                      episode_elapsed_s=round(timestamp - self.active.start, 3),
                      stationary_s=round(self.stationary_seconds, 3),
                      low_posture_s=round(self.low_seconds, 3), motion=motion)
        return result


def draw_monitor_panel(canvas, observation, roi):
    height, width = canvas.shape[:2]
    start, end = (np.array(roi) * [width, height, width, height]).astype(int).reshape(2, 2)
    end = np.minimum(end, [width - 1, height - 1])
    cv2.rectangle(canvas, tuple(start), tuple(end), (239, 191, 91), 1)
    panel = np.full((height, 270, 3), (28, 24, 22), dtype=np.uint8)
    episode = observation["episode_id"]
    lines = [
        "VISUAL MONITOR",
        f"ROI episode {episode}" if episode else "Waiting for cat",
        "Motion " + observation["state"].replace("_", " "),
        "Pose " + observation["posture"].replace("_", " "),
        f"Observed span {observation['episode_elapsed_s']:.1f} s" if episode else "Awaiting observation",
        f"Stationary {observation['stationary_s']:.1f} s",
        f"Low posture {observation['low_posture_s']:.1f} s",
        "Visual evidence only",
    ]
    spacing = min(34, height / 9)
    font_scale = min(0.55, height / 650)
    for index, label in enumerate(lines):
        color = (239, 191, 91) if index == 2 else (230, 230, 230)
        cv2.putText(panel, label, (12, round((index + 1) * spacing)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA)
    return np.concatenate([canvas, panel], axis=1)
