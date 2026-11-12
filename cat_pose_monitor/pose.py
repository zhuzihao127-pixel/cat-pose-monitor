from dataclasses import dataclass

import cv2
import numpy as np
from rtmlib import RTMPose, YOLOX

from .models import prepare_models


KEYPOINTS = (
    "left_eye", "right_eye", "nose", "neck", "tail_base",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_hind_paw",
    "right_hip", "right_knee", "right_hind_paw",
)

SKELETON = (
    (0, 2), (1, 2), (2, 3), (3, 4),
    (3, 5), (5, 6), (6, 7), (3, 8), (8, 9), (9, 10),
    (4, 11), (11, 12), (12, 13), (4, 14), (14, 15), (15, 16),
)

COLORS = {
    "standing_like": (103, 222, 120),
    "sitting_like": (246, 190, 93),
    "low_posture": (99, 198, 255),
    "uncertain": (195, 195, 195),
}


def estimate_posture(points, visible, box_width):
    """Estimate a coarse side-view posture from shoulder, hip and paw height."""
    postures = set()
    for side in ((5, 7, 11, 13), (8, 10, 14, 16)):
        indices = list(side)
        if not visible[indices].all():
            continue
        shoulder, front_paw, hip, hind_paw = points[indices]
        torso_width = abs(shoulder[0] - hip[0])
        if torso_width < max(1.0, box_width * 0.25):
            continue
        # In a side view of a level surface, the front and rear paws should
        # sit at similar image heights. Reject steep angles and steps.
        if abs(front_paw[1] - hind_paw[1]) > torso_width * 0.85:
            continue

        front = (front_paw[1] - shoulder[1]) / torso_width
        hind = (hind_paw[1] - hip[1]) / torso_width
        if front >= 0.50 and hind >= 0.45:
            postures.add("standing_like")
        elif front >= 0.65 and -0.10 <= hind <= 0.35:
            postures.add("sitting_like")
        elif -0.10 <= front <= 0.37 and -0.10 <= hind <= 0.37:
            postures.add("low_posture")

    if len(postures) == 1:
        return postures.pop()
    return "uncertain"


@dataclass
class CatPose:
    box: np.ndarray
    points: np.ndarray
    scores: np.ndarray
    visible: np.ndarray
    posture: str

    def as_dict(self):
        keypoints = {}
        for name, (x, y), score, visible in zip(
            KEYPOINTS, self.points, self.scores, self.visible
        ):
            keypoints[name] = {
                "x": round(float(x), 2) if visible else None,
                "y": round(float(y), 2) if visible else None,
                "score": round(float(score), 4) if np.isfinite(score) else None,
            }
        return {
            "bbox_xyxy": self.box.round(2).tolist(),
            "posture": self.posture,
            "keypoints": keypoints,
        }


class CatPoseMonitor:
    def __init__(self, model_dir, detection_threshold=0.25, keypoint_threshold=0.35):
        detector_path, pose_path = prepare_models(model_dir)
        self.detector = YOLOX(
            str(detector_path), model_input_size=(416, 416),
            det_mode="multiclass", score_thr=detection_threshold,
        )
        self.estimator = RTMPose(str(pose_path), model_input_size=(256, 256))
        self.cat_class = YOLOX.COCO_CLASSES.index("cat")
        self.keypoint_threshold = keypoint_threshold

    def __call__(self, frame):
        height, width = frame.shape[:2]
        boxes, classes = self.detector(frame)
        if not len(boxes):
            return []
        boxes = boxes[classes == self.cat_class]
        boxes = np.clip(boxes, [0, 0, 0, 0], [width - 1, height - 1] * 2)
        boxes = boxes[((boxes[:, 2:] - boxes[:, :2]) >= 4).all(axis=1)]
        if not len(boxes):
            return []

        poses = []
        # The AP-10K checkpoint uses RGB normalization. YOLOX uses BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        points, scores = self.estimator(rgb, bboxes=boxes)
        for box, joints, confidence in zip(boxes, points, scores):
            visible = (
                np.isfinite(confidence) & (confidence >= self.keypoint_threshold)
                & ((joints >= 0) & (joints < [width, height])).all(axis=1)
            )
            posture = estimate_posture(joints, visible, box[2] - box[0])
            poses.append(CatPose(box, joints, confidence, visible, posture))
        return poses


def annotate(frame, poses):
    canvas = frame.copy()
    scale = max(0.50, min(frame.shape[1] / 1000, 1.25))
    thickness = max(1, round(scale * 2))
    for pose in poses:
        color = COLORS[pose.posture]
        x1, y1, x2, y2 = pose.box.astype(int)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        for first, second in SKELETON:
            if pose.visible[first] and pose.visible[second]:
                start, end = pose.points[[first, second]].astype(int)
                cv2.line(canvas, tuple(start), tuple(end), color, thickness, cv2.LINE_AA)
        for point in pose.points[pose.visible].astype(int):
            cv2.circle(canvas, tuple(point), thickness + 2, (246, 246, 246), -1, cv2.LINE_AA)

        label = pose.posture.replace("_", " ")
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
        top = max(0, y1 - text_height - baseline - 8)
        left = min(x1, max(0, canvas.shape[1] - text_width - 12))
        cv2.rectangle(canvas, (left, top), (left + text_width + 12, top + text_height + baseline + 8), color, -1)
        cv2.putText(canvas, label, (left + 6, top + text_height + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (19, 24, 27), thickness, cv2.LINE_AA)

    if not poses:
        cv2.putText(canvas, "No cat detected", (12, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (240, 240, 240), thickness, cv2.LINE_AA)
    return canvas
