import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import cv2

from .pose import CatPoseMonitor, annotate


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WINDOW = "Cat Pose Monitor"


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor cat posture in a camera feed, video or image.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path, help="Local image or video file")
    inputs.add_argument("--camera", type=int, help="Camera index, usually 0")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Model cache directory")
    parser.add_argument("--output", type=Path, help="Annotated image or .mp4 video")
    parser.add_argument("--jsonl", type=Path, help="One record per input frame")
    parser.add_argument("--no-display", action="store_true", help="Run without a preview window")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    parser.add_argument("--keypoint-threshold", type=float, default=0.35)
    args = parser.parse_args()

    if args.source is not None and not args.source.is_file():
        parser.error(f"Input file does not exist: {args.source}")
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be 0 or greater")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if not 0 <= args.detection_threshold <= 1:
        parser.error("--detection-threshold must be between 0 and 1")
    if not math.isfinite(args.keypoint_threshold) or args.keypoint_threshold < 0:
        parser.error("--keypoint-threshold must be finite and nonnegative")
    if (not args.no_display and sys.platform.startswith("linux")
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
        parser.error("No graphical display detected. Use --no-display.")

    paths = [path.resolve() for path in (args.source, args.output, args.jsonl) if path is not None]
    if len(paths) != len(set(paths)):
        parser.error("Input, annotated output and JSONL must use distinct paths")

    args.image = args.source is not None and args.source.suffix.lower() in IMAGE_SUFFIXES
    valid_output = IMAGE_SUFFIXES if args.image else {".mp4"}
    if args.output and args.output.suffix.lower() not in valid_output:
        parser.error(f"Output extension must be one of: {', '.join(sorted(valid_output))}")
    return args


def record_frame(stream, index, timestamp, poses):
    if stream is not None:
        record = {
            "frame_index": index,
            "timestamp_s": round(timestamp, 6),
            "cats": [pose.as_dict() for pose in poses],
        }
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def run_image(args, monitor, journal):
    image = cv2.imread(str(args.source))
    if image is None:
        raise RuntimeError(f"Could not decode image: {args.source}")

    poses = monitor(image)
    record_frame(journal, 0, 0.0, poses)
    if args.output or not args.no_display:
        canvas = annotate(image, poses)
        if args.output and not cv2.imwrite(str(args.output), canvas):
            raise RuntimeError(f"Could not write image: {args.output}")
        if not args.no_display:
            cv2.imshow(WINDOW, canvas)
            cv2.waitKey(0)
    return 1, Counter(pose.posture for pose in poses)


def run_video(args, monitor, journal):
    camera = args.camera is not None
    source = args.camera if camera else str(args.source)
    capture = cv2.VideoCapture(source)
    writer = None
    frame_index = 0
    counts = Counter()
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open {'camera' if camera else 'video'}: {source}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0

        first_frame_at = None
        while args.max_frames is None or frame_index < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                if camera:
                    raise RuntimeError(f"Camera {source} stopped returning frames")
                break

            now = time.monotonic()
            if first_frame_at is None:
                first_frame_at = now
            timestamp = now - first_frame_at if camera else frame_index / fps
            poses = monitor(frame)
            record_frame(journal, frame_index, timestamp, poses)
            counts.update(pose.posture for pose in poses)
            frame_index += 1

            if args.output or not args.no_display:
                canvas = annotate(frame, poses)
                if args.output:
                    if writer is None:
                        height, width = canvas.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                            fps, (width, height),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open video output: {args.output}")
                    writer.write(canvas)
                if not args.no_display:
                    cv2.imshow(WINDOW, canvas)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

        if not frame_index:
            raise RuntimeError(f"No readable frames in {source}")
        return frame_index, counts
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main():
    args = parse_args()
    try:
        monitor = CatPoseMonitor(args.model_dir, args.detection_threshold, args.keypoint_threshold)
        for path in (args.output, args.jsonl):
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
        journal = args.jsonl.open("w", encoding="utf-8", buffering=1) if args.jsonl else nullcontext()
        with journal as stream:
            runner = run_image if args.image else run_video
            frames, counts = runner(args, monitor, stream)
        print(json.dumps({"frames": frames, "posture_counts": dict(counts)}, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        print(f"cat-pose-monitor: {error}", file=sys.stderr)
        return 1
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
