from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Sequence, overload

import cv2
import numpy as np


def _normalize_frame_indices(frame_indices: int | Iterable[int]) -> list[int]:
    if isinstance(frame_indices, int):
        indices = [frame_indices]
    else:
        indices = [int(frame_idx) for frame_idx in frame_indices]

    if not indices:
        raise ValueError("frame_indices must contain at least one frame index")
    if any(frame_idx < 0 for frame_idx in indices):
        raise ValueError("frame_indices must be non-negative")
    return indices


def _read_frame(capture: cv2.VideoCapture, frame_idx: int) -> np.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    if ok and frame is not None:
        return frame

    if frame_idx > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
    return None


def _normalize_output_names(
    frame_count: int,
    file_names: Sequence[str | Path] | None,
) -> list[str]:
    if file_names is None:
        return [str(idx) for idx in range(frame_count)]

    if len(file_names) != frame_count:
        raise ValueError("file_names must have the same length as frames")

    names = [Path(str(file_name)).stem for file_name in file_names]
    if any(not name for name in names):
        raise ValueError("file_names must not contain empty names")
    return names


def save_frames_with_predictions(
    frames: Sequence[np.ndarray],
    detection_boxes: Sequence[Sequence[dict[str, Any]]],
    keypoints: Sequence[Sequence[dict[str, Any]]],
    output_folder: str | Path,
    file_names: Sequence[str | Path] | None = None,
) -> list[Path]:
    if len(frames) != len(detection_boxes) or len(frames) != len(keypoints):
        raise ValueError("frames, detection_boxes, and keypoints must have the same length")

    output_dir = Path(output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = _normalize_output_names(len(frames), file_names)
    saved_paths: list[Path] = []

    for frame_idx, (frame, detections, pose_results, name) in enumerate(zip(frames, detection_boxes, keypoints, names)):
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Frame at index {frame_idx} must have shape (H, W, 3)")

        canvas = np.ascontiguousarray(frame.copy())

        for detection in detections:
            bbox = detection.get("bbox")
            if bbox is None or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
            if "score" in detection:
                label = f"det {float(detection['score']):.2f}"
                cv2.putText(
                    canvas,
                    label,
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        for pose_result in pose_results:
            bbox = pose_result.get("bbox")
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 1)

            keypoint_map = pose_result.get("keypoints") or {}
            if not isinstance(keypoint_map, dict):
                continue

            for keypoint_name, point in keypoint_map.items():
                if not isinstance(point, dict):
                    continue
                x = point.get("x")
                y = point.get("y")
                if x is None or y is None:
                    continue
                center = (int(round(float(x))), int(round(float(y))))
                cv2.circle(canvas, center, 3, (0, 255, 0), -1)
                cv2.putText(
                    canvas,
                    str(keypoint_name),
                    (center[0] + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        output_path = output_dir / f"{name}.jpg"
        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"Failed to write visualization image to {output_path}")
        saved_paths.append(output_path)

    return saved_paths



def extract_frames(
    video_path: str | Path,
    frame_indices: int | Iterable[int],
    output_dir: str | Path | None = None,
    overwrite: bool = False,
) -> list[Path] | list[np.ndarray]:
    video_path = Path(video_path)
    indices = _normalize_frame_indices(frame_indices)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    if resolved_output_dir is not None:
        resolved_output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames: list[np.ndarray] = []
    saved_paths: list[Path] = []
    try:
        for frame_idx in indices:
            frame = _read_frame(capture, frame_idx)
            if frame is None:
                raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

            if resolved_output_dir is None:
                frames.append(frame)
                continue

            output_path = resolved_output_dir / f"{frame_idx:08d}.jpg"
            if output_path.exists() and not overwrite:
                saved_paths.append(output_path)
                continue

            if not cv2.imwrite(str(output_path), frame):
                raise RuntimeError(f"Failed to write frame to {output_path}")
            saved_paths.append(output_path)
    finally:
        capture.release()

    if resolved_output_dir is None:
        return frames
    return saved_paths
