"""Report unique labeled frames per video, grouped by source-video existence."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
label_dir = PROJECT_ROOT / "input" / "labels"
video_dir = PROJECT_ROOT / "videos"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def is_labeled_keypoint(point: object) -> bool:
    """Require visibility and a complete, finite coordinate pair."""
    if not isinstance(point, (list, tuple)) or len(point) != 3 or point[0] != 1:
        return False
    return all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in point[1:]
    )


def count_label_frames(label_path: Path) -> dict[str, int]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of frames in {label_path}")
    frames: set[int] = set()
    labeled: set[int] = set()
    for item in payload:
        frame_idx = item["frame_idx"]
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
            raise ValueError(f"Invalid frame index {frame_idx!r} in {label_path}")
        labels = item.get("labels", {})
        if not isinstance(labels, dict):
            raise ValueError(f"Expected keypoint mapping for frame {frame_idx} in {label_path}")
        frames.add(frame_idx)
        if any(is_labeled_keypoint(point) for point in labels.values()):
            labeled.add(frame_idx)
    return {
        "total_frames": len(frames),
        "labeled_frames": len(labeled),
        "empty_frames": len(frames - labeled),
    }


def build_frame_report(label_root: Path, video_root: Path) -> pd.DataFrame:
    """Include every label JSON, even when its source video is absent.

    total_frames refers to unique indices in the JSON, not the video duration.
    Duplicate indices count once; any labeled entry makes that frame labeled.
    """
    label_paths = sorted(Path(label_root).glob("*.json"))
    if not label_paths:
        raise FileNotFoundError(f"No label JSON files found in {label_root}")
    video_names = {
        path.stem for path in Path(video_root).iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    } if Path(video_root).is_dir() else set()
    return pd.DataFrame([
        {
            "video": path.stem,
            "video_exists": path.stem in video_names,
            **count_label_frames(path),
        }
        for path in label_paths if path.is_file()
    ])


def summarize_frame_report(report: pd.DataFrame) -> pd.DataFrame:
    """Describe labeled-frame counts across videos in each existence group."""
    groups = [("all", report)] + [
        (name, report.loc[report["video_exists"] == exists])
        for name, exists in (("exists", True), ("missing", False))
    ]
    rows = []
    for name, group in groups:
        counts = group["labeled_frames"]
        stats = counts.describe()
        rows.append({
            "source_video": name,
            "videos": len(group),
            "total_labeled": int(counts.sum()),
            "total_empty": int(group["empty_frames"].sum()),
            **{key: stats[key] for key in ("mean", "std", "min", "25%", "50%", "75%", "max")},
        })
    return pd.DataFrame(rows).set_index("source_video")


report = build_frame_report(label_dir, video_dir)
summary = summarize_frame_report(report)
print("A labeled frame has at least one visible keypoint with finite x/y coordinates.")
print("Frame counts refer to unique indices in the label JSON, not all video frames.")
for exists, name in ((True, "Source video exists"), (False, "Source video missing")):
    print(f"\n{name}:")
    group = report.loc[report["video_exists"] == exists].drop(columns="video_exists")
    print(group.to_string(index=False) if not group.empty else "(none)")
print("\nDescriptive statistics for labeled frames per video (50% = median):")
print(summary.to_string(float_format=lambda value: f"{value:.2f}"))
