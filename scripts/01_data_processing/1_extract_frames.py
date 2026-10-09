# This script extracts frames from videos based on frame indices specified in label JSON files.
from __future__ import annotations

import json
import sys
from pathlib import Path


if "__file__" in globals():
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
else:
    PROJECT_ROOT = Path.cwd()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.videos import extract_frames


# Inputs
label_dir = PROJECT_ROOT / "input" / "labels"
video_dir = PROJECT_ROOT / "videos"
output_dir = PROJECT_ROOT / "output" / "extracted_frames"
label_names = None  # Example: ["ai1.json"]
overwrite = False

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")




def to_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def read_frame_indices(label_path: Path) -> list[int]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {label_path} to contain a list, got {type(payload).__name__}")
    frame_indices = sorted({int(item["frame_idx"]) for item in payload})
    if not frame_indices:
        raise ValueError(f"No frame indices found in {label_path}")
    return frame_indices


def find_video_path(video_root: Path, video_name: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = video_root / f"{video_name}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find a video for '{video_name}' under {video_root}")


def process_label_file(label_path: Path, video_root: Path, output_root: Path) -> int:
    video_name = label_path.stem
    video_path = find_video_path(video_root, video_name)
    frame_indices = read_frame_indices(label_path)
    target_dir = output_root / video_name
    saved_paths = extract_frames(video_path, frame_indices, target_dir, overwrite=overwrite)
    return len(saved_paths)


label_dir = to_repo_path(label_dir)
video_dir = to_repo_path(video_dir)
output_dir = to_repo_path(output_dir)

if not label_dir.is_dir():
    raise FileNotFoundError(f"Label directory not found: {label_dir}")
if not video_dir.is_dir():
    raise FileNotFoundError(f"Video directory not found: {video_dir}")

label_paths = sorted(path for path in label_dir.glob("*.json") if path.is_file())
if label_names:
    selected = {name if name.endswith(".json") else f"{name}.json" for name in label_names}
    label_paths = [path for path in label_paths if path.name in selected]
if not label_paths:
    raise FileNotFoundError(f"No label JSON files found in {label_dir}")

total_saved = 0
failures = []
for label_path in label_paths:
    try:
        saved = process_label_file(label_path, video_dir, output_dir)
        total_saved += saved
        print(f"[ok] {label_path.stem}: saved {saved}")
    except Exception as exc:  # noqa: BLE001
        failures.append((label_path.name, str(exc)))
        print(f"[error] {label_path.name}: {exc}")

print(f"Finished. Processed {len(label_paths)} label files, saved {total_saved} frames.")
if failures:
    raise SystemExit("Some files failed:\n" + "\n".join(f"- {name}: {message}" for name, message in failures))
