"""Save full-size skeleton overlays and one representative signature per video."""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()

# Inputs and rendering options (editable in an interactive terminal).
label_dir = PROJECT_ROOT / "input" / "labels"
frames_dir = PROJECT_ROOT / "output" / "extracted_frames"
output_dir = PROJECT_ROOT / "output" / "skeleton_frame_overlay"
config_path = PROJECT_ROOT / "config.yaml"
signature_dir = output_dir / "signature"
signature_min_completeness = 0.5
label_names = None  # Example: ["Rat 9 2025-11-11 11-09-35.json"]
show_names = True
point_radius = 3
line_thickness = 2


def visible_point(point: object) -> tuple[int, int] | None:
    """Labels store [visibility, x, y]; ignore absent or invalid points."""
    if not isinstance(point, (list, tuple)) or len(point) != 3 or point[0] != 1:
        return None
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in point[1:]
    ):
        return None
    return int(round(point[1])), int(round(point[2]))


def draw_skeleton(image, labels: dict, skeleton: list):
    canvas = image.copy()
    points = {name: xy for name, value in labels.items() if (xy := visible_point(value)) is not None}
    for left, right in skeleton:
        if left in points and right in points:
            cv2.line(canvas, points[left], points[right], (255, 255, 0), line_thickness, cv2.LINE_AA)
    for name, (x, y) in points.items():
        cv2.circle(canvas, (x, y), point_radius, (0, 255, 0), -1, cv2.LINE_AA)
        if show_names:
            cv2.putText(canvas, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def process_label_file(label_path: Path, skeleton: list) -> dict[str, int]:
    payload = json.loads(label_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list of frames in {label_path}")
    # Validate labels before removing this video's previous overlays.
    for item in payload:
        frame_idx = item["frame_idx"]
        if isinstance(frame_idx, bool) or not isinstance(frame_idx, int) or frame_idx < 0:
            raise ValueError(f"Invalid frame index: {frame_idx!r}")
        labels = item.get("labels", {})
        if not isinstance(labels, dict):
            raise ValueError(f"Expected keypoint mapping for frame {frame_idx}")
    video_output_dir = output_dir / label_path.stem
    if video_output_dir == signature_dir:
        raise ValueError("Video name 'signature' is reserved for signature output")
    if video_output_dir.is_symlink():
        raise ValueError(f"Refusing to clear a symlink: {video_output_dir}")
    if video_output_dir.resolve().is_relative_to(frames_dir.resolve()):
        raise ValueError("Overlay output must be outside the extracted frames folder")
    if video_output_dir.exists():
        shutil.rmtree(video_output_dir)
    video_output_dir.mkdir(parents=True, exist_ok=True)
    signature_dir.mkdir(parents=True, exist_ok=True)
    signature_target = signature_dir / f"{label_path.stem}.jpg"
    signature_target.unlink(missing_ok=True)
    best_signature = None
    best_distance_squared = -1.0
    required_points = math.ceil(signature_min_completeness * len(bodyparts))
    counts = dict(saved=0, missing=0, unlabeled=0, signature=0)
    for item in payload:
        frame_idx = item["frame_idx"]
        labels = item.get("labels", {})
        if not any(visible_point(point) is not None for point in labels.values()):
            counts["unlabeled"] += 1
            continue
        source = frames_dir / label_path.stem / f"{frame_idx:08d}.jpg"
        target = video_output_dir / source.name
        if not source.is_file():
            counts["missing"] += 1
            continue
        image = cv2.imread(str(source))
        if image is None:
            raise RuntimeError(f"Failed to read {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), draw_skeleton(image, labels, skeleton)):
            raise RuntimeError(f"Failed to write {target}")
        counts["saved"] += 1
        # Rank qualifying frames by the centroid's distance from the top-left.
        # Keep coordinates in their original resolution for visual inspection.
        points = [xy for name in bodyparts
                  if (xy := visible_point(labels.get(name))) is not None]
        if len(points) >= required_points:
            center_x = sum(x for x, _ in points) / len(points)
            center_y = sum(y for _, y in points) / len(points)
            distance_squared = center_x ** 2 + center_y ** 2
            if distance_squared > best_distance_squared:
                best_distance_squared = distance_squared
                best_signature = target
    if best_signature is not None:
        shutil.copyfile(best_signature, signature_target)
        counts["signature"] = 1
        print(f"[signature] {label_path.stem}: frame={best_signature.stem}")
    else:
        print(f"[signature] {label_path.stem}: no available frame with at least "
              f"{required_points}/{len(bodyparts)} visible bodyparts")
    return counts


if frames_dir.resolve() == output_dir.resolve():
    raise ValueError("Output folder must differ from the extracted frames folder")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
skeleton = config["skeleton"]
bodyparts = list(dict.fromkeys(config["bodyparts"]))
if not bodyparts:
    raise ValueError("Config must define at least one bodypart")
label_paths = sorted(label_dir.glob("*.json"))
if label_names:
    selected = {name if name.endswith(".json") else f"{name}.json" for name in label_names}
    label_paths = [path for path in label_paths if path.name in selected]
if not label_paths:
    raise FileNotFoundError(f"No selected label JSON files found in {label_dir}")

totals = dict(saved=0, missing=0, unlabeled=0, signature=0)
failures = []
for label_path in label_paths:
    try:
        counts = process_label_file(label_path, skeleton)
        for key, value in counts.items():
            totals[key] += value
        print(f"[ok] {label_path.stem}: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    except Exception as exc:
        failures.append(f"{label_path.name}: {exc}")
        print(f"[error] {failures[-1]}")
print(f"Output: {output_dir}")
print("Finished: " + ", ".join(f"{key}={value}" for key, value in totals.items()))
if failures:
    raise SystemExit("Some files failed:\n" + "\n".join(failures))
