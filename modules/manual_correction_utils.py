from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from modules.label_csv_utils import load_keypoints, save_keypoints


def get_latest_prediction_root(predicted_frames_root: Path) -> Path:
    predicted_frames_root = Path(predicted_frames_root)
    if not predicted_frames_root.is_dir():
        raise FileNotFoundError(f"Prediction root not found: {predicted_frames_root}")

    model_dirs = [path for path in predicted_frames_root.iterdir() if path.is_dir()]
    if not model_dirs:
        raise FileNotFoundError(f"No model folders found under {predicted_frames_root}")

    return max(model_dirs, key=lambda path: path.stat().st_mtime)


def load_prediction_map(prediction_json_path: Path) -> dict[int, dict]:
    payload = json.loads(Path(prediction_json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(
            f"Expected a list of predictions in {prediction_json_path}, got {type(payload).__name__}"
        )
    return {int(item["frame_idx"]): item for item in payload}


def load_video_labels(csv_path: Path) -> pd.DataFrame:
    return load_keypoints(csv_path)


def load_label_json(label_json_path: Path) -> pd.DataFrame:
    payload = json.loads(Path(label_json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected {label_json_path} to contain a list, got {type(payload).__name__}")

    rows = []
    for item in payload:
        frame_idx = int(item["frame_idx"])
        row = {"frame": frame_idx}
        labels = item.get("labels") or {}
        for keypoint, value in labels.items():
            visible = int(value[0]) if value else 0
            x = value[1] if len(value) > 1 else None
            y = value[2] if len(value) > 2 else None
            row[f"{keypoint}_x"] = float(x) if visible and x is not None else np.nan
            row[f"{keypoint}_y"] = float(y) if visible and y is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_label_json_payload(label_df: pd.DataFrame, keypoints: list[str]) -> list[dict]:
    payload = []
    for _, row in label_df.iterrows():
        frame_idx = int(float(row["frame"]))
        labels = {}
        for keypoint in keypoints:
            x_col = f"{keypoint}_x"
            y_col = f"{keypoint}_y"
            x_val = pd.to_numeric(pd.Series([row.get(x_col)]), errors="coerce").iloc[0]
            y_val = pd.to_numeric(pd.Series([row.get(y_col)]), errors="coerce").iloc[0]
            if pd.isna(x_val) or pd.isna(y_val):
                labels[keypoint] = [0, None, None]
            else:
                labels[keypoint] = [1, int(round(float(x_val))), int(round(float(y_val)))]
        payload.append({"frame_idx": frame_idx, "labels": labels})
    return payload


def list_keypoints(prediction_map: dict[int, dict], label_df: pd.DataFrame) -> list[str]:
    prediction_keypoints: set[str] = set()
    for item in prediction_map.values():
        prediction_keypoints.update((item.get("keypoints") or {}).keys())

    label_keypoints = {
        col[:-2]
        for col in label_df.columns
        if col.endswith("_x") and f"{col[:-2]}_y" in label_df.columns
    }
    return sorted(prediction_keypoints & label_keypoints)


def get_prediction_point(prediction_map: dict[int, dict], frame_idx: int, keypoint: str) -> tuple[float, float] | None:
    item = prediction_map.get(int(frame_idx))
    if item is None:
        return None
    kp = (item.get("keypoints") or {}).get(keypoint)
    if not kp:
        return None
    x = kp.get("x")
    y = kp.get("y")
    if x is None or y is None:
        return None
    return float(x), float(y)


def get_label_point(label_df: pd.DataFrame, frame_idx: int, keypoint: str) -> tuple[float, float] | None:
    frame_mask = pd.to_numeric(label_df["frame"], errors="coerce") == int(frame_idx)
    if not frame_mask.any():
        return None
    row = label_df.loc[frame_mask].iloc[0]
    x_col = f"{keypoint}_x"
    y_col = f"{keypoint}_y"
    if x_col not in label_df.columns or y_col not in label_df.columns:
        return None
    x = pd.to_numeric(pd.Series([row[x_col]]), errors="coerce").iloc[0]
    y = pd.to_numeric(pd.Series([row[y_col]]), errors="coerce").iloc[0]
    if pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def compute_distance(prediction_map: dict[int, dict], label_df: pd.DataFrame, frame_idx: int, keypoint: str) -> float | None:
    pred_point = get_prediction_point(prediction_map, frame_idx, keypoint)
    label_point = get_label_point(label_df, frame_idx, keypoint)
    if pred_point is None or label_point is None:
        return None
    dx = label_point[0] - pred_point[0]
    dy = label_point[1] - pred_point[1]
    return float(np.hypot(dx, dy))


def find_flagged_frames(
    prediction_map: dict[int, dict],
    label_df: pd.DataFrame,
    keypoint: str,
    cutoff: float,
) -> list[int]:
    frames = sorted(set(int(frame) for frame in pd.to_numeric(label_df["frame"], errors="coerce").dropna().astype(int)))
    flagged = []
    for frame_idx in frames:
        distance = compute_distance(prediction_map, label_df, frame_idx, keypoint)
        if distance is not None and distance > float(cutoff):
            flagged.append(int(frame_idx))
    return flagged


def update_label_point(label_df: pd.DataFrame, frame_idx: int, keypoint: str, x: float, y: float) -> None:
    frame_mask = pd.to_numeric(label_df["frame"], errors="coerce") == int(frame_idx)
    if not frame_mask.any():
        raise KeyError(f"Frame {frame_idx} not found in labels")
    label_df.loc[frame_mask, f"{keypoint}_x"] = float(x)
    label_df.loc[frame_mask, f"{keypoint}_y"] = float(y)


def remove_label_point(label_df: pd.DataFrame, frame_idx: int, keypoint: str) -> None:
    frame_mask = pd.to_numeric(label_df["frame"], errors="coerce") == int(frame_idx)
    if not frame_mask.any():
        raise KeyError(f"Frame {frame_idx} not found in labels")
    label_df.loc[frame_mask, f"{keypoint}_x"] = np.nan
    label_df.loc[frame_mask, f"{keypoint}_y"] = np.nan


def save_video_labels(label_df: pd.DataFrame, video_name: str, csv_path: Path) -> None:
    save_keypoints(label_df, video_name, csv_path)


def get_video_pairs(prediction_root: Path, labels_root: Path) -> list[str]:
    prediction_root = Path(prediction_root)
    labels_root = Path(labels_root)
    video_names = []
    for prediction_path in sorted(prediction_root.glob("*.json")):
        video_name = prediction_path.stem
        csv_path = labels_root / video_name / "CollectedData_rats.csv"
        if csv_path.is_file():
            video_names.append(video_name)
    return video_names


def get_label_json_video_pairs(prediction_root: Path, label_json_root: Path) -> list[str]:
    prediction_root = Path(prediction_root)
    label_json_root = Path(label_json_root)
    video_names = []
    for prediction_path in sorted(prediction_root.glob("*.json")):
        video_name = prediction_path.stem
        label_path = label_json_root / f"{video_name}.json"
        if label_path.is_file():
            video_names.append(video_name)
    return video_names


def get_frame_image_path(frames_root: Path, video_name: str, frame_idx: int) -> Path:
    return Path(frames_root) / video_name / f"{int(frame_idx):08d}.jpg"
