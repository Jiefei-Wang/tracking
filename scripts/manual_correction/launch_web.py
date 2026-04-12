from __future__ import annotations

import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from hashlib import sha1
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import yaml
from nicegui import app, events, ui

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.manual_correction_utils import (
    compute_distance,
    build_label_json_payload,
    get_frame_image_path,
    get_label_point,
    get_label_json_video_pairs,
    get_latest_prediction_root,
    get_prediction_point,
    list_keypoints,
    load_label_json,
    load_prediction_map,
    remove_label_point,
    update_label_point,
)

PREDICTED_FRAMES_ROOT = PROJECT_ROOT / "output" / "predicted_frames"
LABELED_FRAMES_ROOT = PROJECT_ROOT / "output" / "labeled_frames"
LABEL_JSON_ROOT = PROJECT_ROOT / "input" / "labels"
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
CACHE_ROOT = PROJECT_ROOT / ".cache" / "manual_correction_web"
CACHE_VERSION = 1

ALL_VIDEOS = "__ALL_VIDEOS__"
ALL_KEYPOINTS = "__ALL_KEYPOINTS__"

app.add_static_files("/labeled_frames", str(LABELED_FRAMES_ROOT))
ui.add_head_html("<style>body{overflow:hidden;} .nicegui-content{overflow:hidden;}</style>", shared=True)


@dataclass
class EditorState:
    prediction_root: Path
    video_names: list[str]
    video_filter: str = ALL_VIDEOS
    keypoint_filter: str = ALL_KEYPOINTS
    current_keypoint: str | None = None
    current_item_key: tuple[str, int, str] | None = None
    syncing_current_keypoint_options: bool = False
    show_all_keypoints: bool = False
    show_prediction: bool = True
    show_label_text: bool = False
    cutoff: float = 20.0
    current_index: int = 0
    flagged_items: list[tuple[str, int, str, float]] = field(default_factory=list)  # video, frame, keypoint, distance
    dragging: bool = False
    drag_item: tuple[str, int, str] | None = None
    drag_point: tuple[float, float] | None = None
    drag_moved: bool = False
    drag_existing_label: bool = False
    right_button_down: bool = False
    right_button_start: tuple[float, float] | None = None
    right_drag_moved: bool = False
    prediction_by_video: dict[str, dict[int, dict[str, Any]]] = field(default_factory=dict)
    working_by_video: dict[str, pd.DataFrame] = field(default_factory=dict)
    original_by_video: dict[str, pd.DataFrame] = field(default_factory=dict)
    keypoints_by_video: dict[str, list[str]] = field(default_factory=dict)
    all_label_keypoints_by_video: dict[str, list[str]] = field(default_factory=dict)
    label_index_by_video: dict[str, dict[int, int]] = field(default_factory=dict)
    label_count_by_video: dict[str, int] = field(default_factory=dict)
    tracked_columns_by_video: dict[str, list[str]] = field(default_factory=dict)
    changed_by_video: dict[str, set[int]] = field(default_factory=dict)


def plus_svg(x: float, y: float, color: str, alpha: float = 0.5, size: float = 10) -> str:
    return (
        f'<line x1="{x - size}" y1="{y}" x2="{x + size}" y2="{y}" '
        f'stroke="{color}" stroke-width="3" stroke-opacity="{alpha}" />'
        f'<line x1="{x}" y1="{y - size}" x2="{x}" y2="{y + size}" '
        f'stroke="{color}" stroke-width="3" stroke-opacity="{alpha}" />'
    )


def label_text_svg(x: float, y: float, text: str) -> str:
    return (
        f'<text x="{x + 12}" y="{y - 12}" fill="lime" font-size="16" '
        'stroke="#111" stroke-width="0.6" paint-order="stroke">'
        f"{escape(text)}</text>"
    )


def legend_svg(image_width: float, show_prediction: bool) -> str:
    width = 210.0
    height = 70.0 if show_prediction else 46.0
    margin = 12.0
    x0 = max(margin, float(image_width) - width - margin)
    y0 = margin
    prediction_y = y0 + 24.0
    label_y = y0 + (48.0 if show_prediction else 24.0)
    icon_x = x0 + 16.0
    text_x = x0 + 30.0

    svg = (
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
        'fill="white" fill-opacity="0.72" stroke="#222" stroke-width="1" rx="6" />'
    )
    if show_prediction:
        svg += (
            f"{plus_svg(icon_x, prediction_y, color='red', alpha=0.5, size=7)}"
            f'<text x="{text_x}" y="{prediction_y + 4}" fill="#111" font-size="16">Prediction (red)</text>'
        )
    svg += (
        f"{plus_svg(icon_x, label_y, color='lime', alpha=0.5, size=7)}"
        f'<text x="{text_x}" y="{label_y + 4}" fill="#111" font-size="16">Label (green)</text>'
    )
    return svg


def clamp_point(point: tuple[float, float], image_width: float, image_height: float) -> tuple[float, float]:
    return (
        max(0.0, min(float(image_width), float(point[0]))),
        max(0.0, min(float(image_height), float(point[1]))),
    )


def image_url(video_name: str, frame_idx: int, image_path: Path, nonce: str = "") -> str:
    stamp = image_path.stat().st_mtime_ns
    return f"/labeled_frames/{quote(video_name)}/{frame_idx:08d}.jpg?v={stamp}&n={quote(nonce)}"


def extract_all_label_keypoints(label_df: pd.DataFrame) -> list[str]:
    return sorted(col[:-2] for col in label_df.columns if col.endswith("_x") and f"{col[:-2]}_y" in label_df.columns)


def load_config_video_names(config_path: Path) -> set[str]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return set(payload.get("train_videos", [])) | set(payload.get("test_videos", []))


def file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_size, stat.st_mtime_ns


def cache_path(kind: str, parts: tuple[Any, ...]) -> Path:
    key = sha1(repr((CACHE_VERSION, kind, parts)).encode("utf-8")).hexdigest()
    return CACHE_ROOT / f"{kind}_{key}.pkl"


def read_cache(path: Path, signature: Any) -> Any | None:
    if not path.is_file():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception:
        return None
    if payload.get("signature") != signature:
        return None
    return payload.get("data")


def write_cache(path: Path, signature: Any, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"signature": signature, "data": data}, protocol=pickle.HIGHEST_PROTOCOL))


def build_ui() -> None:
    prediction_root = get_latest_prediction_root(PREDICTED_FRAMES_ROOT)
    configured_video_names = load_config_video_names(CONFIG_PATH)
    video_names = [name for name in get_label_json_video_pairs(prediction_root, LABEL_JSON_ROOT) if name in configured_video_names]
    state = EditorState(prediction_root=prediction_root, video_names=video_names)

    with ui.column().classes("w-full h-screen p-2 gap-2 overflow-hidden"):
        with ui.row().classes("items-center gap-3"):
            current_keypoint_select = ui.select({}, value=None, label="Current keypoint").classes("min-w-48")
            show_all_keypoints_checkbox = ui.checkbox("show all keypoints", value=False)
            show_prediction_checkbox = ui.checkbox("show prediction", value=True)
            show_label_text_checkbox = ui.checkbox("show label text", value=False)
            reset_button = ui.button("Reset frame")
            save_button = ui.button("Save (0)")

        with ui.row().classes("w-full grow gap-3 overflow-hidden"):
            with ui.column().classes("w-72 shrink-0 gap-3 overflow-auto"):
                video_options = {ALL_VIDEOS: "All videos", **{name: name for name in video_names}}
                video_select = ui.select(video_options, value=ALL_VIDEOS, label="Video").classes("w-full")
                keypoint_select = ui.select({ALL_KEYPOINTS: "All keypoints"}, value=ALL_KEYPOINTS, label="Keypoint").classes("w-full")
                cutoff_input = ui.number("Cutoff", value=20.0, step=1.0, min=0.0).classes("w-full")
                counter_label = ui.label("0 / 0").classes("text-subtitle2")
                video_info_label = ui.label("Video: n/a").classes("text-body2")
                frame_info_label = ui.label("Frame: n/a | Label: n/a").classes("text-body2")
                keypoint_info_label = ui.label("Flagged keypoint: n/a").classes("text-body2")
                distance_label = ui.label("Distance: n/a").classes("text-body2")
                message_label = ui.label("")
                instruction_label = ui.label(
                    "left-drag move, right click remove, a/d navigate, arrow keys nudge"
                ).classes("text-caption")

            with ui.column().classes("grow gap-2 overflow-hidden"):
                image_view = ui.interactive_image(
                    size=(1280, 720),
                    events=["mousedown", "mousemove", "mouseup", "contextmenu"],
                    cross=False,
                ).classes("w-full border rounded").style(
                    "height: calc(100vh - 120px); object-fit: contain; overflow: hidden; user-select: none;"
                )

                with ui.row().classes("items-center gap-2"):
                    prev_button = ui.button("<- Prev")
                    next_button = ui.button("Next ->")

    def ensure_video_loaded(video_name: str) -> None:
        if video_name in state.working_by_video:
            return
        label_path = LABEL_JSON_ROOT / f"{video_name}.json"
        prediction_path = prediction_root / f"{video_name}.json"
        signature = (file_signature(label_path), file_signature(prediction_path))
        path = cache_path("video", (video_name, signature))
        cached = read_cache(path, signature)

        if cached is None:
            label_df = load_label_json(label_path)
            pred_map = load_prediction_map(prediction_path)
            cached = {
                "label_df": label_df,
                "pred_map": pred_map,
                "keypoints": list_keypoints(pred_map, label_df),
                "all_label_keypoints": extract_all_label_keypoints(label_df),
                "label_index": {
                    int(float(frame_idx)): idx + 1 for idx, frame_idx in enumerate(label_df["frame"].tolist())
                },
                "label_count": len(label_df),
                "tracked_columns": [c for c in label_df.columns if c.endswith("_x") or c.endswith("_y")],
            }
            write_cache(path, signature, cached)

        label_df = cached["label_df"]
        state.working_by_video[video_name] = label_df.copy(deep=True)
        state.original_by_video[video_name] = label_df.copy(deep=True)
        state.prediction_by_video[video_name] = cached["pred_map"]
        state.keypoints_by_video[video_name] = cached["keypoints"]
        state.all_label_keypoints_by_video[video_name] = cached["all_label_keypoints"]
        state.label_index_by_video[video_name] = cached["label_index"]
        state.label_count_by_video[video_name] = cached["label_count"]
        state.tracked_columns_by_video[video_name] = cached["tracked_columns"]
        state.changed_by_video[video_name] = set()

    def get_selected_videos() -> list[str]:
        if state.video_filter == ALL_VIDEOS:
            return list(state.video_names)
        return [state.video_filter] if state.video_filter in state.video_names else []

    def get_current_item() -> tuple[str, int, str, float] | None:
        if not state.flagged_items:
            return None
        if state.current_index < 0 or state.current_index >= len(state.flagged_items):
            return None
        return state.flagged_items[state.current_index]

    def update_save_button_text() -> None:
        if state.video_filter == ALL_VIDEOS:
            count = sum(len(v) for v in state.changed_by_video.values())
        else:
            count = len(state.changed_by_video.get(state.video_filter, set()))
        save_button.set_text(f"Save ({count})")

    def refresh_keypoint_options() -> None:
        videos = get_selected_videos()
        keypoint_set: set[str] = set()
        for video in videos:
            ensure_video_loaded(video)
            keypoint_set.update(state.keypoints_by_video.get(video, []))
        keypoint_values = sorted(keypoint_set)
        keypoint_options = {ALL_KEYPOINTS: "All keypoints", **{kp: kp for kp in keypoint_values}}
        if state.keypoint_filter not in keypoint_options:
            state.keypoint_filter = ALL_KEYPOINTS
        keypoint_select.set_options(keypoint_options, value=state.keypoint_filter)

    def refresh_current_keypoint_options(video_name: str | None, flagged_keypoint: str | None) -> None:
        keypoints = state.all_label_keypoints_by_video.get(video_name or "", [])
        options = {kp: kp for kp in keypoints}
        if flagged_keypoint is not None and flagged_keypoint in options:
            state.current_keypoint = flagged_keypoint
        elif state.current_keypoint not in options:
            state.current_keypoint = keypoints[0] if keypoints else None
        state.syncing_current_keypoint_options = True
        try:
            current_keypoint_select.set_options(options, value=state.current_keypoint)
        finally:
            state.syncing_current_keypoint_options = False

    def get_current_edit_keypoint(video_name: str, fallback_keypoint: str) -> str:
        keypoints = state.all_label_keypoints_by_video.get(video_name, [])
        if state.current_keypoint in keypoints:
            return str(state.current_keypoint)
        return fallback_keypoint

    def get_image_size(video_name: str, frame_idx: int) -> tuple[float, float]:
        pred_item = state.prediction_by_video[video_name].get(frame_idx)
        if pred_item is None:
            return 1280.0, 720.0
        return float(pred_item.get("image_width", 1280.0)), float(pred_item.get("image_height", 720.0))

    def get_row(df: pd.DataFrame, frame_idx: int) -> pd.Series | None:
        frame_mask = pd.to_numeric(df["frame"], errors="coerce") == int(frame_idx)
        if not frame_mask.any():
            return None
        return df.loc[frame_mask].iloc[0]

    def get_video_candidates(video_name: str) -> list[tuple[float, str, int, str]]:
        label_path = LABEL_JSON_ROOT / f"{video_name}.json"
        prediction_path = prediction_root / f"{video_name}.json"
        signature = (
            file_signature(label_path),
            file_signature(prediction_path),
            state.keypoint_filter,
            float(state.cutoff),
        )
        path = cache_path("candidates", (video_name, signature))
        cached = read_cache(path, signature)
        if cached is not None:
            return cached

        prediction_map = state.prediction_by_video[video_name]
        base_df = state.original_by_video[video_name]
        frame_values = pd.to_numeric(base_df["frame"], errors="coerce").dropna().astype(int)
        frame_indices = sorted(set(int(v) for v in frame_values))
        candidates: list[tuple[float, str, int, str]] = []
        if state.keypoint_filter == ALL_KEYPOINTS:
            keypoints = state.keypoints_by_video[video_name]
            for frame_idx in frame_indices:
                best_keypoint: str | None = None
                best_distance: float | None = None
                for keypoint in keypoints:
                    distance = compute_distance(prediction_map, base_df, frame_idx, keypoint)
                    if distance is None or distance <= float(state.cutoff):
                        continue
                    if best_distance is None or float(distance) > best_distance:
                        best_distance = float(distance)
                        best_keypoint = keypoint
                if best_keypoint is not None and best_distance is not None:
                    candidates.append((best_distance, video_name, int(frame_idx), best_keypoint))
        else:
            keypoints = [state.keypoint_filter] if state.keypoint_filter in state.keypoints_by_video[video_name] else []
            for keypoint in keypoints:
                for frame_idx in frame_indices:
                    distance = compute_distance(prediction_map, base_df, frame_idx, keypoint)
                    if distance is not None and distance > float(state.cutoff):
                        candidates.append((float(distance), video_name, int(frame_idx), keypoint))

        write_cache(path, signature, candidates)
        return candidates

    def frame_is_changed(video_name: str, frame_idx: int) -> bool:
        working_df = state.working_by_video[video_name]
        original_df = state.original_by_video[video_name]
        current_row = get_row(working_df, frame_idx)
        original_row = get_row(original_df, frame_idx)
        if current_row is None or original_row is None:
            return False
        for col in state.tracked_columns_by_video[video_name]:
            cur = pd.to_numeric(pd.Series([current_row[col]]), errors="coerce").iloc[0]
            org = pd.to_numeric(pd.Series([original_row[col]]), errors="coerce").iloc[0]
            if pd.isna(cur) and pd.isna(org):
                continue
            if pd.isna(cur) != pd.isna(org):
                return True
            if abs(float(cur) - float(org)) > 1e-6:
                return True
        return False

    def mark_frame_dirty(video_name: str, frame_idx: int) -> None:
        changed_frames = state.changed_by_video[video_name]
        if frame_is_changed(video_name, frame_idx):
            changed_frames.add(int(frame_idx))
        else:
            changed_frames.discard(int(frame_idx))
        update_save_button_text()

    def refresh_image() -> None:
        item = get_current_item()
        if item is None:
            image_view.set_source("")
            image_view.content = ""
            counter_label.set_text("0 / 0")
            video_info_label.set_text("Video: n/a")
            frame_info_label.set_text("Frame: n/a | Label: n/a")
            keypoint_info_label.set_text("Flagged keypoint: n/a")
            distance_label.set_text("Distance: n/a")
            message_label.set_text("No frames exceed cutoff.")
            refresh_current_keypoint_options(None, None)
            state.current_item_key = None
            return

        video_name, frame_idx, flagged_keypoint, _ = item
        ensure_video_loaded(video_name)
        item_key = (video_name, frame_idx, flagged_keypoint)
        if state.current_item_key != item_key:
            state.current_item_key = item_key
            refresh_current_keypoint_options(video_name, flagged_keypoint)
        else:
            refresh_current_keypoint_options(video_name, None)

        image_path = get_frame_image_path(LABELED_FRAMES_ROOT, video_name, frame_idx)
        if not image_path.is_file():
            image_view.set_source("")
            image_view.content = ""
            counter_label.set_text(f"{state.current_index + 1} / {len(state.flagged_items)}")
            video_info_label.set_text(f"Video: {video_name}")
            frame_info_label.set_text(f"Frame: {frame_idx} | Label: n/a")
            keypoint_info_label.set_text(f"Flagged keypoint: {flagged_keypoint}")
            distance_label.set_text("Distance: n/a")
            message_label.set_text(f"Missing image: {image_path}")
            return

        prediction_map = state.prediction_by_video[video_name]
        label_df = state.working_by_video[video_name]
        image_width, image_height = get_image_size(video_name, frame_idx)
        edit_keypoint = get_current_edit_keypoint(video_name, flagged_keypoint)
        nonce = (
            f"{state.current_index}-{video_name}-{frame_idx}-{flagged_keypoint}-"
            f"{edit_keypoint}-{state.show_all_keypoints}-{state.show_prediction}-{state.show_label_text}"
        )
        image_view.set_source(image_url(video_name, frame_idx, image_path, nonce=nonce))

        svg = ""
        shown_keypoints = state.all_label_keypoints_by_video[video_name] if state.show_all_keypoints else [edit_keypoint]
        for keypoint in shown_keypoints:
            if state.show_prediction:
                prediction_point = get_prediction_point(prediction_map, frame_idx, keypoint)
                if prediction_point is not None:
                    svg += plus_svg(prediction_point[0], prediction_point[1], color="red", alpha=0.5)
            label_point = (
                state.drag_point
                if state.dragging and state.drag_item == (video_name, frame_idx, keypoint) and state.drag_point is not None
                else get_label_point(label_df, frame_idx, keypoint)
            )
            if label_point is not None:
                label_point = clamp_point(label_point, image_width, image_height)
                svg += plus_svg(label_point[0], label_point[1], color="lime", alpha=0.5)
                if state.show_label_text:
                    svg += label_text_svg(label_point[0], label_point[1], keypoint)
        svg += legend_svg(image_width=image_width, show_prediction=state.show_prediction)
        image_view.content = svg

        working_distance = compute_distance(prediction_map, label_df, frame_idx, flagged_keypoint)
        distance_text = "n/a" if working_distance is None else f"{working_distance:.1f}"
        label_index = state.label_index_by_video.get(video_name, {}).get(int(frame_idx))
        label_count = state.label_count_by_video.get(video_name, 0)
        label_text = "n/a" if label_index is None else f"{label_index} / {label_count}"
        counter_label.set_text(f"{state.current_index + 1} / {len(state.flagged_items)}")
        video_info_label.set_text(f"Video: {video_name}")
        frame_info_label.set_text(f"Frame: {frame_idx} | Label: {label_text}")
        keypoint_info_label.set_text(f"Flagged keypoint: {flagged_keypoint} | Current: {edit_keypoint}")
        distance_label.set_text(f"Distance: {distance_text}")
        message_label.set_text("")

    def refresh_frames(reset_index: bool, preferred_item: tuple[str, int, str] | None = None) -> None:
        old_item = preferred_item
        if old_item is None:
            current = get_current_item()
            old_item = (current[0], current[1], current[2]) if current is not None else None

        candidates: list[tuple[float, str, int, str]] = []
        selected_videos = get_selected_videos()
        for video_name in selected_videos:
            ensure_video_loaded(video_name)
            candidates.extend(get_video_candidates(video_name))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
        state.flagged_items = [(video, frame, keypoint, distance) for distance, video, frame, keypoint in candidates]

        if not state.flagged_items:
            state.current_index = 0
        elif reset_index or old_item is None:
            state.current_index = 0
        else:
            old_idx = next(
                (
                    idx
                    for idx, (video, frame, keypoint, _) in enumerate(state.flagged_items)
                    if (video, frame, keypoint) == old_item
                ),
                None,
            )
            if old_idx is not None:
                state.current_index = old_idx
            else:
                state.current_index = min(state.current_index, len(state.flagged_items) - 1)

        state.dragging = False
        state.drag_item = None
        state.drag_point = None
        state.drag_moved = False
        state.drag_existing_label = False
        refresh_image()

    def apply_label_update(video_name: str, frame_idx: int, keypoint: str, x: float, y: float) -> None:
        label_df = state.working_by_video[video_name]
        update_label_point(label_df, frame_idx, keypoint, float(x), float(y))
        mark_frame_dirty(video_name, frame_idx)
        current = get_current_item()
        preferred_keypoint = current[2] if current is not None and current[0] == video_name and current[1] == frame_idx else keypoint
        refresh_frames(reset_index=False, preferred_item=(video_name, frame_idx, preferred_keypoint))

    def apply_remove(video_name: str, frame_idx: int, keypoint: str) -> None:
        label_df = state.working_by_video[video_name]
        remove_label_point(label_df, frame_idx, keypoint)
        mark_frame_dirty(video_name, frame_idx)
        current = get_current_item()
        preferred_keypoint = current[2] if current is not None and current[0] == video_name and current[1] == frame_idx else keypoint
        refresh_frames(reset_index=False, preferred_item=(video_name, frame_idx, preferred_keypoint))

    def commit_drag_if_needed() -> None:
        if not state.dragging:
            return
        target_item = state.drag_item
        new_point = state.drag_point
        should_update = new_point is not None and (state.drag_moved or not state.drag_existing_label)
        state.dragging = False
        state.drag_item = None
        state.drag_point = None
        state.drag_moved = False
        state.drag_existing_label = False
        if target_item is not None and should_update:
            apply_label_update(target_item[0], target_item[1], target_item[2], new_point[0], new_point[1])
        else:
            refresh_image()

    def reset_current_frame() -> None:
        item = get_current_item()
        if item is None:
            return
        video_name, frame_idx, _, _ = item
        working_df = state.working_by_video[video_name]
        original_df = state.original_by_video[video_name]
        tracked_columns = state.tracked_columns_by_video[video_name]

        current_mask = pd.to_numeric(working_df["frame"], errors="coerce") == int(frame_idx)
        original_mask = pd.to_numeric(original_df["frame"], errors="coerce") == int(frame_idx)
        if not current_mask.any() or not original_mask.any():
            return
        working_df.loc[current_mask, tracked_columns] = original_df.loc[original_mask, tracked_columns].to_numpy()[0]
        mark_frame_dirty(video_name, frame_idx)
        refresh_frames(reset_index=False, preferred_item=(video_name, frame_idx, item[2]))

    def save_scope_json(notify: bool = True) -> None:
        if state.video_filter == ALL_VIDEOS:
            target_videos = [v for v in state.video_names if state.changed_by_video.get(v)]
        else:
            target_videos = [state.video_filter] if state.changed_by_video.get(state.video_filter) else []
        if not target_videos:
            if notify:
                ui.notify("No changes to save.")
            return

        total_frames = 0
        for video_name in target_videos:
            ensure_video_loaded(video_name)
            working_df = state.working_by_video[video_name]
            keypoints = state.all_label_keypoints_by_video[video_name]
            payload = build_label_json_payload(working_df, keypoints)
            output_path = LABEL_JSON_ROOT / f"{video_name}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            total_frames += len(payload)

            state.original_by_video[video_name] = working_df.copy(deep=True)
            state.changed_by_video[video_name].clear()

        update_save_button_text()
        refresh_frames(reset_index=False)
        if notify:
            ui.notify(f"Saved {len(target_videos)} json files ({total_frames} frames).")

    def move_frame(step: int) -> None:
        if not state.flagged_items:
            return
        commit_drag_if_needed()
        save_scope_json(notify=False)
        if not state.flagged_items:
            return
        state.current_index = max(0, min(len(state.flagged_items) - 1, state.current_index + step))
        refresh_image()

    def nudge(dx: float, dy: float) -> None:
        item = get_current_item()
        if item is None:
            return
        video_name, frame_idx, flagged_keypoint, _ = item
        keypoint = get_current_edit_keypoint(video_name, flagged_keypoint)
        label_df = state.working_by_video[video_name]
        label_point = get_label_point(label_df, frame_idx, keypoint)
        if label_point is None:
            return
        image_width, image_height = get_image_size(video_name, frame_idx)
        new_point = clamp_point((label_point[0] + dx, label_point[1] + dy), image_width, image_height)
        apply_label_update(video_name, frame_idx, keypoint, new_point[0], new_point[1])

    def handle_mouse(e: events.MouseEventArguments) -> None:
        item = get_current_item()
        if item is None:
            return
        video_name, frame_idx, flagged_keypoint, _ = item
        keypoint = get_current_edit_keypoint(video_name, flagged_keypoint)
        image_width, image_height = get_image_size(video_name, frame_idx)

        if e.type == "mousedown" and e.button == 2:
            state.right_button_down = True
            state.right_button_start = (float(e.image_x), float(e.image_y))
            state.right_drag_moved = False
            return

        if e.type == "mousemove" and state.right_button_down and e.buttons == 2 and state.right_button_start is not None:
            dx = float(e.image_x) - state.right_button_start[0]
            dy = float(e.image_y) - state.right_button_start[1]
            if math.hypot(dx, dy) > 5:
                state.right_drag_moved = True
            return

        if e.type == "mouseup" and e.button == 2:
            state.right_button_down = False
            return

        if e.type == "contextmenu":
            should_remove = not state.right_drag_moved
            state.right_button_down = False
            state.right_button_start = None
            state.right_drag_moved = False
            if should_remove:
                apply_remove(video_name, frame_idx, keypoint)
            return

        if e.type == "mousedown" and e.button == 0:
            label_df = state.working_by_video[video_name]
            label_point = get_label_point(label_df, frame_idx, keypoint)
            if label_point is None:
                state.dragging = True
                state.drag_item = (video_name, frame_idx, keypoint)
                state.drag_point = clamp_point((float(e.image_x), float(e.image_y)), image_width, image_height)
                state.drag_moved = False
                state.drag_existing_label = False
                refresh_image()
                return

            clamped_label_point = clamp_point(label_point, image_width, image_height)
            distance = math.hypot(float(e.image_x) - clamped_label_point[0], float(e.image_y) - clamped_label_point[1])
            if distance <= 20:
                state.dragging = True
                state.drag_item = (video_name, frame_idx, keypoint)
                state.drag_point = clamped_label_point
                state.drag_moved = False
                state.drag_existing_label = True
            return

        if e.type == "mousemove" and state.dragging and state.drag_item == (video_name, frame_idx, keypoint):
            state.drag_point = clamp_point((float(e.image_x), float(e.image_y)), image_width, image_height)
            state.drag_moved = True
            refresh_image()
            return

        if e.type == "mouseup" and state.dragging:
            commit_drag_if_needed()

    def handle_video_change(e: events.ValueChangeEventArguments) -> None:
        value = str(e.value) if e.value is not None else ALL_VIDEOS
        state.video_filter = value
        state.current_item_key = None
        refresh_keypoint_options()
        refresh_frames(reset_index=True)
        update_save_button_text()

    def handle_keypoint_change(e: events.ValueChangeEventArguments) -> None:
        value = str(e.value) if e.value is not None else ALL_KEYPOINTS
        state.keypoint_filter = value
        state.current_item_key = None
        refresh_frames(reset_index=True)

    def handle_current_keypoint_change(e: events.ValueChangeEventArguments) -> None:
        if state.syncing_current_keypoint_options:
            return
        state.current_keypoint = str(e.value) if e.value is not None else None
        refresh_image()

    def handle_show_all_keypoints_change(e: events.ValueChangeEventArguments) -> None:
        state.show_all_keypoints = bool(e.value)
        refresh_image()

    def handle_show_prediction_change(e: events.ValueChangeEventArguments) -> None:
        state.show_prediction = bool(e.value)
        refresh_image()

    def handle_show_label_text_change(e: events.ValueChangeEventArguments) -> None:
        state.show_label_text = bool(e.value)
        refresh_image()

    def handle_cutoff_change(e: events.ValueChangeEventArguments) -> None:
        try:
            state.cutoff = float(e.value)
        except (TypeError, ValueError):
            state.cutoff = 20.0
            cutoff_input.value = 20.0
        state.current_item_key = None
        refresh_frames(reset_index=True)

    def handle_keyboard(e: events.KeyEventArguments) -> None:
        if not e.action.keydown:
            return
        if e.key == "a":
            move_frame(-1)
            return
        if e.key == "d":
            move_frame(1)
            return
        if e.key.arrow_left:
            nudge(-1.0, 0.0)
            return
        if e.key.arrow_right:
            nudge(1.0, 0.0)
            return
        if e.key.arrow_up:
            nudge(0.0, -1.0)
            return
        if e.key.arrow_down:
            nudge(0.0, 1.0)
            return

    image_view.on_mouse(handle_mouse)
    image_view.on(
        "wheel",
        js_handler="""
            (e) => {
                e.preventDefault();
                const el = e.currentTarget;
                const img = el.querySelector('img');
                const svg = el.querySelector('svg');
                if (!img || !svg) return;
                const minScale = 1.0;
                const maxScale = 8.0;
                const factor = e.deltaY < 0 ? 1.12 : 0.9;
                const s0 = parseFloat(el.dataset.zoomScale || '1');
                const s1 = Math.min(maxScale, Math.max(minScale, s0 * factor));
                if (Math.abs(s1 - s0) < 1e-9) return;
                const tx0 = parseFloat(el.dataset.panX || '0');
                const ty0 = parseFloat(el.dataset.panY || '0');
                const cx = e.offsetX;
                const cy = e.offsetY;
                const ratio = s1 / s0;
                const tx1 = cx - (cx - tx0) * ratio;
                const ty1 = cy - (cy - ty0) * ratio;
                el.dataset.zoomScale = String(s1);
                el.dataset.panX = String(tx1);
                el.dataset.panY = String(ty1);
                const transform = `translate(${tx1}px, ${ty1}px) scale(${s1})`;
                img.style.transformOrigin = '0 0';
                svg.style.transformOrigin = '0 0';
                img.style.transform = transform;
                svg.style.transform = transform;
            }
        """,
    )
    image_view.on(
        "mousedown",
        js_handler="""
            (e) => {
                if (e.button !== 2) return;
                const el = e.currentTarget;
                const scale = parseFloat(el.dataset.zoomScale || '1');
                e.preventDefault();
                const img = el.querySelector('img');
                const svg = el.querySelector('svg');
                if (!img || !svg) return;
                el.dataset.panActive = '1';
                el.dataset.panStartClientX = String(e.clientX);
                el.dataset.panStartClientY = String(e.clientY);
                el.dataset.panStartX = String(parseFloat(el.dataset.panX || '0'));
                el.dataset.panStartY = String(parseFloat(el.dataset.panY || '0'));
            }
        """,
    )
    image_view.on(
        "mousemove",
        js_handler="""
            (e) => {
                const el = e.currentTarget;
                if (el.dataset.panActive !== '1') return;
                e.preventDefault();
                const img = el.querySelector('img');
                const svg = el.querySelector('svg');
                if (!img || !svg) return;
                const startClientX = parseFloat(el.dataset.panStartClientX || '0');
                const startClientY = parseFloat(el.dataset.panStartClientY || '0');
                const startX = parseFloat(el.dataset.panStartX || '0');
                const startY = parseFloat(el.dataset.panStartY || '0');
                const scale = parseFloat(el.dataset.zoomScale || '1');
                const tx = startX + (e.clientX - startClientX);
                const ty = startY + (e.clientY - startClientY);
                el.dataset.panX = String(tx);
                el.dataset.panY = String(ty);
                const transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
                img.style.transformOrigin = '0 0';
                svg.style.transformOrigin = '0 0';
                img.style.transform = transform;
                svg.style.transform = transform;
            }
        """,
    )
    image_view.on(
        "mouseup",
        js_handler="""
            (e) => {
                const el = e.currentTarget;
                if (e.button === 2) {
                    el.dataset.panActive = '0';
                }
            }
        """,
    )
    image_view.on(
        "mouseleave",
        js_handler="""
            (e) => {
                const el = e.currentTarget;
                el.dataset.panActive = '0';
            }
        """,
    )
    image_view.on(
        "contextmenu",
        js_handler="""
            (e) => {
                e.preventDefault();
            }
        """,
    )

    prev_button.on_click(lambda _: move_frame(-1))
    next_button.on_click(lambda _: move_frame(1))
    reset_button.on_click(lambda _: reset_current_frame())
    save_button.on_click(lambda _: save_scope_json())
    video_select.on_value_change(handle_video_change)
    keypoint_select.on_value_change(handle_keypoint_change)
    current_keypoint_select.on_value_change(handle_current_keypoint_change)
    show_all_keypoints_checkbox.on_value_change(handle_show_all_keypoints_change)
    show_prediction_checkbox.on_value_change(handle_show_prediction_change)
    show_label_text_checkbox.on_value_change(handle_show_label_text_change)
    cutoff_input.on_value_change(handle_cutoff_change)
    ui.keyboard(on_key=handle_keyboard, active=True, repeating=True)

    refresh_keypoint_options()
    update_save_button_text()
    refresh_frames(reset_index=True)


@ui.page("/")
def main_page() -> None:
    build_ui()


def main() -> None:
    ui.run(
        title="Manual Correction",
        reload=False,
        host="0.0.0.0",
        port=1234,
        uvicorn_logging_level="info",
        access_log=True,
        show=False,
        show_welcome_message=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
