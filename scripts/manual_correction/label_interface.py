# %%
"""
Streamlit label editor z

"""

import base64
import copy
import json
import cv2
import math
import hashlib
import os
import re
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from streamlit.runtime.scriptrunner import get_script_run_ctx

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
import sys as _sys
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

# Support the IDE's ordinary Python run command as well as `streamlit run`.
if __name__ == "__main__" and get_script_run_ctx(suppress_warning=True) is None:
    raise SystemExit(subprocess.call([
        _sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
        *_sys.argv[1:],
    ], cwd=str(_PROJECT_ROOT)))

try:
    st.set_page_config(layout="wide")
except Exception:
    pass

KEYPOINT_SCHEMA = [
    "nose", "head", "spine1", "spine2", "spine3", "tailbase", "tail1",
    "tail2", "tail_tip", "L_shoulder", "L_frontpaw", "R_shoulder",
    "R_frontpaw", "L_hip", "L_knee", "L_backpaw", "R_hip", "R_knee",
    "R_backpaw",
]

MAX_W = 900


# ─────────────────────────────────────────────────────────────────────────────
# Static component HTML 
# ─────────────────────────────────────────────────────────────────────────────

COMPONENT_DIR = Path(__file__).resolve().parent / "custom_canvas_component"
COMPONENT_DIR.mkdir(exist_ok=True)

_STATIC_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; }
    body { background: #1a1a2e; font-family: monospace; overflow: hidden; display: flex; align-items: flex-start; gap: 12px; padding: 8px; position: relative; }
    #image-area { flex: 1; min-width: 0; position: relative; }
    canvas { display: block; width: 100%; height: auto; touch-action: none; }
    #panel { width: 180px; flex-shrink: 0; color: #eee; font: 13px sans-serif; }
    #panel p { margin: 8px 0; line-height: 1.4; }
    #palette { overflow-y: auto; max-height: 360px; }
    .keypoint { padding: 7px; margin: 4px 0; background: #30304b; border-radius: 4px; cursor: grab; touch-action: none; }
    #trash { margin-top: 12px; padding: 18px 8px; border: 2px dashed #ff7e91; border-radius: 6px; }
    #trash.active { background: #70283b; }
    #reset { width: 100%; margin-top: 12px; padding: 9px; cursor: pointer; }
    #reset:disabled { opacity: .4; cursor: default; }
    #timeline { width: 100%; margin-top: 12px; accent-color: #80ffdb; }
    #ticks { position: relative; height: 10px; margin: 0 7px; }
    #ticks span { position: absolute; width: 2px; height: 8px; background: #ffd60a; }
    #frame-caption { color: #eee; font: 12px sans-serif; margin-top: 5px; }
    #status {
        position: absolute; top: 8px; left: 8px;
        color: #a0c4ff; font-size: 12px;
        padding: 4px 8px; background: #12122a; min-height: 20px; border-radius: 4px;
    }
</style>
</head>
<body>
<div id="image-area"><div id="status">Loading…</div><canvas id="c"></canvas><input id="timeline" type="range" min="0" step="1" aria-label="Video frame"><div id="ticks"></div><div id="frame-caption"></div></div>
<aside id="panel"><strong>Keypoints</strong><p>Drag a keypoint onto the image to add it.</p><div id="palette"></div><div id="trash">Drag a point here to delete</div><button id="reset" disabled>Reset current frame</button></aside>
<script>
var POINT_R = 7;
var TEXT_SIZE = 11;
var COLORS  = [
  "#ff4d6d","#ffd60a","#80ffdb","#4cc9f0","#f72585",
  "#7b2fff","#06d6a0","#ff9f1c","#e9c46a","#a8dadc"
];

var canvas = document.getElementById("c");
var ctx    = canvas.getContext("2d");
var status = document.getElementById("status");

var schema = [];
var eventCounter = 0;
var instanceId = Math.random().toString(36).slice(2);
var points     = [];
var drag       = null;
var hoveredIdx = -1;
var DISPLAY_W  = 0;
var DISPLAY_H  = 0;
var img        = new Image();

img.onload = function() { draw(); };

// ── Streamlit protocol ────────────────────────────────────────────────────
// Step 1: announce ready
window.parent.postMessage({ 
  isStreamlitMessage: true, 
  type: "streamlit:componentReady", 
  apiVersion: 1 
}, "*");
// Step 2: receive args on every render
window.addEventListener("message", function(e) {
  if (!e.data || e.data.type !== "streamlit:render") return;
  var args = e.data.args;

  document.getElementById("reset").disabled = !args.can_reset;
  var timeline = document.getElementById("timeline");
  timeline.max = Math.max(0, args.frame_count - 1);
  timeline.value = args.frame_idx;
  timeline.disabled = args.frame_count <= 1;
  document.getElementById("frame-caption").textContent = "Frame " + args.frame_idx + " / " + timeline.max;
  var ticks = document.getElementById("ticks");
  ticks.replaceChildren();
  (args.label_indices || []).forEach(function(index) {
    var tick = document.createElement("span");
    tick.style.left = (100 * index / Math.max(1, args.frame_count - 1)) + "%";
    tick.title = "Labeled frame " + index;
    ticks.appendChild(tick);
  });
  schema = args.schema || [];
  DISPLAY_W = args.display_w;
  DISPLAY_H = args.display_h;
  POINT_R   = args.point_size || 7;
  TEXT_SIZE = args.text_size || 11;

  canvas.width  = DISPLAY_W;
  canvas.height = DISPLAY_H;

  // Rebuild points array from args (display-space coords, labels)
  points = (args.points || []).map(function(p, i) {
    return {
      label : p.label,
      x     : p.x,
      y     : p.y,
      color : COLORS[i % COLORS.length]
    };
  });

  // Reload image only when it actually changes
  var newSrc = args.img_src;
  if (img.src !== newSrc) {
    img.src = newSrc;               // onload → draw()
    if (img.complete && img.naturalWidth) { draw(); }
  } else {
    draw();
  }

  status.textContent = "Drag points to move or delete";
  renderPalette();
  window.parent.postMessage({isStreamlitMessage:true, type:"streamlit:setFrameHeight", height:Math.max(canvas.getBoundingClientRect().height + 85, document.getElementById("panel").scrollHeight + 20)}, "*");
});

// Step 3: send display-space coords back to Python
function sendPositions() {
  var out = points.map(function(p) {
    return { label: p.label, x: p.x, y: p.y };
  });
  window.parent.postMessage({ 
    isStreamlitMessage: true, 
    type: "streamlit:setComponentValue", 
    value: {event_id: instanceId + ":" + (++eventCounter), points: out}
  }, "*");
}

function sendAction(action, extra) {
  window.parent.postMessage({isStreamlitMessage:true, type:"streamlit:setComponentValue",
    value:Object.assign({event_id:instanceId + ":" + (++eventCounter), action:action}, extra || {})}, "*");
}
document.getElementById("reset").addEventListener("click", function() { sendAction("reset"); });
document.getElementById("timeline").addEventListener("input", function(e) {
  document.getElementById("frame-caption").textContent = "Frame " + e.target.value + " / " + e.target.max;
});
document.getElementById("timeline").addEventListener("change", function(e) {
  sendAction("navigate", {frame_idx:Number(e.target.value)});
});

// ── Hit test ──────────────────────────────────────────────────────────────
function pointAt(cx, cy) {
  for (var i = points.length - 1; i >= 0; i--) {
    var dx = cx - points[i].x, dy = cy - points[i].y;
    if (Math.sqrt(dx*dx + dy*dy) <= POINT_R + 3) return i;
  }
  return -1;
}

// ── Draw ──────────────────────────────────────────────────────────────────
function draw() {
  if (!DISPLAY_W || !DISPLAY_H) return;
  ctx.clearRect(0, 0, DISPLAY_W, DISPLAY_H);
  if (img.complete && img.naturalWidth) {
    ctx.drawImage(img, 0, 0, DISPLAY_W, DISPLAY_H);
  }

  points.forEach(function(p, i) {
    var r = (i === hoveredIdx || (drag && drag.idx === i)) ? POINT_R + 3 : POINT_R;

    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.7)";
    ctx.shadowBlur  = 6;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle   = p.color;
    ctx.fill();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth   = 1.5;
    ctx.stroke();
    ctx.restore();

    var label = p.label || "?";
    ctx.font = "bold " + TEXT_SIZE + "px monospace";
    var tw = ctx.measureText(label).width;
    var px = p.x + r + 4, py = p.y - r - 2;
    
    // Dynamic text box sizing based on font size
    var boxHeight = TEXT_SIZE + 6;
    var boxOffsetY = TEXT_SIZE + 1;
    
    ctx.fillStyle = "rgba(20,20,40,0.82)";
    ctx.beginPath();
    ctx.roundRect(px - 3, py - boxOffsetY, tw + 8, boxHeight, 4);
    ctx.fill();
    ctx.fillStyle = p.color;
    ctx.fillText(label, px + 1, py);
  });
}

function renderPalette() {
  var palette = document.getElementById("palette");
  palette.replaceChildren();
  schema.filter(function(label) { return !points.some(function(p) { return p.label === label; }); }).forEach(function(label) {
    var item = document.createElement("div");
    item.className = "keypoint";
    item.textContent = label;
    item.addEventListener("pointerdown", function(e) {
      e.preventDefault();
      drag = {label: label};
      item.setPointerCapture(e.pointerId);
    });
    palette.appendChild(item);
  });
  if (!palette.children.length) palette.textContent = "All keypoints placed";
}
function position(e) {
  var r = canvas.getBoundingClientRect();
  return {x:(e.clientX-r.left)*DISPLAY_W/r.width, y:(e.clientY-r.top)*DISPLAY_H/r.height};
}
function inside(e, element) {
  var r = element.getBoundingClientRect();
  return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
}
canvas.addEventListener("pointerdown", function(e) {
  var p = position(e), idx = pointAt(p.x, p.y);
  if (idx < 0) return;
  e.preventDefault();
  drag = {idx:idx, offX:p.x-points[idx].x, offY:p.y-points[idx].y, original:{...points[idx]}};
  canvas.setPointerCapture(e.pointerId);
});
window.addEventListener("pointermove", function(e) {
  var p = position(e);
  if (drag && drag.idx !== undefined) {
    points[drag.idx].x = Math.max(0, Math.min(DISPLAY_W, p.x-drag.offX));
    points[drag.idx].y = Math.max(0, Math.min(DISPLAY_H, p.y-drag.offY));
  }
  hoveredIdx = drag ? -1 : pointAt(p.x,p.y);
  document.getElementById("trash").classList.toggle("active", !!drag && drag.idx !== undefined && inside(e, document.getElementById("trash")));
  draw();
});
window.addEventListener("pointerup", function(e) {
  if (!drag) return;
  var changed = false;
  if (drag.idx !== undefined) {
    if (inside(e, document.getElementById("trash"))) {
      points.splice(drag.idx, 1);
      changed = true;
    } else if (inside(e, canvas)) {
      changed = true;
    } else {
      points[drag.idx] = drag.original;
    }
  } else if (inside(e, canvas) && !points.some(function(p) {return p.label === drag.label;})) {
    var p = position(e);
    points.push({label:drag.label, x:p.x, y:p.y, color:COLORS[points.length % COLORS.length]});
    changed = true;
  }
  drag = null;
  document.getElementById("trash").classList.remove("active");
  renderPalette();
  draw();
  if (changed) sendPositions();
});
window.addEventListener("pointercancel", function() {
  if (drag && drag.idx !== undefined) points[drag.idx] = drag.original;
  drag = null;
  document.getElementById("trash").classList.remove("active");
  draw();
});
</script>
</body>
</html>"""

# Write the static HTML once at startup 
(COMPONENT_DIR / "index.html").write_text(_STATIC_HTML, encoding="utf-8")

# Declare component once 
_canvas_component = components.declare_component(
    "label_canvas",
    path=str(COMPONENT_DIR),
)


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def list_json_label_files() -> List[str]:
    candidates = []
    # Added output/labels_edited so you can reload and resume working on your changes later
    for base in ("input/labels", "output/predicted_frames", "output/labels_edited"):
        p = _PROJECT_ROOT / base
        if p.exists() and p.is_dir():
            for f in sorted(p.glob("*.json")):
                candidates.append(str(f))
    return candidates


@st.cache_data
def list_image_files(images_dir: str) -> List[str]:
    p = Path(images_dir)
    if not p.exists() or not p.is_dir():
        return []
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(sorted(p.glob(ext)))
    # Deduplicate (case-insensitive FS can double-match)
    seen = set()
    out  = []
    for f in files:
        key = str(f).lower()
        if key not in seen:
            seen.add(key)
            out.append(str(f))
    return out


def load_labels(path: str) -> List[Dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    with open(path) as fh:
        return json.load(fh)


def save_labels(path: str, data: List[Dict[str, Any]]):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def get_output_label_path(original_path: str, output_dir: str, suffix: str) -> str:
    stem = Path(original_path).stem
    ext  = Path(original_path).suffix or ".json"
    return str(Path(output_dir) / f"{stem}{suffix}{ext}")


def frame_list_to_map(frames: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    return {
        int(e.get("frame_idx", -1)): e.get("labels", {})
        for e in frames
        if int(e.get("frame_idx", -1)) >= 0
    }


def map_to_frame_list(m: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"frame_idx": idx, "labels": m[idx]} for idx in sorted(m)]


def digits_from_filename(name: str) -> Optional[int]:
    nums = re.findall(r"(\d+)", name)
    return int(nums[-1]) if nums else None


def resolve_frame_idx(image_path: str, labels_data: List[Dict[str, Any]], fallback: Optional[int] = None) -> int:
    parsed_idx = digits_from_filename(Path(image_path).name) if image_path else None
    if parsed_idx is None:
        parsed_idx = fallback

    label_idxs = sorted(
        {
            int(entry.get("frame_idx"))
            for entry in labels_data
            if isinstance(entry, dict) and entry.get("frame_idx") is not None
        }
    )
    if not label_idxs:
        return int(parsed_idx) if parsed_idx is not None else int(fallback or 0)

    if parsed_idx is not None:
        return int(parsed_idx)

    if label_idxs:
        return int(label_idxs[0])

    return int(fallback or 0)


def guess_images_dir(label_path: str) -> Optional[str]:
    if not label_path:
        return None
    candidate = _PROJECT_ROOT / "output" / "extracted_frames" / Path(label_path).stem
    return str(candidate) if candidate.is_dir() else None


def pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def labels_to_points(
    lbls: Dict[str, Any],
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Convert label dict → list of point dicts in DISPLAY-space.
    scale_x / scale_y convert original-image coords to display coords on load.
    """
    pts = []
    for name, v in lbls.items():
        if not (isinstance(v, list) and len(v) >= 3):
            continue
        if not bool(v[0]):
            continue
        x = v[1][0] if isinstance(v[1], (list, tuple)) else v[1]
        y = v[2][0] if isinstance(v[2], (list, tuple)) else v[2]
        if x is not None and y is not None:
            pts.append({
                "label": name,
                "x": float(x) * scale_x,
                "y": float(y) * scale_y,
            })
    return pts


def compute_scale_for_image(image_path: str, max_w: int = MAX_W):
    """
    Recompute the display<->original scale factor for a given image path.
    Used at save time so each cached frame is converted back to original-image
    coordinates using ITS OWN resolution, not whatever image happens to be
    on screen when Save is clicked.
    """
    with Image.open(image_path) as im:
        w, h = im.width, im.height
    s = min(1.0, max_w / w)
    disp_w, disp_h = int(w * s), int(h * s)
    return disp_w / w, disp_h / h


# Video documents keep a saved snapshot and a working copy in original pixels.
# Merely viewing a frame never inserts a row.
def video_label_path(video_path):
    name = Path(video_path).stem + ".json"
    for folder in ("input/labels", "output/labels_edited", "output/predicted_frames"):
        candidate = _PROJECT_ROOT / folder / name
        if candidate.exists():
            return candidate
    return _PROJECT_ROOT / "input/labels" / name


def open_document(video_path):
    path = video_label_path(video_path)
    if not path.exists():
        save_labels(str(path), [])
    frames = frame_list_to_map(load_labels(str(path)))
    return {"path": str(path), "saved": copy.deepcopy(frames), "frames": frames, "history": {}}


def save_document(document):
    save_labels(document["path"], map_to_frame_list(document["frames"]))
    document["saved"] = copy.deepcopy(document["frames"])


def change_frame(document, index, labels):
    before = copy.deepcopy(document["frames"].get(index))
    if before == labels:
        return
    history = document["history"].setdefault(index, {"states": [before], "position": 0})
    history["states"] = history["states"][:history["position"] + 1] + [copy.deepcopy(labels)]
    history["position"] += 1
    if labels is None:
        document["frames"].pop(index, None)
    else:
        document["frames"][index] = labels


@st.cache_data
def video_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            raise ValueError("Unable to open video")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise ValueError("Video has no readable frames")
        return count
    finally:
        cap.release()


@st.cache_data(max_entries=16)
def read_video_frame(video_path, index):
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Unable to read frame {index}")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()


st.title("Label Editor")
videos = sorted(str(p) for p in (_PROJECT_ROOT / "videos").rglob("*")
                if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"})
if not videos:
    st.info("No videos found in the videos folder.")
    st.stop()
state = st.session_state
if "video_editor_active" not in state:
    state.video_editor_active = videos[0]
    state.video_editor_document = open_document(videos[0])
    state.video_editor_frame = min(state.video_editor_document["frames"], default=0)
if "video_editor_selection_update" in state:
    state.video_editor_choice = state.pop("video_editor_selection_update")
if "video_editor_frame_update" in state:
    state.video_editor_frame = state.pop("video_editor_frame_update")


def switch_video(target):
    document = open_document(target)
    state.video_editor_active = target
    state.video_editor_document = document
    state.video_editor_frame_update = min(document["frames"], default=0)
    state.video_editor_selection_update = target


@st.dialog("Save changes before switching video?", dismissible=False)
def confirm_video_switch(target):
    st.write(f"Save changes to {Path(state.video_editor_active).stem}?")
    yes, no, cancel = st.columns(3)
    if yes.button("Yes", type="primary", use_container_width=True):
        try:
            save_document(state.video_editor_document)
            switch_video(target)
        except (OSError, ValueError) as exc:
            st.error(str(exc))
            return
        st.rerun()
    if no.button("No", use_container_width=True):
        switch_video(target)
        st.rerun()
    if cancel.button("Cancel", use_container_width=True):
        state.video_editor_selection_update = state.video_editor_active
        st.rerun()


left, image_column = st.columns([1, 4])
with left:
    selected = st.selectbox("Video name", videos, format_func=lambda v: Path(v).stem, key="video_editor_choice")
    if selected != state.video_editor_active:
        if state.video_editor_document["frames"] != state.video_editor_document["saved"]:
            confirm_video_switch(selected)
        else:
            switch_video(selected)
            st.rerun()
    document = state.video_editor_document
    try:
        frame_count = video_frame_count(state.video_editor_active)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    state.video_editor_frame = max(0, min(frame_count - 1, int(state.get("video_editor_frame", state.get("video_editor_position", 0)))))
    st.markdown(
        "<style>.st-key-video_editor_frame button { display: none; }</style>",
        unsafe_allow_html=True,
    )
    json_indices = sorted(document["frames"])
    current_json_index = (
        json_indices.index(state.video_editor_frame) + 1
        if state.video_editor_frame in document["frames"] else "—"
    )
    frame_index_label = (
        f"Frame index (total in json: {len(json_indices)}, "
        f"current json index: {current_json_index})"
    )
    frame_idx = int(st.number_input(frame_index_label, min_value=0, max_value=frame_count - 1,
                                   step=1, key="video_editor_frame"))
    state.video_editor_position = frame_idx
    indices = sorted(i for i in document["frames"] if 0 <= i < frame_count)
    previous = [i for i in indices if i < frame_idx]
    following = [i for i in indices if i > frame_idx]
    prev_col, next_col = st.columns(2)
    if prev_col.button("◀ Previous", disabled=not previous, use_container_width=True):
        state.video_editor_frame_update = previous[-1]
        st.rerun()
    if next_col.button("Next ▶", disabled=not following, use_container_width=True):
        state.video_editor_frame_update = following[0]
        st.rerun()
    if st.button("Save", type="primary", disabled=document["frames"] == document["saved"], use_container_width=True):
        try:
            save_document(document)
        except OSError as exc:
            st.error(f"Unable to save labels: {exc}")
        else:
            st.rerun()
    st.markdown("---")
    st.markdown("**Frame Management**")
    if st.button("Delete this frame", disabled=frame_idx not in document["frames"], use_container_width=True):
        change_frame(document, frame_idx, None)
        st.rerun()
    st.markdown("---")
    st.markdown("**Appearance Toolbar**")
    a, b = st.columns(2)
    point_size = a.number_input("Point Size", min_value=2, max_value=30, value=4, step=1)
    text_size = b.number_input("Label Size", min_value=6, max_value=40, value=8, step=1)

with image_column:
    try:
        pil_img = read_video_frame(state.video_editor_active, frame_idx)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    orig_w, orig_h = pil_img.size
    scale = min(1.0, MAX_W / orig_w)
    display_w, display_h = int(orig_w * scale), int(orig_h * scale)
    scale_x, scale_y = display_w / orig_w, display_h / orig_h
    display_img = pil_img.resize((display_w, display_h), Image.Resampling.LANCZOS)
    labels = document["frames"].get(frame_idx, {})
    points = labels_to_points(labels, scale_x, scale_y)
    frame_id = hashlib.sha256(f"{state.video_editor_active}:{frame_idx}".encode()).hexdigest()
    value = _canvas_component(
        img_src=f"data:image/jpeg;base64,{pil_to_base64(display_img)}",
        display_w=display_w, display_h=display_h, schema=KEYPOINT_SCHEMA,
        points=points, point_size=point_size, text_size=text_size,
        frame_idx=frame_idx, frame_count=frame_count, label_indices=indices,
        can_reset=document["frames"].get(frame_idx) != document["saved"].get(frame_idx),
        key=f"video_canvas_{frame_id}", default=None,
    )
    event_key = f"video_event_{frame_id}"
    if isinstance(value, dict) and value.get("event_id") != state.get(event_key):
        state[event_key] = value.get("event_id")
        if value.get("action") == "navigate":
            target = value.get("frame_idx")
            if isinstance(target, int) and 0 <= target < frame_count:
                state.video_editor_frame_update = target
                st.rerun()
        elif value.get("action") == "reset":
            change_frame(document, frame_idx, copy.deepcopy(document["saved"].get(frame_idx)))
            st.rerun()
        elif isinstance(value.get("points"), list):
            incoming = value["points"]
            allowed = set(KEYPOINT_SCHEMA) | set(labels)
            try:
                valid = (len({p["label"] for p in incoming}) == len(incoming)
                         and all(p["label"] in allowed and math.isfinite(p["x"]) and math.isfinite(p["y"])
                                 and 0 <= p["x"] <= display_w and 0 <= p["y"] <= display_h for p in incoming))
                if valid:
                    # Preserve invisible annotations and exact original coordinates
                    # for points that did not move during this gesture.
                    updated = {k: copy.deepcopy(v) for k, v in labels.items() if k not in {p["label"] for p in points}}
                    existing = {p["label"]: p for p in points}
                    for p in incoming:
                        old = existing.get(p["label"])
                        if old and old["x"] == p["x"] and old["y"] == p["y"]:
                            updated[p["label"]] = copy.deepcopy(labels[p["label"]])
                        else:
                            updated[p["label"]] = [
                                1,
                                min(orig_w - 1, max(0, round(p["x"] / scale_x))),
                                min(orig_h - 1, max(0, round(p["y"] / scale_y))),
                            ]
                    if updated or frame_idx in document["frames"]:
                        change_frame(document, frame_idx, updated)
                        st.rerun()
            except (KeyError, TypeError, ValueError):
                st.error("Invalid keypoint update; please try again.")
    history = document["history"].get(frame_idx, {"states": [], "position": 0})
    undo, redo = st.columns(2)
    direction = 0
    if undo.button("↩ Undo", disabled=history["position"] <= 0):
        direction = -1
    if redo.button("↪ Redo", disabled=history["position"] >= len(history["states"]) - 1):
        direction = 1
    if direction:
        history["position"] += direction
        snapshot = copy.deepcopy(history["states"][history["position"]])
        if snapshot is None:
            document["frames"].pop(frame_idx, None)
        else:
            document["frames"][frame_idx] = snapshot
        st.rerun()
