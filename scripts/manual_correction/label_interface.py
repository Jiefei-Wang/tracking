# %%
"""
Streamlit label editor z

"""

import base64
import copy
import json
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
    body { background: #1a1a2e; font-family: monospace; overflow: hidden; display: flex; align-items: center; justify-content: center; position: relative; }
    canvas { display: block; max-width: 100%; max-height: 100%; }
    #status {
        position: absolute; top: 8px; left: 8px;
        color: #a0c4ff; font-size: 12px;
        padding: 4px 8px; background: #12122a; min-height: 20px; border-radius: 4px;
    }
</style>
</head>
<body>
<div id="status">Loading…</div>
<canvas id="c"></canvas>
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

  status.textContent = "Move mode — drag points to reposition";
});

// Step 3: send display-space coords back to Python
function sendPositions() {
  var out = points.map(function(p) {
    return { x: Math.round(p.x), y: Math.round(p.y) };
  });
  window.parent.postMessage({ 
    isStreamlitMessage: true, 
    type: "streamlit:setComponentValue", 
    value: out 
  }, "*");
}

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

// ── Mouse ─────────────────────────────────────────────────────────────────
canvas.addEventListener("mousedown", function(e) {
  var r   = canvas.getBoundingClientRect();
  var cx  = e.clientX - r.left, cy = e.clientY - r.top;
  var idx = pointAt(cx, cy);
  if (idx >= 0) drag = { idx: idx, offX: cx - points[idx].x, offY: cy - points[idx].y };
});

canvas.addEventListener("mousemove", function(e) {
  var r  = canvas.getBoundingClientRect();
  var cx = e.clientX - r.left, cy = e.clientY - r.top;
  if (drag) {
    points[drag.idx].x = Math.max(0, Math.min(DISPLAY_W, cx - drag.offX));
    points[drag.idx].y = Math.max(0, Math.min(DISPLAY_H, cy - drag.offY));
    draw();
  } else {
    var h = pointAt(cx, cy);
    if (h !== hoveredIdx) {
      hoveredIdx = h;
      canvas.style.cursor = h >= 0 ? "grab" : "default";
      draw();
    }
  }
});

canvas.addEventListener("mouseup",    function() { if (drag) { drag = null; sendPositions(); } });
canvas.addEventListener("mouseleave", function() { if (drag) { drag = null; sendPositions(); } });

// ── Touch ─────────────────────────────────────────────────────────────────
canvas.addEventListener("touchstart", function(e) {
  e.preventDefault();
  var r = canvas.getBoundingClientRect(), t = e.touches[0];
  var cx = t.clientX - r.left, cy = t.clientY - r.top;
  var idx = pointAt(cx, cy);
  if (idx >= 0) drag = { idx: idx, offX: cx - points[idx].x, offY: cy - points[idx].y };
}, { passive: false });

canvas.addEventListener("touchmove", function(e) {
  e.preventDefault();
  if (!drag) return;
  var r = canvas.getBoundingClientRect(), t = e.touches[0];
  var cx = t.clientX - r.left, cy = t.clientY - r.top;
  points[drag.idx].x = Math.max(0, Math.min(DISPLAY_W, cx - drag.offX));
  points[drag.idx].y = Math.max(0, Math.min(DISPLAY_H, cy - drag.offY));
  draw();
}, { passive: false });

canvas.addEventListener("touchend", function() { if (drag) { drag = null; sendPositions(); } });
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


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

st.title("Label Editor")

col1, col2 = st.columns([1, 3])

with col1:
    label_files  = list_json_label_files()
    # Option to initialize brand new tracking profiles completely from scratch
    options = ["-- Create New Labels --"] + label_files
    
    if "label_choice_state" not in st.session_state:
        st.session_state.label_choice_state = options[0]
        
    label_choice = st.selectbox("Labels JSON", options, key="label_choice_state")

    if label_choice == "-- Create New Labels --":
        new_label_name = st.text_input("New Labels Filename", "new_labels.json")
        label_choice = str(Path("output/labels_edited") / new_label_name)

    default_images_dir = guess_images_dir(label_choice) or str(_PROJECT_ROOT / "output/extracted_frames")
    images_dir  = st.text_input("Images folder", default_images_dir)
    image_files = list_image_files(images_dir)
    st.write(f"{len(image_files)} images found")

    st.markdown("---")
    output_labels_dir  = st.text_input("Save labels to folder", str(_PROJECT_ROOT / "output/labels_edited"))
    output_name_suffix = st.text_input("Filename suffix", "_new")

    # Interactive Frame Stepping Buttons (Prev / Next Frame)
    if "selected_image_state" not in st.session_state and image_files:
        st.session_state.selected_image_state = image_files[0]
    elif image_files and st.session_state.selected_image_state not in image_files:
        st.session_state.selected_image_state = image_files[0]

    if image_files:
        try:
            curr_idx = image_files.index(st.session_state.selected_image_state)
        except ValueError:
            curr_idx = 0

        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("◀ Previous Frame", disabled=(curr_idx <= 0), use_container_width=True):
                st.session_state.selected_image_state = image_files[curr_idx - 1]
                st.rerun()
        with nav_col2:
            if st.button("Next Frame ▶", disabled=(curr_idx >= len(image_files) - 1), use_container_width=True):
                st.session_state.selected_image_state = image_files[curr_idx + 1]
                st.rerun()

        selected_image = st.selectbox("Choose image", image_files, key="selected_image_state")
    else:
        selected_image = st.selectbox("Choose image", [])

    filename   = Path(selected_image).name if selected_image else ""
    parsed_idx = digits_from_filename(filename) if filename else None

    if image_files and selected_image:
        st.caption(f"Frame {curr_idx + 1} of {len(image_files)} — {filename}")

    resolved_frame_idx = resolve_frame_idx(selected_image, labels_data=load_labels(label_choice), fallback=parsed_idx)

    frame_idx = st.number_input(
        "Frame index (override)",
        value=resolved_frame_idx,
        step=1,
        key=f"frame_idx_override_{selected_image}",
    )

    # ── Delete Frame ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Frame Management**")
    if selected_image and image_files:
        if st.button("Delete this frame", type="primary", use_container_width=True):
            deleted_path = Path(selected_image)
            if deleted_path.is_file():
                deleted_path.unlink()

            deleted_fidx = parsed_idx
            output_path = get_output_label_path(label_choice, output_labels_dir, output_name_suffix)
            if deleted_fidx is not None:
                lbl_data = load_labels(label_choice)
                lbl_data = [e for e in lbl_data if int(e.get("frame_idx", -1)) != deleted_fidx]
                save_labels(output_path, lbl_data)

            list_image_files.clear()
            remaining = list_image_files(images_dir)
            if remaining:
                new_idx = min(curr_idx, len(remaining) - 1)
                st.session_state.selected_image_state = remaining[new_idx]
            st.rerun()

    # ── Appearance Toolbar ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Appearance Toolbar**")
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        point_size = st.number_input("Point Size", min_value=2, max_value=30, value=4, step=1)
    with t_col2:
        text_size = st.number_input("Label Size", min_value=6, max_value=40, value=8, step=1)


with col2:
    if not selected_image:
        st.info("No image selected or images folder empty.")
        st.stop()

    # ── Image ──────────────────────────────────────────────────────────────
    try:
        with Image.open(selected_image) as source_image:
            pil_img = source_image.convert("RGB")
    except (OSError, ValueError) as exc:
        st.error(f"Unable to open image: {exc}")
        st.stop()
    orig_w, orig_h = pil_img.width, pil_img.height

    scale = min(1.0, MAX_W / orig_w)
    display_w = int(orig_w * scale)
    display_h = int(orig_h * scale)
    scale_x   = display_w / orig_w
    scale_y   = display_h / orig_h

    display_img = (
        pil_img.resize((display_w, display_h), Image.LANCZOS)
        if scale < 1.0 else pil_img
    )
    img_b64 = pil_to_base64(display_img)
    img_src = f"data:image/jpeg;base64,{img_b64}"

    # ── Labels ─────────────────────────────────────────────────────────────
    labels_data      = load_labels(label_choice)
    frame_map        = frame_list_to_map(labels_data)
    labels_for_frame = frame_map.get(int(frame_idx), {})

    sess_key = f"pts_{frame_idx}_{selected_image}"
    hist_key = f"hist_{frame_idx}_{selected_image}"   # undo/redo stack key
    hptr_key = f"hptr_{frame_idx}_{selected_image}"   # pointer into stack

    # ── History helpers ─────────────────────────────────────────────────────

    def _init_history(initial_points: List[Dict]):
        """Seed the stack with the loaded state."""
        st.session_state[hist_key] = [copy.deepcopy(initial_points)]
        st.session_state[hptr_key] = 0

    def push_history(new_points: List[Dict]):
        """
        Append a snapshot after a mutating action.
        Truncates any redo-future, updates the live pointer.
        """
        stack = st.session_state.get(hist_key, [])
        ptr   = st.session_state.get(hptr_key, 0)
        # drop any states that were ahead of the pointer (redo-branch)
        stack = stack[: ptr + 1]
        stack.append(copy.deepcopy(new_points))
        st.session_state[hist_key] = stack
        st.session_state[hptr_key] = len(stack) - 1

    def current_from_history() -> List[Dict]:
        stack = st.session_state.get(hist_key, [])
        ptr   = st.session_state.get(hptr_key, 0)
        return copy.deepcopy(stack[ptr]) if stack else []

    # ── Bootstrap session for this frame ────────────────────────────────────
    if sess_key not in st.session_state:
        initial = labels_to_points(labels_for_frame, scale_x=scale_x, scale_y=scale_y)
        st.session_state[sess_key] = initial
        _init_history(initial)
    elif hist_key not in st.session_state:
        # History wiped (e.g. hot-reload) but live state survives — re-seed.
        _init_history(st.session_state[sess_key])

    current_points: List[Dict] = st.session_state[sess_key]

    # ── Render component — 
    component_value = _canvas_component(
        # runtime data — received by JS via streamlit:render args
        img_src    = img_src,
        display_w  = display_w,
        display_h  = display_h,
        points     = current_points,   # display-space
        point_size = point_size,       # dynamic node sizing
        text_size  = text_size,        # dynamic text sizing
        # widget identity — changing this forces a fresh iframe
        key        = f"canvas_{sess_key}",
        default    = None,
        height     = display_h + 24,
    )

    # ── Merge returned positions into session_state ─────────────────────────
    if (
        isinstance(component_value, list)
        and len(component_value) == len(current_points)
    ):
        changed = False
        for i, pos in enumerate(component_value):
            new_x = float(pos.get("x", current_points[i]["x"]))
            new_y = float(pos.get("y", current_points[i]["y"]))
            if new_x != current_points[i]["x"] or new_y != current_points[i]["y"]:
                current_points[i]["x"] = new_x
                current_points[i]["y"] = new_y
                changed = True

        if changed:
            st.session_state[sess_key] = current_points
            push_history(current_points)  # drag is an undoable action
            st.rerun()

    # ── Undo / Redo bar (below canvas) ─────────────────────────────────────
    st.markdown("---")
    stack = st.session_state.get(hist_key, [])
    ptr   = st.session_state.get(hptr_key, 0)

    undo_col, redo_col = st.columns([1, 1])
    with undo_col:
        if st.button("↩ Undo", key=f"undo_{sess_key}", disabled=(ptr <= 0)):
            ptr -= 1
            st.session_state[hptr_key] = ptr
            st.session_state[sess_key] = current_from_history()
            st.rerun()
    with redo_col:
        if st.button("↪ Redo", key=f"redo_{sess_key}", disabled=(ptr >= len(stack) - 1)):
            ptr += 1
            st.session_state[hptr_key] = ptr
            st.session_state[sess_key] = current_from_history()
            st.rerun()

    # ── Add keypoint (constrained to the fixed schema) ──────────────────────
    st.markdown("---")
    used_labels       = {p["label"] for p in current_points}
    missing_keypoints = [k for k in KEYPOINT_SCHEMA if k not in used_labels]

    add_col1, add_col2 = st.columns([3, 1])
    with add_col1:
        kp_to_add = st.selectbox(
            "Add keypoint",
            missing_keypoints,
            key=f"add_select_{sess_key}",
            label_visibility="collapsed",
            placeholder=(
                "All keypoints already labeled"
                if not missing_keypoints
                else "Choose a keypoint to add"
            ),
            index=None if missing_keypoints else None,
        ) if missing_keypoints else None
    with add_col2:
        add_clicked = st.button(
            "Add point",
            key=f"add_btn_{sess_key}",
            disabled=not missing_keypoints or not kp_to_add,
        )

    if add_clicked and kp_to_add:
        new_pts = current_points + [{
            "label": kp_to_add,
            "x": display_w / 2.0,
            "y": display_h / 2.0,
        }]
        push_history(new_pts)
        st.session_state[sess_key] = new_pts
        st.rerun()

    # ── Point table ────────────────────────────────────────────────────────
    st.markdown("---")
    st.write("**Point positions** — drag on canvas to reposition, rename below.")

    deleted_label = None   # will hold the label name to delete, if any
    for pt in current_points:
        lbl  = pt["label"]
        cols = st.columns([2, 3, 1])

        with cols[0]:
            new_label = st.text_input(
                "Label",
                value=lbl,
                key=f"lbl_{sess_key}_{lbl}",
                label_visibility="collapsed",
            )
            if new_label != lbl:
                if not new_label.strip() or new_label in used_labels:
                    st.error("Enter a nonempty, unique keypoint label.")
                    continue
                updated_pts = [
                    {**p, "label": new_label} if p["label"] == lbl else p
                    for p in current_points
                ]
                push_history(updated_pts)
                st.session_state[sess_key] = updated_pts
                st.rerun()

        with cols[1]:
            st.caption(f"x={int(pt['x'] / scale_x)}  y={int(pt['y'] / scale_y)}")

        with cols[2]:
            if st.button("Delete", key=f"del_{sess_key}_{lbl}", help=f"Delete {lbl}"):
                deleted_label = lbl

    if deleted_label is not None:
        new_pts = [p for p in current_points if p["label"] != deleted_label]
        push_history(new_pts)
        st.session_state[sess_key] = new_pts
        st.rerun()

    # ── Save ───────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("Save labels"):
        for key, val in list(st.session_state.items()):
            if not key.startswith("pts_"):
                continue
            parts = key.split("_", 2)
            if len(parts) < 3:
                continue
            try:
                fidx = int(parts[1])
            except ValueError:
                continue

            frame_image_path = parts[2]
            try:
                f_scale_x, f_scale_y = compute_scale_for_image(frame_image_path, MAX_W)
            except Exception:
                f_scale_x, f_scale_y = scale_x, scale_y  # fallback

            frame_map[fidx] = {
                pt["label"] or f"pt_{i}": [
                    1,
                    int(pt["x"] / f_scale_x),
                    int(pt["y"] / f_scale_y),
                ]
                for i, pt in enumerate(val)
            }

        out_list    = map_to_frame_list(frame_map)
        output_path = get_output_label_path(
            label_choice, output_labels_dir, output_name_suffix
        )

        existing = []
        if Path(output_path).exists():
            with open(output_path) as fh:
                existing = json.load(fh)

        if existing == out_list:
            st.info(f"No changes detected — {output_path} not modified.")
        else:
            save_labels(output_path, out_list)
            # FClear cache so newly created/saved file registers in the selector instantly
            list_json_label_files.clear()
            st.success(f"Saved labels to {output_path}")
# %%
