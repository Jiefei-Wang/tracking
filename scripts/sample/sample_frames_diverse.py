#!/usr/bin/env python3
"""
Sample visually diverse frames from videos using brightness filtering
and embedding-based diversity selection.

For each video:
  1. Count existing extracted frames for the video (labels / JPGs).
  2. Sample a pool of evenly-spaced frames from the remaining video.
  3. Filter by mean grayscale brightness (discard dark frames).
  4. Downsample each bright frame to 32x32 grayscale -> 1024-d vector.
  5. Pick the shortfall (target - existing) via greedy furthest-point sampling,
     maximizing distance to this video's existing frames AND every other
     video's already-sampled frames.

Existing labels preserved as much as possible, but two cleanups run
first:
  * Dark existing frames removed from the labels (unless --keep-dark is given)
  * Near-duplicate existing frames (closer than --dedupe-existing in the 32x32
    embedding space) are collapsed to the representative carrying the most
    visible keypoints (set --dedupe-existing 0 to disable)
Only after these cleanups are new frames sampled (use --resample to wipe a video and
re-pick everything from scratch)

Three operating modes (mutually exclusive):
  * --filter-only     : clean up the existing labels (drop dark frames, dedupe
                        near-duplicates) and write them back. No new frames are
                        sampled. Fast: never decodes the pool.
  * --target-frames N : sample up to a TOTAL of N frames per video
  * --min-diversity D : keep sampling until even the best remaining candidate is
                        closer than D to everything already chosen (distance in
                        32x32 embedding space, ~max 32.0) (use --max-frames to cap it)

By default a sampling run BOTH prunes the existing set and adds new frames.
Pass --append-only to keep existing frames untouched (skip dark/dupe pruning)
and only sample + append new ones (--keep-dark / --dedupe-existing are then
ignored). Mutually exclusive with --filter-only.

Usage:
    Filter existing only: drop dark + dedupe, write cleaned labels, sample nothing
    python scripts/sample/sample_frames_diverse.py \
        --video-names "Rat 4 2025-11-12 12-25-01" \
        --filter-only --min-brightness 20 --dedupe-existing 0.5

    Append only: keep existing labels exactly as-is, sample + append toward 300 total
    python scripts/sample/sample_frames_diverse.py \
        --video-names "Rat 4 2025-11-12 12-25-01" \
        --append-only --target-frames 300 --min-brightness 20

    Target of 300 frames, filter existing for brightness and diversity (default: both)
    python scripts/sample/sample_frames_diverse.py \
        --video-names "Rat 4 2025-11-12 12-25-01" "Rat 9 2025-11-11 11-09-35" \
        --target-frames 300 \
        --min-brightness 50 \
        --pool-factor 3

    Keeps adding frames until best remaining candidate is >= 1.5, filter existing for brightness & div
    python scripts/sample/sample_frames_diverse.py \
        --video-names "Rat 4 2025-11-12 12-25-01" \
        --min-diversity 1.5 --max-frames 500 --dedupe-existing 1.0

    keeps adding frames till div threshold, filters existing for div only & keeps dark
    python scripts/sample/sample_frames_diverse.py \
        --video-names "Rat 4 2025-11-12 12-25-01" \
        --min-diversity 1.5 --max-frames 500 --dedupe-existing 1.0 --keep-dark
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VIDEOS_DIR = PROJECT_ROOT / "videos"
EXTRACTED_DIR = PROJECT_ROOT / "output" / "extracted_frames"
LABELS_DIR = PROJECT_ROOT / "input" / "labels"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
EMBED_SIZE = 32


def find_video(video_name: str) -> Path | None:
    # Match a video file by its stem across any supported extension.
    for ext in VIDEO_EXTENSIONS:
        p = VIDEOS_DIR / f"{video_name}{ext}"
        if p.is_file():
            return p
    return None


def get_video_info(video_path: Path) -> tuple[int, float]:
    # OpenCV metadata: frame count (for even spacing / pool sizing) and FPS.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return total, fps
    finally:
        cap.release()


def even_indices(total_frames: int, count: int) -> list[int]:
    # Pick `count` frame indices spread evenly across the video; is the sampling pool
    if count >= total_frames:
        return list(range(total_frames))
    stride = total_frames / count
    return [min(int(round(i * stride)), total_frames - 1) for i in range(count)]


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    # Random access decode of a single frame (reused for pool candidates and 
    # as a fallback when an existing label has no JPG on disk).
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def compute_brightness(frame: np.ndarray) -> float:
    # Mean grayscale intensity over the whole frame. Used to drop "lights off"
    # / dark frames that carry no useful content.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def embed_frame(frame: np.ndarray) -> np.ndarray:
    # Normalize the frame to a fixed 32x32 grayscale vector so two frames can
    # be compared with a plain Euclidean distance. Scale is ~0-1.0, so L2
    # distances across this space range roughly 0.0-32.0.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA)
    vec = small.astype(np.float32).flatten()
    vec /= 255.0
    return vec


def select_additional(
    vectors: np.ndarray,
    indices: list[int],
    fixed_vectors: np.ndarray,
    count: int,
) -> tuple[list[int], str]:
    """Greedily pick `count` candidates farthest from `fixed_vectors` and each other."""
    # --target-frames mode: never rejects for low diversity, just takes the
    # `count` most distinct remaining candidates (or fewer if pool runs out).
    if count <= 0:
        return [], "target already met"
    if len(indices) == 0:
        return [], "no candidates"

    count = min(count, len(indices))
    return _greedy_select(vectors, indices, fixed_vectors, count=count)


def select_until_min_diversity(
    vectors: np.ndarray,
    indices: list[int],
    fixed_vectors: np.ndarray,
    min_diversity: float,
    max_frames: int | None,
) -> tuple[list[int], str]:
    """Keep picking the farthest candidate while its distance to the existing
    set stays >= min_diversity. Stops when the best candidate is too similar."""
    # --min-diversity mode: keeps adding frames until even the best remaining
    # candidate is < min_diversity away from the accepted set (or --max-frames
    # / pool exhaustion stops it first).
    if len(indices) == 0:
        return [], "no candidates"
    return _greedy_select(vectors, indices, fixed_vectors,
                          min_diversity=min_diversity, max_frames=max_frames)


def _greedy_select(
    vectors: np.ndarray,
    indices: list[int],
    fixed_vectors: np.ndarray,
    count: int | None = None,
    min_diversity: float | None = None,
    max_frames: int | None = None,
) -> list[int]:
    # Farthest-point sampling (a greedy max-min heuristic): each round picks the
    # candidate whose closest distance to everything already accepted (fixed
    # anchors + previously selected) is largest. Picking the max of the min
    # distances spreads new frames as far as possible from what exists.
    fixed = np.asarray(fixed_vectors, dtype=np.float32)
    remaining = set(range(len(indices)))

    # dists[i] = distance from candidate i to the nearest already-chosen frame.
    # Initialized against the fixed anchors; if there are none (e.g. a full
    # resample of the very first video), seed with distance to the centroid so
    # the first picks still spread out.
    if len(fixed) > 0:
        dists: np.ndarray = cdist(vectors, fixed, metric="euclidean").min(axis=1)
    else:
        center = vectors.mean(axis=0)
        dists = np.linalg.norm(vectors - center, axis=1)

    selected = []
    limit_reason = "exhausted pool"
    while remaining:
        # Stop conditions: reached the requested count, hit --max-frames, or
        # (diversity mode) even the best candidate is now too similar.
        if count is not None and len(selected) >= count:
            limit_reason = f"reached target count {count}"
            break
        if max_frames is not None and len(selected) >= max_frames:
            limit_reason = f"reached max frames {max_frames}"
            break

        # Take the candidate farthest from everything picked so far.
        best = max(remaining, key=lambda i: float(dists[i]))
        best_dist = float(dists[best])

        if min_diversity is not None and best_dist < min_diversity:
            limit_reason = f"best candidate diversity {best_dist:.2f} < cutoff {min_diversity:.2f}"
            break

        selected.append(indices[best])
        remaining.remove(best)

        # The newly picked frame becomes a new anchor: shrink every remaining
        # candidate's distance if this frame is closer than its current best.
        new_dist = np.linalg.norm(vectors[best] - vectors, axis=1)
        for i in remaining:
            if new_dist[i] < dists[i]:
                dists[i] = new_dist[i]

    return selected, limit_reason


def embed_jpgs(video_dir: Path, min_brightness: float = 0.0) -> tuple[list[int], np.ndarray]:
    # Embed every extracted JPG in one video's folder. The filename is the
    # frame index, which is what identifies each frame in the labels.
    indices = []
    vectors = []
    for img_path in sorted(video_dir.glob("*.jpg")):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        if compute_brightness(frame) < min_brightness:
            continue
        try:
            idx = int(img_path.stem)
        except ValueError:
            continue
        indices.append(idx)
        vectors.append(embed_frame(frame))
    if not indices:
        return [], np.empty((0, EMBED_SIZE * EMBED_SIZE), dtype=np.float32)
    return indices, np.array(vectors, dtype=np.float32)


def embed_other_videos(excluded_video: str, min_brightness: float = 0.0) -> np.ndarray:
    """Embeddings of every already-sampled frame in other videos' extract dirs."""
    # Cross-video diversity: the current video's pool is also kept away from
    # frames already sampled for every OTHER video, so one frame choice doesn't
    # get reused across the whole dataset. The JPGs on disk ARE the other
    # videos' current selection -- which is why the extract dir must be kept in
    # sync with the labels (extract_frames.py prunes stale JPGs).
    if not EXTRACTED_DIR.is_dir():
        return np.empty((0, EMBED_SIZE * EMBED_SIZE), dtype=np.float32)
    vectors = []
    for video_dir in sorted(EXTRACTED_DIR.iterdir()):
        if not video_dir.is_dir() or video_dir.name == excluded_video:
            continue
        _, vecs = embed_jpgs(video_dir, min_brightness)
        vectors.append(vecs)
    if not vectors:
        return np.empty((0, EMBED_SIZE * EMBED_SIZE), dtype=np.float32)
    return np.concatenate(vectors, axis=0)


def embed_existing_frames(
    video_path: Path,
    video_name: str,
    indices: list[int],
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Embed this video's existing frames, reading JPGs when available.

    Returns (indices, vectors, brightness) aligned per-frame."""
    # These are the frames already in the labels: they're "kept" (not re-picked)
    # and act as anchors the new picks must stay away from. Reads the JPG when
    # present; otherwise decodes from the video just for the embedding.
    out_dir = EXTRACTED_DIR / video_name
    kept_idx = []
    vectors = []
    brightness = []
    for idx in indices:
        img_path = out_dir / f"{idx:08d}.jpg"
        frame = cv2.imread(str(img_path)) if img_path.is_file() else extract_frame(video_path, idx)
        if frame is None:
            continue
        kept_idx.append(idx)
        vectors.append(embed_frame(frame))
        brightness.append(compute_brightness(frame))
    if not kept_idx:
        return (
            [],
            np.empty((0, EMBED_SIZE * EMBED_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    return kept_idx, np.array(vectors, dtype=np.float32), np.array(brightness, dtype=np.float32)


def dedupe_existing_frames(
    vectors: np.ndarray,
    indices: list[int],
    labels: list[dict],
    min_dist: float,
) -> tuple[list[int], list[int]]:
    """Keep existing frames at least `min_dist` apart, preferring frames that
    carry more visible keypoints. Returns (kept_indices, dropped_indices)."""
    # Greedy on the existing frames: visit them by descending keypoint count
    # (ties by frame order), keep one if it's >= min_dist from all already-kept,
    # else drop it. This preserves the annotated frames worth keeping and
    # collapses bursts of near-identical shots to a single representative.
    if min_dist <= 0 or len(indices) <= 1:
        return indices, []

    def visible_count(idx: int) -> int:
        item = next((e for e in labels if isinstance(e, dict) and int(e["frame_idx"]) == idx), None)
        if not item or not item.get("labels"):
            return 0
        return sum(1 for v in item["labels"].values() if isinstance(v, (list, tuple)) and v and v[0] == 1)

    order = sorted(range(len(indices)), key=lambda i: (-visible_count(indices[i]), indices[i]))
    kept_pos = []
    for i in order:
        if kept_pos and min_dist > 0:
            dists = np.linalg.norm(vectors[i] - vectors[kept_pos], axis=1)
            if dists.min() < min_dist:
                continue
        kept_pos.append(i)
    kept_set = set(kept_pos)
    kept_idx = [indices[i] for i in sorted(kept_pos)]
    dropped_idx = [indices[i] for i in range(len(indices)) if i not in kept_set]
    return kept_idx, dropped_idx


def prune_existing(
    indices: list[int],
    vectors: np.ndarray,
    brightness: np.ndarray,
    labels: list[dict],
    min_brightness: float,
    dedupe_min_dist: float,
) -> tuple[list[int], np.ndarray, list[int], list[int]]:
    """Drop dark existing frames and near-duplicates among the rest.

    Returns (kept_indices, kept_vectors, dropped_dark, dropped_dupes)."""
    # Cleanup applied to the CURRENT label set before any new sampling: remove
    # frames that are too dark to be useful, then dedupe the survivors.
    dropped_dark: list[int] = []
    dropped_dupes: list[int] = []
    if not indices:
        return [], vectors, dropped_dark, dropped_dupes

    mask = brightness >= min_brightness
    dropped_dark = [idx for idx, keep in zip(indices, mask) if not keep]
    ok_idx = [idx for idx, keep in zip(indices, mask) if keep]
    ok_vecs = np.asarray(vectors)[mask]
    if ok_idx:
        kept_idx, dropped_dupes = dedupe_existing_frames(ok_vecs, ok_idx, labels, dedupe_min_dist)
        vec_map = dict(zip(ok_idx, ok_vecs))
        kept_vecs = np.array([vec_map[i] for i in kept_idx], dtype=np.float32)
    else:
        kept_idx, kept_vecs = [], np.empty((0, EMBED_SIZE * EMBED_SIZE), dtype=np.float32)
    return kept_idx, kept_vecs, dropped_dark, dropped_dupes


def label_path_for(video_name: str) -> Path:
    # The labels JSON is the source of truth for which frames are selected.
    return LABELS_DIR / f"{video_name}.json"


def label_has_annotations(label_path: Path) -> bool:
    # True if any frame carries actual keypoint annotations (non-empty labels).
    # Used to gate --resample, which would silently erase that human work.
    if not label_path.is_file():
        return False
    data = json.loads(label_path.read_text())
    for item in data:
        if isinstance(item, dict) and item.get("labels"):
            return True
    return False


def load_label_data(label_path: Path) -> list[dict]:
    # Read the existing label list; a missing/corrupt file just means "no
    # existing frames", which behaves like a fresh sampling.
    if not label_path.is_file():
        return []
    data = json.loads(label_path.read_text())
    if not isinstance(data, list):
        return []
    return data


def save_labels(video_name: str, data: list[dict]):
    # The ONLY write this script performs: a new list of {frame_idx, labels}
    # entries. JPGs are materialized separately by extract_frames.py.
    label_path = label_path_for(video_name)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        json.dump(data, f, indent=2)


def _kept_label_entries(existing_labels: list[dict], keep_indices: set[int]) -> list[dict]:
    # Existing label entries whose frame is still in the kept set. The full
    # dicts (with their annotations) are preserved verbatim.
    return [
        e for e in existing_labels
        if isinstance(e, dict) and int(e["frame_idx"]) in keep_indices
    ]


def process_video(
    video_name: str,
    target_frames: int | None,
    min_diversity: float | None,
    max_frames: int | None,
    min_brightness: float,
    pool_factor: int,
    dry_run: bool,
    force: bool,
    resample: bool,
    keep_dark: bool,
    dedupe_existing: float,
    append_only: bool,
    filter_only: bool,
):
    video_path = find_video(video_name)
    if video_path is None:
        print(f"  ERROR: No video found for '{video_name}' in {VIDEOS_DIR}", file=sys.stderr)
        return

    # Step 0: video metadata -- total frames sizes the sampling pool. Skipped
    # in --filter-only mode, which never decodes the video.
    if not filter_only:
        total_frames, fps = get_video_info(video_path)
        print(f"  Video: {video_path.name}")
        print(f"  Total frames: {total_frames}  FPS: {fps:.1f}")

    label_path = label_path_for(video_name)
    existing_labels = load_label_data(label_path)

    # Step 1: decide what "existing" means. Normal mode keeps the current label
    # set (anchors for selection). --resample discards it and re-picks from
    # scratch, but refuses to wipe annotated frames unless --force is given.
    if resample:
        if label_has_annotations(label_path):
            if not force:
                print(f"  SKIP: {video_name} already has annotations in {label_path}. "
                      f"Re-sampling would erase them; pass --force to override.", file=sys.stderr)
                return
            print(f"  WARNING: Overwriting annotated labels in {label_path} (--force)")
        existing_indices = []
    else:
        has_annotations = any(isinstance(item, dict) and item.get("labels") for item in existing_labels)
        if has_annotations:
            print(f"  NOTE: preserving annotations in existing labels for {video_name}")
        existing_indices = sorted({int(item["frame_idx"]) for item in existing_labels if isinstance(item, dict)})

    # Step 2: embed this video's existing frames so the new picks stay far
    # from them. No existing frames (fresh video or resample) -> no anchors.
    existing_vectors: np.ndarray
    if resample or not existing_indices:
        existing_indices = []
        existing_vectors = np.empty((0, EMBED_SIZE * EMBED_SIZE), dtype=np.float32)
        existing_brightness = np.empty((0,), dtype=np.float32)
    else:
        existing_indices, existing_vectors, existing_brightness = embed_existing_frames(
            video_path, video_name, existing_indices)

    # Step 3: clean up the existing set before adding anything new -- drop dark
    # frames and collapse near-duplicates (both optional via flags). Skipped
    # entirely in --append-only mode so existing labels are left untouched.
    dropped_dark: list[int] = []
    dropped_dupes: list[int] = []
    if not resample and existing_indices and not append_only:
        effective_min_brightness = 0.0 if keep_dark else min_brightness
        existing_indices, existing_vectors, dropped_dark, dropped_dupes = prune_existing(
            existing_indices, existing_vectors, existing_brightness,
            existing_labels, effective_min_brightness, dedupe_existing)
        if dropped_dark:
            print(f"  Pruned existing dark frames: {len(dropped_dark)}")
        if dropped_dupes:
            print(f"  Pruned existing duplicate frames: {len(dropped_dupes)}")

    # Step 3b: --filter-only mode stops here. Write the cleaned labels back and
    # never touch the pool / other videos. Fast because nothing is decoded.
    if filter_only:
        if dry_run:
            print(f"  Would write cleaned labels to: {label_path_for(video_name)}")
            return
        final_labels = _kept_label_entries(existing_labels, set(existing_indices))
        save_labels(video_name, final_labels)
        print(f"  Labels: {LABELS_DIR / f'{video_name}.json'}")
        return

    # Step 4: size the pool. In target mode the pool budget is at least the
    # shortfall; in diversity mode it's --max-frames or a flat 1000. The actual
    # pool is budget x --pool-factor (evenly spaced across the whole video).
    if min_diversity is None:
        missing = target_frames - len(existing_indices)
        if missing <= 0:
            print(f"  Already at {len(existing_indices)} >= target {target_frames}; nothing to add. "
                  f"Raise --target-frames to sample more.")
            # The cleanups above still changed the label set, so write the
            # pruned labels back even though nothing new is being sampled.
            if dry_run:
                print(f"  Would write cleaned labels to: {label_path_for(video_name)}")
                return
            final_labels = _kept_label_entries(existing_labels, set(existing_indices))
            save_labels(video_name, final_labels)
            print(f"  Labels: {LABELS_DIR / f'{video_name}.json'}")
            return
        pool_budget = max(target_frames, missing)
    else:
        missing = None
        pool_budget = max_frames if max_frames is not None else 1000

    pool_size = pool_budget * pool_factor
    pool_indices = even_indices(total_frames, pool_size)
    print(f"  Pool: {len(pool_indices)} evenly-spaced frames")

    # Step 5: decode the pool once and filter. Candidates that are already in
    # the label set (normal mode) or too dark are skipped; the rest become the
    # vectors the greedy selection will draw from.
    existing_set = set(existing_indices)
    bright_indices = []
    bright_vectors = []

    for i, idx in enumerate(pool_indices):
        if not resample and idx in existing_set:
            continue
        frame = extract_frame(video_path, idx)
        if frame is None:
            continue
        brightness = compute_brightness(frame)
        if brightness < min_brightness:
            continue
        vec = embed_frame(frame)
        bright_indices.append(idx)
        bright_vectors.append(vec)

    print(f"  Bright candidates: {len(bright_indices)} (filtered from {len(pool_indices)})")

    if len(bright_indices) == 0:
        print(f"  WARNING: No bright candidate frames found for {video_name}")
        return

    vectors = np.array(bright_vectors, dtype=np.float32)

    # Step 6: gather the fixed anchors = this video's (cleaned) existing frames
    # + every already-sampled frame from ALL other videos, then greedily pick.
    other_vectors = embed_other_videos(video_name, min_brightness)
    fixed = np.concatenate([existing_vectors, other_vectors], axis=0)
    print(f"  Diversity anchors: {len(existing_vectors)} from this video, "
          f"{len(other_vectors)} from other videos")

    if min_diversity is not None:
        selected, limit_reason = select_until_min_diversity(
            vectors, bright_indices, fixed, min_diversity, max_frames)
        print(f"  Stopped: {limit_reason}")
        print(f"  Selected {len(selected)} new frames "
              f"({len(existing_indices)} existing -> {len(existing_indices) + len(selected)} total)")
    else:
        selected, limit_reason = select_additional(vectors, bright_indices, fixed, missing)
        selected.sort()
        print(f"  Selected {len(selected)} new frames "
              f"({len(existing_indices)} existing -> {len(existing_indices) + len(selected)} of {target_frames} total)")

    if dry_run:
        print(f"  Would write labels to: {label_path_for(video_name)}")
        print(f"  Would keep {len(existing_indices)} existing frames untouched")
        return

    # Step 7: build the new label list. Resample = the picked frames only.
    # Normal mode = surviving existing entries (labels preserved verbatim) +
    # the newly picked frames.
    if resample:
        final_labels = [{"frame_idx": idx} for idx in selected]
    else:
        keep_set = set(existing_indices)
        final_labels = _kept_label_entries(existing_labels, keep_set) + [
            {"frame_idx": idx} for idx in selected if idx not in keep_set
        ]

    save_labels(video_name, final_labels)
    print(f"  Labels: {LABELS_DIR / f'{video_name}.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-names", nargs="+", required=True,
                        help="Video stems to process")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--target-frames", type=int, default=None,
                      help="Desired TOTAL number of frames per video (default: 150). "
                           "Existing frames are kept and only the shortfall is sampled.")
    mode.add_argument("--min-diversity", type=float, default=None,
                      help="Sample until the best remaining candidate is closer than this "
                           "to everything already chosen (distance in 32x32 embedding units). "
                           "Mutually exclusive with --target-frames.")
    mode.add_argument("--filter-only", action="store_true",
                      help="Only prune existing labels (drop dark frames, dedupe near-duplicates) "
                           "and write them back. No new frames are sampled.")
    parser.add_argument("--append-only", action="store_true",
                        help="Keep existing frames untouched (skip dark/dupe pruning) and only "
                             "sample + append new frames. --keep-dark / --dedupe-existing are "
                             "ignored. Mutually exclusive with --filter-only.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Safety cap on how many new frames to add (default: unbounded). "
                             "Only used with --min-diversity.")
    parser.add_argument("--min-brightness", type=float, default=50.0,
                        help="Minimum mean grayscale brightness (default: 50). "
                             "Existing frames darker than this are pruned from the labels "
                             "unless --keep-dark is given.")
    parser.add_argument("--keep-dark", action="store_true",
                        help="Keep existing frames darker than --min-brightness in the labels "
                             "(only dedupe them, never drop for darkness)")
    parser.add_argument("--dedupe-existing", type=float, default=1.0,
                        help="Min L2 distance between retained existing frames (32x32 embedding "
                             "space). Near-duplicates are collapsed to the representative with "
                             "the most visible keypoints (default: 1.0, use 0 to disable)")
    parser.add_argument("--pool-factor", type=int, default=3,
                        help="Pool size = frame budget * pool_factor (default: 3)")
    parser.add_argument("--resample", action="store_true",
                        help="Wipe this video's extracted frames and labels, then re-pick from scratch")
    parser.add_argument("--force", action="store_true",
                        help="Allow --resample to erase a video that already has annotated labels")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing files")
    args = parser.parse_args()

    if args.filter_only and args.append_only:
        parser.error("--filter-only and --append-only are mutually exclusive")
    if args.filter_only and args.resample:
        parser.error("--filter-only and --resample are mutually exclusive")
    if args.append_only and args.resample:
        parser.error("--append-only and --resample are mutually exclusive")
    if args.target_frames is None and args.min_diversity is None and not args.filter_only:
        args.target_frames = 150

    for name in args.video_names:
        print(f"\n{'='*60}")
        print(f"Processing: {name}")
        print(f"{'='*60}")
        process_video(name, args.target_frames, args.min_diversity, args.max_frames,
                      args.min_brightness, args.pool_factor, args.dry_run, args.force, args.resample,
                      args.keep_dark, args.dedupe_existing, args.append_only, args.filter_only)

    print(f"\nDone.")


if __name__ == "__main__":
    main()
