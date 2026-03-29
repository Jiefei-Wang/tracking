"""
Filter raw SAM2 masks by mask area threshold + per-video quantiles.

Reads raw cache files from output/sam2/sam2_pickle_raw.
Writes filtered masks to output/sam2/sam2_pickle_filtered.
Saves masked overlays to output/sam2/diverse_masked and kept original frames
to output/sam2/final (using output/sam2/sam2_training_merge_frames as frame source).
Labeled frames are always kept; only unlabeled frames are filtered out.
"""

import os
import pickle
import shutil
import re
import sys
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

# Allow running this script via path while importing project modules from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from modules.label_csv_utils import load_keypoints

MASK_AREA_LOWER_QUANTILE = 0.10
MASK_AREA_UPPER_QUANTILE = 0.90
MIN_MASK_AREA_PIXELS = 1000
LABELED_DATA_DIR = os.path.join("input", "labeled-data")


def rebuild_output_root(output_root):
    if os.path.exists(output_root):
        print(f"Removing existing output root: {output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)


def get_mask_color(obj_id):
    """Get a consistent color for an object ID."""
    cmap = plt.get_cmap("tab10")
    idx = 0 if obj_id is None else obj_id
    return np.array(cmap(idx)[:3])


def apply_mask_overlay(image_array, masks, transparency=0.5):
    """Apply colored mask overlays onto an image array."""
    result = image_array.astype(np.float32)
    for obj_id, mask in masks.items():
        mask_2d = np.asarray(mask).squeeze().astype(bool)
        color = get_mask_color(obj_id) * 255.0
        for channel in range(3):
            result[:, :, channel] = np.where(
                mask_2d,
                result[:, :, channel] * transparency + color[channel] * (1 - transparency),
                result[:, :, channel],
            )
    return result.clip(0, 255).astype(np.uint8)


def get_frame_names(video_dir):
    """Scan image frame names in a directory and return sorted names + indices."""
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]

    items = []
    for frame_name in frame_names:
        frame_idx = extract_index_from_name(frame_name)
        if frame_idx is None:
            continue
        items.append((frame_idx, frame_name))

    items.sort(key=lambda item: item[0])
    frame_names = [name for _, name in items]
    frame_indices = [idx for idx, _ in items]
    return frame_names, frame_indices


def extract_index_from_name(filename):
    """Extract the last integer from a frame filename stem."""
    stem = os.path.splitext(filename)[0]
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    return int(matches[-1])


def compute_total_mask_area(masks: Mapping[int, np.ndarray]) -> int:
    """Compute the union mask area for one frame."""
    union_mask = None
    for mask in masks.values():
        mask_2d = np.asarray(mask).squeeze().astype(bool)
        if union_mask is None:
            union_mask = mask_2d.copy()
        else:
            union_mask |= mask_2d
    if union_mask is None:
        return 0
    return int(union_mask.sum())


def get_labeled_frame_indices(csv_path):
    """Load labeled frame indices from a DLC CSV."""
    if not os.path.isfile(csv_path):
        return set()
    label_df = load_keypoints(csv_path)
    if label_df.empty or "frame" not in label_df.columns:
        return set()
    return set(label_df["frame"].astype(int).tolist())


def filter_video_segments(video_segments, labeled_frame_indices):
    """
    Filter masks while always preserving labeled frames.

    Statistics (including quantiles) are computed from all frames.
    Only unlabeled frames are eligible for removal.
    """
    frame_areas = {
        frame_idx: compute_total_mask_area(masks)
        for frame_idx, masks in video_segments.items()
    }

    filtered_segments = {}
    lower_cutoff = None
    upper_cutoff = None
    removed_unlabeled_by_min = 0
    removed_by_quantile = 0

    if frame_areas:
        area_values = np.array(list(frame_areas.values()), dtype=np.float32)
        lower_cutoff = float(np.quantile(area_values, MASK_AREA_LOWER_QUANTILE))
        upper_cutoff = float(np.quantile(area_values, MASK_AREA_UPPER_QUANTILE))

    for frame_idx, masks in video_segments.items():
        area = frame_areas[frame_idx]
        is_labeled = frame_idx in labeled_frame_indices

        if is_labeled:
            filtered_segments[frame_idx] = masks
            continue

        if area < MIN_MASK_AREA_PIXELS:
            removed_unlabeled_by_min += 1
            continue

        if lower_cutoff is None or upper_cutoff is None:
            filtered_segments[frame_idx] = masks
            continue

        if area < lower_cutoff or area > upper_cutoff:
            removed_by_quantile += 1
            continue
        filtered_segments[frame_idx] = masks

    return (
        filtered_segments,
        lower_cutoff,
        upper_cutoff,
        frame_areas,
        removed_unlabeled_by_min,
        removed_by_quantile,
    )


def rebuild_masked_frames(video_name, video_dir, filtered_segments, output_dir, transparency=0.5):
    """Rebuild masked frame overlays for one video."""
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frame_names, frame_indices = get_frame_names(video_dir)
    idx_to_name = dict(zip(frame_indices, frame_names))

    saved = 0
    for frame_idx, masks in tqdm(sorted(filtered_segments.items()), desc=f"  Overlaying {video_name}"):
        frame_name = idx_to_name.get(frame_idx)
        if frame_name is None:
            continue
        frame_path = os.path.join(video_dir, frame_name)
        image = np.array(Image.open(frame_path).convert("RGB"))
        result = apply_mask_overlay(image, masks, transparency=transparency)
        Image.fromarray(result).save(os.path.join(output_dir, frame_name), quality=95)
        saved += 1
    return saved


def copy_final_frames(video_name, video_dir, filtered_segments, output_dir):
    """Copy kept original frames into final output directory."""
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    frame_names, frame_indices = get_frame_names(video_dir)
    idx_to_name = dict(zip(frame_indices, frame_names))

    saved = 0
    for frame_idx in sorted(filtered_segments):
        frame_name = idx_to_name.get(frame_idx)
        if frame_name is None:
            continue
        src_path = os.path.join(video_dir, frame_name)
        dst_path = os.path.join(output_dir, frame_name)
        shutil.copy2(src_path, dst_path)
        saved += 1

    return saved


def process_video(
    video_name,
    raw_pkl_path,
    video_dir,
    csv_path,
    filtered_pkl_path,
    masked_output_dir,
    final_output_dir,
):
    """Filter a single raw SAM2 cache file and rebuild its overlays."""
    print(f"\nProcessing: {video_name}")

    with open(raw_pkl_path, "rb") as f:
        video_segments = pickle.load(f)

    if not isinstance(video_segments, dict) or not video_segments:
        print("  Skipping: empty or invalid raw SAM2 cache")
        return False

    labeled_frame_indices = get_labeled_frame_indices(csv_path)
    if labeled_frame_indices:
        print(f"  Labeled frames loaded from CSV: {len(labeled_frame_indices)}")
    else:
        print("  Labeled frames loaded from CSV: 0 (no CSV or empty CSV)")

    (
        filtered_segments,
        lower_cutoff,
        upper_cutoff,
        frame_areas,
        removed_unlabeled_by_min,
        removed_by_quantile,
    ) = filter_video_segments(video_segments, labeled_frame_indices)

    os.makedirs(os.path.dirname(filtered_pkl_path), exist_ok=True)
    with open(filtered_pkl_path, "wb") as f:
        pickle.dump(filtered_segments, f)

    saved_overlays = rebuild_masked_frames(video_name, video_dir, filtered_segments, masked_output_dir)
    saved_final = copy_final_frames(video_name, video_dir, filtered_segments, final_output_dir)

    print(f"  Total frames in raw pickle: {len(frame_areas)}")
    print(f"  Total labeled frames (from CSV): {len(labeled_frame_indices)}")
    print(f"  Removed unlabeled by min area (<{MIN_MASK_AREA_PIXELS}): {removed_unlabeled_by_min}")
    if lower_cutoff is None or upper_cutoff is None:
        print("  Quantile cutoffs: not applied (no frames available)")
    else:
        print(f"  Quantile cutoffs: lower={lower_cutoff:.2f}, upper={upper_cutoff:.2f}")
    print(f"  Removed unlabeled by quantile filter: {removed_by_quantile}")
    print(f"  Final masks saved: {len(filtered_segments)}")
    print(f"  Masked overlays saved: {saved_overlays}")
    print(f"  Final original frames saved: {saved_final}")
    return True


def main():
    base_dir = ""
    frames_root = os.path.join(base_dir, "output", "sam2", "sam2_training_merge_frames")
    raw_masks_root = os.path.join(base_dir, "output", "sam2", "sam2_pickle_raw")
    final_masks_root = os.path.join(base_dir, "output", "sam2", "sam2_pickle_filtered")
    masked_frames_root = os.path.join(base_dir, "output", "sam2", "diverse_masked")
    final_frames_root = os.path.join(base_dir, "output", "sam2", "final")

    rebuild_output_root(final_masks_root)
    rebuild_output_root(masked_frames_root)
    rebuild_output_root(final_frames_root)

    if not os.path.isdir(raw_masks_root):
        print(f"Raw SAM2 cache directory not found: {raw_masks_root}")
        return

    raw_pkl_files = sorted(
        f for f in os.listdir(raw_masks_root)
        if f.endswith(".pkl")
    )
    print(f"Found {len(raw_pkl_files)} raw SAM2 cache files")

    processed = 0
    skipped = 0

    for pkl_file in raw_pkl_files:
        video_name = pkl_file[:-4]
        raw_pkl_path = os.path.join(raw_masks_root, pkl_file)
        video_dir = os.path.join(frames_root, video_name)
        csv_path = os.path.join(base_dir, LABELED_DATA_DIR, video_name, "CollectedData_rats.csv")
        filtered_pkl_path = os.path.join(final_masks_root, pkl_file)
        masked_output_dir = os.path.join(masked_frames_root, video_name)
        final_output_dir = os.path.join(final_frames_root, video_name)

        if not os.path.isdir(video_dir):
            print(f"Skipping {video_name}: video frames directory not found")
            skipped += 1
            continue

        try:
            success = process_video(
                video_name,
                raw_pkl_path,
                video_dir,
                csv_path,
                filtered_pkl_path,
                masked_output_dir,
                final_output_dir,
            )
            if success:
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            skipped += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
