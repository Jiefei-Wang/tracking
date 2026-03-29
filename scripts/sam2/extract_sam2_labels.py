"""
Extract raw SAM2 segmentation labels for all videos in output/sam2/diverse.
Saves unfiltered mask maps under output/sam2/sam2_pickle_raw.
"""

import os
import sys
import pickle
import re
import shutil
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from modules.sam_utils import get_coordinates
from modules.label_csv_utils import load_keypoints
from sam2.build_sam import build_sam2_video_predictor

LABELED_DATA_DIR = "input/labeled-data"
DLC_FRAMES_DIR = "output/sam2/DLC_frames"
DIVERSE_FRAMES_DIR = "output/sam2/diverse"
OUTPUT_PICKLE_DIR = "output/sam2/sam2_pickle_raw"
MERGED_FRAMES_DIR = "output/sam2/sam2_training_merge_frames"
CONFIG_PATH = "config.yaml"


def rebuild_output_root(output_root):
    if os.path.exists(output_root):
        print(f"Removing existing output root: {output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)


def extract_index_from_name(filename):
    """Extract the last integer from a frame filename stem."""
    stem = os.path.splitext(filename)[0]
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    return int(matches[-1])


def get_frame_names(video_dir):
    """Scan image frame names in a directory and return sorted names + indices."""
    all_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
    ]

    items = []
    for name in all_names:
        frame_idx = extract_index_from_name(name)
        if frame_idx is None:
            continue
        items.append((frame_idx, name))

    items.sort(key=lambda x: x[0])
    frame_names = [name for _, name in items]
    frame_indices = [frame_idx for frame_idx, _ in items]
    return frame_names, frame_indices


def build_frame_map(video_dir):
    """Build frame_idx -> absolute image path mapping for one directory."""
    frame_map = {}
    if not os.path.isdir(video_dir):
        return frame_map

    for name in os.listdir(video_dir):
        if os.path.splitext(name)[-1].lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        frame_idx = extract_index_from_name(name)
        if frame_idx is None:
            continue
        frame_map[frame_idx] = os.path.join(video_dir, name)
    return frame_map


def prepare_merged_video_frames(video_name, dlc_video_dir, diverse_video_dir, merged_root):
    """
    Build merged per-video frame directory from DLC training frames + diverse frames.
    Returns merged_dir and set of diverse frame indices.
    """
    dlc_map = build_frame_map(dlc_video_dir)
    diverse_map = build_frame_map(diverse_video_dir)
    diverse_indices = set(diverse_map.keys())

    if not dlc_map:
        raise ValueError(f"no images found in {dlc_video_dir}")
    if not diverse_map:
        raise ValueError(f"no images found in {diverse_video_dir}")

    merged_map = dict(diverse_map)
    merged_map.update(dlc_map)  # prefer DLC image if same frame index exists

    merged_video_dir = os.path.join(merged_root, video_name)
    os.makedirs(merged_video_dir, exist_ok=True)

    copied = 0
    skipped_existing = 0
    for frame_idx, src_path in sorted(merged_map.items()):
        dst_path = os.path.join(merged_video_dir, f"{frame_idx:08d}.jpg")
        if os.path.isfile(dst_path):
            skipped_existing += 1
            continue
        shutil.copy2(src_path, dst_path)
        copied += 1

    print(
        f"  Merged frames prepared: total={len(merged_map)} "
        f"(copied={copied}, skipped_existing={skipped_existing}), "
        f"diverse_targets={len(diverse_indices)}"
    )
    return merged_video_dir, diverse_indices


def load_train_video_names(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    values = payload.get("train_videos", [])
    if not isinstance(values, list):
        raise ValueError(f"Expected 'train_videos' to be a list in {config_path}")

    names = set(str(item) for item in values)
    if not names:
        raise ValueError(f"No videos found in train_videos list in {config_path}")
    return names


def process_video(predictor, video_name, video_dir, csv_path, save_path, diverse_indices):
    """Run SAM2 segmentation on a single video and save results."""
    print(f"\n{'='*60}")
    print(f"Processing: {video_name}")
    print(f"{'='*60}")

    # Load frames
    frame_names, frame_indices = get_frame_names(video_dir)
    if len(frame_names) == 0:
        print(f"  Skipping: no JPEG frames found in {video_dir}")
        return False

    # Load labels
    label_df = load_keypoints(csv_path)
    if label_df.empty:
        print(f"  Skipping: no labels found in {csv_path}")
        return False

    print(
        f"  Found merged_frames={len(frame_names)}, "
        f"labeled_rows={len(label_df)}, diverse_targets={len(diverse_indices)}"
    )

    # Initialize predictor state
    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    # Add conditioning points from labeled frames
    n_idx = len(label_df)
    ann_obj_id = 1

    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in tqdm(range(n_idx), desc="  Adding labels"):
            ann_frame_idx = label_df["frame"].iloc[i]
            if ann_frame_idx not in frame_indices:
                print(f"  Warning: frame {ann_frame_idx} not found in video frames, skipping")
                continue

            jpg_idx = frame_indices.index(ann_frame_idx)

            points = get_coordinates(label_df, i)
            if len(points) == 0:
                print(f"  Warning: no valid coordinates for row {i} (frame {ann_frame_idx}), skipping")
                continue

            points = np.array(points, dtype=np.float32)
            labels = np.array([1] * len(points), np.int32)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=jpg_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels,
            )

        # Propagate through all frames
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in tqdm(
            predictor.propagate_in_video(inference_state, start_frame_idx=0),
            desc="  Propagating",
            total=len(frame_names),
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    print(f"  Segmented {len(video_segments)}/{len(frame_names)} frames")

    # Map keys back to original frame indices
    video_segments_mapped = {
        frame_indices[k]: v
        for k, v in video_segments.items()
    }

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(video_segments_mapped, f)
    print(
        f"  Saved raw masks (labeled + unlabeled): {len(video_segments_mapped)} "
        f"(diverse targets in set: {len(diverse_indices)}) -> {save_path}"
    )
    return True


def main():
    dlc_frames_dir = DLC_FRAMES_DIR
    diverse_frames_dir = DIVERSE_FRAMES_DIR
    labeled_data_dir = LABELED_DATA_DIR
    output_dir = OUTPUT_PICKLE_DIR
    merged_frames_dir = MERGED_FRAMES_DIR
    configured_video_names = load_train_video_names(CONFIG_PATH)

    rebuild_output_root(output_dir)
    rebuild_output_root(merged_frames_dir)

    # Build predictor
    device = torch.device("cuda")
    sam2_checkpoint = os.path.join("input", "sam2_checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print("Loading SAM2 model...")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Get all video directories
    video_names = sorted(
        video_name
        for video_name in configured_video_names
        if os.path.isdir(os.path.join(diverse_frames_dir, video_name))
    )

    print(f"Configured train videos in {CONFIG_PATH}: {len(configured_video_names)}")
    print(f"Found {len(video_names)} diverse video directories for inference")

    processed = 0
    skipped = 0

    for video_name in video_names:
        dlc_video_dir = os.path.join(dlc_frames_dir, video_name)
        diverse_video_dir = os.path.join(diverse_frames_dir, video_name)
        csv_path = os.path.join(labeled_data_dir, video_name, "CollectedData_rats.csv")
        save_path = os.path.join(output_dir, f"{video_name}.pkl")

        if not os.path.isdir(dlc_video_dir):
            print(f"Skipping {video_name}: DLC training frames directory not found")
            skipped += 1
            continue

        if not os.path.isdir(diverse_video_dir):
            print(f"Skipping {video_name}: diverse inference frames directory not found")
            skipped += 1
            continue

        # Check if labeled data exists
        if not os.path.isfile(csv_path):
            print(f"Skipping {video_name}: no labeled data CSV found")
            skipped += 1
            continue

        try:
            merged_video_dir, diverse_indices = prepare_merged_video_frames(
                video_name,
                dlc_video_dir,
                diverse_video_dir,
                merged_frames_dir,
            )
            success = process_video(
                predictor,
                video_name,
                merged_video_dir,
                csv_path,
                save_path,
                diverse_indices,
            )
            if success:
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Done! Processed: {processed}, Skipped: {skipped}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
