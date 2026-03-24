"""
Extract SAM2 segmentation labels for all videos in the video_frames folder.
Skips videos that don't have corresponding labeled-data or video frames.
"""

import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm

from modules.sam_utils import get_coordinates
from modules.label_csv_utils import load_keypoints
from sam2.build_sam import build_sam2_video_predictor


def get_frame_names(video_dir):
    """Scan JPEG frame names in a directory and return sorted names + indices."""
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_indices = [int(os.path.splitext(p)[0]) for p in frame_names]
    return frame_names, frame_indices


def process_video(predictor, video_name, video_dir, csv_path, save_path):
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

    print(f"  Found {len(frame_names)} frames, {len(label_df)} labeled rows")

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
        frame_indices[k]: v for k, v in video_segments.items()
    }

    # Save results
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(video_segments_mapped, f)
    print(f"  Saved to {save_path}")
    return True


def main():
    base_dir = ""
    video_frames_dir = os.path.join(base_dir, "output", "video_frames_for_sam")
    labeled_data_dir = os.path.join(base_dir, "input", "labeled-data")
    output_dir = os.path.join(base_dir, "output", "sam2_labels")

    # Build predictor
    device = torch.device("cuda")
    sam2_checkpoint = os.path.join(base_dir, "input", "sam2_checkpoints", "sam2.1_hiera_large.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    print("Loading SAM2 model...")
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # Get all video directories
    video_names = sorted([
        d for d in os.listdir(video_frames_dir)
        if os.path.isdir(os.path.join(video_frames_dir, d)) and not d.startswith(".")
    ])

    print(f"Found {len(video_names)} video directories")

    processed = 0
    skipped = 0

    for video_name in video_names:
        video_dir = os.path.join(video_frames_dir, video_name)
        csv_path = os.path.join(labeled_data_dir, video_name, "CollectedData_rats.csv")
        save_path = os.path.join(output_dir, f"{video_name}.pkl")

        # Check if video frames exist
        if not os.path.isdir(video_dir):
            print(f"Skipping {video_name}: video_frames directory not found")
            skipped += 1
            continue

        # Check if labeled data exists
        if not os.path.isfile(csv_path):
            print(f"Skipping {video_name}: no labeled data CSV found")
            skipped += 1
            continue

        # Skip if already processed
        if os.path.exists(save_path):
            print(f"Skipping {video_name}: already processed ({save_path})")
            skipped += 1
            continue

        try:
            success = process_video(predictor, video_name, video_dir, csv_path, save_path)
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
    # delete all existing .pkl files in sam2_labels to avoid confusion
    sam2_labels_dir = os.path.join("", "sam2_labels")
    if os.path.isdir(sam2_labels_dir):
        for f in os.listdir(sam2_labels_dir):
            if f.endswith(".pkl"):
                os.remove(os.path.join(sam2_labels_dir, f))
    main()
