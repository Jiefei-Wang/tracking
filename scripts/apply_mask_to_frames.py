"""
Apply SAM2 masks to video frames for visual quality inspection.
Creates masked_frames/<video_name>/ with overlay images for each frame.
"""

import os
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_mask_color(obj_id):
    """Get a consistent color for an object ID."""
    cmap = plt.get_cmap("tab10")
    idx = 0 if obj_id is None else obj_id
    return np.array(cmap(idx)[:3])


def apply_mask_overlay(image_array, masks, transparency=0.5):
    """Apply colored mask overlays onto an image array.

    Args:
        image_array: (H, W, 3) uint8 array
        masks: dict of {obj_id: (1, H, W) bool array}
        transparency: 0 = fully opaque mask, 1 = no mask visible
    """
    result = image_array.astype(np.float32)
    for obj_id, mask in masks.items():
        mask_2d = mask.squeeze()  # (H, W)
        color = get_mask_color(obj_id) * 255.0
        for c in range(3):
            result[:, :, c] = np.where(
                mask_2d,
                result[:, :, c] * transparency + color[c] * (1 - transparency),
                result[:, :, c],
            )
    return result.clip(0, 255).astype(np.uint8)


def get_frame_names(video_dir):
    """Scan JPEG frame names in a directory and return sorted names + indices."""
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_indices = [int(os.path.splitext(p)[0]) for p in frame_names]
    return frame_names, frame_indices


def process_video(video_name, video_dir, pkl_path, output_dir, transparency=0.5):
    """Apply masks to all frames of a single video."""
    print(f"\nProcessing: {video_name}")

    # Load masks
    with open(pkl_path, "rb") as f:
        video_segments = pickle.load(f)

    # Load frame names
    frame_names, frame_indices = get_frame_names(video_dir)
    if len(frame_names) == 0:
        print(f"  Skipping: no frames found")
        return False

    # Build frame index -> filename mapping
    idx_to_name = dict(zip(frame_indices, frame_names))

    os.makedirs(output_dir, exist_ok=True)

    applied = 0
    for frame_idx, masks in tqdm(sorted(video_segments.items()), desc=f"  Overlaying"):
        if frame_idx not in idx_to_name:
            continue

        frame_path = os.path.join(video_dir, idx_to_name[frame_idx])
        img = np.array(Image.open(frame_path))
        result = apply_mask_overlay(img, masks, transparency=transparency)

        out_path = os.path.join(output_dir, idx_to_name[frame_idx])
        Image.fromarray(result).save(out_path, quality=95)
        applied += 1

    print(f"  Saved {applied} masked frames to {output_dir}")
    return True


def main():
    base_dir = ""
    video_frames_dir = os.path.join(base_dir, "output", "video_frames_for_sam")
    sam2_labels_dir = os.path.join(base_dir, "output", "sam2_labels")
    masked_frames_dir = os.path.join(base_dir, "output", "masked_frames")

    # Find all pickle files
    pkl_files = [f for f in os.listdir(sam2_labels_dir) if f.endswith(".pkl")]
    print(f"Found {len(pkl_files)} label files")

    processed = 0
    skipped = 0

    for pkl_file in sorted(pkl_files):
        video_name = pkl_file[:-4]  # remove .pkl
        video_dir = os.path.join(video_frames_dir, video_name)
        pkl_path = os.path.join(sam2_labels_dir, pkl_file)
        output_dir = os.path.join(masked_frames_dir, video_name)

        if not os.path.isdir(video_dir):
            print(f"Skipping {video_name}: video_frames directory not found")
            skipped += 1
            continue

        # Skip if already processed
        if os.path.isdir(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"Skipping {video_name}: already processed")
            skipped += 1
            continue

        try:
            success = process_video(video_name, video_dir, pkl_path, output_dir)
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