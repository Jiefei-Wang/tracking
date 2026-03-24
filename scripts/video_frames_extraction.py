import os
import re
import math
import shutil
from pathlib import Path

import cv2
from PIL import Image

INPUT_DIR = "videos"
OUTPUT_BASE_DIR = os.path.join("output", "video_frames_for_sam")
LABELED_DIR = "input/labeled-data"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v")

DEFAULT_FPS = 1.0
MAX_FRAMES = 100


# return image, gets video path, idx frame
def get_image_from_video(video_path, idx_frame):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx_frame)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"ERROR: Could not read frame {idx_frame} from {video_path}")
        return None

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def save_image(file_path, frame):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, frame)
    return file_path


def get_video_info(video_path):
    """
    Returns:
        total_frames (int)
        source_fps (float)
        duration (float)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if source_fps <= 0:
        raise ValueError(f"Invalid FPS for video: {video_path}")

    duration = total_frames / source_fps if total_frames > 0 else 0.0
    return total_frames, source_fps, duration


def compute_sampling_fps(duration, default_fps=DEFAULT_FPS, max_frames=MAX_FRAMES):
    """
    Use 1 fps when possible.
    If that would exceed max_frames, reduce the sampling fps.
    """
    if duration <= 0:
        return default_fps

    estimated = math.ceil(duration * default_fps)
    if estimated <= max_frames:
        return default_fps

    return max_frames / duration


def compute_frame_indices(total_frames, source_fps, sampling_fps):
    """
    Return original video frame indices to extract.
    These indices are based on timestamps sampled at sampling_fps.
    """
    if total_frames <= 0 or source_fps <= 0 or sampling_fps <= 0:
        return []

    duration = total_frames / source_fps
    step_sec = 1.0 / sampling_fps

    indices = []
    used = set()
    t = 0.0

    while t < duration and len(indices) < MAX_FRAMES:
        idx = int(round(t * source_fps))

        if idx >= total_frames:
            idx = total_frames - 1

        if idx not in used:
            indices.append(idx)
            used.add(idx)

        t += step_sec

    return indices


def extract_frames_manually(video_path, output_dir):
    """
    Delete existing output folder, compute sampling indices,
    and save extracted frames using original frame indices as filenames.
    """
    if os.path.exists(output_dir):
        print(f"  Removing existing folder: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    total_frames, source_fps, duration = get_video_info(video_path)
    sampling_fps = compute_sampling_fps(duration)
    frame_indices = compute_frame_indices(total_frames, source_fps, sampling_fps)

    print(f"  Total frames: {total_frames}")
    print(f"  Source FPS: {source_fps:.6f}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Sampling FPS: {sampling_fps:.6f}")
    print(f"  Frames to extract: {len(frame_indices)}")

    for idx in frame_indices:
        frame = get_image_from_video(video_path, idx)
        if frame is None:
            print(f"  Skipped frame {idx}")
            continue

        file_path = os.path.join(output_dir, f"{idx:08d}.jpg")
        save_image(file_path, frame)
        print(f"  Saved: {file_path}")


def extract_index_from_name(path: Path):
    """
    Extract the last integer from the file stem.
    Example:
      img000123.png -> 123
      000456.png -> 456
      frame_42.png -> 42
    """
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def copy_labeled_frames_preserve_index(src_root: Path, dst_root: Path):
    """
    Copy PNG labeled frames into video_frames/<folder>/ using the
    original index found in the filename.
    """
    if not src_root.exists():
        print(f"Labeled-data folder not found: {src_root}")
        return

    for folder in src_root.iterdir():
        if not folder.is_dir():
            continue

        out_folder = dst_root / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        png_files = sorted(folder.glob("*.png"))

        for png_file in png_files:
            frame_idx = extract_index_from_name(png_file)
            if frame_idx is None:
                print(f"Skipped (no numeric index): {png_file}")
                continue

            jpg_path = out_folder / f"{frame_idx:08d}.jpg"

            try:
                with Image.open(png_file) as img:
                    img.convert("RGB").save(jpg_path, "JPEG", quality=95)
                print(f"Saved labeled frame: {jpg_path}")
            except Exception as e:
                print(f"Failed: {png_file} ({e})")


def main():
    if not os.path.isdir(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    video_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )

    if not video_files:
        print(f"No video files found in '{INPUT_DIR}'.")
        return

    for filename in video_files:
        video_path = os.path.join(INPUT_DIR, filename)
        video_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(OUTPUT_BASE_DIR, video_name)

        print(f"\nProcessing video: {filename}")

        try:
            extract_frames_manually(video_path, output_dir)
            print("  Video extraction done.")
        except Exception as e:
            print(f"  Failed for {filename}: {e}")

    print("\nCopying labeled frames...")
    copy_labeled_frames_preserve_index(Path(LABELED_DIR), Path(OUTPUT_BASE_DIR))

    print("\nAll videos processed.")


if __name__ == "__main__":
    main()