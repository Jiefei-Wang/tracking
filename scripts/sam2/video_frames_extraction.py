import argparse
import concurrent.futures
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

INPUT_DIR = "videos"
OUTPUT_BASE_DIR = os.path.join("output", "sam2", "raw_frames")
CONFIG_PATH = "config.yaml"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v")
DEFAULT_WORKERS = 8
DEFAULT_CHUNK_SIZE_FRAMES = 3000


def rebuild_output_root(output_root):
    if os.path.exists(output_root):
        print(f"Removing existing output root: {output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)


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


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fps <= 0:
        raise ValueError(f"Invalid FPS for video: {video_path}")

    return total_frames, fps


def save_image(file_path, frame_bgr):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cv2.imwrite(file_path, frame_bgr)


def build_chunk_tasks(video_entries, chunk_size_frames):
    tasks = []
    for entry in video_entries:
        total_frames = entry["total_frames"]
        for start in range(0, total_frames, chunk_size_frames):
            end = min(start + chunk_size_frames, total_frames)
            tasks.append((entry["video_name"], entry["video_path"], entry["output_dir"], start, end))
    return tasks


def extract_chunk(task):
    video_name, video_path, output_dir, start_frame, end_frame = task
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    saved = 0
    idx = start_frame
    try:
        while idx < end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            file_path = os.path.join(output_dir, f"{idx:08d}.jpg")
            save_image(file_path, frame)
            saved += 1
            idx += 1
    finally:
        cap.release()

    return video_name, saved


def rebuild_output_dirs(video_entries):
    for entry in tqdm(video_entries, desc="Preparing output folders"):
        output_dir = entry["output_dir"]
        if os.path.exists(output_dir):
            print(f"  Removing existing folder: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract all frames from train videos with CPU chunked parallel processing."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--chunk-size-frames",
        type=int,
        default=DEFAULT_CHUNK_SIZE_FRAMES,
        help="Chunk size (number of frames per worker task).",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.workers <= 0:
        raise ValueError("--workers must be >= 1")
    if args.chunk_size_frames <= 0:
        raise ValueError("--chunk-size-frames must be >= 1")

    if not os.path.isdir(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    configured_video_names = load_train_video_names(CONFIG_PATH)

    all_video_files = sorted(
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    )
    video_files = sorted(
        filename for filename in all_video_files
        if os.path.splitext(filename)[0] in configured_video_names
    )
    skipped_video_files = sorted(
        filename for filename in all_video_files
        if os.path.splitext(filename)[0] not in configured_video_names
    )

    if not video_files:
        print(f"No configured video files found in '{INPUT_DIR}'.")
        return

    video_entries = []
    for filename in video_files:
        video_name = os.path.splitext(filename)[0]
        video_path = os.path.join(INPUT_DIR, filename)
        total_frames, fps = get_video_info(video_path)
        video_entries.append(
            {
                "video_name": video_name,
                "video_path": video_path,
                "output_dir": os.path.join(OUTPUT_BASE_DIR, video_name),
                "total_frames": total_frames,
                "fps": fps,
            }
        )

    print(f"Configured videos in {CONFIG_PATH}: {len(configured_video_names)}")
    print(f"Input videos matched config: {len(video_files)}")
    print(f"Input videos skipped (not in config): {len(skipped_video_files)}")
    print(f"Total videos scanned: {len(video_entries)}")
    print(f"Workers: {args.workers}")
    print(f"Chunk size (frames): {args.chunk_size_frames}")

    rebuild_output_root(OUTPUT_BASE_DIR)
    rebuild_output_dirs(video_entries)

    chunk_tasks = build_chunk_tasks(video_entries, args.chunk_size_frames)
    print(f"Total extraction chunks: {len(chunk_tasks)}")

    saved_counts = defaultdict(int)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for video_name, saved in tqdm(
            executor.map(extract_chunk, chunk_tasks),
            total=len(chunk_tasks),
            desc="Extracting frame chunks",
        ):
            saved_counts[video_name] += saved

    total_saved = sum(saved_counts.values())
    print(f"Total saved frames: {total_saved}")
    for entry in video_entries:
        print(
            f"  {entry['video_name']}: "
            f"saved={saved_counts.get(entry['video_name'], 0)} "
            f"expected~={entry['total_frames']}"
        )

    print("\nAll train videos extracted.")


if __name__ == "__main__":
    main()
