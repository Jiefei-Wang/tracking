import argparse
import concurrent.futures
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

RAW_FRAMES_DIR = os.path.join("output", "sam2", "raw_frames")
DIVERSE_OUTPUT_DIR = os.path.join("output", "sam2", "diverse")
LABELED_DIR = "input/labeled-data"
CONFIG_PATH = "config.yaml"
TOTAL_DIVERSE_FRAMES = 1000
THUMBNAIL_SIZE = 32
DEFAULT_WORKERS = 8
CANDIDATE_MULTIPLIER = 10


def rebuild_output_root(output_root):
    if os.path.exists(output_root):
        print(f"Removing existing output root: {output_root}")
        shutil.rmtree(output_root)
    os.makedirs(output_root, exist_ok=True)


def extract_index_from_name(path: Path):
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[-1])


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


def load_labeled_frame_indices_for_video(labeled_root: Path, video_name: str):
    video_label_dir = labeled_root / video_name
    if not video_label_dir.is_dir():
        return set()

    labeled_indices = set()
    for png_file in sorted(video_label_dir.glob("*.png")):
        frame_idx = extract_index_from_name(png_file)
        if frame_idx is not None:
            labeled_indices.add(frame_idx)
    return labeled_indices


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available in this environment.")
    return torch.device(device_arg)


def discover_candidates(raw_root: Path, labeled_root: Path, allowed_video_names):
    video_entries = []
    global_candidates = []
    total_labeled = 0

    for video_name in sorted(allowed_video_names):
        video_dir = raw_root / video_name
        if not video_dir.is_dir():
            continue

        labeled_indices = load_labeled_frame_indices_for_video(labeled_root, video_name)
        frame_paths = sorted(video_dir.glob("*.jpg"))
        unlabeled = []
        for frame_path in frame_paths:
            frame_idx = extract_index_from_name(frame_path)
            if frame_idx is None:
                continue
            if frame_idx in labeled_indices:
                continue
            unlabeled.append((frame_idx, frame_path))

        entry = {
            "video_name": video_name,
            "video_dir": str(video_dir),
            "output_dir": os.path.join(DIVERSE_OUTPUT_DIR, video_name),
            "labeled_indices": labeled_indices,
            "unlabeled": unlabeled,
        }
        video_entries.append(entry)
        total_labeled += len(labeled_indices)

        for frame_idx, frame_path in unlabeled:
            global_candidates.append(
                {
                    "video_name": video_name,
                    "frame_idx": frame_idx,
                    "frame_path": str(frame_path),
                }
            )

    video_entries.sort(key=lambda x: x["video_name"])
    global_candidates.sort(key=lambda x: (x["video_name"], x["frame_idx"]))
    return video_entries, global_candidates, total_labeled


def allocate_presample_counts(video_entries, target_total):
    positive_entries = [entry for entry in video_entries if entry["unlabeled"]]
    if target_total <= 0 or not positive_entries:
        return {entry["video_name"]: 0 for entry in video_entries}

    total_unlabeled = sum(len(entry["unlabeled"]) for entry in positive_entries)
    allocations = {entry["video_name"]: 0 for entry in video_entries}
    remainders = []
    assigned = 0

    for entry in positive_entries:
        count = len(entry["unlabeled"])
        exact = target_total * count / total_unlabeled
        base = min(count, int(exact))
        allocations[entry["video_name"]] = base
        assigned += base
        if base < count:
            remainders.append((exact - base, entry["video_name"]))

    remaining = target_total - assigned
    remainders.sort(key=lambda item: (-item[0], item[1]))

    while remaining > 0:
        progress = False
        for _, video_name in remainders:
            entry = next(item for item in positive_entries if item["video_name"] == video_name)
            max_count = len(entry["unlabeled"])
            if allocations[video_name] >= max_count:
                continue
            allocations[video_name] += 1
            remaining -= 1
            progress = True
            if remaining <= 0:
                break
        if not progress:
            break

    for entry in positive_entries:
        allocations[entry["video_name"]] = min(allocations[entry["video_name"]], len(entry["unlabeled"]))

    return allocations


def select_uniform_items(items, count):
    if count <= 0 or not items:
        return []
    if count >= len(items):
        return list(items)

    positions = np.linspace(0, len(items) - 1, num=count, dtype=int)
    selected = []
    seen = set()
    for pos in positions.tolist():
        if pos in seen:
            continue
        selected.append(items[pos])
        seen.add(pos)
    return selected


def build_presampled_candidates(video_entries, target_total_frames):
    total_unlabeled = sum(len(entry["unlabeled"]) for entry in video_entries)
    if total_unlabeled <= target_total_frames:
        return [
            {
                "video_name": entry["video_name"],
                "frame_idx": frame_idx,
                "frame_path": str(frame_path),
            }
            for entry in video_entries
            for frame_idx, frame_path in entry["unlabeled"]
        ]

    presample_target = min(
        total_unlabeled,
        max(target_total_frames * CANDIDATE_MULTIPLIER, target_total_frames),
    )
    allocations = allocate_presample_counts(video_entries, presample_target)

    presampled = []
    for entry in video_entries:
        count = allocations[entry["video_name"]]
        selected_items = select_uniform_items(entry["unlabeled"], count)
        for frame_idx, frame_path in selected_items:
            presampled.append(
                {
                    "video_name": entry["video_name"],
                    "frame_idx": frame_idx,
                    "frame_path": str(frame_path),
                }
            )

    presampled.sort(key=lambda x: (x["video_name"], x["frame_idx"]))
    return presampled


def build_thumbnail_descriptor_from_path(frame_path):
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    thumb = cv2.resize(gray, (THUMBNAIL_SIZE, THUMBNAIL_SIZE), interpolation=cv2.INTER_AREA)
    return thumb.astype(np.float32).reshape(-1) / 255.0


def group_candidates_by_video(candidates):
    grouped = defaultdict(list)
    for item in candidates:
        grouped[item["video_name"]].append(item)
    for video_name in grouped:
        grouped[video_name].sort(key=lambda x: x["frame_idx"])
    return grouped


def load_descriptors_for_video(args):
    video_name, items = args
    valid_candidates = []
    descriptors = []

    for item in items:
        desc = build_thumbnail_descriptor_from_path(item["frame_path"])
        if desc is None:
            continue
        valid_candidates.append(item)
        descriptors.append(desc)

    if not descriptors:
        return video_name, valid_candidates, None

    return video_name, valid_candidates, np.stack(descriptors, axis=0)


def load_candidate_descriptors(candidates, workers):
    valid_candidates = []
    descriptors = []
    grouped = group_candidates_by_video(candidates)
    worker_items = [(video_name, grouped[video_name]) for video_name in sorted(grouped)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        for _, video_candidates, descriptor_block in tqdm(
            executor.map(load_descriptors_for_video, worker_items),
            total=len(worker_items),
            desc="Loading candidate descriptors",
        ):
            valid_candidates.extend(video_candidates)
            if descriptor_block is not None:
                descriptors.append(descriptor_block)

    if not descriptors:
        return [], None

    valid_candidates.sort(key=lambda x: (x["video_name"], x["frame_idx"]))
    return valid_candidates, np.concatenate(descriptors, axis=0)


def select_global_diverse_candidates(candidates, descriptor_matrix, requested_count, device):
    if requested_count <= 0 or not candidates:
        return []
    if len(candidates) <= requested_count:
        return list(candidates)

    with torch.no_grad():
        descriptors = torch.from_numpy(descriptor_matrix).to(device=device, dtype=torch.float32)
        mean_descriptor = descriptors.mean(dim=0)
        dist_to_mean = torch.sum((descriptors - mean_descriptor) ** 2, dim=1)

        n = descriptors.shape[0]
        available_mask = torch.ones(n, dtype=torch.bool, device=device)
        selected_positions = []

        seed_pos = int(torch.argmax(dist_to_mean).item())
        selected_positions.append(seed_pos)
        available_mask[seed_pos] = False

        min_distances = torch.sum((descriptors - descriptors[seed_pos]) ** 2, dim=1)

        progress = tqdm(total=requested_count - 1, desc="Selecting diverse frames")
        while available_mask.any().item() and len(selected_positions) < requested_count:
            masked_scores = min_distances.masked_fill(~available_mask, float("-inf"))
            next_pos = int(torch.argmax(masked_scores).item())
            selected_positions.append(next_pos)
            available_mask[next_pos] = False
            distances_to_new = torch.sum((descriptors - descriptors[next_pos]) ** 2, dim=1)
            min_distances = torch.minimum(min_distances, distances_to_new)
            progress.update(1)
        progress.close()

    selected = [candidates[pos] for pos in selected_positions]
    selected.sort(key=lambda x: (x["video_name"], x["frame_idx"]))
    return selected


def rebuild_output_dirs(video_entries):
    for entry in tqdm(video_entries, desc="Preparing output folders"):
        output_dir = entry["output_dir"]
        if os.path.exists(output_dir):
            print(f"  Removing existing folder: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)


def copy_selected_frames(selected_candidates):
    saved_counts = defaultdict(int)
    for item in tqdm(selected_candidates, desc="Saving diverse frames"):
        video_name = item["video_name"]
        frame_idx = int(item["frame_idx"])
        src = item["frame_path"]
        dst = os.path.join(DIVERSE_OUTPUT_DIR, video_name, f"{frame_idx:08d}.jpg")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        saved_counts[video_name] += 1
    return saved_counts


def build_parser():
    parser = argparse.ArgumentParser(
        description="Select globally diverse frames from extracted raw frames."
    )
    parser.add_argument(
        "--total-diverse-frames",
        type=int,
        default=TOTAL_DIVERSE_FRAMES,
        help="Total number of diverse frames to select globally.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Selection device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of workers for descriptor loading.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.total_diverse_frames < 0:
        raise ValueError("--total-diverse-frames must be >= 0")
    if args.workers <= 0:
        raise ValueError("--workers must be >= 1")

    raw_root = Path(RAW_FRAMES_DIR)
    if not raw_root.is_dir():
        print(f"Raw frames directory not found: {raw_root}")
        return

    allowed_video_names = load_train_video_names(CONFIG_PATH)
    labeled_root = Path(LABELED_DIR)
    device = resolve_device(args.device)

    video_entries, global_candidates, total_labeled = discover_candidates(
        raw_root,
        labeled_root,
        allowed_video_names,
    )

    if not video_entries:
        print(f"No matching video frame folders found in: {raw_root}")
        return

    print(f"Train videos in {CONFIG_PATH}: {len(allowed_video_names)}")
    print(f"Video folders found: {len(video_entries)}")
    print(f"Labeled frames excluded: {total_labeled}")
    print(f"Total unlabeled candidates: {len(global_candidates)}")
    print(f"Selection device: {device}")

    rebuild_output_root(DIVERSE_OUTPUT_DIR)
    rebuild_output_dirs(video_entries)

    if args.total_diverse_frames <= 0 or not global_candidates:
        presampled_candidates = []
        selected_candidates = []
    elif len(global_candidates) <= args.total_diverse_frames:
        presampled_candidates = list(global_candidates)
        selected_candidates = list(global_candidates)
    else:
        presampled_candidates = build_presampled_candidates(video_entries, args.total_diverse_frames)
        print(f"Presampled candidate count: {len(presampled_candidates)}")
        valid_candidates, descriptor_matrix = load_candidate_descriptors(
            presampled_candidates,
            args.workers,
        )
        if descriptor_matrix is None:
            selected_candidates = []
        else:
            selected_candidates = select_global_diverse_candidates(
                valid_candidates,
                descriptor_matrix,
                args.total_diverse_frames,
                device,
            )

    if args.total_diverse_frames <= 0 or not global_candidates:
        print("Presampled candidate count: 0")
    elif len(global_candidates) <= args.total_diverse_frames:
        print(f"Presampled candidate count: {len(presampled_candidates)}")

    saved_counts = copy_selected_frames(selected_candidates)

    print(f"Final selected diverse frames: {sum(saved_counts.values())}")
    for entry in video_entries:
        print(
            f"  {entry['video_name']}: "
            f"saved_diverse={saved_counts.get(entry['video_name'], 0)} "
            f"labeled_excluded={len(entry['labeled_indices'])}"
        )

    print("\nDiverse frame extraction complete.")


if __name__ == "__main__":
    main()
