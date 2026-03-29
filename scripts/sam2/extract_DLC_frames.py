import argparse
import re
import shutil
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

LABELED_DIR = Path("input/labeled-data")
OUTPUT_BASE_DIR = Path("output/sam2/DCL_frames")
CONFIG_PATH = Path("config.yaml")


def extract_index_from_name(path: Path):
    """Extract the last integer from a file stem."""
    matches = re.findall(r"\d+", path.stem)
    if not matches:
        return None
    return int(matches[-1])


def load_configured_video_names(config_path: Path):
    """Load train/val/test video names from config.yaml."""
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    configured_names = set()
    for key in ("train_videos", "val_videos", "test_videos"):
        values = payload.get(key, [])
        if not isinstance(values, list):
            raise ValueError(f"Expected '{key}' to be a list in {config_path}")
        configured_names.update(str(item) for item in values)

    if not configured_names:
        raise ValueError(f"No videos found in train/val/test lists in {config_path}")

    return configured_names


def copy_labeled_frames_preserve_index(src_root: Path, dst_root: Path, allowed_video_names: set[str]):
    """Copy labeled PNG frames to JPEG while preserving numeric frame index."""
    if not src_root.exists():
        print(f"Labeled-data folder not found: {src_root}")
        return

    folders = sorted(
        folder for folder in src_root.iterdir()
        if folder.is_dir() and folder.name in allowed_video_names
    )
    for folder in tqdm(folders, desc="Copying labeled frame folders"):
        out_folder = dst_root / folder.name
        out_folder.mkdir(parents=True, exist_ok=True)

        png_files = sorted(folder.glob("*.png"))
        for png_file in tqdm(png_files, desc=f"  {folder.name}", leave=False):
            frame_idx = extract_index_from_name(png_file)
            if frame_idx is None:
                print(f"Skipped (no numeric index): {png_file}")
                continue

            jpg_path = out_folder / f"{frame_idx:08d}.jpg"
            try:
                with Image.open(png_file) as img:
                    img.convert("RGB").save(jpg_path, "JPEG", quality=95)
            except Exception as exc:
                print(f"Failed: {png_file} ({exc})")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract DLC labeled frames into output/sam2/DCL_frames."
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete existing output folder before extraction.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    configured_video_names = load_configured_video_names(CONFIG_PATH)
    print(f"Configured videos in {CONFIG_PATH}: {len(configured_video_names)}")

    if args.clean_output and OUTPUT_BASE_DIR.exists():
        print(f"Removing existing folder: {OUTPUT_BASE_DIR}")
        shutil.rmtree(OUTPUT_BASE_DIR)

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    copy_labeled_frames_preserve_index(LABELED_DIR, OUTPUT_BASE_DIR, configured_video_names)
    print(f"Done. Output written to: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()
