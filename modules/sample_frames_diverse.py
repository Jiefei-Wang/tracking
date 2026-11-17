"""Top up a shared frame budget across videos, preserving images and labels.

    from modules.sample_frames_diverse import extract_diverse_frames
    summary = extract_diverse_frames(root=".", total_frames=2000)

Frames go in output/extracted_frames/<video stem>/<zero-based index>.jpg.
Brightness filtering uses OpenCV CUDA when available; diversity distances use
PyTorch CUDA when available. Both fall back to CPU. No work runs on import.
"""
from pathlib import Path
import re
import warnings

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
EMBED_SIZE = 32
MIN_BRIGHTNESS = 50.0


def _frame_index(path):
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def _embedding(frame, cuda=False, filter_dark=False):
    if cuda:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(frame)
        gray = cv2.cuda.cvtColor(gpu, cv2.COLOR_BGR2GRAY)
        if filter_dark and cv2.cuda.meanStdDev(gray).download()[0, 0] < MIN_BRIGHTNESS:
            return None
        small = cv2.cuda.resize(gray, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_LINEAR).download()
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if filter_dark and float(gray.mean()) < MIN_BRIGHTNESS:
            return None
        small = cv2.resize(gray, (EMBED_SIZE, EMBED_SIZE), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32).ravel() / 255.0


def _diverse_order(vectors, anchors, count, device="cpu"):
    """Greedy farthest-point selection against all existing and added frames."""
    if device == "cuda":
        import torch
        data = torch.as_tensor(vectors, device=device)
        distances = torch.full((len(data),), float("inf"), device=device)
        # Chunk anchors to avoid a large pairwise matrix.
        for start in range(0, len(anchors), 128):
            seeds = torch.as_tensor(np.asarray(anchors[start:start + 128]), device=device)
            distances = torch.minimum(distances, torch.cdist(data, seeds).square().amin(dim=1))
        if not len(anchors):
            distances = (data - data.mean(dim=0)).square().sum(dim=1)
        for _ in range(min(count, len(data))):
            chosen = int(distances.argmax().item())
            yield chosen
            distances = torch.minimum(distances, (data - data[chosen]).square().sum(dim=1))
            distances[chosen] = -float("inf")
    else:
        distances = np.full(len(vectors), np.inf, dtype=np.float32)
        for anchor in anchors:
            delta = vectors - anchor
            np.minimum(distances, np.einsum("ij,ij->i", delta, delta), out=distances)
        if not len(anchors):
            distances = ((vectors - vectors.mean(axis=0)) ** 2).sum(axis=1)
        for _ in range(min(count, len(vectors))):
            chosen = int(np.argmax(distances))
            yield chosen
            delta = vectors - vectors[chosen]
            np.minimum(distances, np.einsum("ij,ij->i", delta, delta), out=distances)
            distances[chosen] = -np.inf


def extract_diverse_frames(root=".", total_frames=2000):
    """Reach a TOTAL of ``total_frames`` across all source videos, not per video.

    Count unique numbered images in output/extracted_frames/<video stem> for
    videos directly in root/videos. Skip all processing when their shared count
    reaches the target. Other folders do not count. Existing files, annotations,
    config and train/test assignments are never changed.

    Spread a candidate pool of about three times the missing count across videos,
    then select globally for diversity against every existing frame. Candidate
    allocation is balanced across videos, but final counts need not be equal.
    If too few bright frames are found, scan remaining indices for replacements.
    Warn if the shared target cannot be reached. Return aggregate and per-video
    counts. Grayscale appearance diversity does not guarantee pose diversity.
    """
    if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames < 0:
        raise ValueError("total_frames must be a nonnegative integer")
    root = Path(root).expanduser().resolve()
    video_dir = root / "videos"
    if not video_dir.is_dir():
        raise FileNotFoundError(f"Videos directory does not exist: {video_dir}")
    videos = sorted(p for p in video_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if len({p.stem for p in videos}) != len(videos):
        raise ValueError("Multiple videos share a filename stem; output folders would collide")
    existing, destinations, per_video = {}, {}, {}
    for video in videos:
        dest = root / "output/extracted_frames" / video.stem
        destinations[video.stem] = dest
        frames = {}
        if dest.exists():
            for image in sorted(dest.iterdir()):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    idx = _frame_index(image)
                    if idx is None:
                        raise ValueError(f"Cannot determine frame index from {image}")
                    frames.setdefault(idx, image)
        existing[video.stem] = frames
        per_video[video.stem] = {"existing": len(frames), "added": 0, "total": len(frames)}
    before = sum(len(frames) for frames in existing.values())
    result = {"existing": before, "added": 0, "total": before,
              "shortfall": max(0, total_frames - before), "videos": per_video,
              "filter_backend": None, "diversity_backend": None}
    if before >= total_frames:
        print(f"{before} existing frames across all videos; target {total_frames} reached, skipped")
        return result
    cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0 and all(
        hasattr(cv2.cuda, name) for name in ("cvtColor", "resize", "meanStdDev"))
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except ImportError:
        pass
    result.update(filter_backend="opencv_cuda" if cuda else "cpu", diversity_backend=device)
    print(f"Filter: {result['filter_backend']}; diversity: {device}; shared target: {total_frames}")
    anchors = []
    for frames in existing.values():
        for image in frames.values():
            frame = cv2.imread(str(image))
            if frame is None:
                raise ValueError(f"Cannot read existing image: {image}")
            anchors.append(_embedding(frame, cuda=cuda))
    needed = total_frames - before
    candidates, vectors = [], []
    captures, remaining, checked = {}, {}, {}
    try:
        for video in videos:
            cap = cv2.VideoCapture(str(video))
            captures[video.stem] = cap
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {video}")
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if length <= 0:
                raise RuntimeError(f"Invalid frame count for {video}")
            remaining[video.stem] = np.array([i for i in range(length) if i not in existing[video.stem]], dtype=np.int64)
            checked[video.stem] = set()

        def collect(stem, indices, stop_at=None):
            cap = captures[stem]
            for idx in indices:
                idx = int(idx)
                checked[stem].add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                vector = _embedding(frame, cuda=cuda, filter_dark=True)
                if vector is None:
                    continue
                candidates.append((stem, idx))
                vectors.append(vector)
                if stop_at is not None and len(candidates) >= stop_at:
                    break

        quota = max(1, (needed * 3 + len(videos) - 1) // len(videos)) if videos else 0
        for video in videos:
            available = remaining[video.stem]
            count = min(quota, len(available))
            if count:
                collect(video.stem, available[np.linspace(0, len(available) - 1, count, dtype=int)])
        for video in videos:
            if len(candidates) >= needed:
                break
            stem = video.stem
            collect(stem, (i for i in remaining[stem] if int(i) not in checked[stem]), stop_at=needed)
        if vectors:
            for pos in _diverse_order(np.asarray(vectors), anchors, len(vectors), device=device):
                stem, idx = candidates[pos]
                cap = captures[stem]
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok:
                    continue
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    continue
                destinations[stem].mkdir(parents=True, exist_ok=True)
                with (destinations[stem] / f"{idx:08d}.jpg").open("xb") as handle:
                    handle.write(encoded.tobytes())
                per_video[stem]["added"] += 1
                per_video[stem]["total"] += 1
                result["added"] += 1
                if result["added"] >= needed:
                    break
    finally:
        for cap in captures.values():
            cap.release()
    result["total"] = before + result["added"]
    result["shortfall"] = max(0, total_frames - result["total"])
    print(f"All videos: {before} existing + {result['added']} new = {result['total']} frames")
    if result["shortfall"]:
        warnings.warn(f"Shared target short by {result['shortfall']}: not enough bright, readable frames", stacklevel=2)
    return result
