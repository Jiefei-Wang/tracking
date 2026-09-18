import sys
sys.path.insert(0, r"D:\rat\tracking")
from modules.videos import extract_frames, get_frame_numbers
from modules.feature_utils import *  # get_bar_press_times, build_windows, get_frame_range, frames_to_video, etc.
import os, cv2, pandas as pd

video_path = r"D:\rat\tracking\videos\Rat 4 2025-11-12 12-25-01.mkv"
video_id = os.path.splitext(os.path.basename(video_path))[0]

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

file_paths = {
    "FR": r'D:\rat\tracking\FR\FR\Backup of Box 04 11-12-2025 12.31 Subject 4 DA',
    "EXT": r'D:\rat\tracking\EXT\EXT\Backup of Box 04 11-12-2025 14.32 Subject 4 DA',
    "RELIEF": r'D:\rat\tracking\RELIEF\RELIEF\Backup of Box 04 11-12-2025 14.47 Subject 4 DA',
}

times = []
stages = []
offsets = {"FR": 0, "EXT": 5, "RELIEF": 125}

for stage, file in file_paths.items():
    subject_id, stage_times = get_bar_press_times(file_path=file, stage=stage, offsets=offsets, pad_seconds=5)
    times.extend(stage_times)
    stages.extend([stage] * len(stage_times))

print(f"{len(times)} clips found")
os.makedirs(f"output/clips/{video_id}", exist_ok=True)

# ── LOOP 1: build metadata CSV (computes start_frame/end_frame ONCE) ──
metadata_rows = []
for idx, t in enumerate(times):
    stage = stages[idx]
    start_frame, end_frame = get_frame_range(video_path, t[0], t[1])
    print(f"clip {idx} [{stage}]  {t[0]} -> {t[1]}   frames {start_frame} -> {end_frame}")

    metadata_rows.append({
        "clip_id": idx,
        "stage": stage,
        "t_start": t[0],
        "t_end": t[1],
        "start_frame": start_frame,
        "end_frame": end_frame,
        "video_id": video_id,
        "video_path": video_path,
    })

metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_csv(f"output/clips/{video_id}/clip_metadata.csv", index=False)
print(f"Saved output/clips/{video_id}/clip_metadata.csv")
print(metadata_df.head(20))

print("Start generating clips...")
# ── LOOP 2: generate clip videos — reuses frame numbers already in metadata_df ──
for _, row in metadata_df.iterrows():
    idx = row["clip_id"]
    stage = row["stage"]
    out_path = f"output/clips/{video_id}/clip_{idx}_{stage}.mp4"

    if os.path.exists(out_path):
        print(f"clip {idx} [{stage}] already exists, skipping")
        continue

    frame_indices = list(range(int(row["start_frame"]), int(row["end_frame"]) + 1))
    clip_frames = extract_frames(row["video_path"], frame_indices=frame_indices)
    frames_to_video(clip_frames, out_path, fps=fps)
    print(f"clip {idx} [{stage}] saved -> {out_path}")

    del clip_frames

print("Done generating clips")