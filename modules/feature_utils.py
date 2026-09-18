import copy, math ,os, cv2 ,pandas as pd, numpy as np 
import torch
import gc

def extract_raw_data(file_path, array_letter="Y"):
    data = []
    start_processing = False
    with open(file_path, 'r') as file:
        lines = file.readlines()

    subject_id = None
    marker = f"{array_letter}:"
    for line in lines:
        if "Subject:" in line:
            subject_id = line.split("Subject:")[1].strip()

        if marker in line:
            start_processing = True
            continue

        if start_processing:
            if len(line) > 1 and line[0].isalpha() and line[1] == ':':
                break
            if not line:
                break
            parts = line.split()
            if len(parts) > 1:
                numbers = parts[1:]
                data.extend(map(float, numbers))

    while data and data[-1] == 0:
        data.pop()

    return subject_id, data

def frames_to_video(frames, output_path, fps=30):
    """
    frames: list/array of frames (as returned by extract_frames), each a
            HxWx3 numpy array (BGR, same format cv2 reads/writes natively)
    output_path: e.g. "output/clip_0.mp4"
    fps: frame rate for the output video
    """
    if len(frames) == 0:
        raise ValueError("No frames provided")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # codec for .mp4 output
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        writer.write(frame)

    writer.release()

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise IOError(f"Video file was not created or is empty: {output_path}")
    
    print(f"Saved {len(frames)} frames -> {output_path}")
    
def apply_stage_offset(timestamps_min, stage, offsets_min=None):

    if offsets_min is None:
        offsets_min = {"FR": 0, "EXT": 5, "RELIEF": 125}

    return [t + offsets_min[stage] for t in timestamps_min]

def get_bar_press_times(file_path, stage, offsets, pad_seconds=5):
    """
    MED-PC file -> bar press timestamps -> stage offset -> merged +/-pad_seconds
    windows -> returns time intervals as "M:SS" strings (matches  main file's
    `times` format, which it already feeds into get_frame_range itself).
    """
    subject_id, Y = extract_raw_data(file_path, array_letter="Y")
    Y_offset = apply_stage_offset(Y, stage, offsets)
    windows_min = build_windows(Y_offset, pad_seconds=pad_seconds)

    times = [[minutes_to_mmss(start_min), minutes_to_mmss(end_min)]
             for start_min, end_min in windows_min]

    return subject_id, times

def build_windows(timestamps_min, pad_seconds=5, clamp_start=0.0):
    pad_min = pad_seconds / 60
    raw_windows = [(max(clamp_start, t - pad_min), t + pad_min) for t in timestamps_min]
    raw_windows.sort(key=lambda w: w[0])

    merged = [raw_windows[0]]
    for start, end in raw_windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def minutes_to_mmss(t_min):
    """Bridges decimal-minutes MED-PC timestamps into the 'M:SS' format"""
    m = int(t_min)
    s = (t_min - m) * 60
    return f"{m}:{s:.3f}"


def parse_timestamp(ts):
    parts = [float(p) for p in ts.split(':')]
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    elif len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    else:
        raise ValueError(f"Unrecognized timestamp format: {ts}")


def get_frame_range(video_path, start_ts, end_ts):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_sec = parse_timestamp(start_ts)
    end_sec = parse_timestamp(end_ts)

    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames - 1)

    return start_frame, end_frame





#Things to consider, what threshold to enforce to be able to eliminate "missing" key points? the threshold we can use is 0.5
# Original keypoints format: 'head': {'x': 743.35546875, 'y': 491.76171875, 'score': 0.9911382794380188, 'visibility_score': 0.9613568782806396}
def normalize_keypoints (keypoints, width, height):
    for i in keypoints:
        if keypoints[i]['visibility_score'] >= 0.5:
            keypoints[i]['x'] = keypoints[i]['x'] / width 
            keypoints[i]['y'] = keypoints[i]['y'] / height
        else: 
            keypoints[i]['x'] = np.nan
            keypoints[i]['y'] = np.nan

    return keypoints



def aggregate_velocity_features(clip_df, bodyparts,video_id,stage, stats=("mean", "var", "max")):
    """Collapse one clip's velocity rows into a single flat feature vector."""
    feature_vector = {}
    feature_vector = {"video_id": video_id,"stage": stage}

    for bp in bodyparts:
        bp_df = clip_df[clip_df["bodypart"] == bp]

        for col in ["speed"]:
            values = bp_df[col].to_numpy()
            for stat in stats:
                key = f"{bp}_{col}_{stat}"
                if stat == "mean":
                    feature_vector[key] = np.nanmean(values)
                # elif stat == "std":
                #     feature_vector[key] = np.nanstd(values)
                elif stat == "var":
                    feature_vector[key] = np.nanvar(values)
                # elif stat == "max":
                #     feature_vector[key] = np.nanmax(values) if not np.all(np.isnan(values)) else np.nan
                # elif stat == "min":
                #     feature_vector[key] = np.nanmin(values) if not np.all(np.isnan(values)) else np.nan

    return feature_vector

