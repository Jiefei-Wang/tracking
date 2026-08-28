import sys
sys.path.insert(0, r"D:\rat\tracking")
from modules.videos import extract_frames
from modules.keypoint_rtmpose_predict_common import keypoint_extraction_rtmpose, load_model_from_checkpoint_for_inference
from modules.detector_ssdlite_model import detector_extraction_ssdlite, load_detector
from modules.feature_utils import *  # normalize_keypoints
import os, pickle, pandas as pd, numpy as np, torch, gc

video_path = r"D:\rat\tracking\videos\Rat 4 2025-11-12 12-25-01.mkv"
video_id = os.path.splitext(os.path.basename(video_path))[0]

device_obj = "cuda"
detector_path = "output/ssdlite/weak_20260326_090250"
keypoint_path = "output/RTMPose/no_weak_20260328_174401"

metadata_df = pd.read_csv(f"output/clips/{video_id}/clip_metadata.csv")

detector = load_detector(detector_path=detector_path, device=device_obj)
model, _ = load_model_from_checkpoint_for_inference(model_path=keypoint_path, device=device_obj)

os.makedirs(f"output/keypoints/{video_id}", exist_ok=True)
bodyparts = None

print("Start keypoint generation...")
for _, row in metadata_df.iterrows():
    idx = row["clip_id"]
    video_path = row["video_path"]
    frame_indices = list(range(int(row["start_frame"]), int(row["end_frame"]) + 1))

    clip_frames = extract_frames(video_path, frame_indices=frame_indices)
    np_frames = np.stack(clip_frames)
    HEIGHT, WIDTH, _ = np_frames[0].shape

    print(f"clip {idx} [{row['stage']}] — DETECTION")
    with torch.no_grad():
        detection_boxes = detector_extraction_ssdlite(detector, np_frames, score_threshold=0.1)
    torch.cuda.empty_cache()

    print(f"clip {idx} [{row['stage']}] — POSE")
    with torch.no_grad():
        keypoints = keypoint_extraction_rtmpose(model, np_frames, detection_boxes)
    torch.cuda.empty_cache()

    if bodyparts is None:
        bodyparts = list(keypoints[0][0]['keypoints'].keys())

    for prediction in keypoints:
        for item in prediction:
            bbox, score, cropbox, kp = item.values()
            if len(bbox) > 4:
                bbox = bbox[:4] or bbox[0]
            normalize_keypoints(kp, WIDTH, HEIGHT)

    # persist this clip's keypoints + frame_indices for stage 3
    with open(f"output/keypoints/{video_id}/clip_{idx}.pkl", "wb") as f:
        pickle.dump({"frame_indices": frame_indices, "keypoints": keypoints}, f)

    del clip_frames, np_frames, detection_boxes, keypoints
    gc.collect()
    torch.cuda.empty_cache()

with open(f"output/bodyparts.pkl", "wb") as f:
    pickle.dump(bodyparts, f)

print(f"Done — keypoints saved per clip in output/keypoints/{video_id}")