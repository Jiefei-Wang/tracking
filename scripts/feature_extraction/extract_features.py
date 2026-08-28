import sys
sys.path.insert(0, r"D:\rat\tracking")
import os, pickle, math, pandas as pd, numpy as np
from modules.feature_utils import aggregate_velocity_features

video_path = r"D:\rat\tracking\videos\Rat 4 2025-11-12 12-25-01.mkv"
video_id = os.path.splitext(os.path.basename(video_path))[0]

metadata_df = pd.read_csv(f"output/clips/{video_id}/clip_metadata.csv")

output_dir =  os.path.join("output", "clips", video_id)
os.makedirs(output_dir,exist_ok=True)


with open("output/bodyparts.pkl", "rb") as f:
    bodyparts = pickle.load(f)
kp_weights = {bp: 1 for bp in bodyparts}

velocity_feature = []
clip_feature_vectors = {}

for _, row in metadata_df.iterrows():
    idx = row["clip_id"]
    stage = row["stage"]
    video_id = row["video_id"]

    with open(f"output/keypoints/{video_id}/clip_{idx}.pkl", "rb") as f:
        clip_data = pickle.load(f)
    frame_indices = clip_data["frame_indices"]
    keypoints = clip_data["keypoints"]

    for i in range(1, len(keypoints)):
        if frame_indices[i] != frame_indices[i-1] + 1:
            continue

        curr_kp = keypoints[i][0]['keypoints']
        prev_kp = keypoints[i-1][0]['keypoints']

        for bp in bodyparts:
            x_curr, x_prev = curr_kp[bp]['x'], prev_kp[bp]['x']
            y_curr, y_prev = curr_kp[bp]['y'], prev_kp[bp]['y']

            if np.isnan(x_prev) or np.isnan(y_prev) or np.isnan(x_curr) or np.isnan(y_curr):
                vx = vy = speed = np.nan
            else:
                vx = x_curr - x_prev
                vy = y_curr - y_prev
                speed = math.sqrt(kp_weights[bp] * ((vx)**2 + (vy)**2))

            velocity_feature.append({
                "video_id": video_id, "clip_id": idx,
                "frame_prev": frame_indices[i-1], "frame_curr": frame_indices[i],
                "bodypart": bp, "x_prev": x_prev, "y_prev": y_prev,
                "x_curr": x_curr, "y_curr": y_curr,
                "vx": vx, "vy": vy, "speed": speed, "weight": kp_weights[bp],
            })

    clip_velocity_df = pd.DataFrame([r for r in velocity_feature if r["clip_id"] == idx])
    clip_feature_vectors[idx] = aggregate_velocity_features(clip_velocity_df, bodyparts, video_id, stage)

velocity_df = pd.DataFrame(velocity_feature)
velocity_df.to_csv(os.path.join(output_dir, "velocity_per_bodypart.csv"), index=False)


clip_features_df = pd.DataFrame.from_dict(clip_feature_vectors, orient="index")
clip_features_df.index.name = "clip_id"
clip_features_df.to_csv(os.path.join(output_dir, "clip_features.csv"), index=False)

# print(velocity_df.head(20))
# print(clip_features_df.head(20))