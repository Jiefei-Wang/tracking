- `video_frames_extraction.py`: do two things:
    1. Extract frames from the video and save them in `output/video_frames_for_sam` (max 100 frames).
    2. Copy the labeled frames (with keypoints) from `input/labeled-data` to `output/video_frames_for_sam`.
- `scripts/extract_sam2_labels.py`: Use SAM2 to extract segmentation masks for all frames in `output/video_frames_for_sam` and save the masks in `output/sam2_labels`. 
- `scripts/detector_yolo.py`: Train YOLOv8 on the `input/labeled-data`
- `scripts/keypoint_preprocess.py`: Clear `output/keypoint_cache`, precompute both object ROI and augmentation ROI, save cropped augmentation ROI images in `crops/`, and save debug images with the object ROI drawn in `debug/`.



```
python scripts/keypoint_preprocess.py --batch-size 256
python scripts/keypoint_HRNet.py train --train-config training_config/default.yaml
```
