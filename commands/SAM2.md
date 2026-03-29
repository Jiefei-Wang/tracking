
Idea: 
1. Extract frames from videos
    - Create: `output/sam2/raw_frames`
2. For all frames, pool them and extract diverse frames to maximize the diversity of the data
    - Create: `output/sam2/diverse`
3. Use labeled-data to train SAM2 and extract SAM2 labels for all diverse frames
    - Create: `output/sam2/DLC_frames`, `output/sam2/sam2_training_merge_frames`, and `output/sam2/sam2_pickle_raw`
4. Filter low-quality SAM2 labels and use them for training
    - Create: `output/sam2/diverse_masked`, `output/sam2/final`, `output/sam2/sam2_pickle_filtered` 


```
python scripts/sam2/video_frames_extraction.py
python scripts/sam2/extract_diverse_frames.py
python scripts/sam2/extract_sam2_labels.py
python scripts/sam2/filter_low_quality_sam2.py
```