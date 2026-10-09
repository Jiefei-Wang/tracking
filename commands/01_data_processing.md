
```bash
conda activate tracking
```


Extracting manual labels.
```bash
python scripts/01_data_processing/1_extract_frames.py
```

describe the number of labeled/unlabeled frames per video 
```bash
python scripts/01_data_processing/2_descriptive.py
```

```bash
python scripts/01_data_processing/3_skeleton_frame_overlay.py
```

