# RTMPose Commands

# debug
```bash
python scripts/keypoints/RTMPose.py \
  --model-config input/RTMPose/model_rtmpose_x.yaml \
  --config-overwrite input/RTMPose/config_with_weak.yaml \
  debug
```



# Train
```bash
python scripts/keypoints/RTMPose.py \
  --model-config input/RTMPose/model_rtmpose_x.yaml \
  --config-overwrite input/RTMPose/config_no_weak.yaml \
  prepare


python scripts/keypoints/RTMPose.py \
  --model-config input/RTMPose/model_rtmpose_x.yaml \
  --config-overwrite input/RTMPose/config_no_weak.yaml \
  train --prefix no_weak 

```


```bash
python scripts/keypoints/RTMPose.py \
  --model-config input/RTMPose/model_rtmpose_x.yaml \
  --config-overwrite input/RTMPose/config_with_weak.yaml \
  --init-checkpoint output/RTMPose/no_weak_20260328_174401/checkpoint_best.pt\
  train --prefix with_weak

```






```bash
python scripts/keypoints/RTMPose.py prepare
python scripts/keypoints/RTMPose.py \
  --detector-checkpoint output/ssdlite/no_weak_20260325_224005/checkpoint_best.pt \
  train \
  --prefix weak_run
```

# Eval

```bash
python scripts/keypoints/RTMPose.py \
  eval \
  --checkpoint output/RTMPose/<run>/checkpoint_best.pt \
  --split val
```


```bash
python scripts/keypoints/RTMPose.py \
  eval \
  --checkpoint output/RTMPose/<run>/checkpoint_best.pt \
  --split test
```

# Predict

```bash
python scripts/keypoints/RTMPose_predict.py \
  --score-cutoff 0.2 \
  --visibility-cutoff 0.5 \
  --checkpoint output/RTMPose/no_weak_20260328_174401/checkpoint_best.pt \
  --video test_videos/Camera4_stitched_600_660.mp4
```


```bash
python scripts/keypoints/RTMPose.py --help
python scripts/keypoints/RTMPose.py train --help
python scripts/keypoints/RTMPose.py eval --help
python scripts/keypoints/RTMPose.py debug --help
python scripts/keypoints/RTMPose_predict.py --help
```
