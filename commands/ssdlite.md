# SSDLite Commands

Run all commands from the project root with the `track` environment.

## Train

Default training:

```bash
python scripts/detector_ssdlite.py train
```

Training with custom batch sizes and weak-supervision weight:

```bash
python scripts/detector_ssdlite.py \
  --weak-sample-weight 0.25 \
  --workers 8 \
  --pin-memory \
  --persistent-workers \
  --preload-images \
  train \
  --batch-size 16 \
  --weak-batch-size 16 \
  --eval-batch-size 8 \
  --epochs 100 \
  --device cuda:0
```

Training with the default configs:

```bash
python scripts/detector_ssdlite.py train
```

Defaults:

```bash
model config: input/ssdlite/model.yaml
run config: input/ssdlite/config.yaml
```

You can control checkpointing and full evaluation frequency from the run config:

```yaml
save_every_n_epoch: 5
eval_every_n_epoch: 5
max_save: 5
```

`save_every_n_epoch` controls checkpoint frequency, `eval_every_n_epoch` controls when epoch summaries are reported, and `max_save` keeps only the most recent periodic `checkpoint_epoch_*.pt` files. The reported train/val losses on eval epochs come from deterministic no-augmentation evaluation passes.

Training with a small override config:

```bash
python scripts/detector_ssdlite.py \
  --config_overwrite input/ssdlite/config_with_weak.yaml \
  train
```

If the override YAML contains:

```yaml
prefix: weak
weak_sample_weight: 0.5
```

then outputs are saved to:

```bash
output/ssdlite/weak_<YYYYMMDD_HHMMSS>/
```

You can still override prefix from CLI if you want:

```bash
python scripts/detector_ssdlite.py train --prefix experiment1
```

### current training command
```
python scripts/detector_ssdlite.py \
  --config_overwrite input/ssdlite/config_no_weak.yaml \
  train
python scripts/detector_ssdlite.py \
  --config_overwrite input/ssdlite/config_with_weak.yaml \
  train
```



## Debug

Export transformed training inputs right before model input.

This deletes and recreates:

```bash
output/ssdlite/debug/
```

Default debug export:

```bash
python scripts/detector_ssdlite.py debug
```

Limit the number of saved examples per source:

```bash
python scripts/detector_ssdlite.py debug --limit-per-source 10
```

Debug outputs are written to:

```bash
output/ssdlite/debug/label/
output/ssdlite/debug/sam2/
```

## Evaluate

Evaluate a checkpoint on validation:

```bash
python scripts/detector_ssdlite.py eval \
  --checkpoint output/ssdlite/20260325_153000/checkpoint_best.pt \
  --split val
```

Evaluate on test:

```bash
python scripts/detector_ssdlite.py eval \
  --checkpoint output/ssdlite/no_weak_20260325_224005/checkpoint_best.pt \
  --split test
```

## Predict

Predict on the test split:

```bash
python scripts/detector_ssdlite.py predict \
  --checkpoint output/ssdlite/no_weak_20260325_224005/checkpoint_best.pt \
  --split test
```

Prediction uses a confidence cutoff of `0.5` by default. Override it if needed:

```bash
python scripts/detector_ssdlite.py predict \
  --checkpoint output/ssdlite/20260325_153000/checkpoint_best.pt \
  --split test \
  --prediction-score-threshold 0.7
```

Predict for one video only:

```bash
python scripts/detector_ssdlite.py predict \
  --checkpoint output/ssdlite/weak_20260326_090250/checkpoint_best.pt \
  --video "test_videos/Camera4_stitched_600_660.mp4"
```


## Help

Show all options:

```bash
python scripts/detector_ssdlite.py --help
python scripts/detector_ssdlite.py train --help
python scripts/detector_ssdlite.py debug --help
python scripts/detector_ssdlite.py eval --help
python scripts/detector_ssdlite.py predict --help
```
