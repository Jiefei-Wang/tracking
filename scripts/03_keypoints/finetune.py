#!/usr/bin/env python3
"""
RTMPpose Finetuning Script

- converts JSON labels to DLC CSV
- runs RTMPose prepare/train
- optionally evaluates

"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BODYPARTS = [
    "head", "nose", "spine1", "spine2", "spine3", "tailbase",
    "tail1", "tail2", "tail_tip",
    "L_hip", "L_backpaw", "R_backpaw",
    "L_shoulder", "R_frontpaw", "R_shoulder",
    "R_hip", "R_knee", "L_knee", "L_frontpaw",
]
SCORER = "rats"
FIRST_COL_VALUE = "labeled-data"


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Local repo-friendly RTMPose training helper."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=repo_root,
        help="Repository root directory. Defaults to the script's parent folder.",
    )
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("input/labels"),
        help="JSON label files root.",
    )
    parser.add_argument(
        "--labeled-data-root",
        type=Path,
        default=Path("input/labeled-data"),
        help="Destination for generated DLC CSV label data.",
    )
    parser.add_argument(
        "--extracted-frames-root",
        type=Path,
        default=Path("output/extracted_frames"),
        help="Destination for extracted frame data.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("input/RTMPose/model_rtmpose_x.yaml"),
        help="RTMPose model config file.",
    )
    parser.add_argument(
        "--config-overwrite",
        type=Path,
        default=Path("input/RTMPose/config_finetune.yaml"),
        help="RTMPose config overwrite file.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=Path("output/RTMPose/no_weak_20260328_174401/checkpoint_best.pt"),
        help="Initial checkpoint path to fine-tune from.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes to use during prepare/train.",
    )
    parser.add_argument(
        "--save-every-n-epoch",
        type=int,
        default=1,
        help="Checkpoint save frequency in epochs.",
    )
    parser.add_argument(
        "--max-save",
        type=int,
        default=100,
        help="Maximum number of checkpoints to keep.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--prefix",
        default="finetune",
        help="Prefix for output RTMPose run directory.",
    )
    parser.add_argument(
        "--selection-metric",
        default="loss",
        choices=["loss", "accuracy", "precision", "recall"],
        help="Metric used to select best checkpoint.",
    )
    parser.add_argument(
        "--selection-mode",
        default="min",
        choices=["min", "max"],
        help="Selection direction for the metric.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation after training.",
    )
    parser.add_argument(
        "--eval-splits",
        default="val,test",
        help="Comma-separated splits to evaluate after training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def load_label_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload:
        frame_idx = int(item["frame_idx"])
        row = {"frame": frame_idx}
        labels = item.get("labels") or {}
        for kp, val in labels.items():
            vis = int(val[0]) if val else 0
            x = val[1] if len(val) > 1 else None
            y = val[2] if len(val) > 2 else None
            row[f"{kp}_x"] = float(x) if vis and x is not None else float("nan")
            row[f"{kp}_y"] = float(y) if vis and y is not None else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_dlc_csv(df: pd.DataFrame, video_name: str, bodyparts: list[str], csv_path: Path) -> None:
    kp_cols = [f"{bp}_x" for bp in bodyparts] + [f"{bp}_y" for bp in bodyparts]
    n = 3 + len(kp_cols)
    h0 = [SCORER] + [""] * (n - 1)
    h1 = ["", "", ""] + [bp for bp in bodyparts for _ in range(2)]
    h2 = ["", "", ""] + ["x" if i % 2 == 0 else "y" for _ in bodyparts for i in range(2)]
    
    max_frame = int(pd.to_numeric(df["frame"], errors="coerce").max())
    fw = max(1, len(str(max_frame)))
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        ",".join(h0),
        ",".join(h1),
        ",".join(h2),
    ]
    for _, row in df.sort_values("frame").iterrows():
        fi = int(float(row["frame"]))
        parts = [FIRST_COL_VALUE, video_name, f"img{str(fi).zfill(fw)}.png"]
        for c in kp_cols:
            v = row.get(c)
            parts.append("" if pd.isna(v) else str(int(round(float(v)))))
        lines.append(",".join(parts))

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_labels(labels_root: Path, labeled_data_root: Path) -> int:
    labels_root = labels_root.expanduser().resolve()
    labeled_data_root = labeled_data_root.expanduser().resolve()
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels root does not exist: {labels_root}")

    count = 0
    for json_path in sorted(labels_root.glob("*.json")):
        video_name = json_path.stem
        df = load_label_json(json_path)
        csv_path = labeled_data_root / video_name / "CollectedData_rats.csv"
        build_dlc_csv(df, video_name, BODYPARTS, csv_path)
        count += 1
        print(f"Converted {json_path.name} -> {csv_path.relative_to(labeled_data_root)} ({len(df)} frames)")
    if count == 0:
        raise FileNotFoundError(f"No JSON label files found in {labels_root}")
    return count


def run_command(cmd: list[str], cwd: Path, dry_run: bool = False) -> None:
    print("\n> ", " ".join(str(x) for x in cmd))
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()

    labels_root = repo_root / args.labels_root
    labeled_data_root = repo_root / args.labeled_data_root
    extracted_frames_root = repo_root / args.extracted_frames_root
    model_config = repo_root / args.model_config
    config_overwrite = repo_root / args.config_overwrite
    init_checkpoint = repo_root / args.init_checkpoint
    rtmpose_script = repo_root / "scripts" / "keypoints" / "RTMPose.py"

    if not rtmpose_script.exists():
        raise FileNotFoundError(f"RTMPose entrypoint not found: {rtmpose_script}")

    print(f"Repository root: {repo_root}")
    print(f"Labels root: {labels_root}")
    print(f"Generated labeled data root: {labeled_data_root}")
    print(f"Extracted frames root: {extracted_frames_root}")
    print(f"Model config: {model_config}")
    print(f"Config overwrite: {config_overwrite}")
    print(f"Initial checkpoint: {init_checkpoint}")

    convert_labels(labels_root, labeled_data_root)
    print(f"Converted JSON label files to DLC CSV in {labeled_data_root}")

    prepare_cmd = [
        sys.executable,
        str(rtmpose_script),
        "--model-config",
        str(model_config),
        "--config-overwrite",
        str(config_overwrite),
        "--labels-root",
        str(labeled_data_root),
        "--labeled-frames-root",
        str(extracted_frames_root),
        "--workers",
        str(args.workers),
        "prepare",
    ]
    print("\nRunning RTMPose prepare...")
    run_command(prepare_cmd, cwd=repo_root, dry_run=args.dry_run)

    train_cmd = [
        sys.executable,
        str(rtmpose_script),
        "--model-config",
        str(model_config),
        "--config-overwrite",
        str(config_overwrite),
        "--labels-root",
        str(labeled_data_root),
        "--labeled-frames-root",
        str(extracted_frames_root),
        "--workers",
        str(args.workers),
    ]
    if init_checkpoint.exists():
        train_cmd.extend(["--init-checkpoint", str(init_checkpoint)])
    train_cmd.extend([
        "--save-every-n-epoch",
        str(args.save_every_n_epoch),
        "--max-save",
        str(args.max_save),
        "train",
        "--prefix",
        args.prefix,
        "--epochs",
        str(args.epochs),
        "--selection-metric",
        args.selection_metric,
        "--selection-mode",
        args.selection_mode,
    ])

    print("\nRunning RTMPose train...")
    run_command(train_cmd, cwd=repo_root, dry_run=args.dry_run)

    if args.no_eval:
        print("Skipping evaluation because --no-eval was provided.")
        return

    if args.dry_run:
        print("Dry run complete. No evaluation performed.")
        return

    split_names = [split.strip() for split in args.eval_splits.split(",") if split.strip()]
    finetune_runs = sorted((repo_root / "output" / "RTMPose").glob(f"{args.prefix}_*"))
    if not finetune_runs:
        raise FileNotFoundError("No RTMPose finetune run directories were found under output/RTMPose.")
    latest_run = finetune_runs[-1]
    history_file = latest_run / "history.json"
    if not history_file.exists():
        raise FileNotFoundError(f"History file not found: {history_file}")

    best_epoch = None
    best_val_loss = float("inf")
    with history_file.open(encoding="utf-8") as fp:
        history = json.load(fp)
    for entry in history:
        val_loss = entry.get("val", {}).get("loss")
        if val_loss is None:
            continue
        try:
            v = float(val_loss)
        except (TypeError, ValueError):
            continue
        if np.isnan(v):
            continue
        if v < best_val_loss:
            best_val_loss = v
            best_epoch = entry["epoch"]

    if best_epoch is None:
        print("WARNING: No valid val loss found in history. Falling back to checkpoint_best.pt")
        best_ckpt = latest_run / "checkpoint_best.pt"
    else:
        best_ckpt = latest_run / f"checkpoint_epoch_{best_epoch:03d}.pt"
        print(f"Best val loss: epoch {best_epoch} with val_loss = {best_val_loss:.6f}")

    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint not found: {best_ckpt}")

    for split in split_names:
        eval_cmd = [
            sys.executable,
            str(rtmpose_script),
            "--model-config",
            str(model_config),
            "--config-overwrite",
            str(config_overwrite),
            "--labels-root",
            str(labeled_data_root),
            "--labeled-frames-root",
            str(extracted_frames_root),
            "--workers",
            str(args.workers),
            "eval",
            "--checkpoint",
            str(best_ckpt),
            "--split",
            split,
        ]
        print(f"\nEvaluating checkpoint on '{split}' split...")
        run_command(eval_cmd, cwd=repo_root, dry_run=args.dry_run)

    print(f"\nFinished evaluation using checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
