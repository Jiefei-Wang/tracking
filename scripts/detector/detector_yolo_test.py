"""
Evaluate the best YOLO model from the last training run on the test set.
Finds the most recent output/yolo/<timestamp> folder automatically.
"""

import os
import sys

import yaml
from ultralytics import YOLO


def find_latest_run(yolo_output_dir="output/yolo"):
    """Find the most recent training run directory."""
    if not os.path.isdir(yolo_output_dir):
        raise FileNotFoundError(f"No YOLO output directory: {yolo_output_dir}")
    runs = sorted([
        d for d in os.listdir(yolo_output_dir)
        if os.path.isdir(os.path.join(yolo_output_dir, d))
    ])
    if not runs:
        raise FileNotFoundError(f"No training runs found in {yolo_output_dir}")
    return os.path.join(yolo_output_dir, runs[-1])


def main():
    run_dir = find_latest_run()
    best_model_path = os.path.join(run_dir, "train", "weights", "best.pt")
    data_yaml_path = os.path.join(run_dir, "dataset", "data.yaml")

    if not os.path.isfile(best_model_path):
        print(f"Error: best.pt not found at {best_model_path}")
        sys.exit(1)
    if not os.path.isfile(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"YOLO Test Evaluation")
    print(f"Run directory: {run_dir}")
    print(f"Model: {best_model_path}")
    print(f"Data:  {data_yaml_path}")
    print("=" * 60)

    model = YOLO(best_model_path)
    test_results = model.val(
        data=os.path.abspath(data_yaml_path),
        split="test",
        imgsz=640,
        batch=16,
        device=0,
        project=os.path.abspath(run_dir),
        name="test_eval",
        exist_ok=True,
    )

    print("\n" + "=" * 60)
    print(f"Test mAP50:    {test_results.box.map50:.4f}")
    print(f"Test mAP50-95: {test_results.box.map:.4f}")
    print(f"Test Precision: {test_results.box.mp:.4f}")
    print(f"Test Recall:    {test_results.box.mr:.4f}")
    print(f"Results saved to: {run_dir}/test_eval")
    print("=" * 60)


if __name__ == "__main__":
    main()
