"""
Fine-tune a YOLO pothole / road-damage detector and export to ONNX for RoadWatch.
Run this on Google Colab (free T4 GPU) or a local machine with CUDA.

Setup:
    pip install ultralytics

Usage:
    python train_pothole_model.py --data /path/to/data.yaml --model yolo11s.pt --epochs 60

After training finishes, copy the exported `best.onnx` into `campus-roadwatch/models/`
or load it directly inside the RoadWatch app via "Load .onnx Model".
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO pothole detector and export to ONNX.")
    parser.add_argument("--data", required=True, help="Path to YOLO dataset data.yaml")
    parser.add_argument("--model", default="yolo11s.pt", help="Base checkpoint (yolo11n.pt, yolo11s.pt, yolov8s.pt, yolov8m.pt)")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (keep 640 for RoadWatch)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (-1 for auto)")
    parser.add_argument("--name", default="roadwatch_pothole", help="Run folder name")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        patience=15,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=5.0,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best checkpoint: {best_pt}")

    best_model = YOLO(str(best_pt))
    onnx_path = best_model.export(format="onnx", imgsz=args.imgsz, opset=12, simplify=True)
    print(f"Exported ONNX model ready for RoadWatch: {onnx_path}")


if __name__ == "__main__":
    main()
