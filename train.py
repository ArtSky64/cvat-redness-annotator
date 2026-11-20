import os
import argparse
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11 instance segmentation for facial erythema")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml dataset config")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt", help="Pretrained weights to fine-tune from")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (e.g. 16 for 16GB GPU, 8 for 6GB GPU)")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device (e.g. '0' or 'cpu')")
    parser.add_argument("--project", type=str, default="runs", help="Project runs directory")
    parser.add_argument("--name", type=str, default="erythema_seg_run", help="Experiment name")
    parser.add_argument("--cache", type=str, default="ram", help="Cache mode ('ram', 'disk', or False)")
    return parser.parse_args()

def main():
    args = parse_args()

    cuda_available = torch.cuda.is_available()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset config not found at: {args.data}")

    print(f"\nInitializing YOLO model from {args.model}...")
    model = YOLO(args.model)

    print(f"Starting training on {args.data} for {args.epochs} epochs...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device if cuda_available else "cpu",
        workers=4,
        cache=args.cache,
        project=args.project,
        name=args.name,
        save=True,
        exist_ok=True,
        pretrained=True,
        # Targeted augmentations for facial skin redness:
        mosaic=0.5,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,
        degrees=10.0,
        fliplr=0.5,
        save_period=-1,
        verbose=True
    )

    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Best weights saved to: {best_path}")
    print("="*50)

if __name__ == "__main__":
    main()
