import os
import sys
import json
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from ultralytics import YOLO

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# MediaPipe landmark indices for the outer boundary of the lips
LIP_CONTOUR = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

def mask_to_cvat_mask_points(binary_mask):
    """
    Encodes a 2D binary numpy mask (H, W) into CVAT native RLE mask format:
    [rle_0, rle_1, ..., xtl, ytl, xbr, ybr]
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    crop = binary_mask[ymin : ymax + 1, xmin : xmax + 1]
    flat = crop.flatten()

    rle = []
    curr_val = 0
    curr_count = 0
    for val in flat:
        val_int = 1 if val else 0
        if val_int == curr_val:
            curr_count += 1
        else:
            rle.append(curr_count)
            curr_val = val_int
            curr_count = 1
    rle.append(curr_count)
    return [float(x) for x in rle] + [float(xmin), float(ymin), float(xmax), float(ymax)]

def parse_args():
    parser = argparse.ArgumentParser(description="Run facial erythema segmentation inference")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to trained YOLO11-seg model weights")
    parser.add_argument("--source", type=str, required=True, help="Path to input image file or directory")
    parser.add_argument("--output-json", type=str, default="predictions.json", help="Path to save CVAT format predictions")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image resolution")
    parser.add_argument("--device", type=str, default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--exclude-lips", action="store_true", default=True, help="Use MediaPipe face landmarks to exclude lip contours")
    parser.add_argument("--skip-frames", type=str, default="", help="Comma-separated frame IDs to skip")
    parser.add_argument("--save-vis", type=str, default=None, help="Directory to save visualized predictions")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found at: {args.weights}")

    if not os.path.exists(args.source):
        raise FileNotFoundError(f"Source path not found at: {args.source}")

    skip_frames = set()
    if args.skip_frames:
        skip_frames = set(int(f.strip()) for f in args.skip_frames.split(",") if f.strip().isdigit())

    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)

    # Initialize MediaPipe FaceLandmarker for lip exclusion
    face_detector = None
    task_model = "face_landmarker.task"
    if args.exclude_lips and MP_AVAILABLE and os.path.exists(task_model):
        print("Initializing MediaPipe FaceLandmarker for anatomical lip protection...")
        base_options = python.BaseOptions(model_asset_path=task_model)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        face_detector = vision.FaceLandmarker.create_from_options(options)
    elif args.exclude_lips and not os.path.exists(task_model):
        print("Notice: 'face_landmarker.task' not found. Lip exclusion will be skipped.")

    print(f"Loading YOLO segmentation model from {args.weights}...")
    model = YOLO(args.weights)

    # Collect images
    if os.path.isdir(args.source):
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = sorted([f for f in os.listdir(args.source) if f.lower().endswith(valid_exts)])
        file_paths = [(f, os.path.join(args.source, f)) for f in files]
    else:
        file_paths = [(os.path.basename(args.source), args.source)]

    print(f"Processing {len(file_paths)} images with conf={args.conf} on device={args.device}...")

    results_data = []

    for fname, fpath in tqdm(file_paths, desc="Inference"):
        # Extract frame ID if file name is formatted like 'frame_0001.jpg'
        frame_id = None
        digits = "".join([c for c in os.path.splitext(fname)[0] if c.isdigit()])
        if digits:
            frame_id = int(digits)
            if frame_id in skip_frames:
                continue

        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]

        res = model.predict(img_bgr, conf=args.conf, imgsz=args.imgsz, device=args.device, verbose=False)[0]
        if res.masks is None or len(res.masks.xy) == 0:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in res.masks.xy:
            if len(poly) >= 3:
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        if mask.sum() < 80:
            continue

        # Exclude lip contour
        if face_detector is not None:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det = face_detector.detect(mp_image)
            if det.face_landmarks:
                lms = det.face_landmarks[0]
                lip_pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in LIP_CONTOUR], dtype=np.int32)
                lip_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(lip_mask, [lip_pts], 1)
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                lip_mask = cv2.dilate(lip_mask, k)
                mask[lip_mask > 0] = 0

        # Morphological smoothing
        k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_smooth)

        if mask.sum() < 80:
            continue

        # Optional visualization
        if args.save_vis:
            vis = img_bgr.copy()
            red_overlay = np.zeros_like(img_bgr)
            red_overlay[:, :] = (0, 0, 255)
            vis[mask > 0] = cv2.addWeighted(img_bgr[mask > 0], 0.5, red_overlay[mask > 0], 0.5, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)
            cv2.imwrite(os.path.join(args.save_vis, f"pred_{fname}"), vis)

        points = mask_to_cvat_mask_points(mask)
        if points is not None:
            results_data.append({
                "frame": frame_id if frame_id is not None else fname,
                "points": points
            })

    print(f"\nInference complete! Masks detected for {len(results_data)} images.")
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results_data, f)
        print(f"Saved predictions to: {args.output_json}")

if __name__ == "__main__":
    main()
