import os
import sys
import json
import random
import shutil
import zipfile
import cv2
import numpy as np
import yaml

def rle_to_binary_mask(rle_counts, h, w):
    mask = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for count in rle_counts:
        if val == 1:
            mask[idx : idx + count] = 1
        idx += count
        val = 1 - val
    return mask.reshape((h, w), order='F')

def mask_to_yolo_polygons(binary_mask, min_area=25):
    h, w = binary_mask.shape
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.squeeze(1)
        if len(pts) < 3:
            continue
        norm = []
        for x, y in pts:
            nx = max(0.0, min(1.0, float(x) / w))
            ny = max(0.0, min(1.0, float(y) / h))
            norm.extend([f"{nx:.6f}", f"{ny:.6f}"])
        polys.append(" ".join(norm))
    return polys

def load_roboflow_yolo_dir(base_dir, prefix):
    """Loads all image and polygon label files from a Roboflow YOLO dataset (train, valid, test)."""
    samples = []
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(base_dir, split, "images")
        lbl_dir = os.path.join(base_dir, split, "labels")
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue
        
        for fname in os.listdir(img_dir):
            if not (fname.lower().endswith(".jpg") or fname.lower().endswith(".png") or fname.lower().endswith(".jpeg")):
                continue
            
            img_path = os.path.join(img_dir, fname)
            base_name, _ = os.path.splitext(fname)
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")
            
            poly_lines = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 7: # class_id + at least 3 points (x, y)
                            # Ensure single class 0
                            poly_lines.append(f"0 {' '.join(parts[1:])}")
            
            if poly_lines: # Only include valid annotated samples
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                save_name = f"{prefix}_{fname}"
                samples.append((save_name, img_bytes, poly_lines))
    return samples

def main():
    root_dir = r"D:\cvat"
    dataset_dir = os.path.join(root_dir, "dataset")
    
    # Clean target folders
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)

    samples = [] # List of tuples: (save_img_name, img_bytes, list_of_poly_lines)

    # 1. Load CVAT COCO Zips
    zip_files = [
        (234, os.path.join(root_dir, "job_234_dataset_coco.zip")),
        (235, os.path.join(root_dir, "job_235_dataset_2026_11_03_18_12_32_coco 1.0.zip")),
        (236, os.path.join(root_dir, "job_236_dataset_2026_11_05_16_06_45_coco 1.0.zip")),
    ]

    for job_id, zip_path in zip_files:
        if not os.path.exists(zip_path):
            print(f"Skipping {zip_path}, file not found")
            continue
        
        print(f"Processing job {job_id} from {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            with z.open("annotations/instances_default.json") as f:
                coco = json.load(f)
            
            categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
            redness_cat_ids = [cid for cid, name in categories.items() if "redness" in name.lower()]
            print(f"  Found redness category IDs: {redness_cat_ids}")

            ann_by_image = {}
            for ann in coco.get("annotations", []):
                if ann.get("category_id") in redness_cat_ids:
                    ann_by_image.setdefault(ann["image_id"], []).append(ann)
            
            for img_info in coco.get("images", []):
                img_id = img_info["id"]
                orig_file_name = img_info["file_name"]
                h = img_info["height"]
                w = img_info["width"]
                
                anns = ann_by_image.get(img_id, [])
                if not anns:
                    continue
                
                zip_img_entry = None
                for candidate in [orig_file_name, "images/default/" + orig_file_name, "images/" + orig_file_name]:
                    if candidate in z.namelist():
                        zip_img_entry = candidate
                        break
                if not zip_img_entry:
                    bname = os.path.basename(orig_file_name)
                    for n in z.namelist():
                        if n.endswith(bname):
                            zip_img_entry = n
                            break
                
                if not zip_img_entry:
                    continue
                
                img_bytes = z.read(zip_img_entry)
                
                poly_lines = []
                for ann in anns:
                    seg = ann.get("segmentation")
                    if not seg:
                        continue
                    
                    if isinstance(seg, dict) and "counts" in seg:
                        rle_h, rle_w = seg.get("size", [h, w])
                        bmask = rle_to_binary_mask(seg["counts"], rle_h, rle_w)
                        polys = mask_to_yolo_polygons(bmask)
                        for p in polys:
                            poly_lines.append(f"0 {p}")
                    elif isinstance(seg, list):
                        for p in seg:
                            norm = []
                            for i in range(0, len(p), 2):
                                nx = max(0.0, min(1.0, float(p[i]) / w))
                                ny = max(0.0, min(1.0, float(p[i+1]) / h))
                                norm.extend([f"{nx:.6f}", f"{ny:.6f}"])
                            poly_lines.append(f"0 {' '.join(norm)}")

                if poly_lines:
                    save_name = f"cvat_job{job_id}_{os.path.basename(orig_file_name)}"
                    samples.append((save_name, img_bytes, poly_lines))

    print(f"CVAT samples loaded: {len(samples)}")

    # 2. Load Roboflow Rosacea Dataset
    rf_rosacea_dir = os.path.join(root_dir, "rosacea-1")
    if os.path.exists(rf_rosacea_dir):
        rf_samples1 = load_roboflow_yolo_dir(rf_rosacea_dir, "rf_rosacea")
        print(f"Roboflow Rosacea samples loaded: {len(rf_samples1)}")
        samples.extend(rf_samples1)

    # 3. Load Roboflow Redness Dataset
    rf_redness_dir = os.path.join(root_dir, "redness-1")
    if os.path.exists(rf_redness_dir):
        rf_samples2 = load_roboflow_yolo_dir(rf_redness_dir, "rf_redness")
        print(f"Roboflow Redness samples loaded: {len(rf_samples2)}")
        samples.extend(rf_samples2)

    # 4. Add Negative Background Samples (Empty Labels) to Suppress False Positives on Normal Skin, Ears, Beards
    cache_dir = os.path.join(root_dir, "job234_cache")
    if os.path.exists(cache_dir):
        try:
            from cvat_sdk import Client
            from cvat_sdk.core.client import PasswordCredentials
            with Client("https://cvat.xlai.ru", check_server_version=False) as client:
                username = os.getenv("CVAT_USERNAME", "")
                password = os.getenv("CVAT_PASSWORD", "")
                client.login(PasswordCredentials(username, password))
                meta = client.jobs.retrieve(234).get_meta()
                deleted_frames = sorted(list(set(meta.deleted_frames or [])))
        except Exception:
            deleted_frames = []

        bg_count = 0
        for f_id in deleted_frames:
            fpath = os.path.join(cache_dir, f"frame_{f_id:04d}.jpg")
            if os.path.exists(fpath) and os.path.getsize(fpath) > 10000:
                with open(fpath, "rb") as f:
                    img_bytes = f.read()
                samples.append((f"bg_neg_{f_id:04d}.jpg", img_bytes, []))
                bg_count += 1
                if bg_count >= 50:
                    break
        print(f"Negative background samples added (empty labels): {bg_count}")

    print(f"\nTotal combined dataset samples: {len(samples)}")
    random.seed(42)
    random.shuffle(samples)

    # Train / Val split (85% / 15%)
    split_idx = int(len(samples) * 0.85)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"Train split: {len(train_samples)} images")
    print(f"Val split:   {len(val_samples)} images")

    for split_name, split_data in [("train", train_samples), ("val", val_samples)]:
        for save_name, img_bytes, poly_lines in split_data:
            base_name, _ = os.path.splitext(save_name)
            img_out = os.path.join(dataset_dir, "images", split_name, save_name)
            lbl_out = os.path.join(dataset_dir, "labels", split_name, base_name + ".txt")
            
            with open(img_out, "wb") as f:
                f.write(img_bytes)
            
            with open(lbl_out, "w", encoding="utf-8") as f:
                if poly_lines:
                    f.write("\n".join(poly_lines) + "\n")
                else:
                    f.write("") # Negative background sample: empty file

    # Create data.yaml
    data_yaml = {
        "path": dataset_dir.replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["redness_rash"]
    }

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\nDataset successfully built and saved at {dataset_dir}!")
    print(f"data.yaml written to {yaml_path}")

if __name__ == "__main__":
    main()
