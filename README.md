# 🔬 Facial Erythema & Redness Rash Instance Segmentation

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://docs.ultralytics.com/)
[![CVAT](https://img.shields.io/badge/Integration-CVAT%20SDK-black.svg)](https://www.cvat.ai/)
[![mAP50](https://img.shields.io/badge/Mask%20mAP50-94.2%25-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end deep learning pipeline for automated, high-precision instance segmentation of **facial erythema, rosacea, and skin inflammation** using **YOLO11-seg**, paired with an automated **CVAT (Computer Vision Annotation Tool)** integration.

---

## 🌟 Highlights

* **Pure Deep Learning Segmentation:** Replaces fragile heuristic color thresholding (e.g. $a^*$ channels, LAB colorimetry) with deep convolutional feature representations capable of generalizing across diverse skin tones (Fitzpatrick scale), lighting conditions, and profile camera angles.
* **State-of-the-Art Accuracy:** Reached **94.2% Mask mAP50** and **90.5% Recall** over a verified validation set of clinical dermatology images.
* **Anatomical Lip Protection:** Integrates Google MediaPipe Face Mesh landmarks to dynamically isolate and exclude vermilion border lip contours, eliminating false-positive rash detections on naturally red lips.
* **CVAT Integration:** Seamlessly exports native RLE masks to remote CVAT server instances via `cvat-sdk`, preserving designated ground-truth annotations without touching project workflow states.
* **Hardware Optimized:** Accelerated with FP16 Tensor Cores and RAM caching for rapid convergence on both consumer GPUs (e.g., GTX 1060 / RTX 30/40 series) and Google Colab (Tesla T4).

---

## 📊 Benchmark & Validation Metrics

Evaluated on a held-out clinical validation set ($N = 226$ facial images, $461$ ground-truth rash instances) after 80 epochs of transfer learning:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Mask mAP@50** | **94.2%** | Mean Average Precision at IoU threshold 0.50 |
| **Mask mAP@50-95** | **63.0%** | Stringent multi-threshold contour fidelity |
| **Mask Precision** | **89.6%** | Reliability of predicted erythema pixels |
| **Mask Recall** | **90.5%** | Total coverage of clinical erythema lesions |
| **Box mAP@50** | **94.0%** | Localization accuracy of lesion bounding regions |

---

## 🛠️ Repository Structure

```
facial-erythema-segmentation/
├── .env.example             # Template for CVAT credentials & configuration
├── .gitignore               # Strict exclusion of proprietary data & weights
├── requirements.txt         # Core dependencies
├── README.md                # Documentation & usage guide
├── infer.py                 # Neural inference with anatomical lip exclusion & CVAT RLE export
├── upload_to_cvat.py        # Automated CVAT annotation uploader via REST SDK
├── train.py                 # Flexible CLI training script (YOLO11s-seg)
├── configs/
│   └── data_example.yaml    # YOLO dataset format reference
└── notebooks/
    └── train_colab.ipynb    # Clean, cloud-ready training notebook for Google Colab (Tesla T4)
```

---

## 🚀 Quick Start

### 1. Installation

Clone this repository and create a Python virtual environment:

```bash
git clone https://github.com/your-username/facial-erythema-segmentation.git
cd facial-erythema-segmentation

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration

Copy `.env.example` to `.env` and fill in your remote CVAT credentials (this file is excluded from Git):

```bash
cp .env.example .env
```

Edit `.env`:
```env
CVAT_URL=https://your-cvat-server.com
CVAT_USERNAME=your_username
CVAT_PASSWORD=your_password
```

---

## 🔍 Running Inference

Run inference on an image or directory of images. This generates native CVAT RLE masks while protecting lip contours:

```bash
python infer.py \
    --weights weights/best.pt \
    --source path/to/images/ \
    --output-json predictions.json \
    --conf 0.25 \
    --imgsz 640 \
    --device 0 \
    --save-vis outputs/visualizations/
```

### CLI Arguments:
* `--weights`: Path to trained YOLO11-seg checkpoint (`.pt`).
* `--source`: Path to an image file or directory of frames.
* `--output-json`: Target JSON file containing CVAT RLE predictions.
* `--conf`: Confidence detection threshold (default: `0.25`).
* `--exclude-lips`: Enable MediaPipe anatomical lip contour suppression (default: `True`).
* `--save-vis`: (Optional) Folder to save visual inspection overlays with highlighted contours.

---

## 📤 Uploading to CVAT

Upload predicted masks directly into an existing CVAT Job, preserving any manual ground-truth frames:

```bash
python upload_to_cvat.py \
    --job-id 234 \
    --predictions predictions.json \
    --manual-frames "1,5,6,7"
```

The script will:
1. Connect securely to your CVAT instance using credentials from `.env` or flags.
2. Fetch existing job labels and resolve target `redness_rash`.
3. Preserve all manual frames specified in `--manual-frames`.
4. Upload all neural masks without modifying job stages or workflow statuses.

---

## 🏋️ Model Training

To train or fine-tune YOLO11s-seg on your custom dataset:

```bash
python train.py \
    --data configs/data.yaml \
    --model yolo11s-seg.pt \
    --epochs 80 \
    --batch 16 \
    --imgsz 640 \
    --cache ram \
    --device 0
```

### Google Colab Training:
A turnkey, self-contained Google Colab notebook is available at [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb). It supports one-click execution with automated RAM caching on Tesla T4 GPUs and direct Google Drive checkpointing.

---

## 🔒 Privacy & Dataset Notice

* **Proprietary Data:** Clinical and proprietary client datasets used to train the production weights are confidential and **not included** in this public repository.
* **Custom Data Format:** To train with your own data, organize your dataset following standard YOLO segmentation layout:
  ```
  dataset/
  ├── images/
  │   ├── train/
  │   └── val/
  └── labels/
      ├── train/
      └── val/
  ```
  Where labels are normalized polygon coordinates: `<class-id> <x1> <y1> <x2> <y2> ... <xn> <yn>`.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
