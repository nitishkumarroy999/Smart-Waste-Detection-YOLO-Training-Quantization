# ♻️ Smart Waste Detection: YOLO Training & Quantization

Real-time waste detection with YOLO, optimized for edge devices through post-training quantization.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLO](https://img.shields.io/badge/Model-YOLO-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **Note:** Items marked `<!-- TODO -->` must be checked or filled in against the actual code and results.

---

## 📖 Table of Contents
1. [Overview](#-overview)
2. [How It Works](#-how-it-works)
3. [Features](#-features)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Dataset](#-dataset)
7. [Training](#-training)
8. [Quantization](#-quantization)
9. [Inference](#-inference)
10. [Results](#-results)
11. [Deployment](#-deployment)
12. [Troubleshooting](#-troubleshooting)
13. [Roadmap](#-roadmap)
14. [Contributing](#-contributing)
15. [License](#-license)
16. [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

Manual waste sorting is slow, costly, and error-prone. This project trains a **YOLO object detection model** to locate and classify waste items (e.g., plastic, paper, glass, metal) in images, video, or a live camera feed.

Full-precision models are often too heavy for low-power hardware such as a Raspberry Pi, Jetson Nano, or a phone. So the project also applies **quantization**, which converts weights from 32-bit floats to lower precision (FP16 / INT8). This shrinks the model and speeds up inference, usually with a small accuracy trade-off.

**Goal:** train an accurate detector, compress it, and compare accuracy, size, and speed before and after.

---

## ⚙️ How It Works

The pipeline has five stages:

```
 ┌──────────┐   ┌────────────┐   ┌───────────┐   ┌──────────────┐   ┌────────────┐
 │ Dataset  │──▶│ Preprocess │──▶│ Training  │──▶│ Quantization │──▶│ Inference  │
 │ (images +│   │ & augment  │   │ (YOLO FP32)│  │ (FP16 / INT8)│   │ & benchmark│
 │  labels) │   │            │   │           │   │              │   │            │
 └──────────┘   └────────────┘   └───────────┘   └──────────────┘   └────────────┘
```

### 1. Data preparation
- Images are annotated with bounding boxes and a class label per waste item.
- Labels use the **YOLO format**: `class_id x_center y_center width height`, all normalized to 0–1.
- The data is split into `train / val / test` sets (e.g., 80/20 or 70/20/10). <!-- TODO: confirm split -->
- A `data.yaml` file tells YOLO where the images are and what the classes are.

### 2. Augmentation
During training, images are randomly flipped, scaled, color-shifted, and mosaic-combined. This helps the model handle different lighting, angles, and cluttered backgrounds. If some classes have few samples, class weighting or resampling can reduce bias.

### 3. Training (YOLO)
YOLO is a **single-stage detector**. In one forward pass it predicts:
- **Boxes:** where each object is.
- **Objectness / confidence:** how likely a box contains an object.
- **Class probabilities:** what the object is.

Training starts from **pretrained weights** (transfer learning) and fine-tunes them on the waste dataset. The loss combines box regression, classification, and objectness terms. The best checkpoint is chosen by validation mAP and saved as `best.pt`.

### 4. Quantization
Quantization reduces numeric precision to make the model smaller and faster:

| Method | Precision | Needs calibration data | Typical effect |
|---|---|---|---|
| Baseline | FP32 | No | Highest accuracy, largest size |
| Half precision | FP16 | No | ~2× smaller, fast on GPU |
| Post-training static | INT8 | Yes (small image set) | ~4× smaller, fastest on CPU/edge |
| Dynamic range | INT8 weights | No | Easy, moderate speedup |

<!-- TODO: keep only the methods your code actually uses (e.g., TFLite, ONNX Runtime, OpenVINO, TensorRT, PyTorch). -->

For INT8, a small **calibration set** of representative images is run through the model so it can estimate the value range of each layer and map floats to integers with minimal loss.

### 5. Inference & benchmarking
The original and quantized models run on the same test data, and the project compares:
- **Accuracy:** mAP@0.5, mAP@0.5:0.95, precision, recall
- **Model size** (MB)
- **Inference speed** (ms/image, FPS)
- **Memory / VRAM usage**

---

## ✨ Features
- Train a YOLO model on a custom waste dataset
- Export to quantized formats for edge deployment
- Detect on images, videos, and webcam streams
- Side-by-side benchmark of original vs. quantized models
- Reproducible configuration via YAML files

---

## 📁 Project Structure

<!-- TODO: replace with your real tree -->
```
Smart-Waste-Detection-YOLO-Training-Quantization/
├── data/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── data.yaml
├── notebooks/            # Training / quantization notebooks
├── models/               # Saved weights (best.pt, quantized models)
├── scripts/
│   ├── train.py
│   ├── quantize.py
│   ├── detect.py
│   └── benchmark.py
├── results/              # Metrics, plots, sample predictions
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
# 1. Clone
git clone https://github.com/nitishkumarroy999/Smart-Waste-Detection-YOLO-Training-Quantization.git
cd Smart-Waste-Detection-YOLO-Training-Quantization

# 2. Create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements** <!-- TODO: confirm -->
- Python 3.9+
- PyTorch + Ultralytics (or the YOLO implementation you used)
- OpenCV, NumPy, Matplotlib
- ONNX / ONNX Runtime / TensorFlow Lite, depending on the quantization route
- A CUDA GPU is recommended for training (Google Colab or Kaggle also works)

---

## 🗂️ Dataset

- **Source:** <!-- TODO: Roboflow / TACO / TrashNet / custom -->
- **Classes:** <!-- TODO: e.g., plastic, paper, glass, metal, cardboard, organic -->
- **Images:** <!-- TODO: count -->
- **Format:** YOLO (`.txt` labels)

Example `data.yaml`:
```yaml
path: ./data
train: images/train
val: images/val
test: images/test

nc: 6
names: [plastic, paper, glass, metal, cardboard, organic]
```

---

## 🏋️ Training

```bash
python scripts/train.py \
  --data data/data.yaml \
  --weights yolov8n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

<!-- TODO: adjust to your real command / notebook cell -->

| Parameter | Description |
|---|---|
| `--weights` | Pretrained starting checkpoint |
| `--epochs` | Number of training passes |
| `--imgsz` | Input image size |
| `--batch` | Batch size (lower it if you run out of memory) |

Outputs are saved to `runs/train/...`, including `best.pt`, `last.pt`, loss curves, a confusion matrix, and PR curves.

---

## 📦 Quantization

```bash
python scripts/quantize.py \
  --weights models/best.pt \
  --format int8 \
  --calib-data data/images/val
```

<!-- TODO: adjust to your real command / notebook cell -->

Steps performed:
1. Load the trained FP32 model.
2. Export to an intermediate format (e.g., ONNX).
3. Apply FP16 or INT8 quantization (INT8 uses the calibration images).
4. Save the quantized model to `models/`.
5. Validate it on the test set to measure the accuracy drop.

---

## 🎯 Inference

```bash
# Single image
python scripts/detect.py --weights models/best_int8.tflite --source sample.jpg

# Video
python scripts/detect.py --weights models/best_int8.tflite --source video.mp4

# Webcam
python scripts/detect.py --weights models/best_int8.tflite --source 0
```

Each detection is drawn as a labeled bounding box with a confidence score. Raise the confidence threshold (e.g., `--conf 0.5`) to reduce false positives.

---

## 📊 Results

<!-- TODO: fill with your real numbers -->

| Model | Precision | mAP@0.5 | Size (MB) | Latency (ms) | FPS |
|---|---|---|---|---|---|
| YOLO FP32 | – | – | – | – | – |
| YOLO FP16 | – | – | – | – | – |
| YOLO INT8 | – | – | – | – | – |

**Key takeaways** <!-- TODO -->
- Quantization cut model size by about **X%**.
- Accuracy dropped by about **Y** mAP points.
- Inference was **Z× faster** on the target device.

Add sample prediction images here:
```markdown
![Sample detection](results/sample_prediction.jpg)
```

---

## 🚀 Deployment

The quantized model can run on:
- **Raspberry Pi / ARM CPUs:** TFLite or ONNX Runtime
- **NVIDIA Jetson:** TensorRT
- **Mobile (Android / iOS):** TFLite / CoreML
- **Intel hardware:** OpenVINO

Typical use cases: smart dustbins, conveyor-belt sorting, recycling plants, and cleanup drones or robots.

---

## 🧯 Troubleshooting

| Problem | Likely fix |
|---|---|
| CUDA out of memory | Lower `--batch` or `--imgsz` |
| Poor accuracy on some classes | Add more data for those classes, or use class weights |
| Large accuracy drop after INT8 | Use more and more diverse calibration images, or try FP16 |
| Slow inference | Use a smaller model (e.g., `n` variant) or a hardware-specific runtime |

---

## 🗺️ Roadmap
- [ ] Add more waste categories
- [ ] Quantization-aware training (QAT)
- [ ] Web or mobile demo app
- [ ] Real-time dashboard for waste statistics

---

## 🤝 Contributing
Contributions are welcome:
1. Fork the repo
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit and push your changes
4. Open a Pull Request

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for details. <!-- TODO: confirm license -->

---

## 🙏 Acknowledgements
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- Dataset providers <!-- TODO -->
- Open-source community

---

**Author:** [Nitish Kumar Roy](https://github.com/nitishkumarroy999)
