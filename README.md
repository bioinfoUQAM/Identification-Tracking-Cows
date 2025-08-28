# Cow Tracking and Social Interaction Analysis

This project provides a complete pipeline for detecting, tracking, identifying, and analyzing cows in videos. It is designed to support behavioral research and Social Network Analysis (SNA) in animal studies, particularly livestock.

<img width="1614" height="872" alt="Captura de Tela 2025-08-28 às 3 55 14 PM" src="https://github.com/user-attachments/assets/1206df52-6066-41cf-8fd1-0bbca49842b8" />
<img width="1433" height="799" alt="Captura de Tela 2025-08-28 às 3 56 19 PM" src="https://github.com/user-attachments/assets/21f8e162-f34c-4acd-a509-923f897cd4d4" />





## Features

* **YOLOv8**: Fast and accurate object detection for locating cows in video frames.
* **Deep SORT**: Multi-object tracking to maintain consistent identities over time.
* **EfficientNet**: Classification model to identify individual cows.
* **Track Smoothing & Interpolation**: Reduces jitter and handles short occlusions.
* **Social Graph Generation**: Builds a proximity-based interaction graph in `.gexf` format.
* **Visual Output**: Generates annotated videos with bounding boxes, labels, confidence scores, and trails.

---

## Requirements

* Python 3.8+
* PyTorch
* OpenCV
* torchvision
* PIL
* Ultralytics (YOLOv8)
* `deep_sort_realtime`
* NetworkX
* SciPy

Install requirements:

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python cow_tracking_sna.py \
  --input-video /path/to/video_or_folder \
  --output-folder /path/to/output \
  --yolo-model yolov8n.pt \
  --num-classes 29 \
  --efficientnet-weights /path/to/model.pth \
  --class-names-dir /path/to/train_folder \
  --graph-output social_graph.gexf
```

### Arguments

| Argument                 | Description                                      |
| ------------------------ | ------------------------------------------------ |
| `--input-video`          | Path to a video file or folder containing videos |
| `--output-folder`        | Directory to store annotated videos              |
| `--yolo-model`           | Path to trained YOLOv8 weights file              |
| `--num-classes`          | Number of individual cows (classes)              |
| `--efficientnet-weights` | Path to EfficientNet `.pth` file                 |
| `--class-names-dir`      | Folder where each subfolder is a cow's name      |
| `--graph-output`         | Path to save social graph in GEXF format         |

---

## Output

* **Annotated Video**: `video_IDENTIFIED.mp4` with tracking, identification, and trails.
* **Social Graph**: `social_graph.gexf` (can be opened with Gephi or Python tools).

---

## Applications

* Social behavior tracking
* Identity monitoring and verification
* Group dynamics and social network analysis in livestock

---

## Example

To process a video `herd.mp4` with 29 cows:

```bash
python cow_tracking_sna.py \
  --input-video ./videos/herd.mp4 \
  --output-folder ./results \
  --yolo-model yolov8n.pt \
  --num-classes 29 \
  --efficientnet-weights ./models/efficientnet_cows.pth \
  --class-names-dir ./dataset/train \
  --graph-output herd_social_graph.gexf
```

---

## Visualization

To visualize the social graph:

* Open the `.gexf` file with **Gephi**
* Use edge weights to analyze interaction frequency
* Apply layout algorithms (ForceAtlas2, etc.) to understand network structure

---
# additional information:
All those paths you can find on CEDAR cluster in compute canada:

  --input-video /home/vonzin/scratch/SNA_2025_v2/Videos__180 
  --output-folder /home/vonzin/scratch/SNA_2025_v2/Videos__180_beste.mp4 \
  --yolo-model /home/vonzin/scratch/SNA_2025_v2/best_yolov8_detection.pt \
  --num-classes 29 \
  --efficientnet-weights /home/vonzin/scratch/SNA_2025_v2/lightning_logs/version_64763588__29ids_99cc_180_360cameras/checkpoints/best_model.ckpt \
  --class-names-dir /home/vonzin/scratch/SNA_2025_v2/Dataset_Augmented_29_class_v1_split_300_robusto_CERTO/train



# TRAINING treinamento_detalhado.py


This code trains and evaluates an **image classification model** for cow identification using **PyTorch Lightning** and **EfficientNet-B0** from `torchvision`.

---

## 🚀 Features

- **Model**
  - EfficientNet-B0 backbone with a custom classification head.
  - Pretrained weights loaded from a local `.pth` file.
  - Optimizer: Adam + ReduceLROnPlateau scheduler.
  - Metrics tracked with `MulticlassAccuracy`.

- **Data Handling**
  - Supports `train/`, `val/`, and optional `test/` splits.
  - Data augmentation: resize, random flip, rotation, normalization.
  - Validation/test: resize + normalization only.

- **Training**
  - Mixed precision (fp16) on GPU.
  - Early stopping and model checkpointing by validation accuracy.
  - Metrics logging via a custom callback.

- **Evaluation & Reporting**
  - Training/validation curves (loss & accuracy).
  - Classification report (precision, recall, f1-score).
  - Confusion matrix visualization.
  - Detailed predictions (true label, predicted label, confidence).
  - **Final PDF report** with metrics and plots.

- **Outputs**
  - All results saved in `analise_model_/`
    - `train_val_metrics.csv`
    - `classification_report.csv`
    - `test_predictions.csv`
    - `loss_curve.png`, `accuracy_curve.png`, `confusion_matrix.png`
    - `final_report.pdf`
  - Best model checkpoint saved automatically.

---

## 📂 Project Structure

Dataset_Augmented7_split_TREINAR/
│── train/
│── val/
│── test/ # optional

---

## ⚡ Quick Start

1. **Prepare dataset** with `train/`, `val/`, and optionally `test/` folders.

2. **Run training**:
   ```bash
   python train.py \
       --data-dir Dataset_Augmented7_split_TREINAR \
       --batch-size 64 \
       --lr 1e-3 \
       --epochs 50


# Running: evaluate_accuracy.py


### What this script does

This tool evaluates **cow identification in videos** by comparing **model predictions** against **ground-truth annotations** using a combination of **IoU (Intersection-over-Union)** and **name matching**.

It performs the following:

- **Input cleanup**  
  - Normalizes cow names (case-insensitive, replaces underscores with spaces).  
  - Fixes ground-truth coordinate order if mislabeled as `(x1, x2, y1, y2)` instead of `(x1, y1, x2, y2)`.  

- **Frame-level matching**  
  - For each predicted box, finds the best ground-truth match using IoU.  
  - Prioritizes **same-name matches** when IoU ≥ threshold (default 0.5).  
  - Marks predictions as **correct** only if both box overlap and name match succeed.  

- **Metrics reported**  
  - ✅ Overall accuracy (correct / total)  
  - 📊 Per-cow accuracy (class-wise)  
  - 📈 Precision, Recall, and F1-score (weighted)  
  - Optional detailed CSV with per-frame results (frame, true name, predicted name, IoU, correctness).  

---

### Expected CSV Format

**Ground Truth (`gt.csv`)**  
```csv
frame,x1,y1,x2,y2,true
1,100,150,200,300,Ava
1,220,160,330,310,Bella
2,120,155,210,305,Ava

# pred.csv

frame,x1,y1,x2,y2,predicted
1,98,152,202,298,Ava
1,225,158,328,312,Bella
2,119,150,212,308,Ava

# usage

python evaluate_with_iou.py \
  --pred predictions.csv \
  --gt ground_truth.csv \
  --iou-thresh 0.5 \
  --save results_detailed.csv
--pred → CSV with model predictions

--gt → CSV with ground truth boxes

--iou-thresh → minimum IoU to count a detection as matched (default 0.5)

--save → optional path to save detailed per-frame evaluation

# output

✅ Overall Accuracy (IoU ≥ 0.5): 0.9234
📈 Precision: 0.9200
📈 Recall:   0.9150
📈 F1-Score: 0.9175

📊 Accuracy per cow:
Ava      0.95
Bella    0.90


# Update Compute Canada:

We are reaching out to inform you that the Cedar compute cluster will be retired on September 12, 2025.

Data Access

Files stored on Cedar are already available on Fir because the two clusters share the same file systems; no action is required regarding your stored files.

Starting September 12, please submit your jobs to Fir or another cluster on our new national infrastructure.
