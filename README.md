# Cow Tracking and Social Interaction Analysis

This project provides a complete pipeline for detecting, tracking, identifying, and analyzing cows in videos. It is designed to support behavioral research and Social Network Analysis (SNA) in animal studies, particularly livestock.

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

