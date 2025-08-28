#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import models, transforms
from torch import nn
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pathlib import Path
from collections import defaultdict
import random
import networkx as nx
from scipy.interpolate import interp1d

# -------------------------
# Arguments
# -------------------------
parser = argparse.ArgumentParser(description="Processes ONE video file using YOLO + DeepSort + EfficientNet.")
parser.add_argument('--input-video', type=str, required=True, help="Path to the input video file (e.g., .mp4, .avi, .mov).")
parser.add_argument('--output-folder', type=str, required=True, help="Output folder to save the annotated video.")
parser.add_argument('--yolo-model', type=str, required=True, help="Path or name of the YOLO model (e.g., yolov8n.pt).")
parser.add_argument('--num-classes', type=int, required=True, help="Number of classes in the EfficientNet classifier.")
parser.add_argument('--efficientnet-weights', type=str, required=True, help=".pth file containing EfficientNet weights.")
parser.add_argument('--class-names-dir', type=str, required=True, help="Directory where each subfolder is a class name.")
parser.add_argument('--max-errors', type=int, default=5, help="(Reserved) Max number of errors per class.")
parser.add_argument('--graph-output', type=str, default="social_graph.gexf", help="Output file for the social graph (GEXF).")
args = parser.parse_args()

# -------------------------
# Load cow class names (class -> name)
# -------------------------
def load_class_names(base_dir):
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"--class-names-dir is not a valid directory: {base_dir}")
    cow_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if len(cow_names) == 0:
        raise RuntimeError(f"No class found in {base_dir}. Each class should be a subdirectory.")
    return cow_names

class_names = load_class_names(args.class_names_dir)

# -------------------------
# Transformations
# -------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------
# Utilities for coloring and trails
# -------------------------
def get_color_for_id(track_id):
    random.seed(track_id)
    return tuple(int(random.randint(50, 255)) for _ in range(3))

track_history = defaultdict(list)
interpolation_buffer = defaultdict(lambda: defaultdict(list))  # track_id -> {'frames': [], 'positions': []}

# -------------------------
# Social interaction graph
# -------------------------
social_graph = nx.Graph()

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# -------------------------
# Interpolates missing positions (occlusions)
# -------------------------
def interpolate_missing(track_id):
    frames = interpolation_buffer[track_id]['frames']
    positions = interpolation_buffer[track_id]['positions']
    if len(frames) < 2:
        return []
    try:
        frames_np = np.array(frames)
        positions_np = np.array(positions)
        interp_x = interp1d(frames_np, positions_np[:, 0], kind='linear', fill_value="extrapolate")
        interp_y = interp1d(frames_np, positions_np[:, 1], kind='linear', fill_value="extrapolate")
        all_interp = []
        for i in range(frames[0], frames[-1] + 1):
            all_interp.append((i, (int(interp_x(i)), int(interp_y(i)))))
        return all_interp
    except Exception:
        return []

# -------------------------
# Video processing with tracking and visualization (SINGLE FILE)
# -------------------------
def process_video(video_path, output_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # YOLO (detection)
    yolo = YOLO(args.yolo_model)

    # DeepSort (tracking)
    tracker = DeepSort(max_age=15, max_iou_distance=0.6)

    # EfficientNet (identification)
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, args.num_classes)

    # Load weights
    ckpt = torch.load(args.efficientnet_weights, map_location=device)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if len(unexpected) > 0:
        print(f"[Warning] Unexpected keys in state_dict: {unexpected}")
    if len(missing) > 0:
        print(f"[Warning] Missing keys when loading weights: {missing}")
    model.eval().to(device)

    os.makedirs("errors/wrong", exist_ok=True)
    error_counter = defaultdict(int)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(cap.get(3))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0 or np.isnan(fps):
        fps = 30.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (int(width), int(height)))
    if not out.isOpened():
        raise RuntimeError(f"Failed to open video writer for: {output_path}")

    frame_id = 0
    cow_names_by_id = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        results = yolo(frame)
        detections = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                label = r.names[cls_id] if hasattr(r, "names") else str(cls_id)
                if label == 'cow':
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf.item())
                    if conf < 0.4:
                        continue
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'cow'))

        # Tracking
        tracks = tracker.update_tracks(detections, frame=frame)
        frame_centers = {}
        active_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            active_ids.add(track_id)

            try:
                l, t, r, b = track.to_ltrb()
                x1, y1, x2, y2 = map(int, [l, t, r, b])
            except Exception:
                tlwh = track.to_tlwh()
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h

            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(width, x2 + pad), min(height, y2 + pad)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Identification
            img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                topk = torch.topk(probs, k=min(3, args.num_classes))

            y_offset = 0
            for idx, (prob, class_idx) in enumerate(zip(topk.values, topk.indices)):
                p = float(prob.item())
                cidx = int(class_idx.item())
                if p < 0.7:
                    continue
                cow_name = class_names[cidx] if cidx < len(class_names) else f"Class_{cidx}"
                if idx == 0:
                    cow_names_by_id[track_id] = cow_name

                label_text = f"{cow_name} [ID {track_id}] ({p * 100:.1f}%)"
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                color = get_color_for_id(track_id)
                y_label_top = max(0, y1 - 10 - y_offset - text_size[1])
                cv2.rectangle(frame, (x1, y_label_top - 5), (x1 + text_size[0] + 10, y_label_top + text_size[1] + 5), color, -1)
                cv2.putText(frame, label_text, (x1 + 5, y_label_top + text_size[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += text_size[1] + 12

            color = get_color_for_id(track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            track_history[track_id].append(center)
            interpolation_buffer[track_id]['frames'].append(frame_id)
            interpolation_buffer[track_id]['positions'].append(center)
            frame_centers[track_id] = center

            for i in range(1, len(track_history[track_id])):
                cv2.line(frame, track_history[track_id][i - 1], track_history[track_id][i], color, 2)

        for tid in list(track_history.keys()):
            if tid not in active_ids:
                interpolated = interpolate_missing(tid)
                for fid, pos in interpolated:
                    if abs(fid - frame_id) <= 2:
                        if len(track_history[tid]) == 0 or track_history[tid][-1] != pos:
                            track_history[tid].append(pos)

        ids = list(frame_centers.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                id1, id2 = ids[i], ids[j]
                d = euclidean(frame_centers[id1], frame_centers[id2])
                if d < 150:
                    name1 = cow_names_by_id.get(id1, f"ID_{id1}")
                    name2 = cow_names_by_id.get(id2, f"ID_{id2}")
                    if social_graph.has_edge(name1, name2):
                        social_graph[name1][name2]['weight'] += 1
                    else:
                        social_graph.add_edge(name1, name2, weight=1)

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    print(f"Processed: {os.path.basename(video_path)}")
    return True

# -------------------------
# Main (FILE OR FOLDER)
# -------------------------
if __name__ == "__main__":
    input_path = args.input_video
    os.makedirs(args.output_folder, exist_ok=True)

    video_files = []
    if os.path.isdir(input_path):
        exts = (".mp4", ".avi", ".mov", ".mkv")
        video_files = [os.path.join(input_path, f) for f in os.listdir(input_path)
                       if f.lower().endswith(exts)]
        if len(video_files) == 0:
            raise RuntimeError(f"No videos found in folder: {input_path}")
    elif os.path.isfile(input_path):
        video_files = [input_path]
    else:
        raise FileNotFoundError(f"--input-video is not a valid file or directory: {input_path}")

    for video in video_files:
        base_name = f"{Path(video).stem}_IDENTIFIED.mp4"
        output_path = os.path.join(args.output_folder, base_name)

        try:
            ok = process_video(video, output_path)
            if ok:
                print(f"Finished: {video}")
        except Exception as e:
            print(f"Error processing {video}: {e}")

    nx.write_gexf(social_graph, args.graph_output)
    print(f"Social graph saved to: {args.graph_output}")
    print("All videos processed!")
