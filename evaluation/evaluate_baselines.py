import json
import os
import glob
import numpy as np
import csv
from sklearn.metrics import precision_recall_fscore_support
from PIL import Image

def iou(bb1, bb2):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    x1, y1, x2, y2 = bb1
    x1g, y1g, x2g, y2g = bb2

    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (x2g - x1g) * (y2g - y1g)
    union_area = bb1_area + bb2_area - inter_area

    return inter_area / union_area if union_area else 0

def process_dataset(json1_path, dir1, dir2, baseline, split, output_csv):
    with open(json1_path, 'r') as f:
        json1 = json.load(f)

    y_true, y_pred = [], []
    all_preds, all_gts = [], []
    results_list = []

    results = []

    for q_id in json1.keys():
        if not (split == "all" or q_id.startswith(split)):
            continue

        results_path = os.path.join(dir1, q_id, baseline, "result.json")
        if not os.path.exists(results_path):
            results_path = os.path.join(dir1, q_id, baseline, "results.json")
            if not os.path.exists(results_path):
                continue

        with open(results_path, 'r') as f:
            results_data = json.load(f)

        if baseline == "lisa":
            attributions = results_data.get("segmentations")[0].get("bounding_boxes")
        elif baseline == "sa2va":
            attributions = results_data.get("bounding_boxes")
            attributions = [[bb["x_min"], bb["y_min"], bb["x_max"], bb["y_max"]] for bb in attributions]
        elif baseline == "kosmos":
            image_path = os.path.join(dir2, json1[q_id]["key"], "original.png")
            
            # Open the image and get its dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            entities = results_data.get("entities")

            attributions = []
            for entity in entities:
                bbox_list = entity[2]
                for bbox in bbox_list:
                    # Multiply normalized coordinates by image dimensions
                    x_min = bbox[0] * img_width
                    y_min = bbox[1] * img_height
                    x_max = bbox[2] * img_width
                    y_max = bbox[3] * img_height
                    attributions.append([x_min, y_min, x_max, y_max])
        elif baseline == "gpt4o_zero":
            image_path = os.path.join(dir2, json1[q_id]["key"], "original.png")
            
            # Open the image and get its dimensions
            with Image.open(image_path) as img:
                img_width, img_height = img.size
            
            entities = results_data.get("attributed_nodes")
            attributions = []
            for bbox in entities:
                    x_min = bbox[0] * img_width
                    y_min = bbox[1] * img_height
                    x_max = bbox[2] * img_width
                    y_max = bbox[3] * img_height
                    attributions.append([x_min, y_min, x_max, y_max])

        annotation_path_glob = glob.glob(os.path.join(dir2, json1[q_id]["key"], "annotations.json"))
        if not annotation_path_glob:
            continue

        annotation_path = annotation_path_glob[0]
        with open(annotation_path, 'r') as f:
            annotations_data = json.load(f)

        annotation_bbs = {key: bbs for key, bbs in annotations_data.items()}
        predictions = set()

        for bb in attributions:
            best_match = None
            best_iou = 0
            for key, ann_bb in annotation_bbs.items():
                iou_score = iou(bb, ann_bb)
                if iou_score > 0.5 and iou_score > best_iou:
                    best_iou = iou_score
                    best_match = key
            if best_match:
                predictions.add(best_match)

        ground_truth = set(json1[q_id].get("attribution"))
        y_true.extend([1 if key in ground_truth else 0 for key in predictions])
        y_pred.extend([1] * len(predictions))

        all_preds.append(set(predictions))
        all_gts.append(set(ground_truth))

        precision, recall, f1, _ = precision_recall_fscore_support(
            [1 if lbl in ground_truth else 0 for lbl in predictions],
            [1] * len(predictions),
            average='binary',
            zero_division=0
        )
        
        results.append([q_id, precision, recall, f1])
    
    # Save results to a CSV file
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["subdir", "precision", "recall", "f1"])
        writer.writerows(results)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    

    return {
        "micro": {"precision": precision_micro, "recall": recall_micro, "f1": f1_micro},
        "macro": {"precision": precision_macro, "recall": recall_macro, "f1": f1_macro}
    }

baselines =[ "gpt4o_zero", "lisa", "kosmos", "sa2va"]
splits = ["all"]


for baseline in baselines:
    for split in splits:
        output_csv = f"evaluation_result_{baseline}.csv"

        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Baseline", "Split", "Precision", "Recall", "F1"])

        results = process_dataset(
            "./data/dataset.json",
            "./output/chartpathagent_run",
            "./data/images",
            baseline, split, output_csv
        )
