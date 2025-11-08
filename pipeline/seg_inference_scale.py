import os
import json
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor
from tqdm import tqdm

# Load model and image processor
device = "cuda"
checkpoint = "MananSuri27/finetune-instance-segmentation-flowchartseg-mask2former_20epochs_a6000"

model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint, device_map=device)
image_processor = Mask2FormerImageProcessor.from_pretrained(checkpoint)

def generate_label(index):
    """Generate alphabetic labels (A, B, ..., Z, AA, AB, ...)."""
    label = ""
    while index >= 0:
        label = chr(65 + (index % 26)) + label
        index = index // 26 - 1
    return label

def render_segmentation_labels(image, segmentation_map):
    """Render alphabetic labels (no bounding boxes) for each segment."""
    
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 2:
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        output_image = image.copy()

    unique_ids = np.unique(segmentation_map)
    unique_ids = unique_ids[unique_ids > 0]  # Ignore background (-1)

    segment_positions = []  # Store (y, x, id, bbox)

    for segment_id in unique_ids:
        mask = (segmentation_map == segment_id).astype(np.uint8)
        x, y, w_box, h_box = cv2.boundingRect(mask)
        segment_positions.append((y, x, segment_id, (x, y, w_box, h_box)))

    segment_positions.sort()

    occupied_positions = set()

    for i, (_, _, segment_id, (x, y, w_box, h_box)) in enumerate(segment_positions):
        label = generate_label(i)

        label_x, label_y = x + w_box + 5, y + h_box // 2

        if (label_x, label_y) in occupied_positions:
            label_x, label_y = x - 20, y + h_box // 2

        occupied_positions.add((label_x, label_y))

        cv2.putText(output_image, label, (label_x, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

    return output_image, segment_positions

def process_segmentation_map(segmentation):
    binary = (segmentation != -1).astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result = np.full_like(segmentation, -1)

    filtered_contours = []

    for contour in contours:
        if cv2.contourArea(contour) > 3000:
            filtered_contours.append(contour)
    
    for i, contour in enumerate(filtered_contours):
        cv2.drawContours(result, [contour], -1, i+1, -1)
    
    return result

def process_directory(input_dir):
    # Iterate through subdirectories in the given directory
    for subdir, _, _ in tqdm(os.walk(input_dir)):
        original_image_path = os.path.join(subdir, "original.png")
        
        # Check if the original image exists in the subdirectory
        if os.path.isfile(original_image_path):
            # Load the image
            image = Image.open(original_image_path).convert("RGB")

            # Run inference on the image
            inputs = image_processor(images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)

            # Post-process the outputs
            outputs = image_processor.post_process_instance_segmentation(outputs, target_sizes=[(image.height, image.width)])

            segmentation = outputs[0]["segmentation"].numpy()

            # Process the segmentation map
            segmentation = process_segmentation_map(segmentation)

            # Render segmentation labels (without bounding boxes)
            output_image, segment_positions = render_segmentation_labels(image, segmentation)

            # Save the segmentation image
            segmentation_image_path = os.path.join(subdir, "segmentation.png")
            output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(segmentation_image_path, output_image)

            # Prepare and save the bounding boxes as JSON
            bounding_boxes = {}
            for i, (_, _, segment_id, (x, y, w_box, h_box)) in enumerate(segment_positions):
                label = generate_label(i)
                bounding_boxes[label] = [x, y, x + w_box, y + h_box]

            segmentation_json_path = os.path.join(subdir, "segmentation.json")
            with open(segmentation_json_path, "w") as json_file:
                json.dump(bounding_boxes, json_file, indent=4)

# Example usage (uncomment to run):
# input_directory = "./data/images"  # Replace with the directory you want to process
# process_directory(input_directory)
