import os
import cv2
import json
import numpy as np
from detectron2.data import MetadataCatalog
from scipy.cluster.vq import kmeans, vq
from fastapi import APIRouter
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import torch

cfg = get_cfg()
config_yaml_path = "/home/acer/Downloads/dataset/bar_model/Detectron2_Models/config.yaml"
cfg.merge_from_file(config_yaml_path)
cfg.MODEL.WEIGHTS = "/home/acer/Downloads/dataset/bar_model/Detectron2_Models/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
predictor = DefaultPredictor(cfg)

def get_dominant_color(image, k=1):
    """Extract dominant color using SciPy K-Means clustering."""
    if image.size == 0:
        return [0, 0, 0]

    pixels = image.reshape(-1, 3).astype(float)
    centroids, _ = kmeans(pixels, k)
    labels, _ = vq(pixels, centroids)
    dominant_color = centroids[np.argmax(np.bincount(labels))]
    return dominant_color.astype(int).tolist()

def process_image(image_path, predictor, output_dir):
    """Process single image and save detection results."""
    im = cv2.imread(image_path)
    if im is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    if len(instances) == 0:
        print(f"No objects detected in {image_path}")
        return

    detections = []
    pred_boxes = instances.pred_boxes.tensor.numpy().astype(int)
    scores = instances.scores.numpy()
    pred_classes = instances.pred_classes.numpy()

    for box, score, class_idx in zip(pred_boxes, scores, pred_classes):
        x1, y1, x2, y2 = box
        roi = im[y1:y2, x1:x2]

        detections.append({
            "label": "bars",
            "confidence": float(score),
            "bbox": box.tolist(),
            "dominant_color": get_dominant_color(roi)
        })

    save_results(image_path, detections, output_dir)

def save_results(image_path, detections, output_dir):
    """Save detection results in JSON format matching required structure."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    json_output = {image_path: detections}

    json_path = os.path.join(output_dir, f"{base_name}_detections.json")
    with open(json_path, "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"Results saved to {json_path}")

router = APIRouter()

@router.get("/Bar-bb")
def get_bb():
    image_path = "/home/acer/Desktop/Bar_364.png"
    output_directory = "/home/acer/minor project final/bar_bound_results"
    process_image(image_path, predictor, output_directory)