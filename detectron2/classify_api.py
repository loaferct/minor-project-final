from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import os
import json
import cv2
import torch
import zipfile
import shutil
import uuid
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import zipfile
import glob
import base64


router = APIRouter()

cfg = get_cfg()
config_yaml_path = "/home/acer/Downloads/dataset/lasttry-20250227T155349Z-001/lasttry/config.yaml"
cfg.merge_from_file(config_yaml_path)

cfg.MODEL.WEIGHTS = "/home/acer/Downloads/dataset/lasttry-20250227T155349Z-001/lasttry/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

def delete_files_in_directory():
    folder_path = "/home/acer/minor project final/classification_results"
    if os.path.exists(folder_path):
        # Iterate through the folder's contents
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # Check if it's a file or directory
            if os.path.isfile(item_path):
                os.remove(item_path)  # Remove the file
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

@router.post("/predict/")
async def predict(image: UploadFile = File(...)):
    if os.path.exists("/home/acer/Downloads/results.zip"):
        os.remove("/home/acer/Downloads/results.zip")
    # Create a unique session folder
    session_id = str(uuid.uuid4())
    session_dir = f"/home/acer/Desktop/result/{session_id}"
    os.makedirs(session_dir, exist_ok=True)

    # Save uploaded image
    temp_image_path = os.path.join(session_dir, "temp_image.png")
    with open(temp_image_path, "wb") as f:
        f.write(await image.read())

    original_image_path = os.path.join(session_dir, "original.png")
    shutil.copyfile(temp_image_path, original_image_path)
    # Load image
    new_im = cv2.imread(temp_image_path)
    if new_im is None:
        return {"error": "Invalid image file"}

    # Run prediction
    outputs = predictor(new_im)

    # Extract predictions
    if "instances" not in outputs or len(outputs["instances"]) == 0:
        return {"error": "No objects detected in the image"}

    pred_boxes = outputs["instances"].pred_boxes.tensor.tolist()
    pred_classes = outputs["instances"].pred_classes.tolist()
    scores = outputs["instances"].scores.tolist()

    # Class mapping
    class_mapping = {
        0: "Text",
        1: "Table",
        2: "BarGraph",
        3: "PieChart",
    }

    # Visualize results
    v = Visualizer(new_im[:, :, ::-1], metadata=MetadataCatalog.get("your_dataset_name"), scale=1.0)
    
    # Manually draw each prediction with class name and score
    for box, cls, score in zip(pred_boxes, pred_classes, scores):
        class_name = class_mapping.get(cls, f"Unknown_{cls}")  # Map class ID to name
        label = f"{class_name}: {score:.2f}"  # Combine class name and score

        # Draw the bounding box
        v.draw_box(box, edge_color="g")  # Green bounding box (you can change the color)

        # Draw the label above the box
        x, y = box[0], box[1]  # Top-left corner of the box
        v.draw_text(label, (x+40, y), color="g", font_size=10)  # Adjust position and styling as needed

    # Save the visualized image
    output_image = v.output.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV
    cv2.imwrite(os.path.join(session_dir, "prediction.png"), output_image)

    # Save JSON output
    predictions = []
    for box, cls, score in zip(pred_boxes, pred_classes, scores):
        predictions.append({
            "class": cls,
            "bbox": box,
            "score": score
        })

    json_path = os.path.join(session_dir, "output.json")
    with open(json_path, "w") as f:
        json.dump(predictions, f, indent=4)

    image = cv2.imread(temp_image_path)
    h, w = image.shape[:2]  # Get image dimensions for bounds checking

    for i, detection in enumerate(predictions):
        class_id = detection["class"]
        bbox = detection["bbox"]
        score = detection["score"]

        # Get class name
        class_name = class_mapping.get(class_id, f"Unknown_{class_id}")

        class_dir = os.path.join(session_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        # Ensure bounding box is within image dimensions
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Skip if the crop would be invalid
        if x1 >= x2 or y1 >= y2:
            print(f"Skipping invalid crop for {class_name} at index {i}: {bbox}")
            continue

        # Crop the object
        object_crop = image[y1:y2, x1:x2]

        # Save the cropped image
        filename = f"{class_name}_score_{score:.2f}_{i}.png"
        output_path = os.path.join(class_dir, filename)
        success = cv2.imwrite(output_path, object_crop)
        if not success:
            print(f"Failed to save crop for {class_name} at {output_path}")

    # Remove the temporary uploaded image
    os.remove(temp_image_path)

    # Zip the results
    zip_path = f"{session_dir}.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for root, _, files in os.walk(session_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, session_dir))

    # Cleanup extracted images (keep only the ZIP)
    shutil.rmtree(session_dir)

    return FileResponse(zip_path, filename="results.zip", media_type="application/zip")

@router.get("/unzipfile/")
def unzip():
    delete_files_in_directory()
    with zipfile.ZipFile("/home/acer/Downloads/results.zip", 'r') as zip_ref:
        zip_ref.extractall("/home/acer/minor project final/classification_results")
        return {"message": "successfully extracted"}
        
def image_to_base64(image_path: str) -> str:
    """Helper function to convert image to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

IMAGE_DIR = "/home/acer/minor project final/classification_results"

@router.get("/images/")
async def get_images():
    image_files = glob.glob(os.path.join(IMAGE_DIR, "*.png"))
    
    # Streaming response for the first image
    image1_path = image_files[0]
    image2_path = image_files[1]

    # Convert images to base64 format
    image1_base64 = image_to_base64(image1_path)
    image2_base64 = image_to_base64(image2_path)

    # Return the images as base64 encoded data in a dictionary
    return JSONResponse(content={
        "image1": image1_base64,
        "image2": image2_base64
    })