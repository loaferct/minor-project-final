from fastapi import APIRouter
from paddleocr import PaddleOCR
import os
import json

router = APIRouter()

# Initialize PaddleOCR once to improve performance
ocr = PaddleOCR(use_angle_cls=True, lang="en")

@router.post("/ocr")
async def perform_ocr():
    directory = "/home/acer/minor project final/classification_results/BarGraph"
    files = os.listdir(directory)
    image_file = next((file for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.bmp'))), None)

    if image_file:
        img_path = os.path.join(directory, image_file)
    else:
        return{"Message":"No image file found."}
    
    result = ocr.ocr(img_path, cls=True)
    
    detections = result[0]
    response_data = []
    
    for detection in detections:
        box, (text, score) = detection
        response_data.append({
            "box": box,
            "text": text,
            "confidence": round(score, 4)
        })
    
    json_filename = "/home/acer/minor project final/bar_ocr_results/ocr_results.json"
    with open(json_filename, "w", encoding="utf-8") as jsonfile:
        json.dump(response_data, jsonfile, ensure_ascii=False, indent=4)
    
    return {"message": "OCR results saved to JSON file", "file": json_filename}