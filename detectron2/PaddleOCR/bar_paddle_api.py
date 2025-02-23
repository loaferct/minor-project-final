from fastapi import APIRouter
from paddleocr import PaddleOCR
import os
import json

router = APIRouter()

# Initialize PaddleOCR once to improve performance
ocr = PaddleOCR(use_angle_cls=True, lang="en")

@router.post("/ocr")
async def perform_ocr():
    img_path = "/home/acer/Desktop/Bar_364.png"
    
    # Perform OCR
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
    
    # Save structured results to JSON
    json_filename = "/home/acer/minor project final/bar_ocr_results/ocr_results.json"
    with open(json_filename, "w", encoding="utf-8") as jsonfile:
        json.dump(response_data, jsonfile, ensure_ascii=False, indent=4)
    
    return {"message": "OCR results saved to JSON file", "file": json_filename}