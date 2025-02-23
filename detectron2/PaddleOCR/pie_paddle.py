from fastapi import APIRouter
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR
import os
import urllib.request
import json
import glob

router = APIRouter()

img_dir = "/home/acer/minor project final/classification_results/PieChart/"

image_files = glob.glob(os.path.join(img_dir, "*.*"))
img_path = image_files[0] if image_files else None


def get_font():
    """Download a font file if it doesn't exist and return the path."""
    font_path = "DejaVuSans.ttf"

    if not os.path.exists(font_path):
        font_url = "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.6/doc/fonts/simfang.ttf"
        try:
            urllib.request.urlretrieve(font_url, font_path)
        except:
            fallback_url = "https://github.com/opensourcedesign/fonts/raw/master/gnu-freefont_freesans/FreeSans.ttf"
            urllib.request.urlretrieve(fallback_url, font_path)

    return font_path

@router.get("/perform_ocr")
async def perform_ocr():
    try:
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="en")

        # Perform OCR
        result = ocr.ocr(img_path, cls=True)

        # Extract results from the first page
        detections = result[0]

        # Extract boxes, texts, and scores
        json_data = []
        for detection in detections:
            box = detection[0]  # Bounding box coordinates
            text = detection[1][0]  # Detected text
            score = detection[1][1]  # Confidence score
            json_data.append({
                "box": box,
                "text": text,
                "confidence": float(f"{score:.4f}")  # Format confidence to 4 decimal places
            })

        # Save results to JSON
        with open('/home/acer/minor project final/Pie_ocr_results/ocr_results.json', 'w', encoding='utf-8') as jsonfile:
            json.dump(json_data, jsonfile, ensure_ascii=False, indent=4)

        return JSONResponse(content={"message": "OCR processing completed. Check ocr_results.json"}, status_code=200)

    except Exception as e:
        return JSONResponse(content={"message": f"Error occurred: {str(e)}"}, status_code=500)