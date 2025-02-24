from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import csv
import os
import urllib.request
import json
from fastapi import APIRouter

def get_font():   # font download gara 
    """Download a font file if it doesn't exist and return the path."""
    font_path = "/home/ssp84/Desktop/pie chart/DejaVuSans.ttf"
#     if not os.path.exists(font_path):
#         font_url = "https://github.com/PaddlePaddle/PaddleOCR/raw/release/2.6/doc/fonts/simfang.ttf"
#         try:
#             urllib.request.urlretrieve(font_url, font_path)
#         except Exception as e:
#             # Fallback to a different font if the first one fails
#             fallback_url = "https://github.com/opensourcedesign/fonts/raw/master/gnu-freefont_freesans/FreeSans.ttf"
#             urllib.request.urlretrieve(fallback_url, font_path)
    return font_path

def merge_nearby_texts(detections, vertical_threshold=20, horizontal_overlap_threshold=0.5):
    """
    Merge nearby OCR detections for non-percentage text.

    Two detections will be merged if:
      - They do NOT contain "%" in their text.
      - The vertical gap between the previous box’s bottom and the current box’s top is <= vertical_threshold.
      - Their horizontal boxes overlap significantly (overlap ratio >= horizontal_overlap_threshold).

    The merged detection will combine the texts (separated by a space) and the bounding boxes will be the union.
    """
    # Sort detections by the top coordinate (minimum y of the box)
    sorted_dets = sorted(detections, key=lambda d: min(pt[1] for pt in d["box"]))
    merged = []
    for det in sorted_dets:
        # Do not merge if text is a percentage value (or contains "%")
        if "%" in det["text"]:
            merged.append(det)
        else:
            # If there is an existing merged detection that is also non-percentage,
            # try to merge this detection with it.
            if merged and ("%" not in merged[-1]["text"]):
                last_det = merged[-1]
                # Compute bounding box for last detection
                last_box = last_det["box"]
                last_x1 = min(pt[0] for pt in last_box)
                last_y1 = min(pt[1] for pt in last_box)
                last_x2 = max(pt[0] for pt in last_box)
                last_y2 = max(pt[1] for pt in last_box)

                # Current detection box
                curr_box = det["box"]
                curr_x1 = min(pt[0] for pt in curr_box)
                curr_y1 = min(pt[1] for pt in curr_box)
                curr_x2 = max(pt[0] for pt in curr_box)
                curr_y2 = max(pt[1] for pt in curr_box)

                # Determine vertical gap between last box's bottom and current box's top
                vertical_gap = curr_y1 - last_y2

                # Compute horizontal overlap
                horizontal_overlap = max(0, min(last_x2, curr_x2) - max(last_x1, curr_x1))
                last_width = last_x2 - last_x1
                curr_width = curr_x2 - curr_x1
                min_width = min(last_width, curr_width)
                overlap_ratio = horizontal_overlap / min_width if min_width > 0 else 0

                # If the boxes are close vertically and overlap horizontally, merge them
                if vertical_gap <= vertical_threshold and overlap_ratio >= horizontal_overlap_threshold:
                    merged_text = last_det["text"] + " " + det["text"]
                    # Create union box (simple min/max over coordinates)
                    union_box = [
                        [min(last_x1, curr_x1), min(last_y1, curr_y1)],
                        [max(last_x2, curr_x2), min(last_y1, curr_y1)],
                        [max(last_x2, curr_x2), max(last_y2, curr_y2)],
                        [min(last_x1, curr_x1), max(last_y2, curr_y2)]
                    ]
                    merged[-1] = {
                        "box": union_box,
                        "text": merged_text,
                        "confidence": (last_det["confidence"] + det["confidence"]) / 2
                    }
                else:
                    merged.append(det)
            else:
                merged.append(det)
    return merged

router = APIRouter()
@router.get("/pie_ocr")
def pie_ocr():
    ocr = PaddleOCR(use_angle_cls=True, lang="en")

    # Path to image
    img_path = '/home/acer/Desktop/pie.png'

    # Perform OCR
    result = ocr.ocr(img_path, cls=True)

    # Extract results from the first (and only) page
    detections = result[0]

    # Extract boxes, texts, and scores
    boxes = [detection[0] for detection in detections]
    txts = [detection[1][0] for detection in detections]
    scores = [detection[1][1] for detection in detections]

    # Build a list of detection dictionaries
    raw_detections = []
    for box, text, score in zip(boxes, txts, scores):
        raw_detections.append({
            "box": box,
            "text": text,
            "confidence": float(f"{score:.4f}")
        })

    # Merge detections that belong together (only for non-percentage texts)
    merged_detections = merge_nearby_texts(raw_detections)

    # Get font path for drawing results
    font_path = get_font()


    # Here we save the merged results to a JSON file.
    with open('/home/acer/minor project final/Pie_ocr_results/ocr_results.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(merged_detections, jsonfile, ensure_ascii=False, indent=4)
        
    return {"Message" : "Successfully OCRed"}

