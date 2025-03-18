from fastapi import APIRouter, HTTPException, Body
from paddleocr import PaddleOCR
import json
import cv2
import os
from fastapi.responses import JSONResponse
import shutil

router = APIRouter()

# Function to process the image using PaddleOCR
def process_image(image_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    result = ocr.ocr(image_path, cls=True)

    output = []
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    for line in result:
        for word_info in line:
            text = word_info[1][0]
            bbox = word_info[0]
            style = {
                'coordinates': [int(bbox[0][0]), int(bbox[0][1]),
                                int(bbox[2][0]), int(bbox[2][1])],
                'font_size': int(bbox[3][1] - bbox[0][1]),
                'bold': False,
                'italic': False,
                'color': [0, 0, 0],  # Default black
                'font_family': 'Arial'  # Default font
            }
            output.append({'text': text, 'style': style})

    return {
        'metadata': {'width': w, 'height': h},
        'content': output
    }

# API endpoint to perform OCR on multiple images.
@router.get("/perform-ocr/")
async def perform_ocr():
    # Remove any previous OCR results.
    results_dir = "/home/acer/minor project final/text_ocr_results"
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    input_dir = '/home/acer/minor project final/classification_results/Text'
    os.makedirs(results_dir, exist_ok=True)

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    results = {}

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        json_output = process_image(image_path)
        
        # Save each output as a JSON file.
        output_file = os.path.join(results_dir, f'{os.path.splitext(image_file)[0]}.json')
        with open(output_file, 'w') as f:
            json.dump(json_output, f)
        
        results[image_file] = json_output

    return JSONResponse(content=results)

# API endpoint to return OCR results from saved JSON files.
@router.get("/get-ocr-results/")
async def get_ocr_results():
    results_dir = '/home/acer/minor project final/text_ocr_results'
    if not os.path.exists(results_dir):
        raise HTTPException(status_code=404, detail="Directory not found")
    
    json_files = [f for f in os.listdir(results_dir) if f.lower().endswith('.json')]
    results = {}
    for json_file in json_files:
        json_path = os.path.join(results_dir, json_file)
        with open(json_path, 'r') as f:
            content = json.load(f)
        # Using the full file path as key.
        results[json_path] = content

    return JSONResponse(content=results)

# API endpoint to update a specific OCR JSON file.
@router.put("/update-ocr-result/")
async def update_ocr_result(payload: dict = Body(...)):
    """
    Expects a JSON payload with:
      - filePath: the full path to the JSON file to update.
      - blockIndex: the index (0-based) of the content block to update.
      - textContent: a string containing the updated text.
    
    This endpoint updates only the "text" field of the specified content block.
    The rest of the JSON (including coordinates and style) remains unchanged.
    """
    file_path = payload.get("filePath")
    block_index = payload.get("blockIndex")
    new_text = payload.get("textContent")
    if not file_path or block_index is None or new_text is None:
        raise HTTPException(status_code=400, detail="Missing filePath, blockIndex, or textContent")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Load the existing JSON content.
    with open(file_path, "r") as f:
        data = json.load(f)
    
    if "content" not in data or not isinstance(data["content"], list) or len(data["content"]) <= block_index:
        raise HTTPException(status_code=400, detail="Invalid blockIndex: no content block exists at that index")
    
    # Update only the text field of the specified block.
    data["content"][block_index]["text"] = new_text
    
    # Write the updated data back to file without altering any other fields.
    with open(file_path, "w") as f:
        json.dump(data, f)
    
    return JSONResponse(content={"message": "File updated", "data": data})