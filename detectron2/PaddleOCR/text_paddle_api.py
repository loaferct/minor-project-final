from fastapi import APIRouter
from paddleocr import PaddleOCR
import json
import cv2
import os
from fastapi.responses import JSONResponse

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

# API endpoint to perform OCR on multiple images
@router.get("/perform-ocr/")
async def perform_ocr():
    input_dir = '/home/acer/minor project final/classification_results/Text'
    output_dir = '/home/acer/minor project final/text_ocr_results'

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    results = {}

    # Process each image
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        json_output = process_image(image_path)
        
        # Save each output as a JSON file
        output_file = os.path.join(output_dir, f'{os.path.splitext(image_file)[0]}.json')
        with open(output_file, 'w') as f:
            json.dump(json_output, f)

        # Store the result for response
        results[image_file] = json_output

    # Return a response containing the OCR results for all images
    return JSONResponse(content=results)
