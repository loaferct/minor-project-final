from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse
from paddleocr import PaddleOCR
import cv2
import json
import numpy as np
import os
import io
from typing import Dict, Any
import zipfile

# Initialize FastAPI app
router = APIRouter()

def process_image(image_path):
    # Initialize PaddleOCR
    ocr = PaddleOCR(lang='en', show_log=False)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not read image: {image_path}"}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform OCR
    result = ocr.ocr(gray)
    if not result:
        return {"error": f"No text detected in image: {image_path}"}

    # Extract text and coordinates
    extracted_data = []
    for line in result:
        for word in line:
            text = word[1][0]
            bbox = word[0]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            extracted_data.append({
                'text': text,
                'x': np.mean(x_coords),
                'y': np.mean(y_coords)
            })

    # Sort data into rows and columns
    extracted_data.sort(key=lambda x: (x['y'], x['x']))

    # Group into rows (using y-coordinate clustering)
    current_y = None
    table_data = []
    row = []
    y_threshold = 10  # Adjust based on your table's row height

    for item in extracted_data:
        if current_y is None or abs(item['y'] - current_y) <= y_threshold:
            row.append(item)
        else:
            row.sort(key=lambda x: x['x'])
            table_data.append(row)
            row = [item]
        current_y = item['y']

    if row:
        row.sort(key=lambda x: x['x'])
        table_data.append(row)

    # Convert to 2D array
    final_table = []
    for row in table_data:
        final_row = [cell['text'] for cell in row]
        final_table.append(final_row)

    return final_table

@router.get("/convert-table-to-json/")
async def download_table_json():
    # Define the input directory
    input_dir = "/home/acer/minor project final/classification_results/Table"
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        return {"error": f"Directory {input_dir} not found"}

    # Supported image extensions
    image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")

    # Create a ZIP file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Process each image and add its JSON to the ZIP immediately
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(image_extensions):
                image_path = os.path.join(input_dir, filename)
                try:
                    table_data = process_image(image_path)
                    if not isinstance(table_data, dict):  # Success case
                        image_name = os.path.splitext(filename)[0]
                        json_data = json.dumps(table_data, indent=4)
                        # Add JSON to ZIP immediately after processing
                        zipf.writestr(f"{image_name}.json", json_data)
                    else:
                        print(f"Skipping {filename}: {table_data['error']}")
                except Exception as e:
                    print(f"Failed to process {filename}: {str(e)}")

    # Reset buffer position to the beginning
    zip_buffer.seek(0)

    # Return the ZIP file as a streaming response
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=table_data.zip"}
    )

@router.get("/unzipfile/")
def unzip():
    with zipfile.ZipFile("/home/acer/Downloads/table_data.zip", 'r') as zip_ref:
        zip_ref.extractall("/home/acer/minor project final/table_results")
        return {"message": "successfully extracted"}

@router.get("/get-json-files")
async def get_json_files():
    directory = '/home/acer/minor project final/table_results'
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    json_data = {}
    for file_name in json_files:
        file_path = os.path.join(directory, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
            # Use the file name as the table name (without the '.json' extension)
            table_name = file_name.replace('.json', '')
            json_data[table_name] = data
    
    return JSONResponse(content=json_data)


@router.post("/save/table/{table_name}")
async def edit_table(table_name: str, table_content: dict):
    directory = '/home/acer/minor project final/table_results'
    file_path = os.path.join(directory, f"{table_name}.json")
    
    try:
        # Check if the directory exists, if not create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Ensure the content is a list and handle it properly
        if 'content' not in table_content or not isinstance(table_content['content'], list):
            raise HTTPException(status_code=400, detail="Invalid content format")

        # Store only the content as list in the file
        table_data = table_content['content']

        # Save the content as JSON in the file (no table name in JSON)
        with open(file_path, 'w') as file:
            json.dump(table_data, file, indent=4)
        
        return JSONResponse(content={"message": f"Table '{table_name}' saved successfully!"}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving the table: {str(e)}")
