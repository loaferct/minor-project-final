import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import math
from typing import List, Tuple, Dict, Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse

class PieChartError(Exception):
    """Custom exception for pie chart processing errors"""
    pass

def is_numerical(text: str) -> bool:
    """Check if text is a numerical value (with or without percentage)"""
    text = text.strip().replace('%', '').replace(',', '.').strip()
    try:
        float(text)
        return True
    except ValueError:
        return False

def get_angle(point: Tuple[float, float], center: Tuple[int, int]) -> float:
    """Calculate the angle between a point and the center in degrees"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return (np.degrees(np.arctan2(dy, dx)) + 360) % 360

def kmeans_numpy(X, n_clusters, max_iter=100, n_init=5):
    """K-means clustering implementation using NumPy"""
    best_inertia = float('inf')
    best_centers = None
    best_labels = None

    for _ in range(n_init):
        if len(X) <= n_clusters:
            centers = X.copy().astype(np.float64)
            labels = np.arange(len(X))
            return type('KMeansResult', (), {'cluster_centers_': centers, 'labels_': labels})()

        idx = np.random.choice(len(X), n_clusters, replace=False)
        centers = X[idx].astype(np.float64)

        for _ in range(max_iter):
            distances = np.sqrt(((X[:, np.newaxis] - centers) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            new_centers = np.empty_like(centers)
            for k in range(n_clusters):
                if np.any(labels == k):
                    new_centers[k] = X[labels == k].mean(axis=0)
                else:
                    new_centers[k] = centers[k]
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        inertia = np.sum((X - centers[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers
            best_labels = labels

    return type('KMeansResult', (), {
        'cluster_centers_': best_centers,
        'labels_': best_labels
    })()

def extract_segment_info(
    image: np.ndarray,
    mask: np.ndarray,
    pie_center: Tuple[int, int],
    expected_segments: int
) -> List[Dict]:
    """Extract segment information including centroids from the pie chart"""
    y_coords, x_coords = np.nonzero(mask)
    max_pixels = 10000
    if len(y_coords) > max_pixels:
        sample_indices = np.random.choice(len(y_coords), max_pixels, replace=False)
        y_coords = y_coords[sample_indices]
        x_coords = x_coords[sample_indices]
    pixels = image[y_coords, x_coords]

    if len(pixels) < expected_segments:
        expected_segments = max(1, len(pixels) // 10)

    kmeans = kmeans_numpy(pixels, n_clusters=expected_segments)
    angles = np.degrees(np.arctan2(y_coords - pie_center[1], x_coords - pie_center[0]))
    angles = (angles + 360) % 360
    pixel_labels = kmeans.labels_

    segments = []
    for i in range(expected_segments):
        segment_mask = pixel_labels == i
        segment_x = x_coords[segment_mask]
        segment_y = y_coords[segment_mask]
        if len(segment_x) == 0:
            continue
        centroid_x = np.mean(segment_x)
        centroid_y = np.mean(segment_y)
        segment_angles = angles[segment_mask]
        start_angle = np.min(segment_angles)
        end_angle = np.max(segment_angles)
        # Handle wrap-around segments
        if end_angle - start_angle > 330:
            sorted_angles = np.sort(segment_angles)
            gaps = sorted_angles[1:] - sorted_angles[:-1]
            if len(gaps) > 0:
                max_gap_idx = np.argmax(gaps)
                start_angle = sorted_angles[max_gap_idx + 1]
                end_angle = sorted_angles[max_gap_idx]
        segments.append({
            'start_angle': float(start_angle),
            'end_angle': float(end_angle),
            'color': kmeans.cluster_centers_[i].astype(int),
            'pixel_count': int(np.sum(segment_mask)),
            'centroid': (float(centroid_x), float(centroid_y))
        })

    segments.sort(key=lambda x: x['start_angle'])
    return segments

def match_labels_and_get_title(
    text_regions: List[Dict],
    pie_center: Tuple[int, int]
) -> Tuple[List[str], List[float], str]:
    """Match labels with percentages and determine the title when percentages are present"""
    numericals = []
    non_numericals = []
    for region in text_regions:
        if is_numerical(region['text']):
            numericals.append(region)
        else:
            non_numericals.append(region)

    pairs = []
    used_non_numericals = []
    for num in numericals:
        candidate_non_numericals = [nn for nn in non_numericals if nn not in used_non_numericals]
        if not candidate_non_numericals:
            raise PieChartError(f"Unpaired numerical value: {num['text']}")
        closest = min(
            candidate_non_numericals,
            key=lambda nn: math.hypot(
                num['center'][0] - nn['center'][0],
                num['center'][1] - nn['center'][1]
            )
        )
        pairs.append((closest, num))
        used_non_numericals.append(closest)

    labels = [pair[0]['text'] for pair in pairs]
    percentages = [float(pair[1]['text'].strip('%')) for pair in pairs]
    title_candidates = [nn for nn in non_numericals if nn not in used_non_numericals]
    title = " ".join([tc['text'] for tc in title_candidates]) if title_candidates else "Pie Chart"
    return labels, percentages, title

def process_pie_chart(
    image_path: str,
    paddle_results: List[Dict],
    save_path: Optional[str] = None
) -> Dict:
    """Process the pie chart image and extract data"""
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise PieChartError("Image load failed")

        max_dim = 1000
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # Detect pie chart contour
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            sobel = cv2.magnitude(sobelx, sobely)
            sobel = np.uint8(sobel)
            _, binary = cv2.threshold(sobel, 50, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise PieChartError("No pie chart detected in image")

        pie_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [pie_contour], -1, 255, -1)
        moments = cv2.moments(mask)
        if moments['m00'] == 0:
            raise PieChartError("Invalid pie chart contour moments")
        pie_center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))

        # Process OCR results
        text_regions = []
        for result in paddle_results:
            if result.get('confidence', 1) <= 0.5:
                continue
            box = result['box']
            x_coords = [pt[0] for pt in box]
            y_coords = [pt[1] for pt in box]
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            text_regions.append({
                'text': result['text'],
                'center': ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min)
            })

        has_percentage = any(is_numerical(r['text']) and ('%' in r['text']) for r in text_regions)

        if has_percentage:
            labels, percentages, title = match_labels_and_get_title(text_regions, pie_center)
            expected_segments = len(percentages)
        else:
            # Handle case without percentages
            if text_regions:
                top_text = min(text_regions, key=lambda r: r['center'][1])
                title = top_text['text']
                non_title_regions = [r for r in text_regions if r is not top_text]
            else:
                title = "Pie Chart"
                non_title_regions = []
            expected_segments = len(non_title_regions) if non_title_regions else 3

        if expected_segments < 2:
            expected_segments = 2

        # Extract segments with centroids
        segments = extract_segment_info(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            mask,
            pie_center,
            expected_segments
        )
        if not segments:
            segments = [{
                'start_angle': 0,
                'end_angle': 360,
                'color': np.array([200, 200, 200]),
                'pixel_count': 100,
                'centroid': pie_center
            }]

        if has_percentage:
            # Use OCR-provided percentages and labels
            pass
        else:
            # Compute percentages from pixel counts
            total_pixels = sum(seg['pixel_count'] for seg in segments)
            percentages = [seg['pixel_count'] / total_pixels * 100 for seg in segments]

            # Compute pie radius
            distances = [math.hypot(pt[0][0] - pie_center[0], pt[0][1] - pie_center[1]) for pt in pie_contour]
            radius = np.mean(distances)

            # Add mid-angle to segments
            for seg in segments:
                seg['mid_angle'] = (seg['start_angle'] + seg['end_angle']) / 2.0

            # Compute angles for text regions
            for text_region in non_title_regions:
                text_region['angle'] = get_angle(text_region['center'], pie_center)

            # Match segments to text regions using distance and angle
            pairs = []
            for seg_idx, seg in enumerate(segments):
                for text_idx, text_region in enumerate(non_title_regions):
                    distance = math.hypot(seg['centroid'][0] - text_region['center'][0],
                                        seg['centroid'][1] - text_region['center'][1])
                    angular_diff = min(abs(seg['mid_angle'] - text_region['angle']),
                                     360 - abs(seg['mid_angle'] - text_region['angle']))
                    score = (distance / radius) + (angular_diff / 180)
                    pairs.append((score, seg_idx, text_idx))

            # Sort pairs by score (lower is better)
            pairs.sort(key=lambda x: x[0])

            # Assign labels to segments
            assigned_labels = [None] * len(segments)
            used_text_indices = set()
            for score, seg_idx, text_idx in pairs:
                if assigned_labels[seg_idx] is None and text_idx not in used_text_indices:
                    assigned_labels[seg_idx] = non_title_regions[text_idx]['text']
                    used_text_indices.add(text_idx)

            # Assign default labels to unmatched segments
            for i in range(len(segments)):
                if assigned_labels[i] is None:
                    assigned_labels[i] = f"Segment {i+1}"

            labels = assigned_labels

        # Compile chart data
        chart_data = {
            "title": title,
            "pie_center": list(pie_center),
            "slices": []
        }
        for i, seg in enumerate(segments):
            label = labels[i] if i < len(labels) else f"Segment {i+1}"
            perc = percentages[i] if i < len(percentages) else 0.0
            chart_data["slices"].append({
                "label": label,
                "percentage": float(perc),
                "color": seg['color'].tolist(),
                "start_angle": float(seg['start_angle']),
                "end_angle": float(seg['end_angle']),
                "pixel_count": int(seg['pixel_count'])
            })

        # Save results
        json_path = "/home/acer/minor project final/pie_final_op/pie_construct.json"
        with open(json_path, 'w') as f:
            json.dump(chart_data, f, indent=4)

        # Generate and save reconstructed pie chart
        plt.figure(figsize=(10, 10))
        normalized_colors = [seg['color'] / 255.0 for seg in segments]
        plt.pie(
            percentages,
            labels=labels,
            colors=normalized_colors,
            autopct='%1.1f%%',
            startangle=90
        )
        plt.title(title)
        plt.savefig("/home/acer/reconstruction.png")
        plt.close()

        return chart_data

    except Exception as e:
        print(f"Error processing pie chart: {str(e)}")
        return None

router = APIRouter()

@router.get("/pie_reconstruct")
def pie_reconstruct():
    try:
        directory = "/home/acer/minor project final/classification_results/PieChart"
        files = os.listdir(directory)
        image_file = next((file for file in files if file.endswith(('.png', '.jpg', '.jpeg', '.bmp'))), None)

        if image_file:
            image_path = os.path.join(directory, image_file)
        else:
            return {"Message": "No image file found."}
        
        with open("/home/acer/minor project final/Pie_ocr_results/ocr_results.json", "r") as f:  # Replace with your OCR results path
            paddle_results = json.load(f)
        
        result = process_pie_chart(image_path, paddle_results, save_path="output_chart")
        
        if result:
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content={"Message": "Error processing pie chart."}, status_code=400)

    except Exception as e:
        return JSONResponse(content={"Message": str(e)}, status_code=500)
    
@router.put("/edit_pie")
def edit_pie(request_data: dict):
    try:
        # Define the file path where the pie chart data is stored
        pie_file_path = "/home/acer/minor project final/pie_final_op/pie_construct.json"

        # Check if the JSON file exists
        if not os.path.exists(pie_file_path):
            raise HTTPException(status_code=404, detail="Pie chart JSON file not found.")

        # Load the existing content from the JSON file
        with open(pie_file_path, "r") as f:
            current_data = json.load(f)

        # Update the content based on the incoming request data
        current_data["title"] = request_data.get("title", current_data.get("title"))
        current_data["pie_center"] = request_data.get("pie_center", current_data.get("pie_center"))
        
        # Check if the "slices" field is provided and update the slices
        if "slices" in request_data:
            current_data["slices"] = request_data["slices"]

        # Save the updated content back into the JSON file
        with open(pie_file_path, "w") as f:
            json.dump(current_data, f, indent=4)

        # Return the updated data as the response
        return JSONResponse(content={"message": "Pie chart content updated successfully.", "data": current_data}, status_code=200)

    except Exception as e:
        # Handle errors and return an appropriate response
        raise HTTPException(status_code=500, detail=f"Error updating pie chart: {str(e)}")
