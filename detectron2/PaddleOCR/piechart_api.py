import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
from fastapi import APIRouter

class PieChartError(Exception):
    """Custom exception for pie chart processing errors"""
    pass

def get_angle(point: Tuple[float, float], center: Tuple[int, int]) -> float:
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    return (angle + 360) % 360

def extract_segment_info(image: np.ndarray, mask: np.ndarray, pie_center: Tuple[int, int], expected_segments: int) -> List[Dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    y_coords, x_coords = np.nonzero(mask)
    pixels = image[y_coords, x_coords]
    if expected_segments < 1:
        expected_segments = 1
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=expected_segments, n_init=10, max_iter=300)
    kmeans.fit(pixels)
    angles = np.degrees(np.arctan2(y_coords - pie_center[1], x_coords - pie_center[0]))
    angles = (angles + 360) % 360
    pixel_labels = kmeans.labels_
    segments = []
    for i in range(expected_segments):
        segment_mask = pixel_labels == i
        segment_angles = angles[segment_mask]
        if len(segment_angles) > 0:
            start_angle = np.min(segment_angles)
            end_angle = np.max(segment_angles)
            if end_angle - start_angle > 330:
                angles_sorted = np.sort(segment_angles)
                gaps = angles_sorted[1:] - angles_sorted[:-1]
                max_gap_idx = np.argmax(gaps)
                start_angle = angles_sorted[max_gap_idx + 1]
                end_angle = angles_sorted[max_gap_idx]
            segments.append({
                'start_angle': start_angle,
                'end_angle': end_angle,
                'color': kmeans.cluster_centers_[i].tolist(),
                'pixel_count': np.sum(segment_mask)
            })
    segments.sort(key=lambda x: x['start_angle'])
    return segments

def process_pie_chart(image_path: str, paddle_results: List[Dict]) -> Dict:
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise PieChartError("Failed to load image")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise PieChartError("No pie chart detected in image")
        pie_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [pie_contour], -1, 255, -1)
        moments = cv2.moments(mask)
        if moments['m00'] == 0:
            raise PieChartError("Could not determine pie chart center")
        pie_center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))
        text_regions = []
        for result in paddle_results:
            box = result['box']
            text = result['text']
            confidence = result['confidence']
            if confidence > 0.5:
                x_coords = [point[0] for point in box]
                y_coords = [point[1] for point in box]
                x = min(x_coords)
                y = min(y_coords)
                w = max(x_coords) - x
                h = max(y_coords) - y
                text_regions.append({
                    'text': text,
                    'center': (x + w/2, y + h/2),
                    'bbox': (x, y, w, h),
                    'confidence': confidence,
                    'is_percentage': '%' in text
                })
        title = "Title Not Detected"
        if text_regions:
            title_region = min(text_regions, key=lambda r: r['bbox'][1])
            title = title_region['text']
            text_regions.remove(title_region)
        percentage_count = sum(1 for r in text_regions if r['is_percentage'])
        expected_segments = max(percentage_count, len(text_regions), 1)
        segments = extract_segment_info(image_rgb, mask, pie_center, expected_segments)
        chart_data = {
            "title": title,
            "pie_center": list(pie_center),
            "slices": segments
        }
        return chart_data
    except Exception as e:
        return {"error": str(e)}

router = APIRouter()

@router.get("/reconstruct")
def pie():
    image_path = '/home/acer/minor project final/classification_results/PieChart/PieChart_score_1.00_1.png'
    with open('/home/acer/minor project final/Pie_ocr_results/ocr_results.json', 'r') as f:
        paddle_results = json.load(f)
    chart_data = process_pie_chart(image_path, paddle_results)
    return chart_data
