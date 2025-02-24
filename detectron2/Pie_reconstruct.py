import cv2
import numpy as np
import json
import math
from typing import List, Tuple, Dict, Optional
from fastapi import APIRouter, HTTPException

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
    """Calculate angle between point and center relative to horizontal axis"""
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return np.degrees(np.arctan2(dy, dx)) % 360

def kmeans_numpy(X, n_clusters, max_iter=300, n_init=10):
    """K-means clustering implementation using NumPy"""
    best_inertia = float('inf')
    best_centers = None
    best_labels = None

    for _ in range(n_init):
        idx = np.random.choice(len(X), n_clusters, replace=False)
        centers = X[idx].astype(np.float64)

        for _ in range(max_iter):
            distances = np.sqrt(((X[:, np.newaxis] - centers) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)
            new_centers = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] for k in range(n_clusters)])
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        inertia = np.sum((X - centers[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia, best_centers, best_labels = inertia, centers, labels

    return type('KMeansResult', (), {'cluster_centers_': best_centers, 'labels_': best_labels})()

def extract_segment_info(image: np.ndarray, mask: np.ndarray, pie_center: Tuple[int, int], expected_segments: int) -> List[Dict]:
    """Extract pie segment information using K-means clustering"""
    y_coords, x_coords = np.nonzero(mask)
    pixels = image[y_coords, x_coords]

    if len(pixels) < expected_segments:
        raise PieChartError(f"Not enough pixels ({len(pixels)}) for {expected_segments} segments")

    kmeans = kmeans_numpy(pixels, expected_segments)
    angles = np.degrees(np.arctan2(y_coords - pie_center[1], x_coords - pie_center[0])) % 360
    pixel_labels = kmeans.labels_

    segments = []
    for i in range(expected_segments):
        segment_mask = pixel_labels == i
        segment_angles = angles[segment_mask]
        if not len(segment_angles):
            continue

        start = np.min(segment_angles)
        end = np.max(segment_angles)
        if end - start > 330:  # Handle wrap-around
            sorted_angles = np.sort(segment_angles)
            gaps = sorted_angles[1:] - sorted_angles[:-1]
            split = np.argmax(gaps)
            start, end = sorted_angles[split+1], sorted_angles[split]

        segments.append({
            'start_angle': float(start),
            'end_angle': float(end),
            'color': kmeans.cluster_centers_[i].astype(int),
            'pixel_count': int(np.sum(segment_mask))
        })

    segments.sort(key=lambda x: x['start_angle'])
    return segments

def match_labels_and_get_title(text_regions: List[Dict], pie_center: Tuple[int, int]) -> Tuple[List[str], List[float], str]:
    """Pair numerical values with closest labels and determine title from remaining text"""
    numericals = []
    non_numericals = []

    # Classify text regions
    for region in text_regions:
        if is_numerical(region['text']):
            numericals.append(region)
        else:
            non_numericals.append(region)

    # Pair each numerical with closest non-numerical
    pairs = []
    used_non_numericals = []  # Use a list instead of a set
    for num in numericals:
        closest = min(
            [nn for nn in non_numericals if nn not in used_non_numericals],
            key=lambda nn: math.hypot(
                num['center'][0] - nn['center'][0],
                num['center'][1] - nn['center'][1]
            ),
            default=None
        )
        if not closest:
            raise PieChartError(f"Unpaired numerical value: {num['text']}")
        pairs.append((closest, num))
        used_non_numericals.append(closest)  # Add to list instead of set

    # Extract labels and percentages
    labels = [closest['text'] for closest, _ in pairs]
    percentages = [float(num['text'].strip('%')) for _, num in pairs]

    # Determine title from remaining text
    title_candidates = [nn for nn in non_numericals if nn not in used_non_numericals]
    title = " ".join([tc['text'] for tc in title_candidates]) if title_candidates else "Pie Chart"

    return labels, percentages, title

def process_pie_chart(image_path: str, paddle_results: List[Dict], save_path: Optional[str] = None) -> Dict:
    """Main processing function with improved title handling"""
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise PieChartError("Image load failed")

        # Detect pie chart
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise PieChartError("No pie chart detected")

        # Create mask and find center
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
        moments = cv2.moments(mask)
        pie_center = (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00']))

        # Process text regions
        text_regions = []
        for result in paddle_results:
            if result['confidence'] <= 0.5:
                continue
            box = result['box']
            x_coords = [p[0] for p in box]
            y_coords = [p[1] for p in box]
            text_regions.append({
                'text': result['text'],
                'center': (
                    (min(x_coords) + max(x_coords)) / 2,
                    (min(y_coords) + max(y_coords)) / 2
                ),
                'bbox': (min(x_coords), min(y_coords), max(x_coords)-min(x_coords), max(y_coords)-min(y_coords))
            })

        # Match labels and get title
        labels, percentages, title = match_labels_and_get_title(text_regions, pie_center)

        # Extract segments
        segments = extract_segment_info(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            mask,
            pie_center,
            len(labels)
        )

        # Prepare output
        chart_data = {
            "title": title,
            "pie_center": list(pie_center),
            "slices": [{
                "label": labels[i],
                "percentage": percentages[i],
                "color": seg['color'].tolist(),
                "start_angle": seg['start_angle'],
                "end_angle": seg['end_angle'],
                "pixel_count": seg['pixel_count']
            } for i, seg in enumerate(segments)]
        }

        # Save outputs as pie_construct.json in the specified directory
        save_path = "/home/acer/minor project final/pie_final_op/pie_construct.json"
        with open(save_path, 'w') as f:
            json.dump(chart_data, f, indent=4)

        return chart_data

    except Exception as e:
        print(f"Error processing pie chart: {str(e)}")
        return None

router = APIRouter()

@router.get("/pie_reconstruct")
def pie_reconstruct():
    image_path = "/home/acer/Desktop/pie.png"
    
    with open("/home/acer/minor project final/Pie_ocr_results/ocr_results.json") as f:
        ocr_data = json.load(f)
    
    result = process_pie_chart(image_path, ocr_data)

    if result is None:
        raise HTTPException(status_code=500, detail="Error processing pie chart")
    
    return result
