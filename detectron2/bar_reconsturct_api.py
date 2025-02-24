import json
import numpy as np
import re
from fastapi import APIRouter
from fastapi.responses import JSONResponse

class UniversalChartReconstructor:
    def __init__(self, ocr_path, detection_path):
        self.ocr_data = self.load_json(ocr_path)
        self.detections = self.load_detections(detection_path)
        self.img_width, self.img_height = self.detect_image_dimensions()
        self.title = "Reconstructed Chart"
        self.x_label = "X-Axis"
        self.y_label = "Y-Axis"
        self.y_ticks = []
        self.x_categories = []
        self.x_category_positions = []
        self.bars = []
        self.y_scale = lambda h: h

    @staticmethod
    def load_json(path):
        with open(path) as f:
            return json.load(f)

    def load_detections(self, path):
        data = self.load_json(path)
        if isinstance(data, dict):
            return list(data.values())[0]
        return data

    def detect_image_dimensions(self):
        try:
            all_x = [item['box'][2][0] for item in self.ocr_data] + [det['bbox'][2] for det in self.detections]
            all_y = [item['box'][2][1] for item in self.ocr_data] + [det['bbox'][3] for det in self.detections]
            return max(all_x), max(all_y)
        except Exception:
            return 1000, 800

    def process_ocr_elements(self):
        elements = []
        for item in self.ocr_data:
            text = item['text'].strip()
            box = item['box']
            x_center = (box[0][0] + box[1][0]) / 2
            y_center = (box[0][1] + box[2][1]) / 2
            x1 = box[1][0]
            y3 = box[3][1]
            elements.append({
                'text': text,
                'x_center': x_center,
                'y_center': y_center,
                'x1': x1,
                'y3': y3,
                'used': False
            })
        return elements

    def calculate_scaling(self):
        if not self.y_ticks:
            max_height = 0
            for det in self.detections:
                h = det['bbox'][3] - det['bbox'][1]
                if h > max_height:
                    max_height = h
            if max_height < 1:
                max_height = self.img_height

            def fallback_scale(h):
                return (h / max_height) * 10

            self.y_scale = fallback_scale

            self.y_ticks = []
            for value in range(0, 11, 2):
                h = (value / 10) * max_height
                pos = self.img_height - h
                self.y_ticks.append({'value': value, 'position': pos})
        else:
            y_values = [tick['value'] for tick in self.y_ticks]
            y_positions = [tick['position'] for tick in self.y_ticks]
            try:
                coeff = np.polyfit(y_positions, y_values, 1)
                self.y_scale = np.poly1d(coeff)
            except Exception:
                self.y_scale = lambda h: h

    def process_bars(self):
        for i, det in enumerate(self.detections):
            bbox = det['bbox']
            color = det.get('dominant_color', [i * 50 % 255 for i in range(3)])
            top_px = bbox[1]
            bottom_px = bbox[3]
            self.bars.append({
                'x_center': (bbox[0] + bbox[2]) / 2,
                'top_px': top_px,
                'bottom_px': bottom_px,
                'color': [c / 255 for c in color]
            })
        self.bars.sort(key=lambda x: x['x_center'])

    def map_bars_to_categories(self):
        bar_positions = [bar['x_center'] for bar in self.bars]
        if not self.x_categories:
            self.x_categories = [f"Item {i + 1}" for i in range(len(self.bars))]
            if bar_positions:
                min_pos = min(bar_positions)
                max_pos = max(bar_positions)
            else:
                min_pos = 0
                max_pos = self.img_width
            self.x_category_positions = np.linspace(min_pos, max_pos, len(self.x_categories))
        else:
            if bar_positions:
                min_pos = min(bar_positions)
                max_pos = max(bar_positions)
            else:
                min_pos = 0
                max_pos = self.img_width
            self.x_category_positions = np.linspace(min_pos, max_pos, len(self.x_categories))

        for bar in self.bars:
            idx = np.argmin([abs(bar['x_center'] - pos) for pos in self.x_category_positions])
            bar['category'] = self.x_categories[idx]

    def cluster_labels(self, labels, threshold):
        if not labels:
            return []
        clusters = []
        current_cluster = [labels[0]]
        for label in labels[1:]:
            if label[1] - current_cluster[-1][1] < threshold:
                current_cluster.append(label)
            else:
                clusters.append(current_cluster)
                current_cluster = [label]
        if current_cluster:
            clusters.append(current_cluster)
        return [' '.join([item[0] for item in cluster]) for cluster in clusters]

    def extract_labels(self, ocr_elements):
        x_axis_candidates = [elem for elem in ocr_elements if elem['y_center'] > self.img_height * 0.75]
        if x_axis_candidates:
            x_axis_candidates.sort(key=lambda e: e['y_center'], reverse=True)
            self.x_label = x_axis_candidates[0]['text']
            x_axis_candidates[0]['used'] = True
            tick_candidates = x_axis_candidates[1:]
            if tick_candidates:
                median_y3 = np.median([elem['y3'] for elem in tick_candidates])
                delta = 10
                filtered_x_ticks = [elem for elem in tick_candidates if abs(elem['y3'] - median_y3) <= delta]
                filtered_x_ticks.sort(key=lambda e: e['x_center'])
                self.x_categories = [elem['text'] for elem in filtered_x_ticks]
                for elem in filtered_x_ticks:
                    elem['used'] = True
            else:
                self.x_categories = []
        else:
            x_tick_candidates = [
                elem for elem in ocr_elements
                if (not re.match(r"^\d+(\.\d+)?$", elem['text'])) and elem['y_center'] > self.img_height * 0.75
            ]
            if x_tick_candidates:
                median_y3 = np.median([elem['y3'] for elem in x_tick_candidates])
                delta = 10
                filtered_x_ticks = [elem for elem in x_tick_candidates if abs(elem['y3'] - median_y3) <= delta]
                filtered_x_ticks.sort(key=lambda e: e['x_center'])
                self.x_categories = [elem['text'] for elem in filtered_x_ticks]
                self.x_label = "Categories"
                for elem in filtered_x_ticks:
                    elem['used'] = True

        y_tick_candidates = [elem for elem in ocr_elements if re.match(r"^\d+(\.\d+)?$", elem['text'])]
        if y_tick_candidates:
            median_x1 = np.median([elem['x1'] for elem in y_tick_candidates])
            delta = 10
            filtered_y_ticks = [elem for elem in y_tick_candidates if abs(elem['x1'] - median_x1) <= delta]
            filtered_y_ticks.sort(key=lambda e: e['y_center'], reverse=True)
            self.y_ticks = [{'value': float(elem['text']), 'position': elem['y_center']}
                            for elem in filtered_y_ticks]
            for elem in filtered_y_ticks:
                elem['used'] = True

        y_axis_candidates = [
            elem for elem in ocr_elements
            if elem['x_center'] < self.img_width * 0.1 and
               (self.img_height * 0.2 < elem['y_center'] < self.img_height * 0.8) and
               not re.match(r"^\d+(\.\d+)?$", elem['text'])
        ]
        if y_axis_candidates:
            y_axis_candidates.sort(key=lambda e: e['y_center'])
            self.y_label = ' '.join([e['text'] for e in y_axis_candidates])
            for elem in y_axis_candidates:
                elem['used'] = True
        else:
            if not self.y_label:
                self.y_label = "Values"

        remaining = [elem for elem in ocr_elements if not elem['used']]
        title_candidates = [elem for elem in remaining if elem['y_center'] < self.img_height * 0.5]
        if title_candidates:
            title_candidates.sort(key=lambda e: e['y_center'])
            self.title = " ".join([elem['text'] for elem in title_candidates])
        else:
            self.title = "Reconstructed Chart"

    def reconstruct(self):
        try:
            ocr_elements = self.process_ocr_elements()
            self.extract_labels(ocr_elements)
            self.process_bars()
            self.calculate_scaling()
            
            for bar in self.bars:
                top_val = self.y_scale(bar['top_px'])
                bottom_val = self.y_scale(bar['bottom_px'])
                bar_height = bottom_val - top_val
                if bar_height < 0:
                    bar_height = -bar_height
                bar['value'] = bar_height

            self.map_bars_to_categories()
        except Exception as e:
            print(f"Reconstruction error: {str(e)}")

    def export_json(self, output_path):
        chart_data = {
            "title": self.title,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_ticks": [
                {"label": label, "position": float(pos)}
                for label, pos in zip(self.x_categories, self.x_category_positions)
            ],
            "y_ticks": [
                {"value": tick['value'], "position": float(tick['position'])}
                for tick in self.y_ticks
            ],
            "bars": [
                {
                    "category": bar.get('category', ''),
                    "color": [round(c, 4) for c in bar['color']],
                    "value": float(bar.get('value', 0)),
                    "top_px": float(bar['top_px']),
                    "bottom_px": float(bar['bottom_px']),
                    "x_center": float(bar['x_center'])
                }
                for bar in self.bars
            ]
        }
        with open(output_path, 'w') as f:
            json.dump(chart_data, f, indent=2)

router = APIRouter()

@router.get("/Bar_reconstruct")
def bar_reconstruct():
    reconstructor = UniversalChartReconstructor(
        ocr_path="/home/acer/minor project final/bar_ocr_results/ocr_results.json",
        detection_path="/home/acer/bar_bb.json"
    )
    reconstructor.reconstruct()
    output_path = "/home/acer/minor project final/bar_final_op/final.json"
    reconstructor.export_json(output_path)
    with open(output_path, 'r') as f:
        final_json = json.load(f)
    return JSONResponse(content=final_json)

