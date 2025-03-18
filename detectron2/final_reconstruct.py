import os
import re
import json
import requests
import matplotlib.pyplot as plt
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
input_image_path = "/home/acer/minor project final/classification_results/original.png"

TEXT_JSON_FOLDER = "/home/acer/minor project final/text_ocr_results"
TABLE_JSON_FOLDER = "/home/acer/minor project final/table_results"
PIE_JSON_FILE = "/home/acer/minor project final/pie_final_op/pie_construct.json"
BAR_JSON_FILE = "/home/acer/minor project final/bar_final_op/final.json"
INIT_SEG_JSON = "/home/acer/minor project final/classification_results/output.json"
OUTPUT_PDF = "/home/acer/Desktop/output.pdf"

# --- Utility Functions ---

def register_font(font_name, font_path):
    """
    Register a TTF font for ReportLab.
    If the specified font file is not found, download a free substitute (LiberationSans)
    and register it under the desired font name.
    """
    if not os.path.exists(font_path):
        print(f"{font_path} not found. Downloading a substitute for '{font_name}'...")
        url = "https://github.com/liberationfonts/liberation-fonts/files/600856/LiberationSans-Regular.ttf?raw=true"
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(font_path, "wb") as f:
                f.write(r.content)
            print("Font downloaded and saved as", font_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download font from {url}: {e}")
    pdfmetrics.registerFont(TTFont(font_name, font_path))

def tl_to_pdf(y, page_height):
    """
    Convert a y-coordinate from a top-left origin (common in image processing)
    to the bottom-left origin used by ReportLab.
    """
    return page_height - y

def extract_score(filename):
    """
    Extract a numeric score from the filename.
    Looks for a pattern like 'score_0.99' or 'score_0.90' and returns the float value.
    If no score is found, returns 0.
    """
    m = re.search(r'score_([0-9.]+)', filename)
    if m:
        return float(m.group(1))
    return 0

def select_file_for_score(files_info, seg_score, tolerance=0.01):
    """
    Given a list of tuples (file_score, filename) and a segmentation score,
    return the filename whose file_score is within tolerance of seg_score.
    If none is within tolerance, return the one with the smallest difference.
    """
    if not files_info:
        return None
    # Compute difference for each file
    differences = [(abs(file_score - seg_score), filename) for file_score, filename in files_info]
    differences.sort(key=lambda x: x[0])
    best_diff, best_file = differences[0]
    return best_file

# --- Chart Generation Functions using Matplotlib ---

def generate_pie_chart_image():
    """
    Generate a pie chart image from piechart.json.
    All text (labels, percentages, title) is set to font size defined by pie_text_size.
    """
    with open(PIE_JSON_FILE, "r") as f:
        pie_data = json.load(f)

    title = pie_data.get("title", "")
    slices = pie_data.get("slices", [])
    percentages = [s["percentage"] for s in slices]
    labels = [s["label"] for s in slices]
    colors = []
    for s in slices:
        c = s.get("color", [0, 0, 0])
        colors.append((c[0]/255.0, c[1]/255.0, c[2]/255.0))

    fig, ax = plt.subplots(figsize=(7.2, 6.9), dpi=100)
    ax.pie(
        percentages,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': pie_text_size}
    )
    ax.set_title(title, fontsize=pie_text_size)
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig("piechart_output.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Generated piechart_output.png")

def generate_bar_graph_image():
    """
    Generate a standard bar graph from bargraph.json using plt.bar.
    The chart uses font size defined by bar_text_size for all text.
    Only the top and right spines are removed so that the x-axis and y-axis remain.
    The saved image is trimmed so that when inserted into the PDF, it fits exactly within the segmentation region.
    """
    with open(BAR_JSON_FILE, "r") as f:
        bar_data = json.load(f)

    title = bar_data.get("title", "")
    x_label = bar_data.get("x_label", "")
    y_label = bar_data.get("y_label", "")

    bars = bar_data.get("bars", [])
    if not bars:
        print("No bar data found.")
        return

    bar_values = [bar.get("value", 0) for bar in bars]
    x_positions = list(range(len(bars)))
    bar_colors = []
    for bar in bars:
        col = bar.get("color", [0, 0, 0])
        bar_colors.append((col[0], col[1], col[2]))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.bar(x_positions, bar_values, color=bar_colors, edgecolor='none')
    ax.set_title(title, fontsize=bar_text_size)
    ax.set_xlabel(x_label, fontsize=bar_text_size)
    ax.set_ylabel(y_label, fontsize=bar_text_size)
    ax.tick_params(axis='both', labelsize=bar_text_size)

    # Remove only the top and right spines so that the bottom (x-axis) and left (y-axis) remain.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if "x_ticks" in bar_data:
        x_tick_labels = [tick.get("label", "") for tick in bar_data["x_ticks"]]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_tick_labels, fontsize=bar_text_size)

    plt.tight_layout()
    plt.savefig("bargraph_output.png", bbox_inches='tight', pad_inches=0)
    plt.close()
    print("Generated bargraph_output.png")

# --- Drawing Functions for PDF ---

def draw_text(c, text_data, offset, page_height):
    """
    Draw text elements from a cropped text JSON.
    The coordinates from the cropped region are offset to position them
    in the full-page PDF. The font size is forced to constant_font_size.
    Adjusted vertical position to better center the text.
    """
    constant_font_size = FONT_SIZE
    for item in text_data.get("content", []):
        text = item.get("text", "")
        style = item.get("style", {})
        coords = style.get("coordinates", [0, 0, 0, 0])
        x1, y1, _, _ = coords
        abs_x = offset[0] + x1
        # Adjust vertical offset by subtracting half the font size for centering.
        abs_y = tl_to_pdf(offset[1] + y1, page_height) - constant_font_size / 2
        try:
            c.setFont(style.get("font_family", "Helvetica"), constant_font_size)
        except Exception:
            c.setFont("Helvetica", constant_font_size)
        col = style.get("color", [0, 0, 0])
        c.setFillColor(Color(col[0]/255.0, col[1]/255.0, col[2]/255.0))
        c.drawString(abs_x, abs_y, text)

def draw_table(c, table_data, offset, region_size, page_height):
    """
    Draw a simple table within the specified bounding box.
    The font size for table text is forced to constant_font_size for legibility.
    The first row of the table is drawn in bold.
    """
    constant_font_size = FONT_SIZE
    rows = table_data
    if not rows:
        return
    n_rows = len(rows)
    n_cols = len(rows[0])
    x_min, y_min = offset
    region_width, region_height = region_size
    cell_width = region_width / n_cols
    cell_height = region_height / n_rows

    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            cell_x = x_min + j * cell_width
            cell_y = tl_to_pdf(y_min + i * cell_height, page_height) - cell_height
            c.rect(cell_x, cell_y, cell_width, cell_height)
            # Use Helvetica-Bold for the first row, else regular Helvetica.
            if i == 0:
                c.setFont("Helvetica-Bold", constant_font_size)
            else:
                c.setFont("Helvetica", constant_font_size)
            text_x = cell_x + 2
            text_y = cell_y + cell_height/2 - constant_font_size/2
            c.drawString(text_x, text_y, str(cell))

# --- Main PDF Generation Code ---

def main():
    # Register custom font (Arial) if available.
    try:
        register_font("Arial", "Arial.ttf")
    except Exception as e:
        print("Arial font registration failed; using default Helvetica. Error:", e)

    # Generate the chart images from JSON.
    generate_pie_chart_image()
    generate_bar_graph_image()

    # Gather text and table JSON files from the specified folders.
    text_files = [
        os.path.join(TEXT_JSON_FOLDER, f)
        for f in os.listdir(TEXT_JSON_FOLDER)
        if f.lower().endswith(".json")
    ]
    table_files = [
        os.path.join(TABLE_JSON_FOLDER, f)
        for f in os.listdir(TABLE_JSON_FOLDER)
        if f.lower().endswith(".json")
    ]
    # Build lists of tuples (extracted_score, filename) for text and table files.
    text_files_info = [(extract_score(os.path.basename(f)), f) for f in text_files]
    table_files_info = [(extract_score(os.path.basename(f)), f) for f in table_files]

    # Use the input image to set the PDF page size.
    with Image.open(input_image_path) as im:
        page_width, page_height = im.size

    # Create a PDF canvas with the same size as the input image.
    c = canvas.Canvas(OUTPUT_PDF, pagesize=(page_width, page_height))

    # Load the initial segmentation JSON.
    with open(INIT_SEG_JSON, "r") as f:
        segmentation = json.load(f)

    # Process each segmentation region.
    for seg in segmentation:
        class_id = seg["class"]
        bbox = seg["bbox"]  # [x_min, y_min, x_max, y_max] in top-left coordinates.
        x_min, y_min, x_max, y_max = bbox
        region_width = x_max - x_min
        region_height = y_max - y_min
        offset = (x_min, y_min)
        pdf_y = tl_to_pdf(y_max, page_height)

        if class_id == 3:
            # Insert pie chart image.
            pie_img_file = "piechart_output.png"
            if os.path.exists(pie_img_file):
                c.drawImage(pie_img_file, x_min, pdf_y, width=region_width, height=region_height)
            else:
                print(f"Pie chart image file {pie_img_file} not found.")
        elif class_id == 2:
            # Insert bar graph image.
            bar_img_file = "bargraph_output.png"
            if os.path.exists(bar_img_file):
                c.drawImage(bar_img_file, x_min, pdf_y, width=region_width, height=region_height)
            else:
                print(f"Bar graph image file {bar_img_file} not found.")
        elif class_id == 1:
            # For table regions, select the table JSON whose score best matches the segmentation score.
            seg_score = seg.get("score", 0)
            table_file = select_file_for_score(table_files_info, seg_score, tolerance=0.01)
            if table_file:
                try:
                    with open(table_file, "r") as f:
                        table_data = json.load(f)
                    draw_table(c, table_data, offset, (region_width, region_height), page_height)
                except Exception as e:
                    print(f"Error reading table JSON '{table_file}': {e}")
            else:
                print("No matching table JSON file found.")
        elif class_id == 0:
            # For text regions, select the text JSON whose score best matches the segmentation score.
            seg_score = seg.get("score", 0)
            text_file = select_file_for_score(text_files_info, seg_score, tolerance=0.01)
            if text_file:
                try:
                    with open(text_file, "r") as f:
                        text_data = json.load(f)
                    draw_text(c, text_data, offset, page_height)
                except Exception as e:
                    print(f"Error reading text JSON '{text_file}': {e}")
            else:
                print("No matching text JSON file found.")
        else:
            continue

    c.save()
    print("PDF generated as output.pdf")

class FontSizeRequest(BaseModel):
    font_size: int
    pie_text_size: int
    bar_text_size: int

@router.post("/final_output")  # Changed from .get to .post
async def final_op(request: FontSizeRequest):
    global FONT_SIZE, pie_text_size, bar_text_size  # Declare as global to modify the module-level variables
    FONT_SIZE = request.font_size
    pie_text_size = request.pie_text_size
    bar_text_size = request.bar_text_size
    main()
    return {"message": "PDF generation completed"}