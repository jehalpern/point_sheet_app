from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import qrcode
from io import BytesIO
from reportlab.lib.utils import ImageReader
import cv2
import numpy as np

def generate_aruco_marker(marker_id, size_pixels=200):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, size_pixels)
    is_success, buffer = cv2.imencode(".png", marker_img)
    return ImageReader(BytesIO(buffer))

def draw_point_sheet(c, origin_x, origin_y, width=letter[0], height=letter[1]):
    scale = 1.07

    margin = 0.3 * inch * scale
    box_size = 0.25 * inch * scale
    row_height = 0.45 * inch * scale
    col_width = 1.2 * inch * scale
    score_options = ['0', '1', '2']
    time_blocks = ['WIN', 'Until AM Recess', 'Until Lunch', 'Until PM Recess', 'End of Day', 'Specials']
    behaviors = ['Positive', 'Respectful', 'Responsible', 'Safe']

    table_x_offset = 0.05 * width * scale
    square_size = 0.25 * inch * scale

    start_y = origin_y + height - margin - .95 * inch
    table_top = start_y + row_height * 0.5
    table_left = origin_x + table_x_offset - 0.05 * inch * scale
    table_width = (len(time_blocks)) * col_width + 1.19 * inch * scale
    table_height = len(behaviors) * row_height + 0.48 * inch * scale

    # Draw ArUco markers
    marker_size = square_size
    positions = [
        (table_left - marker_size + 1.2 * inch, table_top - 0.46 * inch),                            # Top-left
        (table_left + table_width + 0.03 * inch, table_top - 0.46 * inch),               # Top-right (moved right)
        (table_left - marker_size + 1.2 * inch, table_top - table_height - marker_size - 0.046 * inch),  # Bottom-left (moved down)
        (table_left + table_width, table_top - table_height - marker_size - 0.01 * inch)   # Bottom-right
    ]
    for i, (x, y) in enumerate(positions):
        marker = generate_aruco_marker(marker_id=i, size_pixels=200)
        c.drawImage(marker, x, y, marker_size, marker_size)

    # Header
    c.setFont("Helvetica-Bold", 14)
    c.drawCentredString(origin_x + width / 2, origin_y + height - margin - 0.1 * inch, "Check-in Check-out")
    c.setFont("Helvetica", 12)
    c.drawString(origin_x + margin, origin_y + height - margin - 0.3 * inch, "Name: test_student Adam")
    c.drawString(origin_x + width / 1.5, origin_y + height - margin - 0.3 * inch, "Date: 2025-07-12")

    # Time block headers
    start_y = origin_y + height - margin - 1.0 * inch
    c.setFont("Helvetica-Bold", 7)
    for i, tb in enumerate(time_blocks):
        c.drawCentredString(origin_x + table_x_offset + 1.75 * inch * scale + i * col_width, start_y, tb)

    # Score cells
    c.setFont("Helvetica", 7)
    for row_idx, behavior in enumerate(behaviors):
        y = start_y - (row_idx + 1.2) * row_height
        c.drawString(origin_x + table_x_offset + 0.01 * width, y + box_size / 2, behavior)
        for col_idx in range(len(time_blocks)):
            for score_idx, score in enumerate(score_options):
                x = origin_x + table_x_offset + 1.2 * inch * scale + col_idx * col_width + score_idx * (box_size + 8 * scale)
                c.drawCentredString(x + box_size / 2, y + box_size / 4, score)

    # Grid
    table_top = start_y + row_height * 0.5
    table_left = origin_x + table_x_offset + 0.1 * inch * scale
    grid_x_offset = -0.15 * inch * scale
    grid_y_offset = 0.5
    table_width = (len(time_blocks)) * col_width + 1.0 * inch * scale
    table_height = len(behaviors) * row_height + 0.4 * inch * scale

    c.setLineWidth(2.5)  # Make all grid lines bold

    for i in range(len(behaviors) + 1):
        y = table_top - i * row_height + grid_y_offset
        c.line(table_left + grid_x_offset, y, table_left + table_width, y)

    for i in range(len(time_blocks) + 1):
        x = table_left + i * col_width + grid_x_offset
        c.line(x, table_top + grid_y_offset, x, table_top - table_height + grid_y_offset)

    c.setLineWidth(2.5)  # Make border bold as well
    offset_table_left = table_left - .15 * inch * scale
    offset_table_width = table_width + 0.15 * inch * scale
    c.rect(offset_table_left, table_top - table_height, offset_table_width, table_height, stroke=1, fill=0)

    # QR Code
    student_id = "ada3bd21"
    date_str = "2025-07-12"
    qr_data = f"{student_id}|{date_str}"
    qr = qrcode.make(qr_data)
    qr_buffer = BytesIO()
    qr.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)
    qr_img = ImageReader(qr_buffer)
    qr_size = 0.3 * inch * 3.9 * scale
    qr_x = table_left + table_width + .25 * inch * scale
    qr_y = table_top - qr_size - .4 * inch * scale
    c.drawImage(qr_img, qr_x, qr_y, qr_size, qr_size)

def generate_pdf(filename="Optimized_Point_Sheet_Aruco.pdf"):
    c = canvas.Canvas(filename, pagesize=landscape(letter))
    sheet_width = landscape(letter)[0]
    sheet_height = landscape(letter)[1] / 2

    positions = [
        (0, sheet_height),
        (0, 0),
    ]
    for pos in positions:
        draw_point_sheet(c, origin_x=pos[0], origin_y=pos[1], width=sheet_width, height=sheet_height)

    c.save()

if __name__ == "__main__":
    generate_pdf()
