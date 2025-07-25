from reportlab.lib.pagesizes import letter, landscape
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import qrcode
from io import BytesIO
from reportlab.lib.utils import ImageReader

def draw_point_sheet(c, origin_x, origin_y, width=letter[0], height=letter[1]):
    margin = 0.2 * inch
    box_size = 0.25 * inch
    row_height = 0.5 * inch
    col_width = 1.2 * inch
    score_options = ['0', '1', '2']
    time_blocks = ['WIN', 'Until AM Recess', 'Until Lunch', 'Until PM Recess', 'End of Day', 'Specials']
    behaviors = ['Positive', 'Respectful', 'Responsible', 'Safe']

    table_x_offset = 0.06 * width  # 10% to the right

    # Alignment corners
    square_size = 0.1 * inch
    c.rect(origin_x + margin, origin_y + margin, square_size, square_size, fill=1)
    c.rect(origin_x + width - margin - square_size, origin_y + margin, square_size, square_size, fill=1)
    c.rect(origin_x + margin, origin_y + height - margin - square_size, square_size, square_size, fill=1)
    c.rect(origin_x + width - margin - square_size, origin_y + height - margin - square_size, square_size, square_size, fill=1)

    # Header
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(origin_x + width / 2, origin_y + height - margin - 0.1 * inch, "Check-in Check-out")
    c.setFont("Helvetica", 8)
    c.drawString(origin_x + margin, origin_y + height - margin - 0.4 * inch, "Name: test_student Adam") 
    c.drawString(origin_x + width / 2, origin_y + height - margin - 0.4 * inch, "Date: 2025-07-12")

    # Table origin
    start_y = origin_y + height - margin - 1.0 * inch
    c.setFont("Helvetica-Bold", 7)
    c.drawString(origin_x + table_x_offset + 0.8 * inch, start_y, "")
    for i, tb in enumerate(time_blocks):
        c.drawCentredString(origin_x + table_x_offset + 1.75 * inch + i * col_width, start_y, tb)

    # Table rows
    c.setFont("Helvetica", 7)
    for row_idx, behavior in enumerate(behaviors):
        y = start_y - (row_idx + 1.2) * row_height
        c.drawString(origin_x + table_x_offset + 0.01 * width, y + box_size / 2, behavior)  # shifted right by 15%
        for col_idx in range(len(time_blocks)):
            for score_idx, score in enumerate(score_options):
                x = origin_x + table_x_offset + 1.3 * inch + col_idx * col_width + score_idx * (box_size + 1)
                c.rect(x, y, box_size, box_size)
                c.drawCentredString(x + box_size / 2, y + box_size / 4, score)

    # Draw grid lines for the table
    table_top = start_y + row_height * 0.5
    table_left = origin_x + table_x_offset + 0.1 * inch
    grid_x_offset = -0.15 * inch  # Move grid lines to the left
    grid_y_offset = 0.5  # You can adjust this value to move grid lines up/down

    table_width = (len(time_blocks)) * col_width + 1.0 * inch
    table_height = len(behaviors) * row_height + 0.4 * inch

    # Horizontal grid lines
    for i in range(len(behaviors) + 1):
        y = table_top - i * row_height + grid_y_offset
        c.line(table_left + grid_x_offset, y, table_left + table_width, y)

    # Vertical grid lines
    num_cols = len(time_blocks)
    for i in range(num_cols + 1):
        x = table_left + i * col_width + grid_x_offset  # Removed + 1 * inch
        c.line(x, table_top + grid_y_offset, x, table_top - table_height + grid_y_offset)

    # Draw border box around table
    c.setLineWidth(1)
    offset_table_left = table_left - .15 * inch 
    offset_table_width = table_width + 0.15 * inch
    c.rect(offset_table_left, table_top - table_height, offset_table_width, table_height, stroke=1, fill=0)

    # Example student_id and date (replace with actual values as needed)
    student_id = "ada3bd21"
    date_str = "2025-07-12"

    # Generate QR code data
    qr_data = f"{student_id}|{date_str}"
    qr = qrcode.make(qr_data)
    qr_buffer = BytesIO()
    qr.save(qr_buffer, format="PNG")
    qr_buffer.seek(0)
    qr_img = ImageReader(qr_buffer)

    # Position QR code to the right of the table and make it 15% bigger
    qr_size = 0.4 * inch * 1.5  # % bigger
    qr_x = table_left + table_width + 0.25 * inch  # a bit of space to the right of the table
    qr_y = table_top - qr_size  # align top of QR with top of table
    c.drawImage(qr_img, qr_x, qr_y, qr_size, qr_size)

def generate_pdf(filename="Optimized_Point_Sheet.pdf"):
    c = canvas.Canvas(filename, pagesize=landscape(letter))
    sheet_width = landscape(letter)[0]
    sheet_height = landscape(letter)[1] / 2

    # 2 sheets per page, stacked vertically
    positions = [
        (0, sheet_height),   # Top half
        (0, 0),              # Bottom half
    ]
    for pos in positions:
        draw_point_sheet(c, origin_x=pos[0], origin_y=pos[1], width=sheet_width, height=sheet_height)

    c.save()

if __name__ == "__main__":
    generate_pdf()
