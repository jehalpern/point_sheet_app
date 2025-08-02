import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_to_corner_boxes(image, box_size=40, padding=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.2 and box_size - 10 <= w <= box_size + 10:
            candidates.append((x, y, x + w, y + h))

    if len(candidates) < 4:
        print("Less than 4 corner boxes found.")
        return image

    candidates = sorted(candidates, key=lambda b: (b[1], b[0]))
    top_two = sorted(candidates[:2], key=lambda b: b[0])
    bottom_two = sorted(candidates[-2:], key=lambda b: b[0])
    corners = [top_two[0], top_two[1], bottom_two[0], bottom_two[1]]

    x_min = min(c[0] for c in corners) - padding
    y_min = min(c[1] for c in corners) - padding
    x_max = max(c[2] for c in corners) + padding
    y_max = max(c[3] for c in corners) + padding

    h, w = image.shape[:2]
    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max = min(x_max, w)
    y_max = min(y_max, h)

    cropped = image[y_min:y_max, x_min:x_max]
    return cropped

def extract_score_regions(cropped_img, start_x, start_y, cell_width, cell_height, x_spacing, y_spacing):
    regions = []
    for row in range(4):
        for col in range(6):
            x1 = start_x + col * x_spacing
            y1 = start_y + row * y_spacing
            x2 = x1 + cell_width
            y2 = y1 + cell_height
            region = cropped_img[y1:y2, x1:x2]
            regions.append(((row, col), region))
    return regions

def display_regions(regions):
    plt.figure(figsize=(15, 10))
    for i, ((row, col), region) in enumerate(regions):
        plt.subplot(4, 6, i + 1)
        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        plt.title(f"{row},{col}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def extract_score_regions(cropped_img, start_x, start_y, cell_width, cell_height, x_spacing, y_spacing):
    regions = []
    overlay_img = cropped_img.copy()

    for row in range(4):
        for col in range(6):
            x1 = start_x + col * x_spacing
            y1 = start_y + row * y_spacing
            x2 = x1 + cell_width
            y2 = y1 + cell_height

            # Draw rectangle on overlay
            cv2.rectangle(overlay_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            region = cropped_img[y1:y2, x1:x2]
            regions.append(((row, col), region))

    return regions, overlay_img


def display_regions(regions, overlay_img):
    # Show the overlay image with all rectangles first
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
    plt.title("Cropped Image with Region Boxes")
    plt.axis("off")
    plt.show()

    # Show individual cropped regions
    plt.figure(figsize=(15, 10))
    for i, ((row, col), region) in enumerate(regions):
        plt.subplot(4, 6, i + 1)
        plt.imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        plt.title(f"{row},{col}", fontsize=8)
        plt.axis("off")
    plt.tight_layout()
    plt.show()



# === MAIN ===
if __name__ == "__main__":
    image_path = "../sheet_generator/photo_test.jpg"  # Replace with your actual path
    image = cv2.imread(image_path)

    cropped = crop_to_corner_boxes(image)

    # 🔧 These are the values you'll adjust:
    start_x = 450   # fine-tune in the cropped image space
    start_y = 350
    cell_width = 300
    cell_height = 115
    x_spacing = 300
    y_spacing = 120

    #regions = extract_score_regions(cropped, start_x, start_y, cell_width, cell_height, x_spacing, y_spacing)
    #display_regions(regions)

    regions, overlay_img = extract_score_regions(
    cropped, start_x, start_y, cell_width, cell_height, x_spacing, y_spacing
    )
    display_regions(regions, overlay_img)

