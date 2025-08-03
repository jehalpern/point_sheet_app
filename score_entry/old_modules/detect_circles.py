import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import easyocr

def detect_and_draw_circles(image):
    output_image = image.copy()
    rows, cols, options = 4, 6, 3
    cell_h = image.shape[0] // rows
    cell_w = image.shape[1] // cols
    box_w = cell_w // options

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 3)

    for row in range(rows):
        for col in range(cols):
            for i in range(options):
                x = col * cell_w + i * box_w
                y = row * cell_h
                box = thresh[y:y + cell_h, x:x + box_w]

                contours, _ = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                is_circled = False

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    if area > 30 and perimeter > 30:
                        circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-5)
                        if 0.8 < circularity < 1.2:
                            is_circled = True
                            break

                if is_circled:
                    cx = x + box_w // 2
                    cy = y + cell_h // 2
                    radius = min(box_w, cell_h) // 2 - 4
                    cv2.circle(output_image, (cx, cy), radius, (0, 0, 255), 2)

    return output_image

def detect_and_draw_circles_with_debug(image):
    output_image = image.copy()
    rows, cols, options = 4, 6, 3
    cell_h = image.shape[0] // rows
    cell_w = image.shape[1] // cols
    box_w = cell_w // options

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morph cleanup (optional)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    for row in range(rows):
        for col in range(cols):
            for i in range(options):
                x = col * cell_w + i * box_w
                y = row * cell_h
                box = thresh[y:y + cell_h, x:x + box_w]

                contours, _ = cv2.findContours(box, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_circle = None
                debug = cv2.cvtColor(box.copy(), cv2.COLOR_GRAY2BGR)

                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if area > 50 and 0.3 < circularity < 1.5:
                        best_circle = cnt
                        break

                if best_circle is not None:
                    cx = x + box_w // 2
                    cy = y + cell_h // 2
                    radius = min(box_w, cell_h) // 2 - 4
                    cv2.circle(output_image, (cx, cy), radius, (0, 0, 255), 2)

    return output_image

def find_printed_digits_debug(image):
    debug_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Boost contrast to isolate printed digits
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    # Threshold for digit isolation
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Denoising
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = h / float(w) if w > 0 else 0

        # Relaxed constraints
        if 8 < h < 70 and 4 < w < 50 and 0.8 < aspect_ratio < 8.0 and area > 20:
            candidates.append((x, y, w, h))
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 255), 2)

    print(f"🔍 Found {len(candidates)} printed digit candidates")
    return debug_image, candidates


import cv2
import pytesseract

def find_digits_012(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Preprocessing to improve OCR
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR config to focus on digits
    config = r'--psm 6 -c tessedit_char_whitelist=012'

    # Get OCR data
    data = pytesseract.image_to_data(binary, config=config, output_type=pytesseract.Output.DICT)

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text in {"0", "1", "2"}:
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return output



def find_handwritten_os(image):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(image)

    output = image.copy()  # always a valid image
    found = False

    for (bbox, text, conf) in results:
        if text.strip().lower() == 'o':
            found = True
            (tl, tr, br, bl) = bbox
            tl = tuple(map(int, tl))
            br = tuple(map(int, br))
            cv2.rectangle(output, tl, br, (0, 0, 255), 2)
            cv2.putText(output, f"'{text}'", tl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    if not found:
        print("⚠️ No handwritten 'o' found, but image is valid.")

    return output




#########################################################################
def select_and_process_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return

    image = cv2.imread(path)
    if image is None:
        print("❌ Failed to load image.")
        return

    processed = find_handwritten_os(image)  # <- assumes the debug version returns 2 values
    
    if processed is None or not isinstance(processed, np.ndarray):
        print("⚠️ Processed image is invalid.")
        return


    img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((1000, 700))
    img_tk = ImageTk.PhotoImage(img_pil)

    panel.config(image=img_tk)
    panel.image = img_tk




# GUI setup
root = tk.Tk()
root.title("🧾 Detect Circled Scores")
root.geometry("1050x800")

btn = tk.Button(root, text="📷 Load Image", command=select_and_process_image, font=("Arial", 14))
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

root.mainloop()
