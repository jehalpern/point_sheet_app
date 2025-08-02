import cv2
import pytesseract
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def detect_os_with_tesseract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=30)

    config = r'--psm 6 -c tessedit_char_whitelist=O,o'
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)

    # Match output size to preprocessed image
    output = cv2.resize(image.copy(), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    o_count = 0

    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if text.lower() == 'o':
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(output, f"{text}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            o_count += 1

    print(f"✅ Found {o_count} 'o'(s) using pytesseract.")
    return output


def select_and_process_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return

    image = cv2.imread(path)
    if image is None:
        print("❌ Failed to load image.")
        return

    processed = detect_os_with_tesseract(image)

    if processed is None or not isinstance(processed, np.ndarray):
        print("⚠️ Processed image is invalid.")
        return

    img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((1000, 700))
    img_tk = ImageTk.PhotoImage(img_pil)

    panel.config(image=img_tk)
    panel.image = img_tk

# === GUI setup ===
root = tk.Tk()
root.title("🔎 Detect Handwritten 'o's with Tesseract")
root.geometry("1050x800")

btn = tk.Button(root, text="📷 Load Image", command=select_and_process_image, font=("Arial", 14))
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

root.mainloop()
