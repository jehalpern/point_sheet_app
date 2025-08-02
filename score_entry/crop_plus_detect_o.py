import cv2
import numpy as np
import pytesseract
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def crop_to_aruco_box_with_perspective_and_deskew(image, padding=20, target_height=600, aspect_ratio=3.78):
    def deskew_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        angles = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle = (theta - np.pi / 2) * 180 / np.pi
                if -45 < angle < 45:
                    angles.append(angle)
        if not angles:
            return img
        mean_angle = np.mean(angles)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 2:
        print("❌ Not enough ArUco markers detected.")
        return None

    ids = ids.flatten()
    marker_centers = {}
    for i, marker_id in enumerate(ids):
        center = np.mean(corners[i][0], axis=0)
        marker_centers[marker_id] = center

    inferred = marker_centers.copy()
    if len(inferred) < 4:
        if 0 in inferred and 1 in inferred and 2 in inferred:
            inferred[3] = inferred[2] + (inferred[1] - inferred[0])
        elif 0 in inferred and 1 in inferred and 3 in inferred:
            inferred[2] = inferred[3] - (inferred[1] - inferred[0])
        elif 0 in inferred and 2 in inferred and 3 in inferred:
            inferred[1] = inferred[3] - (inferred[2] - inferred[0])
        elif 1 in inferred and 2 in inferred and 3 in inferred:
            inferred[0] = inferred[2] - (inferred[3] - inferred[1])
        else:
            print("⚠️ Not enough markers to reliably infer corners.")
            return None

    pts_src = np.array([inferred[0], inferred[1], inferred[3], inferred[2]], dtype="float32")

    h = target_height
    w = int(target_height * aspect_ratio)
    pts_dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    deskewed = deskew_image(warped)

    return deskewed

def detect_os_with_tesseract(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=30)

    config = r'--psm 11 -c tessedit_char_whitelist=O,o'
    data = pytesseract.image_to_data(gray, config=config, output_type=pytesseract.Output.DICT)

    output = cv2.resize(image.copy(), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    o_count = 0

    for i in range(len(data["text"])):
        text = data["text"][i].strip().lower()
        if text.startswith('o'):
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            num_chars = len(text)
            box_width = w // num_chars if num_chars > 0 else w

            for j in range(num_chars):
                cx = x + j * box_width
                cv2.rectangle(output, (cx, y), (cx + box_width, y + h), (0, 0, 255), 2)
                cv2.putText(output, 'o', (cx, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
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

    cropped = crop_to_aruco_box_with_perspective_and_deskew(image)

    if cropped is not None:
        processed = detect_os_with_tesseract(cropped)
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((800, 600))
        img_tk = ImageTk.PhotoImage(img_pil)

        panel.config(image=img_tk)
        panel.image = img_tk
    else:
        print("⚠️ Cropping failed or markers not found.")

# === GUI setup ===
root = tk.Tk()
root.title("🧾 ArUco Crop + Tesseract O Detector")
root.geometry("850x700")

btn = tk.Button(root, text="📷 Load Score Sheet Image", command=select_and_process_image, font=("Arial", 14))
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

root.mainloop()