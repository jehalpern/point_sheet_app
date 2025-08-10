import cv2
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


MODEL_PATH = "cnn_model.keras"  # Update this if needed

model = load_model(MODEL_PATH)

def preprocess_cell(cell_image, target_size=(384, 128)):
    resized = cv2.resize(cell_image, (target_size[1], target_size[0]))  # (width, height)
    normalized = resized / 255.0
    return normalized.reshape(1, *target_size, 3)




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
    return deskew_image(warped)

def detect_cells_and_classify(image):
    cropped = crop_to_aruco_box_with_perspective_and_deskew(image)
    if cropped is None:
        return None

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if 35000 <= area <= 45000:
                raw_boxes.append((x, y, w, h))

    def iou(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        return inter_area / union_area if union_area else 0

    deduped = []
    raw_boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    for box in raw_boxes:
        if all(iou(box, kept) < 0.4 for kept in deduped):
            deduped.append(box)

    for (x, y, w, h) in deduped:
        cell_img = cropped[y:y+h, x:x+w]
        input_tensor = preprocess_cell(cell_img)
        prediction = model.predict(input_tensor)
        label = str(np.argmax(prediction))

        cv2.rectangle(cropped, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(cropped, label, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return cropped

def select_and_process_image():
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not path:
        return

    image = cv2.imread(path)
    if image is None:
        print("❌ Failed to load image.")
        return

    processed = detect_cells_and_classify(image)
    if processed is not None:
        img_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((800, 600))
        img_tk = ImageTk.PhotoImage(img_pil)

        panel.config(image=img_tk)
        panel.image = img_tk
    else:
        print("⚠️ Cropping or detection failed.")

# === GUI setup ===
root = tk.Tk()
root.title("🧾 ArUco Crop + Score Classifier")
root.geometry("850x700")

btn = tk.Button(root, text="📷 Load Score Sheet Image", command=select_and_process_image, font=("Arial", 14))
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

root.mainloop()
