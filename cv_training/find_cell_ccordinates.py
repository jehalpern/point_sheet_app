# === analyze_cells_gui.py ===
import os
import cv2
import numpy as np
import csv
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk

# === CONFIG ===
EXPECTED_ROWS = 4
EXPECTED_COLS = 6
THRESHOLD_Y = 25
MAX_DISPLAY_WIDTH = 1000
CSV_OUTPUT = "cell_features.csv"

# === ArUco Crop & Deskew ===
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
        return None

    ids = ids.flatten()
    marker_centers = {ids[i]: np.mean(corners[i][0], axis=0) for i in range(len(ids))}
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
            return None

    pts_src = np.array([inferred[0], inferred[1], inferred[3], inferred[2]], dtype="float32")
    h = target_height
    w = int(target_height * aspect_ratio)
    pts_dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    deskewed = deskew_image(warped)
    return deskewed

# === Detect and spatially sort cell boxes ===
def detect_and_sort_cells(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    # Group by rows and then columns
    deduped.sort(key=lambda b: b[1])  # sort by y
    rows = []
    for box in deduped:
        placed = False
        for row in rows:
            if abs(row[0][1] - box[1]) < THRESHOLD_Y:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    for row in rows:
        row.sort(key=lambda b: b[0])  # sort by x within row

    sorted_cells = [box for row in rows for box in row]
    return sorted_cells

# === GUI ===
def run_gui():
    def process_folder():
        folder = filedialog.askdirectory()
        if not folder:
            return

        data = [["filename", "cell_index", "x", "y", "w", "h", "aspect_ratio"]]
        for img_path in Path(folder).glob("*.jpg"):
            img = cv2.imread(str(img_path))
            cropped = crop_to_aruco_box_with_perspective_and_deskew(img)
            if cropped is None:
                continue
            sorted_boxes = detect_and_sort_cells(cropped)
            for i, (x, y, w, h) in enumerate(sorted_boxes):
                ar = round(w / h, 4) if h else 0
                data.append([img_path.name, i, x, y, w, h, ar])

        with open(CSV_OUTPUT, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f"✅ CSV saved to {CSV_OUTPUT}")

    root = tk.Tk()
    root.title("Export Detected Cells (Spatial Order)")
    tk.Button(root, text="Select Folder and Export CSV", font=("Arial", 16), command=process_folder).pack(padx=20, pady=40)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
