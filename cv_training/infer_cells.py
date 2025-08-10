# === infer_missing_cells_gui.py ===
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk

# === CONFIG ===
GRID_ROWS = 4
GRID_COLS = 6
CELL_COUNT = GRID_ROWS * GRID_COLS
THRESHOLD_Y = 25
MAX_DISPLAY_WIDTH = 1000

# === Crop and deskew using ArUco markers ===
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

# === Cell detection and spatial sorting ===
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

    # === Spatial indexing ===
    deduped.sort(key=lambda b: b[1])
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
        row.sort(key=lambda b: b[0])

    indexed = {}
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            idx = r * GRID_COLS + c
            try:
                indexed[idx] = rows[r][c]
            except IndexError:
                pass
    return indexed, iou

# === Inference logic ===
def infer_cell(index, detected, iou_fn):
    neighbors = []
    if index >= 6 and index - 6 in detected:
        neighbors.append(detected[index - 6])
    if index + 6 < CELL_COUNT and index + 6 in detected:
        neighbors.append(detected[index + 6])
    if index % 6 > 0 and index - 1 in detected:
        neighbors.append(detected[index - 1])
    if index % 6 < 5 and index + 1 in detected:
        neighbors.append(detected[index + 1])

    if neighbors:
        x = int(np.mean([n[0] for n in neighbors]))
        y = int(np.mean([n[1] for n in neighbors]))
        w = int(np.mean([n[2] for n in neighbors]))
        h = int(np.mean([n[3] for n in neighbors]))
        proposed = (x, y, w, h)

        for existing in detected.values():
            if iou_fn(proposed, existing) > 0.1:
                return None
        return proposed
    else:
        return None

# === GUI logic ===
def run_gui():
    def select_image():
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
        if not path:
            return
        img = cv2.imread(path)
        cropped = crop_to_aruco_box_with_perspective_and_deskew(img)
        if cropped is None:
            return

        detected, iou_fn = detect_and_sort_cells(cropped)
        inferred_grid = {}
        for i in range(CELL_COUNT):
            if i in detected:
                inferred_grid[i] = detected[i]
            else:
                inf = infer_cell(i, detected, iou_fn)
                if inf:
                    inferred_grid[i] = inf

        debug = cropped.copy()
        for i in range(CELL_COUNT):
            if i not in inferred_grid:
                continue
            x, y, w, h = inferred_grid[i]
            color = (255, 0, 0) if i in detected else (0, 255, 0)
            cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
            cv2.putText(debug, str(i), (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        scale = min(MAX_DISPLAY_WIDTH / debug.shape[1], 1.0)
        if scale < 1.0:
            debug = cv2.resize(debug, (int(debug.shape[1] * scale), int(debug.shape[0] * scale)))

        img_rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        panel.config(image=img_tk)
        panel.image = img_tk

    root = tk.Tk()
    root.title("Inferred Cell Debugger Using Neighbors")
    tk.Button(root, text="Select Image", font=("Arial", 16), command=select_image).pack(pady=10)
    global panel
    panel = tk.Label(root)
    panel.pack()
    root.mainloop()

if __name__ == "__main__":
    run_gui()