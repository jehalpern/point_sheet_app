import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from PIL import Image, ImageTk

# === CONFIG ===
expected_rows = 6
expected_cols = 4
threshold_y = 25
LABEL_DIRS = ['0', '1', '2']
for d in LABEL_DIRS:
    os.makedirs(d, exist_ok=True)

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
            return None

    pts_src = np.array([inferred[0], inferred[1], inferred[3], inferred[2]], dtype="float32")
    h = target_height
    w = int(target_height * aspect_ratio)
    pts_dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    deskewed = deskew_image(warped)

    return deskewed

def detect_score_cells(image):
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
            if 30000 <= area <= 50000:
                raw_boxes.append((x, y, w, h))

    # Deduplicate using IoU
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

    return deduped

def extract_cells(image, boxes):
    # Use same logic to group into 4x6 grid
    deduped = boxes.copy()
    deduped.sort(key=lambda b: b[1])
    rows = []
    for box in deduped:
        x, y, w, h = box
        placed = False
        for row in rows:
            if abs(row[0][1] - y) < threshold_y:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    for row in rows:
        row.sort(key=lambda b: b[0])

    filled_rows = []
    for row in rows:
        if len(row) == expected_cols:
            filled_rows.append(row)
            continue
        avg_w = int(np.mean([b[2] for b in row]))
        y_mean = int(np.mean([b[1] for b in row]))
        x_min = min([b[0] for b in row])
        x_max = max([b[0] + b[2] for b in row])
        expected_xs = np.linspace(x_min, x_max - avg_w, expected_cols)
        filled_row = []
        for expected_x in expected_xs:
            match = None
            for b in row:
                if abs(b[0] - expected_x) < avg_w // 2:
                    match = b
                    break
            if match:
                filled_row.append(match)
            else:
                filled_row.append((int(expected_x), y_mean, avg_w, row[0][3]))
        filled_rows.append(filled_row)

    if len(filled_rows) < expected_rows:
        avg_h = int(np.mean([b[3] for row in filled_rows for b in row]))
        row_ys = [np.mean([b[1] for b in row]) for row in filled_rows]
        y_min = min(row_ys)
        y_max = max(row_ys)
        expected_ys = np.linspace(y_min, y_max, expected_rows)
        complete_grid = []
        for expected_y in expected_ys:
            match_row = None
            for row in filled_rows:
                row_y = np.mean([b[1] for b in row])
                if abs(row_y - expected_y) < avg_h // 2:
                    match_row = row
                    break
            if match_row:
                complete_grid.append(match_row)
            else:
                x_template = [b[0] for b in filled_rows[0]]
                w_template = filled_rows[0][0][2]
                h_template = avg_h
                inferred_row = [(x, int(expected_y), w_template, h_template) for x in x_template]
                complete_grid.append(inferred_row)
    else:
        complete_grid = filled_rows

    cells = []
    for row in complete_grid:
        for (x, y, w, h) in row:
            cell = image[y:y+h, x:x+w]
            cells.append(cell)
    return cells

class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Label Training Cells")
        self.img_index = 0
        self.cell_index = 0
        self.cells = []
        self.image_files = []

        self.preview = tk.Label(root)
        self.preview.pack()

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        for label in ['0', '1', '2', 'Discard']:
            tk.Button(
                btn_frame, text=label, font=("Arial", 20), width=6,
                command=lambda l=label: self.save_label(l)
            ).pack(side=tk.LEFT, padx=5)

        tk.Button(root, text="Load Image Folder", font=("Arial", 14), command=self.load_folder).pack(pady=5)

    def load_folder(self):
        folder = filedialog.askdirectory()
        self.image_files = [os.path.join(folder, f)
                            for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))]
        self.img_index = 0
        self.load_next_image()

    def load_next_image(self):
        if self.img_index >= len(self.image_files):
            self.preview.config(image="", text="Done!")
            return

        path = self.image_files[self.img_index]
        image = cv2.imread(path)
        cropped = crop_to_aruco_box_with_perspective_and_deskew(image)
        if cropped is None:
            self.img_index += 1
            self.load_next_image()
            return

        boxes = detect_score_cells(cropped)
        self.cells = extract_cells(cropped, boxes)
        self.cell_index = 0
        self.show_cell()

    def show_cell(self):
        if self.cell_index >= len(self.cells):
            self.img_index += 1
            self.load_next_image()
            return

        cell = self.cells[self.cell_index]
        img = cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((300, 300))
        self.tkimg = ImageTk.PhotoImage(img)
        self.preview.config(image=self.tkimg)

    def save_label(self, label):
        if label in ['0', '1', '2']:
            folder = Path(label)
            count = len(list(folder.glob("*.png")))
            fname = folder / f"{label}_{count:04d}.png"
            cv2.imwrite(str(fname), self.cells[self.cell_index])
        self.cell_index += 1
        self.show_cell()

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelingApp(root)
    root.mainloop()
