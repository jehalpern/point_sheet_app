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

def detect_score_cells(image):
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            area = cv2.contourArea(cnt)
            if 300 < area < 3000 and 0.5 < aspect_ratio < 2.5:
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
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        roi = cv2.resize(roi, (w * 3, h * 3), interpolation=cv2.INTER_LINEAR)
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        _, roi_thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = r'--psm 10 -c tessedit_char_whitelist=012'
        digit = pytesseract.image_to_string(roi_thresh, config=config).strip()
        if digit:
            cv2.putText(output, digit[0], (x + 3, y + h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"Box at ({x}, {y}) — Width: {w}, Height: {h}, Area: {w*h} Digit: {digit}")

    print(f"✅ Final deduplicated box count: {len(deduped)}")
    deduped.sort(key=lambda b: (b[1], b[0]))
    clusters = []
    used_indices = set()
    
    MAX_WIDTH = 250  # <-- adjust this to your limit

    for i in range(len(deduped) - 2):
        if i in used_indices:
            continue
        x0, y0, w0, h0 = deduped[i]
        x1, y1, w1, h1 = deduped[i+1]
        x2, y2, w2, h2 = deduped[i+2]
        if abs(y0 - y1) < 15 and abs(y1 - y2) < 15:
            x_min = min(x0, x1, x2)
            y_min = min(y0, y1, y2)
            x_max = max(x0 + w0, x1 + w1, x2 + w2)
            y_max = max(y0 + h0, y1 + h1, y2 + h2)
            cluster_width = x_max - x_min
            if cluster_width > MAX_WIDTH:
                continue
            width = cluster_width
            height = y_max - y_min
            area = width * height
            clusters.append((x_min, y_min, width, height))
            used_indices.update({i, i+1, i+2})



    single_cells = [box for idx, box in enumerate(deduped) if idx not in used_indices]

    for (x, y, w, h) in single_cells:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)

    for (x, y, w, h) in clusters:
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

    print(f"🟩 Single cell boxes: {len(single_cells)}")
    print(f"🔵 Clustered 3-cell boxes: {len(clusters)}")
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
        processed = detect_score_cells(cropped)
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
root.title("🧾 ArUco Crop + Score Cluster Detector")
root.geometry("850x700")

btn = tk.Button(root, text="📷 Load Score Sheet Image", command=select_and_process_image, font=("Arial", 14))
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

root.mainloop()
