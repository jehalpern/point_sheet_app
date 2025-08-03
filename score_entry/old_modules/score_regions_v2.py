import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk



"""""
def crop_to_aruco_box(image, padding=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        print("❌ Not enough ArUco markers detected.")
        return None

    all_points = []
    for box in corners:
        all_points.extend(box[0])  # Add the four corners of each marker

    all_points = np.array(all_points)
    x_min = max(int(np.min(all_points[:, 0])) - padding, 0)
    y_min = max(int(np.min(all_points[:, 1])) - padding, 0)
    x_max = min(int(np.max(all_points[:, 0])) + padding, image.shape[1])
    y_max = min(int(np.max(all_points[:, 1])) + padding, image.shape[0])

    cropped = image[y_min:y_max, x_min:x_max]
    return cropped 
"""""
def crop_to_aruco_box_with_inference(image, padding=20):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())

    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None or len(ids) < 2:
        print("❌ Not enough markers detected.")
        return None

    ids = ids.flatten()
    marker_centers = {}
    for i, marker_id in enumerate(ids):
        center = np.mean(corners[i][0], axis=0)
        marker_centers[marker_id] = center

    # Known structure of markers (based on your layout)
    # Assume IDs:
    # 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
    expected_ids = [0, 1, 2, 3]

    if len(marker_centers) == 4:
        # All present: warp box directly
        pts = np.array([
            marker_centers[0],
            marker_centers[1],
            marker_centers[3],
            marker_centers[2]
        ], dtype="float32")

    else:
        print(f"🧠 Detected {len(marker_centers)} markers: {list(marker_centers.keys())}")
        print("➕ Inferring missing corners...")

        # Try to infer the missing corners
        inferred = marker_centers.copy()
        if 0 in inferred and 1 in inferred and 2 in inferred:
            # Infer bottom-right (3)
            vec_diag = inferred[1] - inferred[0]
            inferred[3] = inferred[2] + vec_diag
        elif 0 in inferred and 1 in inferred and 3 in inferred:
            # Infer bottom-left (2)
            vec_diag = inferred[1] - inferred[0]
            inferred[2] = inferred[3] - vec_diag
        elif 0 in inferred and 2 in inferred and 3 in inferred:
            # Infer top-right (1)
            vec_diag = inferred[2] - inferred[0]
            inferred[1] = inferred[3] - vec_diag
        elif 1 in inferred and 2 in inferred and 3 in inferred:
            # Infer top-left (0)
            vec_diag = inferred[3] - inferred[1]
            inferred[0] = inferred[2] - vec_diag
        else:
            print("⚠️ Not enough markers to reliably infer.")
            return None

        pts = np.array([
            inferred[0],
            inferred[1],
            inferred[3],
            inferred[2]
        ], dtype="float32")

    # Compute bounding box from inferred rectangle
    x_min = max(int(min(p[0] for p in pts)) - padding, 0)
    y_min = max(int(min(p[1] for p in pts)) - padding, 0)
    x_max = min(int(max(p[0] for p in pts)) + padding, image.shape[1])
    y_max = min(int(max(p[1] for p in pts)) + padding, image.shape[0])

    cropped = image[y_min:y_max, x_min:x_max]
    return cropped



def crop_to_aruco_box_with_perspective(image, padding=20, target_height=600, aspect_ratio=3.78):
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

    # Reference: assume marker layout:
    # 0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right
    inferred = marker_centers.copy()

    # Try to infer missing corners
    if len(marker_centers) == 4:
        pass
    elif 0 in inferred and 1 in inferred and 2 in inferred:
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

    # Define source points: top-left, top-right, bottom-right, bottom-left
    pts_src = np.array([
        inferred[0],
        inferred[1],
        inferred[3],
        inferred[2]
    ], dtype="float32")

    # Compute output size based on desired height and aspect ratio
    h = target_height
    w = int(target_height * aspect_ratio)
    pts_dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    # Perspective transform
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    return warped

def crop_to_aruco_box_with_perspective_and_deskew(image, padding=20, target_height=600, aspect_ratio=3.78):
    import cv2
    import numpy as np

    def deskew_image(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
        angles = []
        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle = (theta - np.pi / 2) * 180 / np.pi
                if -45 < angle < 45:  # ignore vertical lines
                    angles.append(angle)
        if not angles:
            return img  # fallback
        mean_angle = np.mean(angles)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, mean_angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # ArUco detection
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

    # Infer missing corners
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

    pts_src = np.array([
        inferred[0],
        inferred[1],
        inferred[3],
        inferred[2]
    ], dtype="float32")

    h = target_height
    w = int(target_height * aspect_ratio)
    pts_dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(image, M, (w, h))

    # Automatically deskew the warped image
    deskewed = deskew_image(warped)

    return deskewed


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
        img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((800, 600))
        img_tk = ImageTk.PhotoImage(img_pil)

        panel.config(image=img_tk)
        panel.image = img_tk
    else:
        print("⚠️ Cropping failed or markers not found.")

# === GUI setup ===
root = tk.Tk()
root.title("🧾 ArUco Crop Only (No Warp)")
root.geometry("850x700")

btn = tk.Button(root, text="📷 Load Score Sheet Image", command=select_and_process_image, font=("Arial", 14))
btn.pack(pady=10)

panel = tk.Label(root)
panel.pack()

root.mainloop()
