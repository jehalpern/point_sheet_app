import tkinter as tk
from image_loader import load_image_via_dialog
import cv2

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.equalizeHist(blurred)
    return enhanced

def read_qr_code_with_fallback(image):
    detector = cv2.QRCodeDetector()

    # 1. Try full-image decode first
    data, _, _ = detector.detectAndDecode(image)
    if data:
        print("✅ Decoded from full image")
        return data

    # 2. Fallback: detect + crop + resize
    retval, points = detector.detect(image)
    if retval and points is not None:
        points = points[0]
        x_min = int(min(points[:, 0])) - 10
        x_max = int(max(points[:, 0])) + 10
        y_min = int(min(points[:, 1])) - 10
        y_max = int(max(points[:, 1])) + 10
        h, w = image.shape[:2]
        x_min = max(x_min, 0)
        x_max = min(x_max, w)
        y_min = max(y_min, 0)
        y_max = min(y_max, h)

        cropped = image[y_min:y_max, x_min:x_max]
        scaled = cv2.resize(cropped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

        data, _, _ = detector.detectAndDecode(scaled)
        if data:
            print("✅ Decoded from cropped QR region")
            return data
        else:
            print("⚠️ QR code region found but not decoded.")
    else:
        print("❌ No QR code detected.")

    return None


def main():
    root = tk.Tk()
    root.withdraw()

    image, path = load_image_via_dialog()
    if image is not None:
        data = read_qr_code_with_fallback(image)
        if data:
            print(f"Decoded QR Data: {data}")
        else:
            print("QR code not found in image.")
    else:
        print("No image selected.")

if __name__ == "__main__":
    main()
