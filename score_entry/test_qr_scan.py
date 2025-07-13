import tkinter as tk
from image_loader import load_image_via_dialog
import cv2

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    enhanced = cv2.equalizeHist(blurred)
    return enhanced

def read_qr_code_with_crop(image):
    enhanced = preprocess_image(image)
    detector = cv2.QRCodeDetector()
    retval, points = detector.detect(image)  # Use original image for detection

    if retval and points is not None:
        points = points[0]

        x_min = int(min(points[:, 0]))
        x_max = int(max(points[:, 0]))
        y_min = int(min(points[:, 1]))
        y_max = int(max(points[:, 1]))

        pad = 10
        h, w = image.shape[:2]
        x_min = max(x_min - pad, 0)
        x_max = min(x_max + pad, w)
        y_min = max(y_min - pad, 0)
        y_max = min(y_max + pad, h)

        cropped = image[y_min:y_max, x_min:x_max]

        # Resize cropped region to improve decoding resolution
        scaled = cv2.resize(cropped, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        data, _, _ = detector.detectAndDecode(scaled)

        if data:
            print(f"QR Code data: {data}")
            return data
        else:
            print("QR code region detected but not decoded.")
    else:
        print("No QR code detected.")

    return None

def main():
    root = tk.Tk()
    root.withdraw()

    image, path = load_image_via_dialog()
    if image is not None:
        data = read_qr_code_with_crop(image)
        if data:
            print(f"Decoded QR Data: {data}")
        else:
            print("QR code not found in image.")
    else:
        print("No image selected.")

if __name__ == "__main__":
    main()
