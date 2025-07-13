import cv2

def read_qr_code(image):
    """
    Attempts to detect and decode a QR code from the given OpenCV image.
    Returns the decoded data string if found, else None.
    """
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(image)

    if bbox is not None and data:
        print(f"QR Code data: {data}")
        return data
    else:
        print("No QR code detected.")
        return None
