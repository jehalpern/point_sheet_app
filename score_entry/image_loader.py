import cv2
from tkinter import filedialog

def load_image_via_dialog():
    """
    Opens a file dialog to select an image file and loads it using OpenCV.
    Returns the OpenCV image and file path.
    """
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not file_path:
        return None, None

    image = cv2.imread(file_path)
    if image is None:
        raise ValueError("Could not load image from file.")

    return image, file_path
