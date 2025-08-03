import cv2
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from datetime import datetime

# === CONFIG ===
SAVE_DIR = "captured_images"
FOCUS_VALUE = 35  # 3.5 mapped to 35 for V4L2 (0–255 scale)

# Create folder if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

# === CAMERA SETUP ===
cap = cv2.VideoCapture(0)  # use 0 for default camera

# Set manual focus (if supported)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, FOCUS_VALUE)

# === GUI SETUP ===
window = tk.Tk()
window.title("Point Sheet Image Capture")

# Live preview label
preview_label = tk.Label(window)
preview_label.pack()

# Function to update live feed
def update_frame():
    ret, frame = cap.read()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        preview_label.imgtk = imgtk
        preview_label.configure(image=imgtk)
    window.after(10, update_frame)

# Save image to disk
def take_photo():
    ret, frame = cap.read()
    if ret:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.png"
        path = os.path.join(SAVE_DIR, filename)
        cv2.imwrite(path, frame)
        messagebox.showinfo("Photo Saved", f"Saved as {filename}")
    else:
        messagebox.showerror("Error", "Failed to capture image")

# Take Photo Button
btn = tk.Button(window, text="📸 Take Photo", font=("Arial", 24), bg="green", fg="white", command=take_photo)
btn.pack(pady=10)

# Start the loop
update_frame()
window.mainloop()

# Release on exit
cap.release()
cv2.destroyAllWindows()
