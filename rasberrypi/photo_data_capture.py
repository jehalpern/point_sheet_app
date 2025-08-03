import os
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# === CONFIG ===
PHOTO_DIR = Path("photo_stills")
PHOTO_DIR.mkdir(exist_ok=True)

# Get the next available handw_### filename
def get_next_filename():
    existing = list(PHOTO_DIR.glob("handw_*.jpg"))
    numbers = []
    for file in existing:
        try:
            num = int(file.stem.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue
    next_num = max(numbers, default=0) + 1
    return PHOTO_DIR / f"handw_{next_num:03d}.jpg"

# Capture image using libcamera-still
def take_photo():
    global photo_preview

    save_path = get_next_filename()
    cmd = [
        'libcamera-still',
        '--lens-position', '3.5',
        '-o', str(save_path),
        '--timeout', '1000',
        '--nopreview'  # Prevent native preview from stealing focus
    ]
    subprocess.run(cmd)

    # Show preview
    try:
        img = Image.open(save_path)
        img.thumbnail((400, 400))
        img_tk = ImageTk.PhotoImage(img)
        photo_preview.config(image=img_tk)
        photo_preview.image = img_tk
        messagebox.showinfo("Photo Captured", f"Saved to {save_path.name}")
    except Exception as e:
        messagebox.showerror("Error", f"Could not preview image:\n{e}")

# === GUI SETUP ===
root = tk.Tk()
root.title("Capture Photo")
root.geometry("500x600")
root.configure(bg="#f0f0f0")

# Label for preview
photo_preview = tk.Label(root, text="Preview will appear here", bg="#dcdcdc", width=400, height=300)
photo_preview.pack(pady=20)

# Styled button
take_btn = tk.Button(
    root,
    text="📸 Take Photo",
    font=("Arial", 24, "bold"),
    bg="#4CAF50",
    fg="white",
    padx=20,
    pady=10,
    command=take_photo
)
take_btn.pack(pady=20)

root.mainloop()
