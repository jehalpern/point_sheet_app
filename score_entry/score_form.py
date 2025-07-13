import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
from db.models import insert_point_sheet_record, student_exists
import os
import uuid

class ScoreEntryApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Score Sheet Entry")
        self.geometry("900x600")
        self.image_path = None
        self.behaviors = ["Positive", "Respectful", "Responsible", "Safe"]
        self.times = ["WIN", "AM Recess", "Lunch", "PM Recess", "End Day", "Specials"]
        self.scores = {}  # {(behavior, time): score}
        self.create_widgets()

    def create_widgets(self):
        control_frame = ttk.Frame(self)
        control_frame.pack(fill="x", padx=10, pady=10)

        self.student_id_entry = ttk.Entry(control_frame, width=20)
        self.student_id_entry.grid(row=0, column=1, padx=5)
        ttk.Label(control_frame, text="Student ID:").grid(row=0, column=0, padx=5)

        self.date_entry = ttk.Entry(control_frame, width=20)
        self.date_entry.grid(row=0, column=3, padx=5)
        ttk.Label(control_frame, text="Date (YYYY-MM-DD):").grid(row=0, column=2, padx=5)

        ttk.Button(control_frame, text="Upload Image", command=self.load_image).grid(row=0, column=4, padx=5)
        ttk.Button(control_frame, text="Submit to DB", command=self.submit_scores).grid(row=0, column=5, padx=5)

        self.image_panel = ttk.Label(self)
        self.image_panel.pack(pady=10)

        self.table_frame = ttk.Frame(self)
        self.table_frame.pack()
        self.create_score_grid()

    def create_score_grid(self):
        for i, time in enumerate(["Behavior"] + self.times):
            ttk.Label(self.table_frame, text=time, relief="ridge", width=12).grid(row=0, column=i, sticky="nsew")

        for r, behavior in enumerate(self.behaviors):
            ttk.Label(self.table_frame, text=behavior, relief="ridge", width=12).grid(row=r+1, column=0, sticky="nsew")
            for c, time in enumerate(self.times):
                var = tk.StringVar()
                combo = ttk.Combobox(self.table_frame, textvariable=var, values=["0", "1", "2"], width=5)
                combo.grid(row=r+1, column=c+1, padx=2, pady=2)
                combo.set("2")
                self.scores[(behavior, time)] = var

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not file_path:
            return
        self.image_path = file_path
        img = Image.open(file_path)
        img.thumbnail((600, 400))
        self.img_tk = ImageTk.PhotoImage(img)
        self.image_panel.configure(image=self.img_tk)



    def submit_scores(self):
        student_id = self.student_id_entry.get().strip()
        date = self.date_entry.get().strip()

        if not student_id or not date:
            messagebox.showwarning("Missing Info", "Please enter both student ID and date.")
            return
        
        # Check if student exists
        if not student_exists(student_id):
            messagebox.showerror("Student Not Found", f"Student ID '{student_id}' does not exist in the database.")
            return
        
        score_dict = {}
        for (behavior, time), var in self.scores.items():
            key = f"{time.lower().replace(' ', '_')}_{behavior.lower()}"
            score_dict[key] = int(var.get())

        insert_point_sheet_record(student_id, date, **score_dict)
        messagebox.showinfo("Success", "Scores saved to database.")

        if self.image_path:
            save_dir = os.path.join("data", "images", student_id)
            os.makedirs(save_dir, exist_ok=True)
            unique_name = f"{date}_{uuid.uuid4().hex[:6]}.jpg"
            target_path = os.path.join(save_dir, unique_name)
            Image.open(self.image_path).save(target_path)
            print(f"Image saved to: {target_path}")

if __name__ == "__main__":
    app = ScoreEntryApp()
    app.mainloop()
