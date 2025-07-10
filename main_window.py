import tkinter as tk
from tkinter import ttk, messagebox
from db.models import create_student, get_all_students
import uuid

class StudentManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Student Management")
        self.geometry("600x400")
        self.create_widgets()
        self.refresh_student_list()

    def create_widgets(self):
        # Input form
        form_frame = ttk.LabelFrame(self, text="Add Student")
        form_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(form_frame, text="Name:").grid(row=0, column=0, padx=5, pady=5)
        self.name_entry = ttk.Entry(form_frame)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(form_frame, text="Teacher:").grid(row=0, column=2, padx=5, pady=5)
        self.teacher_entry = ttk.Entry(form_frame)
        self.teacher_entry.grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(form_frame, text="Grade:").grid(row=0, column=4, padx=5, pady=5)
        self.grade_entry = ttk.Entry(form_frame)
        self.grade_entry.grid(row=0, column=5, padx=5, pady=5)

        add_button = ttk.Button(form_frame, text="Add Student", command=self.add_student)
        add_button.grid(row=0, column=6, padx=10)

        # Student list
        list_frame = ttk.LabelFrame(self, text="All Students")
        list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.tree = ttk.Treeview(list_frame, columns=("ID", "Name", "Teacher", "Grade"), show="headings")
        for col in ("ID", "Name", "Teacher", "Grade"):
            self.tree.heading(col, text=col)
            self.tree.column(col, stretch=True)
        self.tree.pack(fill="both", expand=True)

    def add_student(self):
        name = self.name_entry.get().strip()
        teacher = self.teacher_entry.get().strip()
        grade = self.grade_entry.get().strip()

        if not name or not teacher or not grade:
            messagebox.showwarning("Missing Info", "Please fill out all fields.")
            return

        student_id = str(uuid.uuid4())[:8]
        create_student(student_id, name, teacher, grade)
        self.refresh_student_list()

        # Clear inputs
        self.name_entry.delete(0, tk.END)
        self.teacher_entry.delete(0, tk.END)
        self.grade_entry.delete(0, tk.END)

    def refresh_student_list(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for student in get_all_students():
            self.tree.insert('', 'end', values=student)

if __name__ == "__main__":
    app = StudentManager()
    app.mainloop()
