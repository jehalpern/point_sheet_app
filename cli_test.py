# cli_test.py

from db.models import create_student, get_all_students
import uuid

def print_menu():
    print("\n=== Point Sheet DB CLI ===")
    print("1. Add student")
    print("2. List students")
    print("3. Exit")

def prompt_student_data():
    name = input("Enter student name: ")
    teacher = input("Enter teacher name: ")
    grade = input("Enter grade: ")
    student_id = str(uuid.uuid4())[:8]  # Short unique ID
    return student_id, name, teacher, grade

def main():
    while True:
        print_menu()
        choice = input("Choose an option: ").strip()

        if choice == "1":
            student_id, name, teacher, grade = prompt_student_data()
            create_student(student_id, name, teacher, grade)
            print(f"✅ Student '{name}' added with ID: {student_id}")
        elif choice == "2":
            students = get_all_students()
            if not students:
                print("No students found.")
            for s in students:
                print(f"{s[0]} | {s[1]} | Teacher: {s[2]} | Grade: {s[3]}")
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
