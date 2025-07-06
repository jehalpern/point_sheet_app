from db import get_connection

def create_student(student_id, name, teacher, grade):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO students (student_id, name, teacher_name, grade) VALUES (?, ?, ?, ?)",
        (student_id, name, teacher, grade)
    )
    conn.commit()
    conn.close()

def get_all_students():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students")
    results = cur.fetchall()
    conn.close()
    return results
