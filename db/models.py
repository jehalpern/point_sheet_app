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

def insert_point_sheet_record(student_id, date, **scores):
    conn = get_connection()
    cur = conn.cursor()

    columns = [
        "win_positive", "win_respectful", "win_responsible", "win_safe",
        "am_recess_positive", "am_recess_respectful", "am_recess_responsible", "am_recess_safe",
        "lunch_positive", "lunch_respectful", "lunch_responsible", "lunch_safe",
        "pm_recess_positive", "pm_recess_respectful", "pm_recess_responsible", "pm_recess_safe",
        "end_day_positive", "end_day_respectful", "end_day_responsible", "end_day_safe",
        "specials_positive", "specials_respectful", "specials_responsible", "specials_safe"
    ]

    values = [scores.get(col, 0) for col in columns]

    query = f"""
        INSERT INTO point_sheets (
            student_id, date,
            {', '.join(columns)}
        ) VALUES (
            ?, ?, {', '.join(['?'] * len(columns))}
        )
    """
    cur.execute(query, (student_id, date, *values))
    conn.commit()
    conn.close()

def get_point_sheets_by_student(student_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM point_sheets
        WHERE student_id = ?
        ORDER BY date DESC
    """, (student_id,))
    results = cur.fetchall()
    conn.close()
    return results

def student_exists(student_id):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists
