CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name TEXT,
    teacher_name TEXT,
    grade TEXT
);

CREATE TABLE IF NOT EXISTS point_sheets (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id TEXT,
    date TEXT,
    
    -- Score columns
    win_positive INTEGER,
    win_respectful INTEGER,
    win_responsible INTEGER,
    win_safe INTEGER,
    am_recess_positive INTEGER,
    am_recess_respectful INTEGER,
    am_recess_responsible INTEGER,
    am_recess_safe INTEGER,
    lunch_positive INTEGER,
    lunch_respectful INTEGER,
    lunch_responsible INTEGER,
    lunch_safe INTEGER,
    pm_recess_positive INTEGER,
    pm_recess_respectful INTEGER,
    pm_recess_responsible INTEGER,
    pm_recess_safe INTEGER,
    end_day_positive INTEGER,
    end_day_respectful INTEGER,
    end_day_responsible INTEGER,
    end_day_safe INTEGER,
    specials_positive INTEGER,
    specials_respectful INTEGER,
    specials_responsible INTEGER,
    specials_safe INTEGER,
    FOREIGN KEY(student_id) REFERENCES students(student_id)
);
