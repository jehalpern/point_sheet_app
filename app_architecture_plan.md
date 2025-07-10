# Point Sheet App Developer Documentation

## Overview
The Point Sheet App is designed to support the generation, capture, and reporting of daily behavior data using structured paper sheets scanned with computer vision. It serves teachers, students, and administrators in managing daily check-in/check-out (CICO) processes with a focus on modularity and expandability.

This document outlines the current architecture and planned features to guide developers contributing to the application.

---

## 📦 Current Architecture

### PDF Sheet Generator
- Implemented with ReportLab.
- Generates printable point sheets (1–4 per page).
- Designed for CV-readability with boxed scores (0–2), consistent layout, and alignment markers.

### Database
- SQLite database located in `/db/pointsheet.db`.
- Tables:
  - `students`: Stores student metadata (ID, name, teacher, grade).
  - `point_sheets`: Stores 28-point daily behavior score data per student.
- Initialization handled by `db/init_db.py` using `schema.sql`.

### CLI for Testing
- CLI script `cli_test.py` allows creating and listing students.
- Interfaces with `db/models.py` functions.

### Project Structure
```
point_sheet_app/
├── app.py              # Main application entry point
├── config.py           # Central configuration
├── main_window.py      # GUI controller (to be expanded)
├── db/                 # Database connection and schema
├── sheet_generator/    # PDF generation logic
├── utils/              # Helper functions, dialogs, logging
├── data/               # Sample CSVs and export files
├── assets/             # Fonts and logos
```

---

## 🧩 Planned Functionality & Architecture

### 1. User Login
- User roles: teachers, administrators.
- Authentication options:
  - Local password-based system.
  - Future option: OAuth (Google, school SSO).
- Table: `users (user_id, name, email, password_hash)`
- Login window to control access.

### 2. Sheet Generation (GUI-controlled)
- Allow users to select:
  - Student names or class roster.
  - Date or date range.
  - Layout (e.g., 1 or 4 sheets per page).
- Embed optional QR code for student/date identification.

### 3. Record Scores via Computer Vision
- Capture photo using Pi camera or upload image.
- Preprocessing with OpenCV:
  - Detect matrix/grid.
  - Segment score boxes.
  - Determine circled value using contour density or trained CNN.
- Save results to `point_sheets` table.

### 4. Manual Score Adjustment
- GUI table for reviewing/editing detected scores.
- Option to approve or override CV predictions.
- Edits saved directly to DB.

### 5. Student Score Summary Viewer
- Visual dashboard per student:
  - Daily scores and trendlines.
  - Filter by date, behavior.
- Charts with `matplotlib` or `plotly`.

### 6. Teacher Reports
- Generate:
  - Individual student reports
  - Class-level summary reports
- Export formats:
  - Password-protected PDF
  - XLSX or CSV
- Customizable filters: behavior categories, date range.

### 7. Score Sheet Archiving
- Save scanned image with metadata:
  - `data/images/<student_id>/<date>.jpg`
- Supports audit logging and model training dataset.

---

## 🔧 Additional Modules & Structure

### Expanded Database
- Add `users` table for login
- Add `scan_logs` for tracking image file metadata

### GUI Modules
- `login_window.py` – user login interface
- `student_form.py` – add/edit student profiles
- `score_editor.py` – score review/adjustment
- `report_viewer.py` – generate/export reports

### ML & CV Logic
- Folder: `cv_model/`
  - `detector.py` – OpenCV rule-based logic
  - `trainer.py` – CNN training script
  - `labeled_data/` – raw training examples

### API / Service Layer 
- `db_interface.py`: decouples GUI from direct DB access.
- Improves testability and scalability.

### Configuration & Environment
- `.env` for local settings and file paths
- `config.py` for shared constants across app

---

## 🔚 Conclusion
This architecture supports a modular, maintainable application with clear separation of concerns: GUI, database, computer vision, and reporting. Developers are encouraged to follow this structure and expand functionality as needed, while maintaining code separation and documentation for each module.

