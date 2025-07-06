import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "pointsheet.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn
