import sqlite3
from db import get_connection

def initialize_db():
    with open("db/schema.sql") as f:
        schema = f.read()
    conn = get_connection()
    conn.executescript(schema)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    initialize_db()
