import os
import sqlite3
import pandas as pd

def read_sqlite_preview(db_path: str, nrows: int = 5) -> dict:
    """
    Connects to SQLite DB and returns first nrows of each table (excluding sqlite_internal tables).
    Returns {"success": bool, "content": str or error message}
    """
    if not os.path.exists(db_path):
        return {"success": False, "error": f"Database file not found: {db_path}"}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get list of normal tables only
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = [row[0] for row in cursor.fetchall()]
        
        previews = []
        for table in tables:
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT {nrows};", conn)
            previews.append(f"--- Table: {table} ---\n{df.to_csv(index=False)}")

        conn.close()
        return {"success": True, "content": "\n".join(previews)}

    except Exception as e:
        return {"success": False, "error": str(e)}
