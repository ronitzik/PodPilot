import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()
DB_PATH = "podcasts.db"

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  
    return conn

def initialize_database():
    """Creates a table to store podcast metadata and embeddings."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS podcasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            podcast_id TEXT UNIQUE,
            title TEXT NOT NULL,
            genre TEXT NOT NULL,
            description TEXT,
            embedding BLOB,
            duration_ms INTEGER
        )
    ''')
    conn.commit()
    conn.close()

initialize_database()
