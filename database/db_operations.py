import os
import sqlite3

class Database:
    def __init__(self, db_file='sentiment_data.db'):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    sentiment_score REAL NOT NULL,
                    touchpoint TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def insert_sentiment_data(self, text, sentiment_score, touchpoint):
        with self.conn:
            self.conn.execute(
                "INSERT INTO sentiment_data (text, sentiment_score, touchpoint) VALUES (?, ?, ?)",
                (text, sentiment_score, touchpoint)
            )

    def get_recent_sentiment_data(self, limit=10):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT text, sentiment_score, touchpoint FROM sentiment_data ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()

    def get_historical_sentiment_data(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT created_at, sentiment_score, touchpoint FROM sentiment_data ORDER BY created_at"
        )
        return cursor.fetchall()

    def close(self):
        self.conn.close()
