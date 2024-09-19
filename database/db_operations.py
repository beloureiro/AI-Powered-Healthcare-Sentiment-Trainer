import os
import sqlite3

class Database:
    def __init__(self, db_file='sentiment_data.db'):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            # Drop the table if it exists to recreate it with the correct columns
            self.conn.execute("DROP TABLE IF EXISTS sentiment_data")
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    pre_sentiment_score REAL,
                    post_sentiment_score REAL,
                    pre_touchpoint TEXT,
                    post_touchpoint TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def insert_sentiment_data(self, text, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint):
        with self.conn:
            self.conn.execute(
                "INSERT INTO sentiment_data (text, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint) VALUES (?, ?, ?, ?, ?)",
                (text, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint)
            )

    def get_recent_sentiment_data(self, limit=10):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT text, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint FROM sentiment_data ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return cursor.fetchall()

    # Corrected function to fetch historical data
    def get_historical_sentiment_data(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT created_at, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint FROM sentiment_data ORDER BY created_at"
        )
        return cursor.fetchall()

    def close(self):
        self.conn.close()
