import os
import sqlite3

class Database:
    def __init__(self, db_file='sentiment_data.db'):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            # Drop the table if it exists to recreate it with the correct columns
            #self.conn.execute("DROP TABLE IF EXISTS sentiment_data")  -- linha comentada para iniciar os registros fora do teste
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    model_name TEXT NOT NULL, 
                    pre_sentiment_score REAL,
                    post_sentiment_score REAL,
                    pre_touchpoint TEXT,
                    post_touchpoint TEXT,
                    pre_sentiment_category TEXT,  -- Adicione essa linha
                    post_sentiment_category TEXT, -- Adicione essa linha
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    # Now, accept model_name as a parameter
    def insert_sentiment_data(self, text, model_name, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint, pre_sentiment_category, post_sentiment_category):
        with self.conn:
            self.conn.execute(
                "INSERT INTO sentiment_data (text, model_name, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint, pre_sentiment_category, post_sentiment_category) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (text, model_name, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint, pre_sentiment_category, post_sentiment_category)
            )
 
    def get_recent_sentiment_data(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT text, model_name, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint, pre_sentiment_category, post_sentiment_category FROM sentiment_data ORDER BY created_at DESC"
        )
        return cursor.fetchall()


    def get_historical_sentiment_data(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT created_at, model_name, pre_sentiment_score, post_sentiment_score, pre_touchpoint, post_touchpoint FROM sentiment_data ORDER BY created_at"
        )
        return cursor.fetchall()

    def close(self):
        self.conn.close()
