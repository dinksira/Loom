"""
loom/storage.py

SQLite storage for Loom. 
This module handles all interactions with the SQLite database to store documents, embeddings, and analysis results.
"""

import sqlite3
from pathlib import Path
from .models import Document, AnalysisResult

class Storage:
    """
    Handles database interactions for Loom.
    """

    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """Create necessary tables in the database if they don't exist."""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    path TEXT UNIQUE NOT NULL,
                    raw_text TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    clean_text TEXT,
                    tokens TEXT,  -- serialized list of tokens
                    word_count INTEGER,
                    cluster_id INTEGER
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY,
                    run_timestamp TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    orphan_threshold REAL NOT NULL,
                    num_clusters INTEGER NOT NULL
                )
            ''')
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY,
                    cluster_id INTEGER NOT NULL,
                    keywords TEXT,  -- serialized list of keywords
                    coherence_score REAL,
                    document_ids TEXT  -- serialized list of document IDs
                )
            ''')

    def insert_document(self, doc: Document):
        """Insert a new document into the documents table."""
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO documents (path, raw_text, content_hash, created_at, file_type, 
                                                  clean_text, tokens, word_count, cluster_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                (str(doc.path), doc.raw_text, doc.content_hash, doc.created_at.isoformat(), 
                 doc.file_type, doc.clean_text, ','.join(doc.tokens), doc.word_count, doc.cluster_id))

    def get_all_documents(self):
        """Retrieve all documents."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM documents')
        return cursor.fetchall()

    def store_analysis_result(self, result: AnalysisResult):
        """Store analysis results in the database."""
        with self.conn:
            cursor = self.conn.execute('''
                INSERT INTO analysis_results (run_timestamp, document_count, orphan_threshold, num_clusters)
                VALUES (?, ?, ?, ?)''', 
                (result.run_timestamp.isoformat(), result.document_count, result.orphan_threshold, result.num_clusters))
            result_id = cursor.lastrowid
            
            for cluster in result.clusters:
                self.conn.execute('''
                    INSERT INTO clusters (cluster_id, keywords, coherence_score, document_ids)
                    VALUES (?, ?, ?, ?)''', 
                    (cluster.cluster_id, ','.join(cluster.keywords), cluster.coherence_score, 
                     ','.join(map(str, cluster.document_ids))))
        return result_id

    def close(self):
        """Close the database connection."""
        self.conn.close()