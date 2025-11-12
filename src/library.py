"""
Library Module

Manages the collection of papers with metadata storage using SQLite.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class PaperLibrary:
    """Manages paper collection with SQLite database."""

    def __init__(self, db_path: str = "data/papers.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                authors TEXT,
                year INTEGER,
                file_path TEXT UNIQUE NOT NULL,
                indexed_date TEXT,
                num_pages INTEGER,
                summary TEXT,
                keywords TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_paper(self, title: str, file_path: str, authors: str = "",
                  year: Optional[int] = None, num_pages: int = 0,
                  keywords: str = "") -> int:
        """
        Add a new paper to the library.

        Args:
            title: Paper title
            file_path: Path to PDF file
            authors: Authors (comma-separated)
            year: Publication year
            num_pages: Number of pages
            keywords: Keywords (comma-separated)

        Returns:
            ID of the inserted paper
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO papers (title, authors, year, file_path,
                                   indexed_date, num_pages, keywords)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (title, authors, year, file_path,
                  datetime.now().isoformat(), num_pages, keywords))

            conn.commit()
            paper_id = cursor.lastrowid
            return paper_id
        finally:
            conn.close()

    def get_paper(self, paper_id: int) -> Optional[Dict]:
        """Get a paper by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT * FROM papers WHERE id = ?', (paper_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_papers(self) -> List[Dict]:
        """List all papers in the library."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        try:
            cursor.execute('SELECT * FROM papers ORDER BY indexed_date DESC')
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def delete_paper(self, paper_id: int) -> bool:
        """Delete a paper from the library."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('DELETE FROM papers WHERE id = ?', (paper_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def update_summary(self, paper_id: int, summary: str):
        """Update the summary for a paper."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('UPDATE papers SET summary = ? WHERE id = ?',
                          (summary, paper_id))
            conn.commit()
        finally:
            conn.close()
