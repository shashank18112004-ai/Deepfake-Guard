"""SQLite persistence for HR deepfake interview summaries and video tests."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / 'interview_history.db'

@contextmanager
def connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def initialize_database():
    with connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            interview_id TEXT NOT NULL,
            candidate_name TEXT,
            candidate_usn TEXT,
            prediction TEXT NOT NULL CHECK (prediction IN ('REAL','FAKE')),
            risk_score REAL NOT NULL CHECK (risk_score >= 0 AND risk_score <= 100),
            frames_analyzed INTEGER NOT NULL CHECK (frames_analyzed >= 0),
            created_at TEXT NOT NULL,
            record_type TEXT NOT NULL DEFAULT 'LIVE_INTERVIEW'
        )""")
        columns = {row[1] for row in conn.execute('PRAGMA table_info(interviews)')}
        if 'record_type' not in columns:
            conn.execute("ALTER TABLE interviews ADD COLUMN record_type TEXT NOT NULL DEFAULT 'LIVE_INTERVIEW'")

def save_interview(interview_id, candidate_name, candidate_usn, prediction, risk_score, frames_analyzed, record_type='LIVE_INTERVIEW'):
    if record_type not in ('LIVE_INTERVIEW', 'VIDEO_TEST'):
        raise ValueError('Invalid record type')
    created_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with connection() as conn:
        cur = conn.execute("""INSERT INTO interviews
            (interview_id,candidate_name,candidate_usn,prediction,risk_score,frames_analyzed,created_at,record_type)
            VALUES (?,?,?,?,?,?,?,?)""",
            (interview_id,candidate_name,candidate_usn,prediction,risk_score,frames_analyzed,created_at,record_type))
        row = conn.execute('SELECT * FROM interviews WHERE id=?', (cur.lastrowid,)).fetchone()
        return dict(row)

def list_interviews(limit=100):
    with connection() as conn:
        return [dict(row) for row in conn.execute('SELECT * FROM interviews ORDER BY id DESC LIMIT ?', (limit,))]

def get_interview(record_id):
    with connection() as conn:
        row = conn.execute('SELECT * FROM interviews WHERE id=?', (record_id,)).fetchone()
        return dict(row) if row else None


def delete_interview(record_id: int) -> bool:
    """Delete exactly one saved result; return False if not found."""
    with connection() as conn:
        cursor = conn.execute('DELETE FROM interviews WHERE id = ?', (record_id,))
        return cursor.rowcount == 1
