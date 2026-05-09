# ============================================
# database.py
# SQLite persistence for study sessions
# ============================================

import os
import json
import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.path.join("data", "study_companion.db")


def get_connection():
    """Get a database connection, creating tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    return conn


def _create_tables(conn):
    """Create all tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            document_name TEXT,
            document_text TEXT,
            summary TEXT,
            keywords TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS quiz_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            score INTEGER,
            total INTEGER,
            percentage REAL,
            grade TEXT,
            difficulty TEXT,
            details TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS qa_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT,
            answer TEXT,
            confidence REAL,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS flashcards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            term TEXT,
            definition TEXT,
            category TEXT DEFAULT 'general',
            mastery_level INTEGER DEFAULT 0,
            next_review TEXT,
            review_count INTEGER DEFAULT 0,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );

        CREATE TABLE IF NOT EXISTS study_progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            action_type TEXT,
            details TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    conn.commit()


# ---- Session CRUD ----

def save_session(session_id, document_name, document_text, summary=None, keywords=None):
    """Save or update a study session."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO sessions (id, document_name, document_text, summary, keywords, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(id) DO UPDATE SET
                document_name=excluded.document_name,
                document_text=excluded.document_text,
                summary=COALESCE(excluded.summary, sessions.summary),
                keywords=COALESCE(excluded.keywords, sessions.keywords),
                updated_at=datetime('now')
        """, (session_id, document_name, document_text,
              summary, json.dumps(keywords) if keywords else None))
        conn.commit()
    finally:
        conn.close()


def load_session(session_id):
    """Load a session from database."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_all_sessions():
    """Get all sessions (for dashboard)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, document_name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_session(session_id):
    """Delete a session and all related data."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM flashcards WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM quiz_results WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM qa_history WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM study_progress WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()


# ---- Quiz Results ----

def save_quiz_result(session_id, score, total, percentage, grade, difficulty, details):
    """Save a quiz result."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO quiz_results (session_id, score, total, percentage, grade, difficulty, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_id, score, total, percentage, grade, difficulty, json.dumps(details)))
        conn.commit()
        _log_progress(conn, session_id, "quiz", f"Score: {score}/{total} ({grade})")
    finally:
        conn.close()


def get_quiz_history(session_id=None):
    """Get quiz history, optionally filtered by session."""
    conn = get_connection()
    try:
        if session_id:
            rows = conn.execute(
                "SELECT * FROM quiz_results WHERE session_id = ? ORDER BY created_at DESC", (session_id,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM quiz_results ORDER BY created_at DESC LIMIT 50"
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


# ---- Q&A History ----

def save_qa_entry(session_id, question, answer, confidence):
    """Save a Q&A interaction."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO qa_history (session_id, question, answer, confidence)
            VALUES (?, ?, ?, ?)
        """, (session_id, question, answer, confidence))
        conn.commit()
    finally:
        conn.close()


# ---- Flashcards ----

def save_flashcards(session_id, flashcards):
    """Save a list of flashcards."""
    conn = get_connection()
    try:
        now = datetime.now().isoformat()
        for fc in flashcards:
            conn.execute("""
                INSERT INTO flashcards (session_id, term, definition, category, next_review)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, fc.get("term", ""), fc.get("definition", ""),
                  fc.get("category", "general"), now))
        conn.commit()
        _log_progress(conn, session_id, "flashcards", f"Generated {len(flashcards)} flashcards")
    finally:
        conn.close()


def get_flashcards(session_id):
    """Get all flashcards for a session."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM flashcards WHERE session_id = ? ORDER BY mastery_level ASC, id ASC",
            (session_id,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_due_flashcards(session_id):
    """Get flashcards due for review (spaced repetition)."""
    conn = get_connection()
    try:
        now = datetime.now().isoformat()
        rows = conn.execute("""
            SELECT * FROM flashcards
            WHERE session_id = ? AND (next_review <= ? OR next_review IS NULL)
            ORDER BY mastery_level ASC, review_count ASC
        """, (session_id, now)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def update_flashcard_mastery(flashcard_id, correct):
    """Update flashcard after review (spaced repetition algorithm)."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM flashcards WHERE id = ?", (flashcard_id,)).fetchone()
        if not row:
            return

        mastery = row["mastery_level"]
        review_count = row["review_count"] + 1

        if correct:
            mastery = min(mastery + 1, 5)
        else:
            mastery = max(mastery - 1, 0)

        # Spaced repetition intervals (in hours)
        intervals = {0: 0.5, 1: 4, 2: 24, 3: 72, 4: 168, 5: 720}
        hours = intervals.get(mastery, 24)

        from datetime import timedelta
        next_review = (datetime.now() + timedelta(hours=hours)).isoformat()

        conn.execute("""
            UPDATE flashcards SET mastery_level = ?, review_count = ?, next_review = ?
            WHERE id = ?
        """, (mastery, review_count, next_review, flashcard_id))
        conn.commit()
    finally:
        conn.close()


# ---- Study Progress ----

def _log_progress(conn, session_id, action_type, details):
    """Internal: log a study action."""
    conn.execute("""
        INSERT INTO study_progress (session_id, action_type, details)
        VALUES (?, ?, ?)
    """, (session_id, action_type, details))
    conn.commit()


def get_study_stats(session_id=None):
    """Get overall study statistics for the dashboard."""
    conn = get_connection()
    try:
        stats = {}

        # Total sessions
        stats["total_sessions"] = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]

        # Quiz stats
        if session_id:
            quiz_rows = conn.execute(
                "SELECT * FROM quiz_results WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,)
            ).fetchall()
        else:
            quiz_rows = conn.execute(
                "SELECT * FROM quiz_results ORDER BY created_at DESC"
            ).fetchall()

        if quiz_rows:
            pcts = [r["percentage"] for r in quiz_rows]
            stats["total_quizzes"] = len(quiz_rows)
            stats["average_score"] = round(sum(pcts) / len(pcts), 1)
            stats["best_score"] = max(pcts)
            stats["worst_score"] = min(pcts)
            stats["total_questions"] = sum(r["total"] for r in quiz_rows)
            stats["total_correct"] = sum(r["score"] for r in quiz_rows)
            # Score history for chart
            stats["score_history"] = [
                {"date": r["created_at"], "score": r["percentage"],
                 "grade": r["grade"], "difficulty": r["difficulty"]}
                for r in quiz_rows[:20]
            ]
        else:
            stats["total_quizzes"] = 0
            stats["average_score"] = 0
            stats["score_history"] = []

        # Flashcard stats
        if session_id:
            fc_rows = conn.execute(
                "SELECT mastery_level, COUNT(*) as cnt FROM flashcards WHERE session_id = ? GROUP BY mastery_level",
                (session_id,)
            ).fetchall()
        else:
            fc_rows = conn.execute(
                "SELECT mastery_level, COUNT(*) as cnt FROM flashcards GROUP BY mastery_level"
            ).fetchall()

        stats["flashcard_mastery"] = {str(r["mastery_level"]): r["cnt"] for r in fc_rows}
        stats["total_flashcards"] = sum(r["cnt"] for r in fc_rows)

        # QA stats
        qa_count = conn.execute(
            "SELECT COUNT(*) FROM qa_history" + (" WHERE session_id = ?" if session_id else ""),
            (session_id,) if session_id else ()
        ).fetchone()[0]
        stats["total_questions_asked"] = qa_count

        return stats
    finally:
        conn.close()
