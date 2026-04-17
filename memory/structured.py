from __future__ import annotations
import sqlite3
import os
import time


class StructuredStore:
    def __init__(self, db_path: str = "~/.aria/structured.db") -> None:
        db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                deadline TEXT,
                status TEXT DEFAULT 'pending',
                created_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS commitments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                to_whom TEXT,
                fulfilled INTEGER DEFAULT 0,
                deadline_epoch REAL,
                created_at REAL DEFAULT (strftime('%s','now'))
            );
            CREATE TABLE IF NOT EXISTS key_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL UNIQUE,
                created_at REAL DEFAULT (strftime('%s','now'))
            );
        """)
        self._conn.commit()

    def add_task(self, text: str, deadline: str | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO tasks (text, deadline) VALUES (?, ?)", (text, deadline)
        )
        self._conn.commit()
        return cur.lastrowid

    def get_tasks(self, status: str | None = None) -> list[dict]:
        if status:
            rows = self._conn.execute("SELECT * FROM tasks WHERE status = ?", (status,)).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM tasks").fetchall()
        return [dict(r) for r in rows]

    def update_task_status(self, task_id: int, status: str) -> None:
        self._conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        self._conn.commit()

    def add_commitment(self, text: str, to_whom: str | None = None, deadline_epoch: float | None = None) -> int:
        cur = self._conn.execute(
            "INSERT INTO commitments (text, to_whom, deadline_epoch) VALUES (?, ?, ?)",
            (text, to_whom, deadline_epoch),
        )
        self._conn.commit()
        return cur.lastrowid

    def get_commitments(self, fulfilled: bool | None = None) -> list[dict]:
        if fulfilled is None:
            rows = self._conn.execute("SELECT * FROM commitments").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM commitments WHERE fulfilled = ?", (int(fulfilled),)
            ).fetchall()
        return [dict(r) for r in rows]

    def fulfill_commitment(self, commitment_id: int) -> None:
        self._conn.execute("UPDATE commitments SET fulfilled = 1 WHERE id = ?", (commitment_id,))
        self._conn.commit()

    def add_key_fact(self, text: str) -> None:
        self._conn.execute("INSERT OR IGNORE INTO key_facts (text) VALUES (?)", (text,))
        self._conn.commit()

    def get_key_facts(self) -> list[dict]:
        return [dict(r) for r in self._conn.execute("SELECT * FROM key_facts").fetchall()]

    def get_at_risk_commitments(self, within_hours: float = 3.0) -> list[dict]:
        cutoff = time.time() + within_hours * 3600
        rows = self._conn.execute(
            "SELECT * FROM commitments WHERE fulfilled = 0 AND deadline_epoch IS NOT NULL AND deadline_epoch <= ?",
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]

    def close(self) -> None:
        self._conn.close()
