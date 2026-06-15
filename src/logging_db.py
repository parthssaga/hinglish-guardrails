"""
SQLite logging layer.

Every message that passes through the pipeline produces one row here,
capturing what was decided, by which guardrail, with what score, in what
language, and how long it took. This is the audit trail the literature
review identified as missing from existing guardrail systems, and it is
what the dashboard reads from.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,
    conversation  TEXT,
    language      TEXT,
    lang_conf     REAL,
    user_input    TEXT,           -- PII-redacted before storage
    final_action  TEXT,           -- 'blocked' | 'allowed'
    blocked_by    TEXT,           -- guardrail name, or NULL
    response      TEXT,           -- model response, or NULL if blocked
    input_results TEXT,           -- JSON list of input guardrail results
    output_results TEXT,          -- JSON list of output guardrail results
    total_ms      REAL
);
"""


class EventLog:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def record(
        self,
        *,
        conversation: str,
        language: str,
        lang_conf: float,
        user_input: str,
        final_action: str,
        blocked_by: str | None,
        response: str | None,
        input_results: list[dict],
        output_results: list[dict],
        total_ms: float,
    ) -> int:
        with closing(sqlite3.connect(self.db_path)) as conn:
            cur = conn.execute(
                """
                INSERT INTO events (
                    ts, conversation, language, lang_conf, user_input,
                    final_action, blocked_by, response,
                    input_results, output_results, total_ms
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    time.time(),
                    conversation,
                    language,
                    lang_conf,
                    user_input,
                    final_action,
                    blocked_by,
                    response,
                    json.dumps(input_results),
                    json.dumps(output_results),
                    total_ms,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def recent(self, limit: int = 100) -> list[dict]:
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["input_results"] = json.loads(d["input_results"] or "[]")
            d["output_results"] = json.loads(d["output_results"] or "[]")
            out.append(d)
        return out

    def stats(self) -> dict:
        with closing(sqlite3.connect(self.db_path)) as conn:
            total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            blocked = conn.execute(
                "SELECT COUNT(*) FROM events WHERE final_action='blocked'"
            ).fetchone()[0]
            by_lang = dict(
                conn.execute(
                    "SELECT language, COUNT(*) FROM events GROUP BY language"
                ).fetchall()
            )
            by_guardrail = dict(
                conn.execute(
                    "SELECT blocked_by, COUNT(*) FROM events "
                    "WHERE blocked_by IS NOT NULL GROUP BY blocked_by"
                ).fetchall()
            )
        return {
            "total": total,
            "blocked": blocked,
            "allowed": total - blocked,
            "by_language": by_lang,
            "by_guardrail": by_guardrail,
        }
