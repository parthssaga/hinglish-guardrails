"""
Analyze guardrail log events from guardrail_logs.db.

Usage:
    python analyze_logs.py
    python analyze_logs.py --all
    python analyze_logs.py --csv out.csv
    python analyze_logs.py --db path.db
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from contextlib import closing
from datetime import datetime

from config import DB_PATH


def fetch_stats(db_path: str) -> dict:
    with closing(sqlite3.connect(db_path)) as conn:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        blocked = conn.execute(
            "SELECT COUNT(*) FROM events WHERE final_action='blocked'"
        ).fetchone()[0]
        avg_ms = conn.execute("SELECT AVG(total_ms) FROM events").fetchone()[0] or 0.0
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
        "avg_latency_ms": avg_ms,
        "by_language": by_lang,
        "by_guardrail": by_guardrail,
    }


def fetch_events(db_path: str, limit: int | None = None) -> list[dict]:
    query = "SELECT * FROM events ORDER BY id DESC"
    if limit is not None:
        query += f" LIMIT {limit}"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query).fetchall()
    return [dict(r) for r in rows]


def flatten_results(json_str: str | None) -> str:
    if not json_str:
        return ""
    try:
        results = json.loads(json_str)
        parts = []
        for r in results:
            name = r.get("name", "?")
            score = r.get("score", 0)
            fired = "Y" if r.get("triggered") else "N"
            parts.append(f"{name}:{score:.2f}({fired})")
        return " | ".join(parts)
    except Exception:
        return str(json_str)


def print_summary(stats: dict) -> None:
    total = stats["total"]
    blocked = stats["blocked"]
    allowed = stats["allowed"]

    print("=" * 62)
    print("GUARDRAIL LOG SUMMARY")
    print("=" * 62)
    print(f"  Total events  : {total}")
    if total:
        print(f"  Blocked       : {blocked:5d}  ({blocked / total * 100:5.1f}%)")
        print(f"  Allowed       : {allowed:5d}  ({allowed / total * 100:5.1f}%)")
    else:
        print("  Blocked       :     0")
        print("  Allowed       :     0")
    print(f"  Avg latency   : {stats['avg_latency_ms']:.1f} ms")

    print("\n  BY LANGUAGE:")
    if stats["by_language"]:
        for lang, count in sorted(stats["by_language"].items()):
            pct = count / total * 100 if total else 0.0
            print(f"    {lang:12s}  {count:5d}  ({pct:.1f}%)")
    else:
        print("    (no data)")

    print("\n  BY GUARDRAIL FIRED:")
    if stats["by_guardrail"]:
        for grl, count in sorted(stats["by_guardrail"].items(), key=lambda x: -x[1]):
            print(f"    {grl:18s}  {count:5d}")
    else:
        print("    (none blocked)")
    print()


def print_events_table(events: list[dict], show_all: bool = False) -> None:
    label = "ALL EVENTS" if show_all else f"RECENT EVENTS (last {len(events)})"
    print("=" * 110)
    print(label)
    print("=" * 110)
    header = f"  {'ID':>5}  {'TIMESTAMP':19}  {'LANG':9}  {'ACTION':8}  {'FIRED BY':12}  INPUT (truncated)"
    print(header)
    print("  " + "-" * 106)
    for ev in reversed(events):
        ts = (
            datetime.fromtimestamp(ev["ts"]).strftime("%Y-%m-%d %H:%M:%S")
            if ev.get("ts")
            else "?"
        )
        lang = (ev.get("language") or "?")[:9]
        action = ev.get("final_action") or "?"
        blocked_by = (ev.get("blocked_by") or "-")[:12]
        snippet = (ev.get("user_input") or "").replace("\n", " ")[:50]
        print(f"  {ev['id']:>5}  {ts:19}  {lang:9}  {action:8}  {blocked_by:12}  {snippet}")
    print()


def export_csv(events: list[dict], path: str) -> None:
    fieldnames = [
        "id", "timestamp", "language", "user_input", "final_action",
        "blocked_by", "response", "total_ms",
        "input_guardrails", "output_guardrails",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ev in reversed(events):
            writer.writerow({
                "id": ev.get("id", ""),
                "timestamp": (
                    datetime.fromtimestamp(ev["ts"]).isoformat()
                    if ev.get("ts") else ""
                ),
                "language": ev.get("language") or "",
                "user_input": ev.get("user_input") or "",
                "final_action": ev.get("final_action") or "",
                "blocked_by": ev.get("blocked_by") or "",
                "response": ev.get("response") or "",
                "total_ms": ev.get("total_ms") or "",
                "input_guardrails": flatten_results(ev.get("input_results")),
                "output_guardrails": flatten_results(ev.get("output_results")),
            })
    print(f"Exported {len(events)} events to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze guardrail log events.")
    ap.add_argument(
        "--all", action="store_true",
        help="Show all events (default: last 25)"
    )
    ap.add_argument(
        "--csv", metavar="FILE",
        help="Export full history to CSV"
    )
    ap.add_argument(
        "--db", metavar="PATH", default=DB_PATH,
        help="Path to the SQLite database (default: %(default)s)"
    )
    args = ap.parse_args()

    stats = fetch_stats(args.db)
    print_summary(stats)

    limit = None if args.all else 25
    events = fetch_events(args.db, limit=limit)
    print_events_table(events, show_all=args.all)

    if args.csv:
        all_events = fetch_events(args.db, limit=None)
        export_csv(all_events, args.csv)


if __name__ == "__main__":
    main()
