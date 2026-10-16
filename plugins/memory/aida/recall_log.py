"""Журнал отбора: что поднялось под каждый вопрос и доехало ли до модели.

Без него калибровка порогов делается на глаз по одному запомнившемуся плохому примеру.
С ним за неделю видно: какие строки поднимаются чаще всего (кандидаты в «знаю всегда») и
какие не поднялись ни разу (кандидаты в архив).

Пишется локально, в SQLite рядом с очередью: отбор идёт на каждом ходу, и ждать базу на
другом конце интернета ради журнала нельзя. Любая ошибка здесь глотается — журнал не имеет
права стоить ответа.

Отчёт за неделю:  python -m plugins.memory.aida.recall_log [--days 7] [--home <HERMES_HOME>]
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Iterable

QUERY_CLIP = 300
DB_NAME = "aida_recall_log.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recalls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    query TEXT NOT NULL,
    hits INTEGER NOT NULL,
    kept INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS hits (
    recall_id INTEGER NOT NULL REFERENCES recalls(id),
    memory_id INTEGER,
    memory_type TEXT,
    block TEXT NOT NULL,
    score REAL,
    chars INTEGER NOT NULL,
    kept INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS hits_memory ON hits(memory_id);
CREATE INDEX IF NOT EXISTS recalls_ts ON recalls(ts);
"""


class RecallLog:
    """Один файл, одна таблица подъёмов и одна — поднятых строк."""

    def __init__(self, path: Path):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(self, query: str, entries: Iterable[dict]) -> int | None:
        """Записать один подъём. ``entries``: id, memory_type, block, score, chars, kept."""
        rows = list(entries)
        try:
            with self._lock:
                cur = self._conn.execute(
                    "INSERT INTO recalls (ts, query, hits, kept) VALUES (?, ?, ?, ?)",
                    (time.time(), (query or "")[:QUERY_CLIP], len(rows), sum(1 for r in rows if r.get("kept"))),
                )
                recall_id = cur.lastrowid
                self._conn.executemany(
                    "INSERT INTO hits (recall_id, memory_id, memory_type, block, score, chars, kept)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [(recall_id, r.get("id"), r.get("memory_type"), r.get("block", ""),
                      r.get("score"), int(r.get("chars") or 0), 1 if r.get("kept") else 0) for r in rows],
                )
                self._conn.commit()
                return recall_id
        except Exception:
            return None

    def top_surfaced(self, days: int = 7, limit: int = 20) -> list[tuple]:
        """(memory_id, memory_type, раз поднялась, раз доехала)."""
        since = time.time() - days * 86400
        with self._lock:
            return self._conn.execute(
                "SELECT h.memory_id, h.memory_type, count(*), sum(h.kept)"
                " FROM hits h JOIN recalls r ON r.id = h.recall_id"
                " WHERE r.ts >= ? AND h.memory_id IS NOT NULL"
                " GROUP BY h.memory_id, h.memory_type ORDER BY count(*) DESC, sum(h.kept) DESC LIMIT ?",
                (since, limit),
            ).fetchall()

    def surfaced_ids(self, days: int = 7) -> set[int]:
        since = time.time() - days * 86400
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT h.memory_id FROM hits h JOIN recalls r ON r.id = h.recall_id"
                " WHERE r.ts >= ? AND h.memory_id IS NOT NULL", (since,)
            ).fetchall()
        return {int(r[0]) for r in rows}

    def totals(self, days: int = 7) -> tuple[int, int, int]:
        """(подъёмов, поднятых строк, доехавших строк) за период."""
        since = time.time() - days * 86400
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*), coalesce(sum(hits), 0), coalesce(sum(kept), 0) FROM recalls WHERE ts >= ?",
                (since,),
            ).fetchone()
        return int(row[0]), int(row[1]), int(row[2])

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def weekly_report(log: RecallLog, active_rows: list[tuple], days: int = 7, limit: int = 20) -> str:
    """Текст отчёта. ``active_rows`` — (id, memory_type, priority, content) активных записей."""
    recalls, hits, kept = log.totals(days)
    lines = [f"Отбор памяти за {days} дн.: подъёмов {recalls}, поднято строк {hits}, доехало до модели {kept}"]
    by_id = {int(r[0]): r for r in active_rows}
    lines.append("")
    lines.append("Поднимались чаще всего (кандидаты в «знаю всегда»):")
    top = log.top_surfaced(days, limit)
    if not top:
        lines.append("  — пока ничего")
    for memory_id, memory_type, times, kept_times in top:
        row = by_id.get(int(memory_id))
        text = " ".join(str(row[3]).split())[:100] if row else "(уже не активна)"
        lines.append(f"  [{memory_id}] {memory_type} ×{times} (доехало {kept_times}) {text}")
    seen = log.surfaced_ids(days)
    never = [r for r in active_rows if int(r[0]) not in seen and r[1] != "dialogue"]
    lines.append("")
    lines.append(f"Не поднялись ни разу: {len(never)} из {len([r for r in active_rows if r[1] != 'dialogue'])}"
                 " не-диалоговых записей (кандидаты в архив, первые по давности):")
    for r in sorted(never, key=lambda r: int(r[0]))[:limit]:
        lines.append(f"  [{r[0]}] {r[1]}/{r[2]} {' '.join(str(r[3]).split())[:100]}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import sys

    parser = argparse.ArgumentParser(description="Отчёт по отбору памяти за период")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--home", type=Path, default=None)
    args = parser.parse_args(argv)
    home = args.home
    if home is None:
        try:
            from hermes_constants import get_hermes_home
            home = get_hermes_home()
        except Exception:
            home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    home = Path(home)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from plugins.memory.aida.nightly_job import load_profile_env

    load_profile_env(home)
    active_rows: list[tuple] = []
    db_url = os.environ.get("SUPABASE_DB_URL", "")
    if db_url:
        from plugins.memory.aida import _Store

        store = _Store(db_url, "")
        try:
            active_rows = store._execute_guarded(
                "SELECT id, memory_type, priority, content FROM memories WHERE status = 'active'", {})
        finally:
            store.close()
    log = RecallLog(home / DB_NAME)
    try:
        print(weekly_report(log, active_rows, args.days))
    finally:
        log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
