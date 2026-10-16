"""Журнал отбора: пишется, считает, строит отчёт за неделю."""

from __future__ import annotations

from plugins.memory.aida.recall_log import RecallLog, weekly_report


def _log(tmp_path):
    return RecallLog(tmp_path / "aida_recall_log.db")


def test_a_recall_is_written_with_what_surfaced_and_what_reached_the_model(tmp_path):
    log = _log(tmp_path)
    try:
        recall_id = log.record("как устроен движок памяти", [
            {"id": 1, "memory_type": "fact", "block": "факты", "score": 0.7, "chars": 80, "kept": True},
            {"id": 2, "memory_type": "dialogue", "block": "цитаты", "score": 0.5, "chars": 400, "kept": False},
        ])

        assert recall_id is not None
        assert log.totals(days=1) == (1, 2, 1)
        assert log.surfaced_ids(days=1) == {1, 2}
        assert log.top_surfaced(days=1)[0][:3] == (1, "fact", 1)
    finally:
        log.close()


def test_the_weekly_report_names_the_most_surfaced_and_the_never_surfaced(tmp_path):
    log = _log(tmp_path)
    try:
        for _ in range(3):
            log.record("вопрос", [{"id": 1, "memory_type": "rule", "block": "факты", "chars": 10, "kept": True}])
        active = [(1, "rule", "P1", "правило, которое всплывает"), (2, "fact", "P2", "факт, который не всплыл"),
                  (3, "dialogue", "P2", "реплика")]

        report = weekly_report(log, active, days=7)

        assert "[1] rule ×3 (доехало 3) правило, которое всплывает" in report
        assert "Не поднялись ни разу: 1 из 2" in report
        assert "[2] fact/P2 факт, который не всплыл" in report
        assert "реплика" not in report  # диалог не кандидат ни в блок, ни в архив по этому признаку
    finally:
        log.close()


def test_a_broken_entry_never_raises(tmp_path):
    log = _log(tmp_path)
    try:
        assert log.record("q", [{"id": "не число", "chars": "тоже"}]) is None or True
    finally:
        log.close()
