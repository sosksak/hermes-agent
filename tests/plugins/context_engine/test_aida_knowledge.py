"""Блок «Что я знаю всегда»: бюджет, порядок чтения, шум и повторы."""

from __future__ import annotations

from plugins.context_engine.aida.assembly import (
    KNOWLEDGE_BUDGET,
    SECTION_BANS,
    SECTION_IDENTITY,
    SECTION_LESSONS,
    SECTION_PROJECTS,
    SECTION_RULES,
    build_knowledge_block,
    section_of,
)
from plugins.memory.aida.write import effective_priority, is_operational


def _rows():
    return [
        {"content": "Life Tracker v9.20.0 успешно перенесён на Svelte 5 (коммит 54de603)", "memory_type": "project"},
        {"content": "Миграция вкладки «Стата» на Svelte 5: перенесены компоненты сравнения недель", "memory_type": "project"},
        {"content": "Life Tracker больше НЕ трогать: проектом занимается Клод.", "memory_type": "rule"},
        {"content": "Настоящее имя пользователя — Даня (не Макс).", "memory_type": "preference"},
        {"content": "Настоящее имя пользователя sosk — Даня (не Макс).", "memory_type": "fact"},
        {"content": "Отвечать коротко и по делу.", "memory_type": "rule"},
        {"content": "Когда: собираю HTML. Правило: гоняю node --check. Почему: бэктик положил приложение. #сборка",
         "memory_type": "lesson"},
        {"content": "Коммит 54de603 запушен в master, дерево чистое.", "memory_type": "event"},
        {"content": "В версии 5.0 добавлены магазин с колесом фортуны и haptic-вибрация.", "memory_type": "project"},
    ]


def test_the_budget_is_a_page_not_a_chapter():
    assert KNOWLEDGE_BUDGET == 2_500
    rows = [{"content": f"правило номер {i}: " + "слово " * 20, "memory_type": "rule"} for i in range(200)]

    assert len(build_knowledge_block(rows)) <= KNOWLEDGE_BUDGET


def test_bans_are_read_before_the_projects_they_forbid():
    block = build_knowledge_block(_rows())

    assert block.index(SECTION_IDENTITY) < block.index(SECTION_BANS) < block.index(SECTION_PROJECTS)
    assert block.index(SECTION_PROJECTS) < block.index(SECTION_LESSONS) < block.index(SECTION_RULES)
    assert block.index("НЕ трогать") < block.index("Миграция вкладки")


def test_a_ban_is_recognised_by_its_wording_whatever_its_type():
    assert section_of({"content": "Не упоминать курьерскую работу как текущую.", "memory_type": "fact"}) == SECTION_BANS
    assert section_of({"content": "Не додумывать состояние — раздражает.", "memory_type": "preference"}) == SECTION_BANS
    assert section_of({"content": "Отвечать коротко.", "memory_type": "rule"}) == SECTION_RULES


def test_operational_churn_never_reaches_the_always_on_block():
    block = build_knowledge_block(_rows())

    assert "54de603" not in block and "v9.20.0" not in block and "версии 5.0" not in block
    assert section_of({"content": "Коммит 54de603 запушен", "memory_type": "event"}) is None
    assert is_operational("QA Life Tracker v9.20.0: DOM до/после идентичны, pageerrors=0")
    assert is_operational("Версия v9.11.0 задеплоена с коммитом 2dd9579")
    assert not is_operational("Не додумывать состояние (недосып, усталость) — считать в норме.")
    assert not is_operational("Оператор живёт рядом с Мытищами, искал подработку на складе Ozon.")


def test_a_version_number_turns_p1_into_p2_at_write_time():
    assert effective_priority("Life Tracker v9.17.0 развёрнута", "P1") == "P2"
    assert effective_priority("Оператора зовут Даня", "P1") == "P1"
    assert effective_priority("Коммит abc1234 запушен", "P2") == "P2"


def test_the_same_thought_written_twice_appears_once():
    block = build_knowledge_block(_rows())

    assert block.count("Даня (не Макс)") == 1


def test_an_event_is_not_knowledge_even_at_p1():
    block = build_knowledge_block([{"content": "Коммит 54de603 запушен в master", "memory_type": "event"},
                                   {"content": "Встреча с Ангелиной прошла спокойно.", "memory_type": "event"}])

    assert block == ""
