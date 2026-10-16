"""Уроки: формат, разбор и подъём по активности, а не по словам."""

from __future__ import annotations

from plugins.memory.aida.lessons import (
    detect_activities,
    format_lesson,
    lesson_tags,
    matches_activity,
    parse_lesson,
    tag_pattern,
)


def test_a_lesson_is_three_fields_and_tags_in_one_line():
    line = format_lesson("собираю HTML с инлайн-скриптами", "гоняю node --check по каждому скрипту",
                         "один лишний бэктик положил всё приложение", ["код", "сборка"])

    assert line == ("Когда: собираю HTML с инлайн-скриптами. Правило: гоняю node --check по каждому скрипту. "
                    "Почему: один лишний бэктик положил всё приложение. #код #сборка")
    assert parse_lesson(line) == {
        "trigger": "собираю HTML с инлайн-скриптами",
        "rule": "гоняю node --check по каждому скрипту",
        "why": "один лишний бэктик положил всё приложение",
        "activities": ["код", "сборка"],
    }


def test_a_plain_fact_is_not_mistaken_for_a_lesson():
    assert parse_lesson("Оператора зовут Даня.") is None
    assert lesson_tags("Оператора зовут Даня.") == []


def test_a_build_question_without_the_word_backtick_still_names_the_build_activity():
    assert "сборка" in detect_activities("Собери index.html из html-частей и проверь, что инлайн-скрипт не сломался")
    assert "код" in detect_activities("почини баг в функции")
    assert "деньги" in detect_activities("Сколько я заработал за октябрь и хватит ли на цель месяца?")
    assert "сообщение" in detect_activities("Напиши сообщение Ангелине, что я сегодня не смогу")
    assert "память" in detect_activities("как устроен твой движок памяти")


def test_a_question_about_nothing_in_particular_names_no_activity():
    assert detect_activities("как дела, что нового") == []
    assert detect_activities("") == []


def test_a_lesson_surfaces_by_tag_and_the_sql_pattern_matches_whole_tags_only():
    import re

    lesson = format_lesson("сборка", "проверить", "ошибался", ["сборка"])
    assert matches_activity(lesson, ["сборка", "код"])
    assert not matches_activity(lesson, ["деньги"])
    pattern = re.compile(tag_pattern(["код"]))
    assert pattern.search("… #код")
    assert pattern.search("… #код #сборка")
    assert not pattern.search("… #кодировка")
