"""Уроки: записи, которые меняют поведение, и подъём их по ситуации, а не по словам.

Самое ценное в базе — не факты, а выводы вида «на любой сборке HTML гонять node --check по
инлайн-скрипту». Такая строка должна приезжать на любую сборку, даже если в вопросе нет ни
слова «бэктик»: совпадение слов её не находит, потому что вопрос про задачу, а урок — про
ошибку. Поэтому у урока есть тег активности, и движок поднимает уроки по тому, ЧТО человек
сейчас делает, — правит код, считает деньги, пишет сообщение, трогает память.

Урок хранится в той же таблице, типом ``lesson``, тремя полями в одной строке:

    Когда: <trigger>. Правило: <rule>. Почему: <why>. #код #сборка

Без дат и номеров инцидентов: урок — это правило на будущее, а не запись о прошлом.
"""

from __future__ import annotations

import re

# Теги активности и слова вопроса, по которым она узнаётся. Слова — основы, сравнение по
# началу слова: «собери», «сборка», «собрать» ловятся одним «собер|сбор».
ACTIVITIES: dict[str, tuple[str, ...]] = {
    "код": ("код", "скрипт", "html", "css", "js", "svelte", "функци", "баг", "ошибк", "коммит",
            "деплой", "задеплой", "репо", "файл", "index", "бандл", "build", "сбор", "собер",
            "собр", "рендер", "компонент", "тест", "pytest", "python", "node", "sql", "запрос"),
    "сборка": ("сбор", "собер", "собр", "html", "index", "бандл", "build", "инлайн", "inline",
               "деплой", "задеплой", "минифи", "terser"),
    "деньги": ("деньг", "заработ", "рубл", "₽", "цель месяца", "зарплат", "доход", "расход",
               "бюджет", "копилк", "смен", "налог", "ton", "крипт", "биткоин", "курс"),
    "сообщение": ("напиши", "сообщени", "ответь", "ответить", "написать", "девушк", "ангелин",
                  "текст для", "письмо", "отправ"),
    "память": ("памят", "вспомни", "запомни", "забыл", "помнишь", "движок", "контекст", "aida"),
}

_LESSON_RE = re.compile(
    r"^\s*Когда:\s*(?P<trigger>.+?)\.\s*Правило:\s*(?P<rule>.+?)\.\s*Почему:\s*(?P<why>.+?)\.?\s*"
    r"(?P<tags>(?:#\S+\s*)*)$",
    re.DOTALL,
)


def detect_activities(question: str) -> list[str]:
    """Какие активности видны в вопросе; порядок — как в ACTIVITIES."""
    text = (question or "").lower()
    found = []
    for tag, stems in ACTIVITIES.items():
        if any(stem in text for stem in stems):
            found.append(tag)
    return found


def format_lesson(trigger: str, rule: str, why: str, activities: list[str] | tuple[str, ...] = ()) -> str:
    """Урок одной строкой, как он ложится в базу."""
    parts = [f"Когда: {_clean(trigger)}.", f"Правило: {_clean(rule)}.", f"Почему: {_clean(why)}."]
    tags = " ".join(f"#{tag.lstrip('#')}" for tag in activities if tag)
    return " ".join(parts) + (f" {tags}" if tags else "")


def parse_lesson(content: str) -> dict | None:
    """Поля урока из строки, или None, если строка не в формате урока."""
    match = _LESSON_RE.match(content or "")
    if not match:
        return None
    tags = [tag.lstrip("#") for tag in match.group("tags").split()]
    return {"trigger": match.group("trigger").strip(), "rule": match.group("rule").strip(),
            "why": match.group("why").strip(), "activities": tags}


def lesson_tags(content: str) -> list[str]:
    return [tag.lstrip("#") for tag in re.findall(r"#\S+", content or "")]


def matches_activity(content: str, activities: list[str]) -> bool:
    tags = set(lesson_tags(content))
    return bool(tags & set(activities))


def _clean(text: str) -> str:
    return " ".join((text or "").split()).rstrip(".")


# Активные уроки по тегу активности. Тег — хвост строки «#код», и регулярное выражение на
# него точнее полнотекста: «russian» режет «#код» в «код» и находит и «кодировался».
LESSONS_SQL = """
SELECT m.id, m.content, m.memory_type, m.priority, m.created_at, m.role, m.chat_id
FROM memories m
WHERE m.status = 'active' AND m.memory_type = 'lesson'
  AND m.content ~ %(pattern)s
ORDER BY m.created_at DESC
LIMIT %(limit)s
"""


def tag_pattern(activities: list[str]) -> str:
    """Регулярное выражение Postgres: хотя бы один из тегов, целым словом."""
    escaped = "|".join(re.escape(tag) for tag in activities)
    return rf"#({escaped})(\s|$)"
