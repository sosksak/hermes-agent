"""Сборка витрины: слоты, бюджеты и безопасная граница хвоста.

Здесь нет ни базы, ни сети — только превращение «что нашлось» в «что уходит в модель».
Вынесено отдельно, потому что самое опасное место движка проверяется именно тут: обращение
к инструменту и его результат — пара, и хвост, разрезанный между ними, отвергается
поставщиком целиком. Разговор при этом не деградирует, а встаёт.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# Бюджеты в символах — как в том движке, что работал на сервере. Символы, а не токены,
# потому что считать их нечем и незачем: это витрина, а не бухгалтерия.
TOTAL_BUDGET = 150_000
# Блок «знаю всегда» на 12 000 знаков (89 строк плоским списком) модель пролистывала, а не
# выполняла: запрет «Life Tracker не трогать» стоял после десяти строк про Life Tracker. Это
# не хранилище, а то, что читается перед каждым ответом, — размер страницы, не главы.
KNOWLEDGE_BUDGET = 2_500
RECALLED_BUDGET = 15_000
LESSONS_BUDGET = 3_000
OLD_HISTORY_BUDGET = 10_000
# Цитата — отрывок, а не документ. На 2 000 знаков цитатой становился целый отчёт ассистента;
# отрывка в 400 знаков хватает, чтобы узнать разговор, а остальное достаётся aida_recall.
OLD_QUOTE_CLIP = 400
MAX_OLD_QUOTES = 2
MIN_TAIL_BUDGET = 20_000
# Бюджет запроса от окна модели: 5% окна, в знаках. Для окна в миллион токенов это прежние
# 150 000; для меньшего окна число меньше, но не ниже MIN_TOTAL_BUDGET. Знаки, а не токены, потому
# что считать токены нечем, а трёх знаков на токен хватает для порядка величины.
WINDOW_SHARE = 0.05
CHARS_PER_TOKEN = 3
MIN_TOTAL_BUDGET = 90_000
# Старение выхлопа инструментов: результаты старше этого числа последних ходов и длиннее порога
# заменяются в запросе заглушкой. Выхлоп — три четверти объёма разговора, а реплики человека и
# агента — остальное. Замер на этой самой сборке (бюджет 150 000 знаков, размеры по замеру
# оператора): хвост вмещает ~24 хода без старения и ~73 с порогом 500 знаков; порог 1 000 даёт
# только ~53, потому что около 40% результатов короче него.
KEEP_FULL_TURNS = 2
AGE_MIN_CHARS = 500
# Когда хвост упёрся в потолок, его начало сдвигается сразу до этой доли бюджета. Сдвиг ломает кеш
# поставщика для всего, что после системной части, поэтому он должен быть редким и крупным.
STEP_KEEP = 0.6

# Заголовки групп — из того же движка. Модель читает их как оглавление, человек — как
# карту того, что о нём знают.
GROUP_LABELS = {
    "preference": "ПРЕДПОЧТЕНИЯ ОПЕРАТОРА",
    "fact": "ФАКТЫ ОБ ОПЕРАТОРЕ",
    "decision": "КЛЮЧЕВЫЕ РЕШЕНИЯ",
    "rule": "ПРАВИЛА РАБОТЫ",
    "context": "КОНТЕКСТ",
    "project": "ПРОЕКТЫ",
    "contact": "КОНТАКТЫ",
    "event": "СОБЫТИЯ",
    "lesson": "УРОКИ",
}

# Разделы блока «знаю всегда» — в порядке чтения, не в порядке типов: кто человек → что
# запрещено → чем занят → что усвоено → как работать. Запрет должен быть прочитан раньше, чем
# десять строк про проект, который он запрещает трогать.
SECTION_IDENTITY = "КТО ОПЕРАТОР"
SECTION_BANS = "ЗАПРЕТЫ И ГРАНИЦЫ"
SECTION_PROJECTS = "ПРОЕКТЫ И РЕШЕНИЯ"
SECTION_LESSONS = "УРОКИ"
SECTION_RULES = "ПРАВИЛА РАБОТЫ"
SECTION_ORDER = (SECTION_IDENTITY, SECTION_BANS, SECTION_PROJECTS, SECTION_LESSONS, SECTION_RULES)
_SECTION_BY_TYPE = {
    "preference": SECTION_IDENTITY, "fact": SECTION_IDENTITY, "contact": SECTION_IDENTITY,
    "project": SECTION_PROJECTS, "decision": SECTION_PROJECTS, "context": SECTION_PROJECTS,
    "lesson": SECTION_LESSONS,
    "rule": SECTION_RULES,
}
# Запрет узнаётся по формулировке, из любого типа: «не трогать», «не упоминать», «нельзя».
_BAN_RE = re.compile(
    r"\b(не трога|нельзя|никогда|запрещ|не упомина|не додумыва|не предлага|не использ|не спрашива|"
    r"не писа|не отправля|не звон|без вступлен|только готовый)",
    re.IGNORECASE,
)
# Событие — не знание: в блок оно не идёт, даже с P1 (см. write.is_operational про шум).
_SKIPPED_TYPES = frozenset({"event", "dialogue"})
# Две строки с такой долей общих значимых слов — одна мысль, записанная дважды. Настоящее
# схлопывание делает ночная работа; здесь — страховка витрины, чтобы повтор не ехал сегодня.
DUPLICATE_OVERLAP = 0.6


def is_system(message: dict) -> bool:
    return message.get("role") == "system"


def split_system(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Отделить системную часть хозяина от разговора.

    Она идёт первой и не трогается: там личность Гермеса и его инструкции, и она же —
    устойчивое начало, на котором держится кеш поставщика.
    """
    head: list[dict] = []
    for index, message in enumerate(messages):
        if is_system(message):
            head.append(message)
            continue
        return head, list(messages[index:])
    return head, []


def _has_tool_call(message: dict) -> bool:
    return bool(message.get("tool_calls") or message.get("function_call"))


def _is_tool_result(message: dict) -> bool:
    return message.get("role") == "tool" or message.get("tool_call_id") is not None


def message_size(message: dict) -> int:
    """Грубый размер сообщения в символах — вместе с его инструментальной частью."""
    content = message.get("content")
    if isinstance(content, str):
        size = len(content)
    elif isinstance(content, list):
        size = sum(len(str(part)) for part in content)
    else:
        size = len(str(content or ""))
    for call in message.get("tool_calls") or ():
        size += len(str(call))
    return size + 16


def is_clean_boundary(messages: list[dict], index: int) -> bool:
    """Можно ли начать хвост с этого сообщения, не оборвав пару «обращение — результат».

    Чистая граница — это реплика человека. Ответ помощника с обращением к инструменту
    тянет за собой результат, а результат без своего обращения поставщик не принимает.
    """
    if index >= len(messages):
        return False
    message = messages[index]
    if _is_tool_result(message):
        return False
    if message.get("role") == "user":
        return True
    # Ответ помощника без инструментов — тоже допустимое начало, но только если следом
    # не идёт осиротевший результат.
    if message.get("role") == "assistant" and not _has_tool_call(message):
        return not (index + 1 < len(messages) and _is_tool_result(messages[index + 1]))
    return False


def safe_tail(messages: list[dict], budget: int) -> list[dict]:
    """Свежий хвост разговора, целиком влезающий в бюджет и не рвущий ни одной пары.

    Отсчитывается с конца; найденная по бюджету граница сдвигается вперёд до ближайшей
    чистой. Если впереди чистой границы нет вовсе (хвост упёрся в длинную
    инструментальную цепочку — например, в снимок экрана, который один больше бюджета),
    отступаем НАЗАД до ближайшей чистой границы, даже ценой бюджета.
    """
    if not messages:
        return []
    spent = 0
    start = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        spent += message_size(messages[index])
        if spent > budget and index != len(messages) - 1:
            break
        start = index
    while start < len(messages) and not is_clean_boundary(messages, start):
        start += 1
    if start >= len(messages):
        # Последнее сообщение брать нельзя: им может оказаться результат инструмента, а
        # результат без своего обращения поставщик отвергает — и запрос уходит пустым.
        # Слишком длинный запрос модель ещё обработает, пустой — нет.
        for index in range(len(messages) - 1, -1, -1):
            if is_clean_boundary(messages, index):
                return messages[index:]
        return list(messages)
    return messages[start:]


def total_budget_for(window_tokens: int | None) -> int:
    """Бюджет запроса в знаках от размера окна модели; окно неизвестно — прежние 150 000."""
    if not window_tokens or window_tokens <= 0:
        return TOTAL_BUDGET
    return max(MIN_TOTAL_BUDGET, int(window_tokens * WINDOW_SHARE * CHARS_PER_TOKEN))


def _is_real_user(message: dict) -> bool:
    return message.get("role") == "user" and not _is_tool_result(message)


def _stub_for(message: dict, size: int) -> str:
    name = message.get("name") or message.get("tool_name")
    label = f"«{name}» " if name else ""
    return (f"[{label}отпущен из запроса: {size} знаков; оригинал в сессии — повтори вызов или "
            "найди через session_search]")


def age_tool_results(messages: list[dict], *, keep_turns: int = KEEP_FULL_TURNS,
                     min_chars: int = AGE_MIN_CHARS) -> tuple[list[dict], int]:
    """Заменить в запросе старые длинные результаты инструментов заглушкой. Хранилище не трогается.

    «Старые» — до реплики человека, с которой начинаются последние keep_turns ходов; текущий ход
    и предыдущий остаются целиком. Обращение к инструменту и его пара не меняются: заменяется
    только текст результата, а `tool_call_id` и роль остаются, поэтому поставщик пару принимает.
    Возвращает новый список и число замен; вход не изменяется.
    """
    user_positions = [i for i, m in enumerate(messages) if _is_real_user(m)]
    if len(user_positions) < keep_turns:
        return messages, 0
    boundary = user_positions[-keep_turns]
    aged = 0
    out = list(messages)
    for index in range(boundary):
        message = messages[index]
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, str) or len(content) <= min_chars:
            continue
        out[index] = {**message, "content": _stub_for(message, len(content))}
        aged += 1
    return (out, aged) if aged else (messages, 0)


def message_key(message: dict) -> int:
    """Отпечаток сообщения по содержимому: им запоминается, с какой реплики начинается хвост.

    У сообщений запроса нет стабильных номеров — хозяин каждый раз собирает их заново, — а
    содержимое реплики между запросами не меняется.
    """
    return hash((message.get("role"), str(message.get("content")),
                 message.get("tool_call_id"), str(message.get("tool_calls"))))


def select_tail(messages: list[dict], budget: int, anchor: int | None = None) -> tuple[list[dict], int | None]:
    """Хвост разговора со ступенчатым началом: неподвижный, пока влезает, и крупный шаг, когда нет.

    Скользящий хвост (начало пересчитывается на каждом запросе) сдвигается почти на каждом ходу, а
    каждый сдвиг делает недействительным кеш поставщика для всего после системной части. Здесь
    начало запоминается: пока хвост от него влезает в бюджет, он остаётся тем же и запрос
    дописывается только в конец. Когда не влезает — режем сразу до STEP_KEEP от бюджета и
    запоминаем новое начало. Возвращает хвост и отпечаток его первой реплики.
    """
    if not messages:
        return [], anchor
    if anchor is not None:
        for index, message in enumerate(messages):
            if message_key(message) != anchor:
                continue
            kept = messages[index:]
            if is_clean_boundary(messages, index) and sum(message_size(m) for m in kept) <= budget:
                return kept, anchor
            break
    if sum(message_size(m) for m in messages) <= budget:
        tail = safe_tail(messages, budget)
    else:
        tail = safe_tail(messages, max(int(budget * STEP_KEEP), 1))
    return tail, message_key(tail[0]) if tail else anchor


def _clip(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    return text[:budget].rstrip() + " […]"


def _is_operational(content: str) -> bool:
    try:
        from plugins.memory.aida.write import is_operational

        return is_operational(content)
    except Exception:
        return False


def _words(text: str) -> set[str]:
    try:
        from plugins.memory.aida.search import content_words

        return set(content_words(text))
    except Exception:
        return {w for w in re.findall(r"\w+", text.lower()) if len(w) >= 4}


def _near_duplicate(item: str, seen: list[set[str]]) -> bool:
    words = _words(item)
    if not words:
        return False
    for other in seen:
        if other and len(words & other) / len(words | other) >= DUPLICATE_OVERLAP:
            return True
    return False


def section_of(row: dict) -> str | None:
    """В какой раздел блока идёт строка; None — не идёт вовсе (событие, шум, диалог)."""
    memory_type = row.get("memory_type") or ""
    content = str(row.get("content") or "")
    if memory_type in _SKIPPED_TYPES or _is_operational(content):
        return None
    if memory_type != "lesson" and _BAN_RE.search(content):
        return SECTION_BANS
    return _SECTION_BY_TYPE.get(memory_type, SECTION_PROJECTS)


def build_knowledge_block(rows: Iterable[dict], budget: int = KNOWLEDGE_BUDGET) -> str:
    """То, что верно всегда: знание с приоритетом P1, разложенное по разделам в порядке чтения.

    Записей больше, чем влезает (на 13.10.2026 — 481 на 67 000 знаков), поэтому отбор, а не
    обрезка: строки берутся по кругу из каждого раздела в порядке прихода (свежие первыми),
    пока есть бюджет, — редкий раздел (запреты, уроки) не вытесняется толпой проектных заметок,
    а обрезанных на полуслове строк нет вовсе. Внутри раздела строки идут по типу: сначала
    редкие (контакты, решения), потом частые. Событие, операционный шум (хеш, версия,
    «задеплоено») и повтор уже взятой строки в блок не попадают.
    """
    grouped: dict[str, list[str]] = {label: [] for label in SECTION_ORDER}
    for row in rows:
        section = section_of(row)
        item = " ".join(str(row.get("content", "")).split())
        if section and item:
            grouped[section].append(item)
    order = [label for label in SECTION_ORDER if grouped[label]]
    if not order:
        return ""
    chosen: dict[str, list[str]] = {label: [] for label in order}
    seen_words: list[set[str]] = []
    spent = sum(len(f"\n\n## {label}") for label in order) + len("# Что я знаю всегда")
    cursors = {label: 0 for label in order}
    progressed = True
    while progressed:
        progressed = False
        for label in order:
            items = grouped[label]
            index = cursors[label]
            if index >= len(items):
                continue
            item = items[index]
            cursors[label] = index + 1
            progressed = True
            if _near_duplicate(item, seen_words):
                continue  # та же мысль уже в блоке
            line = f"\n- {item}"
            if spent + len(line) > budget:
                continue  # эта не влезла; следующая в разделе может быть короче
            chosen[label].append(item)
            seen_words.append(_words(item))
            spent += len(line)
    lines = ["# Что я знаю всегда"]
    for label in order:
        if not chosen[label]:
            continue
        lines.append("")
        lines.append(f"## {label}")
        lines.extend(f"- {item}" for item in chosen[label])
    return "\n".join(lines)


def choose_within_budget(rows: Iterable[dict], render, budget: int) -> list[tuple[dict, str]]:
    """Строки по порядку, пока влезают целиком. Возвращает пары (строка, её текст в блоке)."""
    chosen: list[tuple[dict, str]] = []
    spent = 0
    for row in rows:
        line = render(row)
        if spent + len(line) > budget:
            break
        chosen.append((row, line))
        spent += len(line)
    return chosen


def _tagged_line(row: dict) -> str:
    content = " ".join(str(row.get("content", "")).split())
    created = row.get("created_at")
    when = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else ""
    tag = " · ".join(part for part in (row.get("memory_type"), when) if part)
    return f"- [{tag}] {content}" if tag else f"- {content}"


def build_lessons_block(rows: Iterable[dict], budget: int = LESSONS_BUDGET) -> str:
    """Уроки под то, что человек сейчас делает. Без дат: урок — правило, а не событие."""
    chosen = choose_within_budget(rows, lambda r: "- " + " ".join(str(r.get("content", "")).split()), budget)
    if not chosen:
        return ""
    return "# Уроки под эту задачу\n" + "\n".join(line for _, line in chosen)


def build_recalled_block(rows: Iterable[dict], budget: int = RECALLED_BUDGET) -> str:
    """Найденное по смыслу этого вопроса."""
    chosen = choose_within_budget(rows, _tagged_line, budget)
    if not chosen:
        return ""
    return "# Вспомнил под этот вопрос\n" + "\n".join(line for _, line in chosen)


def _quote_line(row: dict) -> str:
    content = " ".join(str(row.get("content", "")).split())
    created = row.get("created_at")
    when = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else ""
    who = "Оператор" if row.get("role") == "user" else "Я"
    return (f"[{when}] {who}: {_clip(content, OLD_QUOTE_CLIP)}" if when
            else f"{who}: {_clip(content, OLD_QUOTE_CLIP)}")


def select_quotes(rows: Iterable[dict], limit: int = MAX_OLD_QUOTES, question: str = "") -> list[dict]:
    """Какие строки разговора вообще цитировать: не монолог, не обёртка, по одной на эпизод,
    и только те, что делят с вопросом хотя бы две значимые основы слов."""
    rows = list(rows)
    try:
        from plugins.memory.aida.search import collapse_episodes

        return collapse_episodes(rows, limit, question)
    except Exception:
        return [r for r in rows if not str(r.get("content", "")).lstrip().startswith("[")][:limit]


def build_old_history_block(rows: Iterable[dict], budget: int = OLD_HISTORY_BUDGET) -> str:
    """Куски давнего разговора, которые всплыли по смыслу вопроса.

    Дословно, с датой и с указанием, кто говорил: это не пересказ, а цитата из того, что
    было сказано на самом деле. Не больше MAX_OLD_QUOTES и не длиннее OLD_QUOTE_CLIP каждая:
    цитата — чтобы узнать разговор, а не чтобы перечитать его.
    """
    chosen = choose_within_budget(select_quotes(rows), _quote_line, budget)
    if not chosen:
        return ""
    return "# Из давнего разговора, по смыслу вопроса\n" + "\n".join(line for _, line in chosen)


def _prepend_to_question(message: dict, block: str) -> dict:
    """Подмешать динамический слой к самому вопросу, не создавая нового сообщения.

    Системное сообщение в середине разговора принимают не все модели — на той, где
    работает оператор, оно возвращает ошибку и обрывает ответ. Поэтому изменчивая часть
    едет вместе с вопросом, ровно как в том движке, что работал на сервере.
    """
    content = message.get("content")
    if isinstance(content, str):
        merged: Any = f"{block}\n\n---\n\n{content}"
    elif isinstance(content, list):
        merged = [{"type": "text", "text": block}] + list(content)
    else:
        merged = block
    return {**message, "content": merged}


def _attach_to_current_question(tail: list[dict], dynamic: str) -> list[dict]:
    """Подмешать изменчивый слой к вопросу ТЕКУЩЕГО хода — последней реплике человека в хвосте.

    Ход состоит из нескольких запросов к модели (после каждого обращения к инструменту уходит
    новый), а хозяин историю после нашей подмены не меняет: подмешанное живёт ровно один запрос.
    Поэтому на каждом запросе хода подмешиваем заново к той же реплике — байты одинаковые, а
    ответ, который пишется после инструментов, не остаётся без найденного.
    """
    for index in range(len(tail) - 1, -1, -1):
        message = tail[index]
        if message.get("role") == "user" and not _is_tool_result(message):
            return tail[:index] + [_prepend_to_question(message, dynamic)] + tail[index + 1:]
    return tail


def assemble_with_anchor(
    system_head: list[dict],
    knowledge: str,
    recalled: str,
    old_history: str,
    conversation: list[dict],
    *,
    total_budget: int = TOTAL_BUDGET,
    anchor: int | None = None,
    age_tools: bool = True,
    lessons: str = "",
) -> tuple[list[dict], int | None, int]:
    """Собрать запрос заново и вернуть его, отпечаток начала хвоста и число состаренных результатов.

    Порядок не случайный. Системная часть и знание меняются редко и стоят первыми — на них
    держится кеш поставщика. Изменчивое подмешивается к вопросу текущего хода на каждом запросе
    хода, чтобы не вставлять системные сообщения в середину разговора. Начало хвоста
    запоминается между запросами (см. select_tail), чтобы кеш ломался редко и крупно.
    """
    spent = len(knowledge) + len(recalled) + len(old_history) + len(lessons)
    spent += sum(message_size(m) for m in system_head)
    tail_budget = max(MIN_TAIL_BUDGET, total_budget - spent)

    assembled = list(system_head)
    if knowledge:
        assembled.append({"role": "system", "content": knowledge})

    # Сначала стареем, потом режем хвост: бюджет должен считаться по тому, что реально уйдёт.
    aged, aged_count = age_tool_results(conversation) if age_tools else (conversation, 0)
    tail, new_anchor = select_tail(aged, tail_budget, anchor)
    # Уроки — выше всего изменчивого: они про то, как делать, а цитаты — про то, что было.
    dynamic = "\n\n".join(block for block in (lessons, recalled, old_history) if block)
    if dynamic:
        tail = _attach_to_current_question(tail, dynamic)
    assembled.extend(tail)
    return assembled, new_anchor, aged_count


def assemble(
    system_head: list[dict],
    knowledge: str,
    recalled: str,
    old_history: str,
    conversation: list[dict],
    *,
    total_budget: int = TOTAL_BUDGET,
    age_tools: bool = True,
    lessons: str = "",
) -> list[dict]:
    """Собрать запрос без памяти о прошлых запросах (начало хвоста считается заново)."""
    return assemble_with_anchor(system_head, knowledge, recalled, old_history, conversation,
                                total_budget=total_budget, age_tools=age_tools, lessons=lessons)[0]
