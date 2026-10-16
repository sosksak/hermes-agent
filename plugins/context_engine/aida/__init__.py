"""Контекстный движок «витрина»: окна нет, контекст собирается заново на каждый запрос.

Встроенный движок копит разговор в окне, а когда оно заполняется — пересказывает старую
часть и продолжает с пересказа. Дальше модель опирается на пересказ и достраивает утраченные
детали; это прямой источник выдуманных подробностей, и именно поэтому так делать запрещено.

Здесь вместо этого повторено устройство движка, который работал на сервере: контекст — не
хранилище, а витрина, и она собирается заново под каждый вопрос:

* устойчивый слой (личность хозяина и знание, которое верно всегда) стоит первым и меняется
  редко, чтобы поставщик мог попадать в свой кеш;
* изменчивый слой (найденное по смыслу вопроса и всплывшие куски давнего разговора) едет
  вместе с самим вопросом;
* свежий хвост разговора остаётся дословно, настоящими сообщениями.

Ничего не пересказывается: всё, что не поместилось, лежит в общей памяти целиком и находится
поиском. Движок читает ту же базу, что и память Гермеса, поэтому включать его имеет смысл
вместе с плагином памяти `aida`.

Включается строкой `context.engine: aida` в настройке; ошибка внутри безопасна — хозяин
поймает её и отправит запрос так, как отправил бы без движка.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections import OrderedDict
from typing import Any

from agent.context_engine import ContextEngine

from .assembly import (
    KNOWLEDGE_BUDGET,
    LESSONS_BUDGET,
    OLD_HISTORY_BUDGET,
    RECALLED_BUDGET,
    TOTAL_BUDGET,
    _quote_line,
    _tagged_line,
    assemble_with_anchor,
    build_knowledge_block,
    build_lessons_block,
    build_old_history_block,
    build_recalled_block,
    choose_within_budget,
    select_quotes,
    message_key,
    message_size,
    safe_tail,
    split_system,
    total_budget_for,
)

logger = logging.getLogger(__name__)

# Сколько живёт собранное. Внутри одного хода модель ходит к инструментам много раз, и
# каждый такой поход — новый запрос: без этого мы бы искали в базе по десять раз на ход.
DYNAMIC_TTL = 120.0
KNOWLEDGE_TTL = 600.0

RECALL_LIMIT = 8
OLD_HISTORY_LIMIT = 5
MAX_TRACKED_CONVERSATIONS = 64

# Обёртки шлюза вокруг реплики — не часть вопроса. «[Replying to: "…длинная цитата…"]» перед
# «ваще кайфую» делает вопросом цитату, а не реплику; «[Image attached at: C:\…]» ищет по
# пути к файлу. Снимаем, прежде чем искать.
_WRAPPER_RE = re.compile(
    r"\[Replying to: \".*?\"\]"
    r"|\[(?:Replied-to image|Image attached|Gateway message origin)[^\]]*\]",
    re.DOTALL,
)


def strip_wrappers(text: str) -> str:
    """Реплика без служебных обёрток шлюза — то, что человек на самом деле написал."""
    return " ".join(_WRAPPER_RE.sub(" ", text or "").split())

# Всё знание P1 целиком: отбор по важности и свежести делает build_knowledge_block, потому что
# «последние 60 по дате» теряли старые решения, а резались на бюджете посреди строки.
# Событие — не «знаю всегда», ему в этом блоке делать нечего при любом приоритете.
P1_SQL = """
SELECT content, memory_type FROM memories
WHERE status = 'active' AND priority = 'P1' AND memory_type NOT IN ('dialogue', 'event')
ORDER BY created_at DESC
"""

LESSONS_LIMIT = 6


def _detect_activities(question: str) -> list[str]:
    try:
        from plugins.memory.aida.lessons import detect_activities

        return detect_activities(question)
    except Exception:
        return []


def _open_recall_log():
    """Журнал отбора рядом с очередью памяти; без него работаем, как работали."""
    try:
        import os
        from pathlib import Path

        from plugins.memory.aida.recall_log import DB_NAME, RecallLog

        try:
            from hermes_constants import get_hermes_home

            home = get_hermes_home()
        except Exception:
            home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        return RecallLog(Path(home) / DB_NAME)
    except Exception:
        logger.debug("Витрина: журнал отбора недоступен, отбор не записывается", exc_info=True)
        return None


class _NoSignal:
    """Заглушка: память не установлена, значит и показывать отметку некому."""

    @staticmethod
    def announce_engine() -> None:
        pass

    @staticmethod
    def publish(count: int) -> None:
        pass


def _worth_searching(question: str) -> bool:
    """Та же проверка, что у памяти: нет памяти — нет и плагина, тогда считаем сами."""
    try:
        from plugins.memory.aida.search import worth_searching

        return worth_searching(question)
    except Exception:
        return len(question) >= 12


def _worth_quoting(question: str) -> bool:
    """Цитаты давнего разговора — только под настоящий вопрос, не под команду из двух слов."""
    try:
        from plugins.memory.aida.search import worth_quoting

        return worth_quoting(question)
    except Exception:
        return len(question.split()) >= 3


def _load_recall_signal():
    """Ниточка к памяти: она ставит отметку «вспомнил», а ищет теперь движок.

    Объявляемся не здесь, а при первой настоящей сборке: движок создаётся и просто затем,
    чтобы перечислить доступные, и такое перечисление не должно отключать поиск у памяти.
    """
    try:
        from plugins.memory.aida import recall_signal

        return recall_signal
    except Exception:
        logger.debug("Витрина: плагин памяти не найден, отметку о подъёме ставить некому")
        return _NoSignal()


class AidaContextEngine(ContextEngine):
    """Собирает контекст заново на каждый запрос и никогда не пересказывает разговор."""

    # Пересказа нет, поэтому и сообщать не о чем: «отпустил старое» — это норма работы,
    # а не событие. Предупреждения и ошибки хозяин показывает сам.
    emit_automatic_compaction_status = False

    @property
    def name(self) -> str:
        return "aida"

    def __init__(self) -> None:
        self.total_budget = TOTAL_BUDGET
        self._lock = threading.Lock()
        self._knowledge: tuple[float, str] = (0.0, "")
        # время, ключ, (уроки, факты, цитаты)
        self._dynamic: tuple[float, str, tuple[str, str, str]] = (0.0, "", ("", "", ""))
        self._recall_log: Any = None
        self._recall_log_tried = False
        self._store: Any = None
        self._store_tried = False
        self._recall_signal = _load_recall_signal()
        self._announced = False
        # Начало хвоста по разговору: ключ — отпечаток первой реплики разговора, значение —
        # отпечаток реплики, с которой хвост начинается. Один шлюз ведёт несколько разговоров.
        self._anchors: OrderedDict[int, int] = OrderedDict()
        # Сколько результатов инструментов уже состарено в запросе — по тому же ключу разговора.
        self._aged_counts: OrderedDict[int, int] = OrderedDict()

    # -- хозяйство -----------------------------------------------------------------

    def update_from_response(self, usage: dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Почти никогда: запрос уже собран под бюджет. Остаётся как страховка."""
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        return bool(self.threshold_tokens and tokens and tokens > self.threshold_tokens)

    def compress(self, messages: list[dict], current_tokens: int = None,
                 focus_topic: str = None, force: bool = False, memory_context: str = "") -> list[dict]:
        """Отпустить старое, а не пересказать его.

        Модель не зовём вообще: всё, что уходит из запроса, лежит в общей памяти целиком и
        находится поиском. Пересказ на его месте был бы единственным, на что модель дальше
        могла бы опереться — и ровно поэтому его здесь нет.
        """
        system_head, conversation = split_system(messages)
        head_size = sum(message_size(m) for m in system_head)
        kept = safe_tail(conversation, max(self.total_budget - head_size, 1))
        if len(kept) < len(conversation):
            self.compression_count += 1
            logger.info("Витрина: отпущено %d старых сообщений, они остаются в памяти",
                        len(conversation) - len(kept))
        return system_head + kept

    # -- витрина -------------------------------------------------------------------

    def select_context(self, request_messages: list[dict], *, conversation_messages: list[dict] = None,
                       incoming_message: dict = None, budget_tokens: int = 0) -> list[dict] | None:
        """Собрать контекст этого запроса заново. None — оставить всё как есть."""
        if not request_messages:
            return None
        store = self._get_store()
        if store is None:
            return None  # общая память не настроена — ведём себя как обычный движок
        self._recall_signal.announce_engine()
        if not self._announced:
            # Один раз за запуск, чтобы «движок работает» было видно в журнале, а не
            # выводилось из того, что ничего не сломалось.
            self._announced = True
            logger.info("Витрина: контекст этого разговора собирается заново, окна нет")
        try:
            system_head, conversation = split_system(request_messages)
            question = self._question_text(request_messages, incoming_message)
            knowledge = self._knowledge_block(store)
            lessons, recalled, old_history = self._dynamic_blocks(store, question)
            conversation_key = message_key(conversation[0]) if conversation else None
            with self._lock:
                anchor = self._anchors.get(conversation_key)
            if budget_tokens:
                self.total_budget = total_budget_for(budget_tokens)
            assembled, new_anchor, aged_count = assemble_with_anchor(
                system_head, knowledge, recalled, old_history, conversation,
                total_budget=self.total_budget, anchor=anchor, lessons=lessons)
            if conversation_key is not None:
                with self._lock:
                    grew = aged_count > self._aged_counts.get(conversation_key, 0)
                    if new_anchor is not None:
                        self._anchors[conversation_key] = new_anchor
                    self._aged_counts[conversation_key] = aged_count
                    for tracked in (self._anchors, self._aged_counts):
                        if conversation_key in tracked:
                            tracked.move_to_end(conversation_key)
                        while len(tracked) > MAX_TRACKED_CONVERSATIONS:
                            tracked.popitem(last=False)
                if grew:
                    # Модель, которой понадобился отпущенный результат, перечитает файл. Без
                    # сброса хозяин ответил бы «файл не менялся» — в расчёте на содержимое,
                    # которое мы только что убрали из запроса.
                    self._reset_read_dedup()
            return assembled
        except Exception:
            logger.warning("Витрина: собрать не удалось, запрос уходит как есть", exc_info=True)
            return None

    @staticmethod
    def _reset_read_dedup() -> None:
        """Штатный сброс учёта «этот файл уже читали и он не менялся» (для всех задач сразу)."""
        try:
            from tools.file_tools_read_tracking import reset_file_dedup

            reset_file_dedup(None)
        except Exception:
            logger.debug("Витрина: сбросить учёт чтения файлов не удалось", exc_info=True)
        try:
            from tools.skills_tool_dedup import reset_skill_view_dedup

            reset_skill_view_dedup(None)
        except Exception:
            logger.debug("Витрина: сбросить учёт просмотра навыков не удалось", exc_info=True)

    @staticmethod
    def _question_text(request_messages: list[dict], incoming_message: dict | None) -> str:
        source = incoming_message if isinstance(incoming_message, dict) else None
        if source is None:
            for message in reversed(request_messages):
                if message.get("role") == "user":
                    source = message
                    break
        if source is None:
            return ""
        content = source.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(content or "")

    def _knowledge_block(self, store: Any) -> str:
        with self._lock:
            stamped, block = self._knowledge
            # Проверяем время, а не наличие текста: пока знания нет вовсе, пустой ответ —
            # такой же ответ, и ходить за ним в базу на каждом обращении незачем.
            if stamped and time.monotonic() - stamped < KNOWLEDGE_TTL:
                return block
        rows = [{"content": row[0], "memory_type": row[1]} for row in store._execute_guarded(P1_SQL, {})]
        block = build_knowledge_block(rows, KNOWLEDGE_BUDGET)
        with self._lock:
            self._knowledge = (time.monotonic(), block)
        return block

    def _dynamic_blocks(self, store: Any, question: str) -> tuple[str, str, str]:
        """Найденное под вопрос: уроки под эту работу, записи по смыслу и куски давнего разговора.

        Один поиск на факты и цитаты: разговорные строки лежат в той же таблице, что и факты,
        и отличаются только типом. Уроки — отдельной выборкой по тегу активности: урок про
        node --check должен приехать на любую сборку, даже без слова «бэктик» в вопросе.
        """
        question = (question or "").strip()
        if not question:
            return "", "", ""
        question = strip_wrappers(question)
        if not _worth_searching(question):
            # «?», «на», «😂😂😂»: искать нечего. Поиск по такому всегда что-то возвращает —
            # такие же обрывки, — и модель отвечала на них вместо реплики человека.
            self._recall_signal.publish(0)
            return "", "", ""
        key = question[:400]
        with self._lock:
            stamped, cached_key, cached = self._dynamic
            if cached_key == key and time.monotonic() - stamped < DYNAMIC_TTL:
                return cached

        found = store.recall(question, RECALL_LIMIT + OLD_HISTORY_LIMIT)
        activities = _detect_activities(question)
        lesson_rows = (store.lessons_for(activities, LESSONS_LIMIT)
                       if activities and hasattr(store, "lessons_for") else [])
        lesson_ids = {row.get("id") for row in lesson_rows}
        facts = [row for row in found
                 if row.get("memory_type") != "dialogue" and row.get("id") not in lesson_ids][:RECALL_LIMIT]
        talk = (select_quotes([row for row in found if row.get("memory_type") == "dialogue"][:OLD_HISTORY_LIMIT],
                              question=question)
                if _worth_quoting(question) else [])
        kept_lessons = choose_within_budget(
            lesson_rows, lambda r: "- " + " ".join(str(r.get("content", "")).split()), LESSONS_BUDGET)
        kept_facts = choose_within_budget(facts, _tagged_line, RECALLED_BUDGET)
        kept_talk = choose_within_budget(talk, _quote_line, OLD_HISTORY_BUDGET)
        lessons = build_lessons_block([row for row, _ in kept_lessons], LESSONS_BUDGET)
        recalled = build_recalled_block([row for row, _ in kept_facts], RECALLED_BUDGET)
        old_history = build_old_history_block([row for row, _ in kept_talk], OLD_HISTORY_BUDGET)
        # Оператор договорился, что подъём памяти всегда видно. Ищет теперь движок, но
        # отметку по-прежнему ставит память — число едет к ней отсюда.
        self._recall_signal.publish(len(kept_lessons) + len(kept_facts) + len(kept_talk))
        self._log_recall(question, (("уроки", lesson_rows, kept_lessons), ("факты", facts, kept_facts),
                                    ("цитаты", talk, kept_talk)))

        with self._lock:
            # Кэшируем все блоки как есть: внутри хода запрос уходит несколько раз, и раньше
            # повторный запрос получал только факты — цитаты давнего разговора терялись.
            self._dynamic = (time.monotonic(), key, (lessons, recalled, old_history))
        return lessons, recalled, old_history

    def _log_recall(self, question: str, blocks) -> None:
        """Записать в журнал, что поднялось и что доехало; любая ошибка здесь глотается."""
        if not self._recall_log_tried:
            self._recall_log_tried = True
            self._recall_log = _open_recall_log()
        if self._recall_log is None:
            return
        entries = []
        for block, rows, kept in blocks:
            kept_ids = {id(row) for row, _ in kept}
            sizes = {id(row): len(line) for row, line in kept}
            for row in rows:
                entries.append({
                    "id": row.get("id"), "memory_type": row.get("memory_type"), "block": block,
                    "score": row.get("score", row.get("rank")),
                    "chars": sizes.get(id(row), len(" ".join(str(row.get("content", "")).split()))),
                    "kept": id(row) in kept_ids,
                })
        try:
            self._recall_log.record(question, entries)
        except Exception:
            logger.debug("Витрина: журнал отбора не записался", exc_info=True)

    def _get_store(self) -> Any:
        """Та же база, что у памяти Гермеса. Второго подключения не заводим."""
        if self._store is not None or self._store_tried:
            return self._store
        self._store_tried = True
        try:
            from agent.secret_scope import get_secret
            from plugins.memory.aida import _Store

            url = get_secret("SUPABASE_DB_URL", "") or ""
            if not url:
                logger.info("Витрина: общая память не настроена, движок не вмешивается")
                return None
            self._store = _Store(url, get_secret("OPENROUTER_API_KEY", "") or "")
        except Exception:
            logger.warning("Витрина: подключиться к общей памяти не удалось", exc_info=True)
            self._store = None
        return self._store

    def on_session_end(self, session_id: str, messages: list[dict]) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None
            self._store_tried = False


def register(ctx) -> None:
    """Зарегистрировать витрину как контекстный движок."""
    ctx.register_context_engine(AidaContextEngine())
