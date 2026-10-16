"""Цитаты «Из давнего разговора»: что нельзя цитировать и сколько можно.

По одному тесту на каждый класс шума, зафиксированный живым примером 17.10.2026: монолог
ассистента, строка-обёртка, три строки одного эпизода, четыре цитаты по две тысячи знаков.
"""

from __future__ import annotations

from datetime import datetime

from plugins.context_engine.aida.assembly import MAX_OLD_QUOTES, OLD_QUOTE_CLIP, build_old_history_block
from plugins.memory.aida.search import (
    DIALOGUE_MIN_SIM,
    FTS_SQL,
    MAX_QUOTES,
    QUOTE_MAX_ASSISTANT_CHARS,
    VECTOR_MIN_SIM,
    VECTOR_SQL,
    collapse_episodes,
    is_system_payload,
    quotable,
)


def _row(content, *, role="user", day=13, chat="hermes:a", **extra):
    row = {"id": hash(content) & 0xFFFF, "content": content, "memory_type": "dialogue", "role": role,
           "created_at": datetime(2026, 10, day), "chat_id": chat}
    row.update(extra)
    return row


def test_a_long_monologue_of_the_assistant_is_not_a_memory_to_quote():
    monologue = _row("Готово, мой дорогой друг. Фундамент заложен — коммит 4fa92a1 " * 20, role="assistant")
    short_reply = _row("Да, согласен — оно работает по ключевому слову, а не по смыслу", role="assistant")
    operator = _row("я тут думаю, что память должна помнить смысл, а не слова " * 20, role="user")

    assert len(monologue["content"]) > QUOTE_MAX_ASSISTANT_CHARS
    assert not quotable(monologue)
    assert quotable(short_reply)
    assert quotable(operator)  # длинная реплика человека — его воспоминание, её цитировать можно


def test_a_line_starting_with_a_bracket_is_a_wrapper_not_a_person_speaking():
    for payload in ('[Replying to: "…"] ваще кайфую, спасибо за разбор памяти',
                    "[ASYNC DELEGATION BATCH COMPLETE — 2 tasks] результаты субагентов ниже",
                    "[часть 2/5] Ты три года встречался не с ней, а с её будущей версией",
                    "  [Image attached at: C:\\shot.png] что скажешь про этот скриншот"):
        assert is_system_payload(payload)
        assert not quotable(_row(payload))


def test_three_lines_of_one_episode_become_one_quote():
    question = _row("что мне ответить Ангелине на это сообщение про выходные и встречу")
    answer = _row("ответь Ангелине про выходные коротко: встреча в субботу", role="assistant")
    retelling = _row("Это скриншот переписки в Telegram с Ангелиной про выходные", chat="hermes:b")
    other_day = _row("переписка с Ангелиной про выходные, но месяц назад", day=1, chat="hermes:c")

    kept = collapse_episodes([question, answer, retelling, other_day], limit=4)

    assert kept == [question, other_day]


def test_no_more_than_two_quotes_and_none_longer_than_four_hundred_signs():
    rows = [_row(f"реплика номер {i} про переезд и квартиру в Мытищах " * 30, day=i + 1, chat=f"hermes:{i}")
            for i in range(5)]

    block = build_old_history_block(rows)
    quotes = [line for line in block.splitlines() if line.startswith("[2026")]

    assert MAX_QUOTES == MAX_OLD_QUOTES == 2
    assert len(quotes) == 2
    assert all(len(q) <= OLD_QUOTE_CLIP + 40 and q.endswith("[…]") for q in quotes)


def test_the_block_says_who_spoke_from_the_stored_role():
    block = build_old_history_block([_row("я спросил", role="user"), _row("я ответил", role="assistant", day=14)])

    assert "Оператор: я спросил" in block and "Я: я ответил" in block


def test_dialogue_has_its_own_higher_vector_floor_and_needs_two_matched_words():
    assert VECTOR_MIN_SIM == 0.45 and DIALOGUE_MIN_SIM == 0.6
    assert f">= {VECTOR_MIN_SIM}" in VECTOR_SQL and f">= {DIALOGUE_MIN_SIM}" in VECTOR_SQL
    assert "scored.matched >= 2" in FTS_SQL
    # Обёртки и монологи отсекаются уже в SQL, чтобы не занимать мест в выдаче.
    for sql in (FTS_SQL, VECTOR_SQL):
        assert "left(ltrim(m.content), 1) <> '['" in sql
        assert f"char_length(m.content) > {QUOTE_MAX_ASSISTANT_CHARS}" in sql


def test_a_quote_must_share_two_significant_stems_with_the_question():
    question = "Как устроен твой движок памяти и станешь ли ты в итоге цифровой личностью, а не просто ботом?"
    filler_hit = _row("Это просто выдержка из памяти, мой дорогой друг — не вопрос по делу", role="assistant")
    real_hit = _row("движок памяти у тебя пока ищет по словам, личностью это не назовёшь", role="assistant", day=14)

    assert collapse_episodes([filler_hit, real_hit], question=question) == [real_hit]
    assert collapse_episodes([filler_hit, real_hit]) == [filler_hit, real_hit]  # без вопроса — без этого сита
