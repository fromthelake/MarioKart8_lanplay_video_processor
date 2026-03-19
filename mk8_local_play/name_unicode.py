from __future__ import annotations

import json
import unicodedata
from functools import lru_cache
from pathlib import Path


def collapse_name_whitespace(text: str | None) -> str:
    value = "" if text is None else str(text)
    return " ".join(value.split())


def normalize_name_key(text: str | None) -> str:
    collapsed = collapse_name_whitespace(text)
    if not collapsed:
        return ""
    return unicodedata.normalize("NFKC", collapsed).casefold()


def visible_name_characters(text: str | None) -> list[str]:
    return [char for char in collapse_name_whitespace(text) if not char.isspace()]


def visible_name_length(text: str | None) -> int:
    return len(visible_name_characters(text))


def distinct_visible_name_count(text: str | None) -> int:
    return len(set(visible_name_characters(text)))


@lru_cache(maxsize=1)
def load_allowed_name_char_data() -> dict:
    data_path = Path(__file__).resolve().parent / "data" / "mii_allowed_chars.json"
    with data_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        return {}
    return data


@lru_cache(maxsize=1)
def _allowed_name_chars() -> set[str]:
    data = load_allowed_name_char_data()
    chars: set[str] = set()
    for key in ("latin_extended", "japanese", "special_symbols"):
        value = data.get(key, "")
        if isinstance(value, str):
            chars.update(value)
    return chars


ALLOWED_NAME_CHARS = _allowed_name_chars()


def is_allowed_name_char(ch: str) -> bool:
    return ch in ALLOWED_NAME_CHARS


def allowed_name_char_ratio(text: str | None) -> float:
    visible = visible_name_characters(text)
    if not visible:
        return 0.0
    allowed = sum(1 for char in visible if is_allowed_name_char(char))
    return allowed / len(visible)


def unknown_name_chars(text: str | None) -> str:
    seen: set[str] = set()
    unknown_chars: list[str] = []
    for char in visible_name_characters(text):
        if is_allowed_name_char(char) or char in seen:
            continue
        seen.add(char)
        unknown_chars.append(char)
    return "".join(unknown_chars)

