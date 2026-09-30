"""Context/terminology handling for ASR prompts.

Context text is cleaned (trimmed, de-duplicated, empty entries removed), given
a short natural-language frame and truncated at a term boundary so it always
stays within Whisper's 224-token prompt budget. The same cleaned term list is
what providers with non-prompt context support (e.g. keyword biasing) receive.
"""

from __future__ import annotations

MAX_PROMPT_CHARS = 600
PROMPT_PREFIX = "Relevant names and technical terms: "


def clean_glossary_terms(raw: str) -> list[str]:
    if not raw:
        return []
    parts: list[str] = []
    for line in raw.replace(";", "\n").splitlines():
        parts.extend(line.split(","))
    seen: set[str] = set()
    terms: list[str] = []
    for part in parts:
        term = " ".join(part.split())
        if not term:
            continue
        folded = term.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        terms.append(term)
    return terms


def build_initial_prompt_from_terms(
    terms: list[str], max_chars: int = MAX_PROMPT_CHARS
) -> str | None:
    cleaned = []
    seen: set[str] = set()
    for term in terms:
        normalized = " ".join(term.split())
        if not normalized or normalized.casefold() in seen:
            continue
        seen.add(normalized.casefold())
        cleaned.append(normalized)
    if not cleaned:
        return None
    budget = max_chars - len(PROMPT_PREFIX)
    if budget <= 0:
        raise ValueError("max_chars too small for prompt prefix")
    kept: list[str] = []
    length = 0
    for term in cleaned:
        addition = len(term) + (2 if kept else 0)
        if length + addition > budget:
            break
        kept.append(term)
        length += addition
    if not kept:
        return None
    return PROMPT_PREFIX + ", ".join(kept)


def build_initial_prompt(raw: str, max_chars: int = MAX_PROMPT_CHARS) -> str | None:
    return build_initial_prompt_from_terms(clean_glossary_terms(raw), max_chars)


def normalized_glossary(raw: str) -> str:
    return "\n".join(clean_glossary_terms(raw))
