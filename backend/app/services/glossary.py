"""Course vocabulary handling for the Whisper initial prompt.

The glossary is cleaned (trimmed, de-duplicated, empty entries removed), given
a short natural-language frame and truncated at a term boundary so it always
stays within Whisper's 224-token prompt budget.
"""

from __future__ import annotations

MAX_PROMPT_CHARS = 600
_PROMPT_PREFIX = "Politecnico di Milano university lecture. Technical terms: "


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


def build_initial_prompt(raw: str, max_chars: int = MAX_PROMPT_CHARS) -> str | None:
    terms = clean_glossary_terms(raw)
    if not terms:
        return None
    budget = max_chars - len(_PROMPT_PREFIX)
    if budget <= 0:
        raise ValueError("max_chars too small for prompt prefix")
    kept: list[str] = []
    length = 0
    for term in terms:
        addition = len(term) + (2 if kept else 0)
        if length + addition > budget:
            break
        kept.append(term)
        length += addition
    if not kept:
        return None
    return _PROMPT_PREFIX + ", ".join(kept)


def normalized_glossary(raw: str) -> str:
    return "\n".join(clean_glossary_terms(raw))
