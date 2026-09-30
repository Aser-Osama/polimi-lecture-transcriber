"""Hierarchical transcription context: global + course/project + per-file.

The three levels are merged with the *most specific level winning*: terms are
de-duplicated case-insensitively in first-appearance order, and when the same
term appears at multiple levels the spelling from the most specific one is
kept. The merged raw text is what gets stored on the job (so a job is
reproducible even if the project is edited later); the merged term list is what
providers actually receive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.glossary import (
    MAX_PROMPT_CHARS,
    build_initial_prompt_from_terms,
    clean_glossary_terms,
)


@dataclass
class ResolvedContext:
    global_context: str = ""
    project_id: str | None = None
    project_name: str | None = None
    project_context: str = ""
    per_file_context: str = ""
    effective: str = ""
    terms: list[str] = field(default_factory=list)

    @property
    def prompt(self) -> str | None:
        return build_initial_prompt_from_terms(self.terms, MAX_PROMPT_CHARS)


def merge_context_texts(global_context: str = "", project_context: str = "", per_file_context: str = "") -> str:
    """Raw merged context: non-empty levels joined with newlines, specific last."""
    parts = [
        (global_context or "").strip(),
        (project_context or "").strip(),
        (per_file_context or "").strip(),
    ]
    return "\n".join(part for part in parts if part)


def merge_context_terms(levels: list[str]) -> list[str]:
    """Ordered, de-duplicated terms; the most specific level wins the spelling.

    ``levels`` must be ordered broad -> specific (global, project, file).
    """
    order: list[str] = []
    spelling: dict[str, str] = {}
    for level in levels:
        for term in clean_glossary_terms(level):
            folded = term.casefold()
            if folded not in spelling:
                order.append(folded)
            spelling[folded] = term
    return [spelling[folded] for folded in order]


def resolve_context(
    global_context: str = "",
    project: tuple[str | None, str, str] | None = None,
    per_file_context: str = "",
) -> ResolvedContext:
    """project = (project_id, project_name, project_context) or None."""
    project_id, project_name, project_context = project if project else (None, "", "")
    terms = merge_context_terms([global_context, project_context, per_file_context])
    # The effective context is the canonical merged term list: de-duplicated
    # case-insensitively with the most specific spelling winning, so re-cleaning
    # it later (for prompts/providers) is lossless.
    effective = "\n".join(terms)
    return ResolvedContext(
        global_context=(global_context or "").strip(),
        project_id=project_id,
        project_name=project_name or None,
        project_context=(project_context or "").strip(),
        per_file_context=(per_file_context or "").strip(),
        effective=effective,
        terms=terms,
    )
