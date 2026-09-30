from __future__ import annotations

from app.services.context import (
    merge_context_terms,
    merge_context_texts,
    resolve_context,
)


def test_merge_terms_broad_to_specific_order():
    terms = merge_context_terms(["NUMA, TLB", "MESI", "spinlock"])
    assert terms == ["NUMA", "TLB", "MESI", "spinlock"]


def test_specific_context_overrides_spelling():
    terms = merge_context_terms(["numa, tlb", "NUMA, cache coherence"])
    assert terms == ["NUMA", "tlb", "cache coherence"]


def test_duplicate_terms_collapse_case_insensitively():
    terms = merge_context_terms(["TLB\nNUMA", "tlb\nMESI", "TLB"])
    assert terms == ["TLB", "NUMA", "MESI"]


def test_empty_levels_are_ignored():
    assert merge_context_terms(["", "  ", "CUDA"]) == ["CUDA"]
    assert merge_context_terms(["", "", ""]) == []


def test_merge_texts_joins_non_empty_levels():
    merged = merge_context_texts("global term", "", "file term")
    assert merged == "global term\nfile term"
    assert merge_context_texts("", "", "") == ""


def test_resolve_context_applies_specific_spelling_in_effective():
    resolved = resolve_context(
        global_context="numa",
        project=("p1", "OS", "NUMA"),
        per_file_context="",
    )
    assert resolved.effective == "NUMA"
    assert resolved.terms == ["NUMA"]


def test_resolve_context_full_breakdown():
    resolved = resolve_context(
        global_context="Politecnico di Milano",
        project=("p1", "Operating Systems", "NUMA, TLB"),
        per_file_context="guest speaker Daniele Cattaneo",
    )
    assert resolved.project_id == "p1"
    assert resolved.project_name == "Operating Systems"
    assert resolved.global_context == "Politecnico di Milano"
    assert resolved.per_file_context == "guest speaker Daniele Cattaneo"
    assert merged_terms(resolved) == [
        "Politecnico di Milano",
        "NUMA",
        "TLB",
        "guest speaker Daniele Cattaneo",
    ]
    assert resolved.effective == "\n".join(resolved.terms)
    assert resolved.prompt is not None and "NUMA" in resolved.prompt


def merged_terms(resolved) -> list[str]:
    return resolved.terms


def test_resolve_context_without_project():
    resolved = resolve_context(global_context="", project=None, per_file_context="only file")
    assert resolved.project_id is None
    assert resolved.project_name is None
    assert resolved.terms == ["only file"]
    assert resolved.prompt is not None


def test_resolve_context_empty_everything():
    resolved = resolve_context()
    assert resolved.terms == []
    assert resolved.prompt is None
    assert resolved.effective == ""
