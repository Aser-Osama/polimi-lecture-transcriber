from app.services.glossary import (
    PROMPT_PREFIX,
    build_initial_prompt,
    clean_glossary_terms,
    normalized_glossary,
)


def test_parses_lines_and_commas():
    raw = "NUMA, TLB\nMESI, cache coherence\nCUDA\n"
    assert clean_glossary_terms(raw) == ["NUMA", "TLB", "MESI", "cache coherence", "CUDA"]


def test_deduplicates_case_insensitively_preserving_first_spelling():
    raw = "NUMA\nnuma\nTLB\ntlb\n"
    assert clean_glossary_terms(raw) == ["NUMA", "TLB"]


def test_ignores_empty_entries_and_trims():
    raw = "  , ,\n  spinlock  \n\n,\nmutex,,\n"
    assert clean_glossary_terms(raw) == ["spinlock", "mutex"]


def test_compacts_internal_whitespace():
    assert clean_glossary_terms("virtual    memory") == ["virtual memory"]


def test_prompt_none_for_empty():
    assert build_initial_prompt("") is None
    assert build_initial_prompt("  ,  \n") is None


def test_prompt_includes_terms():
    prompt = build_initial_prompt("NUMA, TLB")
    assert prompt is not None
    assert "NUMA, TLB" in prompt


def test_prompt_respects_max_chars():
    raw = "\n".join(f"verylongtechnicalterm{i}" for i in range(200))
    prompt = build_initial_prompt(raw, max_chars=300)
    assert prompt is not None
    assert len(prompt) <= 300
    assert "verylongtechnicalterm0" in prompt


def test_prompt_truncation_keeps_whole_terms():
    raw = "term_one, term_two, term_three"
    prompt = build_initial_prompt(raw, max_chars=len(PROMPT_PREFIX) + 15)
    assert prompt is not None
    assert "term_one" in prompt
    assert "term_three" not in prompt
    assert not prompt.endswith(",")


def test_normalized_glossary_round_trip():
    assert normalized_glossary("a, b\na\n") == "a\nb"
