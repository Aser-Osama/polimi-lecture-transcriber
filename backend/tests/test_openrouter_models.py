from app.services.openrouter_models import (
    DEFAULT_OPENROUTER_MODEL,
    is_known_openrouter_model,
    openrouter_model_choices,
)


def test_catalog_has_the_five_requested_models():
    ids = {entry["id"] for entry in openrouter_model_choices()}
    assert ids == {
        "microsoft/mai-transcribe-2",
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo",
        "qwen/qwen3-asr-1.7b",
        "qwen/qwen3-asr-flash-2026-02-10",
    }


def test_default_is_mai_transcribe_2():
    assert DEFAULT_OPENROUTER_MODEL == "microsoft/mai-transcribe-2"
    defaults = [entry for entry in openrouter_model_choices() if entry["default"]]
    assert len(defaults) == 1
    assert defaults[0]["id"] == DEFAULT_OPENROUTER_MODEL


def test_membership_check():
    assert is_known_openrouter_model("openai/whisper-large-v3")
    assert not is_known_openrouter_model("openai/gpt-4o-transcribe")
