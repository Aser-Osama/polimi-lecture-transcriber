from app import capabilities
from app.providers.base import (
    CancellationToken,
    ProgressCallback,
    RawTranscription,
    TranscriptionProvider,
    TranscriptionRequest,
)
from app.providers.fake import FakeProvider
from app.providers.local_mlx import LocalMLXProvider
from app.providers.openrouter import OpenRouterProvider

_PROVIDERS: dict[str, type[TranscriptionProvider]] = {
    LocalMLXProvider.name: LocalMLXProvider,
    OpenRouterProvider.name: OpenRouterProvider,
    FakeProvider.name: FakeProvider,
}


def create_provider(name: str) -> TranscriptionProvider:
    if name == LocalMLXProvider.name and not capabilities.local_transcription_supported():
        raise ValueError(
            "Local transcription (MLX Whisper) is only available on macOS. "
            "Use the OpenRouter backend on this platform."
        )
    try:
        provider_cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown transcription provider: {name}") from None
    return provider_cls()


def provider_names() -> list[str]:
    return [
        name
        for name in _PROVIDERS
        if name != LocalMLXProvider.name or capabilities.local_transcription_supported()
    ]


__all__ = [
    "CancellationToken",
    "FakeProvider",
    "LocalMLXProvider",
    "OpenRouterProvider",
    "ProgressCallback",
    "RawTranscription",
    "TranscriptionProvider",
    "TranscriptionRequest",
    "create_provider",
    "provider_names",
]
