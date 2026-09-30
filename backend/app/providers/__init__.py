from app.providers.base import (
    CancellationToken,
    ProgressCallback,
    RawTranscription,
    TranscriptionProvider,
    TranscriptionRequest,
)
from app.providers.fake import FakeProvider
from app.providers.local_mlx import LocalMLXProvider

_PROVIDERS: dict[str, type[TranscriptionProvider]] = {
    LocalMLXProvider.name: LocalMLXProvider,
    FakeProvider.name: FakeProvider,
}


def create_provider(name: str) -> TranscriptionProvider:
    try:
        provider_cls = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"Unknown transcription provider: {name}") from None
    return provider_cls()


def provider_names() -> list[str]:
    return list(_PROVIDERS)


__all__ = [
    "CancellationToken",
    "FakeProvider",
    "LocalMLXProvider",
    "ProgressCallback",
    "RawTranscription",
    "TranscriptionProvider",
    "TranscriptionRequest",
    "create_provider",
    "provider_names",
]
