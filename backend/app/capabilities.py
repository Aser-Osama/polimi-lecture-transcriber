"""Platform capability detection.

The local MLX Whisper backend and WhisperX forced alignment require macOS on
Apple silicon. Windows and Linux run the OpenRouter cloud path only: every
feature that needs this machine (MLX models, WhisperX, the native file picker,
Keychain) is gated behind these functions.

All checks read the current process state on every call so tests (and users)
can simulate another platform with ``PT_SIMULATE_PLATFORM``. The override is
meant for development and tests; normal installs leave it unset.
"""

from __future__ import annotations

import os
import sys

_SIMULATED_PLATFORM_ENV = "PT_SIMULATE_PLATFORM"


def sys_platform() -> str:
    """The platform used for capability decisions (``darwin``/``win32``/``linux``)."""
    override = os.environ.get(_SIMULATED_PLATFORM_ENV, "").strip()
    return override or sys.platform


def is_macos() -> bool:
    return sys_platform() == "darwin"


def platform_label() -> str:
    platform = sys_platform()
    if platform == "darwin":
        return "macOS"
    if platform == "win32":
        return "Windows"
    if platform.startswith("linux"):
        return "Linux"
    return platform


def local_transcription_supported() -> bool:
    """MLX Whisper runs only on macOS (Apple silicon)."""
    return is_macos()


def local_alignment_supported() -> bool:
    """WhisperX forced alignment runs only on macOS."""
    return is_macos()


def native_file_picker_supported() -> bool:
    """The OS file picker (osascript) exists only on macOS."""
    return is_macos()


def key_storage_kind() -> str:
    """Where the OpenRouter API key lives: ``keychain`` or ``file``."""
    return "keychain" if is_macos() else "file"


def capabilities() -> dict:
    """Serializable capability summary for the API/UI."""
    return {
        "platform": sys_platform(),
        "platform_label": platform_label(),
        "local_transcription": local_transcription_supported(),
        "local_alignment": local_alignment_supported(),
        "native_file_picker": native_file_picker_supported(),
        "key_storage": key_storage_kind(),
    }
