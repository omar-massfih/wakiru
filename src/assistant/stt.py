"""Local, offline speech-to-text via faster-whisper (CTranslate2 — no torch).

Mirrors :mod:`assistant.memory.embeddings`: the model is loaded lazily and
cached process-wide; the first call downloads it into the HuggingFace cache
once (~180 MB for ``small`` at int8), every call afterwards runs fully offline.
Audio decoding (Telegram's OGG/Opus included) happens inside faster-whisper via
PyAV — no system ffmpeg needed.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .config import Settings, get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def _model(model_name: str):
    from faster_whisper import WhisperModel

    # int8 on CPU: fastest practical setting for a personal server, and the
    # accuracy loss vs float32 is negligible for dictation-length clips.
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe(audio_path: str, settings: Settings | None = None) -> str:
    """The spoken text in the audio file at ``audio_path`` (any common format)."""
    settings = settings or get_settings()
    segments, info = _model(settings.voice_model).transcribe(
        audio_path,
        # None => autodetect (the user may mix Norwegian and English).
        language=settings.voice_language or None,
        vad_filter=True,  # skip leading/trailing silence instead of hallucinating
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    logger.info(
        "transcribed %.1fs of audio (lang=%s, p=%.2f): %.80r",
        info.duration, info.language, info.language_probability, text,
    )
    return text
