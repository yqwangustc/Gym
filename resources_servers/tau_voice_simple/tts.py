# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible GPT-4o mini TTS client for Tau-Voice Simple user turns.

Ported from VLMEvalKitMcore's ``tau_voice_simple/openai_tts.py``: the same
11-voice pool, request payload, instructions, retry schedule, and WAV checks.
Endpoint and key come from the server config instead of environment lookups.
"""

import http.client
import io
import json
import logging
import math
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from time import sleep
from typing import Callable


LOG = logging.getLogger(__name__)


class _EmptyTTSResponseError(RuntimeError):
    """Transient empty audio response."""


# The reference evaluation's fixed voice pool, in round-robin order.
GPT4O_MINI_TTS_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
)
DEFAULT_TTS_INSTRUCTIONS = (
    "Speak naturally and clearly as a customer in a phone conversation. "
    "Preserve the supplied wording exactly: do not add, omit, or paraphrase words."
)


def voice_for_session(session_ordinal: int, voices: tuple[str, ...] = GPT4O_MINI_TTS_VOICES) -> str:
    """Return the round-robin voice for one complete session."""
    if session_ordinal < 0:
        raise ValueError("session_ordinal must be non-negative")
    return voices[session_ordinal % len(voices)]


@dataclass(frozen=True)
class TTSConfig:
    endpoint: str
    api_key: str = field(repr=False)
    model: str
    voice: str
    instructions: str = DEFAULT_TTS_INSTRUCTIONS
    timeout_seconds: float = 120.0
    max_retries: int = -1


@dataclass(frozen=True)
class SynthesisResult:
    pcm_s16le: bytes
    sample_rate: int
    model: str
    voice: str

    def wav_bytes(self) -> bytes:
        return pcm16_wav_bytes(self.pcm_s16le, self.sample_rate)


class TTSClient:
    """Synthesize one session's user turns with a fixed voice."""

    def __init__(self, config: TTSConfig, *, urlopen: Callable | None = None):
        if config.voice not in GPT4O_MINI_TTS_VOICES:
            raise ValueError(f"Unsupported TTS voice {config.voice!r}; choose from {list(GPT4O_MINI_TTS_VOICES)}")
        if not math.isfinite(config.timeout_seconds) or config.timeout_seconds < 0:
            raise ValueError("TTS timeout_seconds must be non-negative and finite")
        if config.max_retries < -1:
            raise ValueError("TTS max_retries must be -1 or non-negative")
        self.config = config
        self._urlopen = urlopen or urllib.request.urlopen

    def synthesize(self, text: str) -> SynthesisResult:
        text = text.strip()
        if not text:
            raise ValueError("TTS cannot synthesize empty text")
        payload = {
            "model": self.config.model,
            "voice": self.config.voice,
            "input": text,
            "response_format": "wav",
        }
        if self.config.instructions:
            payload["instructions"] = self.config.instructions
        request = urllib.request.Request(
            self.config.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
                "Accept": "audio/wav",
            },
        )
        pcm_s16le, sample_rate = decode_pcm16_wav(self._request_with_retries(request))
        return SynthesisResult(pcm_s16le, sample_rate, self.config.model, self.config.voice)

    def _request_with_retries(
        self,
        request: urllib.request.Request,
    ) -> bytes:
        api_key = self.config.api_key
        attempt = 0
        while True:
            try:
                LOG.info("TauVoiceSimple waiting for user TTS (attempt %d)", attempt + 1)
                with self._urlopen(
                    request,
                    timeout=self.config.timeout_seconds or None,
                ) as response:
                    audio = response.read()
                    if not audio:
                        raise _EmptyTTSResponseError("TTS endpoint returned an empty response")
                    if attempt:
                        LOG.info("TauVoiceSimple user TTS recovered after %d retries", attempt)
                    return audio
            except urllib.error.HTTPError as exc:
                # This gateway sometimes labels its own DB saturation as 401.
                # Retry only that known infrastructure error; invalid credentials
                # and budget failures must still stop immediately.
                body = ""
                if exc.code == 401:
                    try:
                        body = body or exc.read(1000).decode("utf-8", errors="replace")
                    except (OSError, http.client.HTTPException):
                        body = "<error body unavailable>"
                gateway_overloaded = (
                    exc.code == 401 and "Error querying the database" in body and "too many clients already" in body
                )
                retryable = gateway_overloaded or exc.code == 429 or 500 <= exc.code < 600
                if not retryable or (self.config.max_retries >= 0 and attempt >= self.config.max_retries):
                    try:
                        body = body or exc.read(1000).decode("utf-8", errors="replace")
                    except (OSError, http.client.HTTPException):
                        body = "<error body unavailable>"
                    finally:
                        exc.close()
                    body = body.replace(api_key, "<redacted>")
                    raise RuntimeError(f"TTS request failed with HTTP {exc.code}: {body}") from exc
                failure = f"HTTP {exc.code}"
                exc.close()
            except (OSError, http.client.HTTPException, _EmptyTTSResponseError) as exc:
                if self.config.max_retries >= 0 and attempt >= self.config.max_retries:
                    reason = str(getattr(exc, "reason", exc)).replace(api_key, "<redacted>")
                    raise RuntimeError(f"TTS request failed: {reason}") from exc
                failure = type(exc).__name__
            delay = min(30.0, 2.0 ** min(attempt, 5))
            # Do not log credentials, request text, URLs, or provider error bodies.
            LOG.warning(
                "TauVoiceSimple transient user TTS failure (%s); retrying same turn in %.0fs (retry %d, limit=%s)",
                failure,
                delay,
                attempt + 1,
                "unlimited" if self.config.max_retries == -1 else self.config.max_retries,
            )
            sleep(delay)
            attempt += 1


def decode_pcm16_wav(wav_bytes: bytes) -> tuple[bytes, int]:
    """Return mono PCM16 samples and the sample rate, rejecting any other WAV."""
    if not wav_bytes.startswith(b"RIFF") or wav_bytes[8:12] != b"WAVE":
        raise RuntimeError("TTS endpoint did not return RIFF/WAV audio")
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            pcm_s16le = wav_file.readframes(wav_file.getnframes())
    except (EOFError, wave.Error) as exc:
        raise RuntimeError(f"TTS endpoint returned invalid WAV audio: {exc}") from exc
    if channels != 1 or sample_width != 2 or sample_rate <= 0:
        raise RuntimeError(
            "Tau-Voice Simple requires mono PCM16 WAV audio; received "
            f"channels={channels}, sample_width={sample_width}, sample_rate={sample_rate}"
        )
    if not pcm_s16le or len(pcm_s16le) % 2:
        raise RuntimeError("TTS endpoint returned invalid or empty PCM16 samples")
    return pcm_s16le, sample_rate


def pcm16_wav_bytes(pcm_s16le: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_s16le)
    return buffer.getvalue()
