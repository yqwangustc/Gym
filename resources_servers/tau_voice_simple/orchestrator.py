# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Keep user TTS waits outside the Tau episode timeout."""

from tau2.orchestrator.orchestrator import Orchestrator


CALLER_PROTOCOL = "caller-tool-priority-v1"


class VoiceOrchestrator(Orchestrator):
    """Retain native communication checks and account separately for TTS waits."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initial_tts_wait = self.user.tts_wait_seconds

    def _check_timeout(self):
        timeout = self.timeout
        if timeout is not None:
            self.timeout += max(0.0, self.user.tts_wait_seconds - self._initial_tts_wait)
        try:
            return super()._check_timeout()
        finally:
            self.timeout = timeout
