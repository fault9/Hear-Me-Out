"""Buffered, server-authoritative study event delivery for VC proxies."""

from __future__ import annotations

import asyncio
import math
import os
import time
from typing import Any

import aiohttp
import numpy as np


class StudyEventBuffer:
    def __init__(self, session_id: str, engine: str, app_api_url: str):
        self.session_id = session_id
        self.engine = engine
        self.app_api_url = app_api_url.rstrip("/")
        self.token = os.environ.get("STUDY_EVENT_TOKEN", "local-study-events")
        self.rows: list[dict[str, Any]] = []
        self.sequence = 0
        self._flush_lock = asyncio.Lock()

    def add(self, event: str, **fields: Any) -> None:
        if not self.session_id or self.session_id.endswith("_CHECK"):
            return
        self.sequence += 1
        self.rows.append({
            "event": event,
            "event_sequence": self.sequence,
            "engine": self.engine,
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        })

    async def flush(self, *, force: bool = False) -> bool:
        async with self._flush_lock:
            if not self.rows or (not force and len(self.rows) < 128):
                return True
            batch = self.rows[:]
            url = f"{self.app_api_url}/api/study/internal/session/{self.session_id}/events"
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, json={"events": batch}, ssl=False,
                        headers={"X-Study-Event-Token": self.token},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as response:
                        if response.status != 200:
                            return False
                del self.rows[:len(batch)]
                return True
            except Exception:
                return False


class EnergySpeechTracker:
    """Deterministic input-energy intervals, labelled as estimates in events."""

    def __init__(self, sample_rate: int, threshold: float | None = None,
                 hangover_ms: int = 250):
        self.sample_rate = sample_rate
        self.threshold = threshold if threshold is not None else float(
            os.environ.get("STUDY_VAD_RMS_THRESHOLD", "0.012"))
        self.hangover_samples = int(sample_rate * hangover_ms / 1000)
        self.speaking = False
        self.last_active_sample = 0

    def update(self, wav: np.ndarray, start_sample: int, end_sample: int) -> list[dict]:
        rms = math.sqrt(float(np.mean(np.square(wav, dtype=np.float64)))) if len(wav) else 0.0
        active = rms >= self.threshold
        events: list[dict] = []
        if active:
            self.last_active_sample = end_sample
            if not self.speaking:
                self.speaking = True
                events.append({"event": "participant_speech_start", "input_sample": start_sample,
                               "detector": "rms", "rms": rms, "threshold": self.threshold})
        elif self.speaking and start_sample - self.last_active_sample >= self.hangover_samples:
            self.speaking = False
            events.append({"event": "participant_speech_end",
                           "input_sample": self.last_active_sample,
                           "detector": "rms", "threshold": self.threshold})
        return events

    def close(self) -> list[dict]:
        if not self.speaking:
            return []
        self.speaking = False
        return [{"event": "participant_speech_end", "input_sample": self.last_active_sample,
                 "detector": "rms", "threshold": self.threshold}]
