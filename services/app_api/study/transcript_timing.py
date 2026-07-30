"""Pure helpers for turning Whisper timestamp tokens into transcript segments."""

from __future__ import annotations

from typing import Any


def whisper_timestamp_segments(token_ids: list[int], tokenizer: Any,
                               duration_s: float | None = None) -> list[dict]:
    """Decode Whisper's 20 ms timestamp tokens without another ASR pass."""
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    if not isinstance(no_timestamps, int) or no_timestamps < 0:
        return []
    timestamp_begin = no_timestamps + 1
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    segments: list[dict] = []
    text_tokens: list[int] = []
    start_s: float | None = None

    def flush(end_s: float) -> None:
        nonlocal text_tokens, start_s
        if not text_tokens:
            return
        text = tokenizer.decode(text_tokens, skip_special_tokens=True).strip()
        if text:
            start = start_s if start_s is not None else 0.0
            segments.append({
                "text": text,
                "start": round(float(start), 3),
                "end": round(max(float(start), end_s), 3),
            })
        text_tokens = []

    for token in token_ids:
        token = int(token)
        if token >= timestamp_begin:
            boundary_s = (token - timestamp_begin) * 0.02
            flush(boundary_s)
            start_s = boundary_s
        elif token not in special_ids:
            text_tokens.append(token)

    if text_tokens:
        flush(float(duration_s) if duration_s is not None
              else (start_s if start_s is not None else 0.0))
    return segments


def whisper_longform_segments(generation_output: Any, tokenizer: Any) -> list[dict]:
    """Decode absolute segments returned by Whisper long-form generation."""
    if not isinstance(generation_output, dict):
        return []
    batches = generation_output.get("segments") or []
    if not batches:
        return []
    segment_rows = batches[0] if isinstance(batches[0], list) else batches
    output: list[dict] = []
    for segment in segment_rows:
        if not isinstance(segment, dict):
            continue
        try:
            start = float(segment["start"])
            end = max(start, float(segment["end"]))
        except (KeyError, TypeError, ValueError):
            continue
        tokens = segment.get("tokens")
        if hasattr(tokens, "detach"):
            tokens = tokens.detach().cpu().tolist()
        text = tokenizer.decode(tokens or [], skip_special_tokens=True).strip()
        if text:
            output.append({
                "text": text,
                "start": round(start, 3),
                "end": round(end, 3),
            })
    return output
