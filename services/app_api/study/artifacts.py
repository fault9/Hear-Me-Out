"""Immutable study artifact helpers.

Original recordings and configuration snapshots are written once. Derived
analysis lives below a versioned analysis directory and can be regenerated.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: str | Path, value: bytes, *, exclusive: bool = False) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and dest.exists():
        raise FileExistsError(dest)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive and dest.exists():
            raise FileExistsError(dest)
        os.replace(tmp_name, dest)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def atomic_write_json(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    atomic_write_bytes(path, json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8"),
                       exclusive=exclusive)


def immutable_copy(source: str | Path, destination: str | Path) -> dict:
    src, dest = Path(source), Path(destination)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", dir=dest.parent)
    os.close(fd)
    try:
        shutil.copyfile(src, tmp_name)
        os.replace(tmp_name, dest)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
    return file_record(dest)


def wav_record(path: str | Path) -> dict:
    result: dict[str, Any] = {}
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            result = {
                "sample_rate_hz": rate,
                "channels": wav.getnchannels(),
                "sample_width_bytes": wav.getsampwidth(),
                "frames": frames,
                "duration_s": frames / rate if rate else None,
            }
    except (wave.Error, EOFError):
        result["wav_metadata_error"] = True
    return result


def file_record(path: str | Path, *, relative_to: str | Path | None = None) -> dict:
    p = Path(path)
    display = p.relative_to(relative_to) if relative_to else p
    result = {
        "path": str(display),
        "size_bytes": p.stat().st_size,
        "sha256": sha256_file(p),
    }
    if p.suffix.lower() == ".wav":
        result.update(wav_record(p))
    return result


def git_revision(repo_root: str | Path) -> str | None:
    override = os.environ.get("HMO_GIT_COMMIT")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True,
            stderr=subprocess.DEVNULL, timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def append_jsonl(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())

