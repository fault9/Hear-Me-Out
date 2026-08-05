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


# Analysis stages record artifact paths with file_record(relative_to=data_root)
# where that data_root is the MEDIA directory (STUDY_DATA_DIR, normally
# <study data root>/media), so a stored path reads "sessions/study_1/...".
# Readers hold either the study data root or the media directory, and offline
# copies relocate both, so resolution tries the media directory first and the
# study data root second (which also keeps older "media/..."-prefixed paths
# working). Candidates must stay inside the base they resolve against.
def artifact_bases(data_root: str | Path) -> list[Path]:
    root = Path(os.path.expanduser(str(data_root)))
    if root.name == "media":
        return [root, root.parent]
    return [root / "media", root]


def resolve_artifact_path(data_root: str | Path,
                          path: str | Path | None) -> Path | None:
    """Locate a manifest-recorded artifact under an approved study-data root.

    Returns None when no path is recorded, when nothing exists at any approved
    base, or when the path escapes the base it resolves against.
    """
    if not path:
        return None
    candidate = Path(os.path.expanduser(str(path)))
    bases = [base.resolve() for base in artifact_bases(data_root)]
    if candidate.is_absolute():
        resolved = candidate.resolve()
        within = any(resolved == base or base in resolved.parents for base in bases)
        return resolved if within and resolved.exists() else None
    for base in bases:
        resolved = (base / candidate).resolve()
        if not (resolved == base or base in resolved.parents):
            continue
        if resolved.exists():
            return resolved
    return None


def load_manifest_artifact(session: dict, data_root: str | Path,
                           key: str) -> dict | None:
    """Read one analysis artifact recorded in a session's artifact manifest."""
    analysis = (session.get("artifact_manifest") or {}).get("analysis") or {}
    record = analysis.get(key) or {}
    path = record.get("path") if isinstance(record, dict) else None
    resolved = resolve_artifact_path(data_root, path)
    if resolved is None:
        return None
    try:
        loaded = json.loads(resolved.read_text())
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


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
