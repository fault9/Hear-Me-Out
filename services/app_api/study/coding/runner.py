"""Run the LLM judge and verifier over blinded packets.

Model, decoding settings, prompts, schema, and codebook are recorded in (and
enforced against) the freeze manifest: `judge` refuses to run without a valid
manifest unless `--pilot` is passed, so main-dataset coding cannot silently
use drifted materials.

The Anthropic client is isolated behind `LLMClient` so tests inject a fake
and so a different provider only needs one adapter. Decoding is temperature 0
with a fixed max-token budget; every stored label file carries full
provenance (model id, decoding, prompt/codebook hashes, attempts, timestamps).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from . import prompts
from .freeze import manifest_status, materials_fingerprint
from .packets import read_index
from .schema import SCHEMA_VERSION, validate_checks, validate_labels

DEFAULT_MODEL = os.environ.get("CODING_JUDGE_MODEL", "claude-sonnet-5")
DEFAULT_MAX_TOKENS = int(os.environ.get("CODING_MAX_TOKENS", "4096"))
# An OpenAI-compatible endpoint (CODING_JUDGE_BASE_URL) selects that transport
# instead of Anthropic's. Which provider coded the data is a data-governance
# fact, so the base URL is recorded in the frozen decoding config.
DEFAULT_BASE_URL = os.environ.get("CODING_JUDGE_BASE_URL", "")
# JSON mode asks the endpoint to constrain decoding to a JSON object. Not
# every OpenAI-compatible provider implements it, so it stays explicit
# configuration (settled by `probe` before freezing) rather than something the
# client discovers mid-run: a decoding setting that changed partway through
# would leave packets coded under settings the freeze manifest does not name.
DEFAULT_JSON_MODE = os.environ.get("CODING_JUDGE_JSON_MODE", "1") != "0"
MAX_ATTEMPTS = 3
RETRY_EXCERPT_CHARS = 2000


class LLMClient:
    """Thin adapter over the Anthropic SDK (constructed lazily so the rest of
    the pipeline runs in environments without the dependency)."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = 0.0
        self._client = None

    def decoding(self) -> dict:
        return {"model": self.model, "temperature": self.temperature,
                "max_tokens": self.max_tokens}

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError(
                    "The 'anthropic' package is required to run the judge. "
                    "Install it in this environment (uv add anthropic / pip "
                    "install anthropic) and set ANTHROPIC_API_KEY.") from exc
            self._client = anthropic.Anthropic()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content
                       if getattr(block, "type", "") == "text")


class OpenAICompatibleClient(LLMClient):
    """Same contract against an OpenAI-compatible endpoint, so the judge can
    run on an EU-hosted provider where the data governance requires it."""

    def __init__(self, model: str = DEFAULT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 base_url: str = DEFAULT_BASE_URL,
                 json_mode: bool = DEFAULT_JSON_MODE):
        super().__init__(model=model, max_tokens=max_tokens)
        self.base_url = base_url
        self.json_mode = json_mode

    def decoding(self) -> dict:
        return {**super().decoding(), "base_url": self.base_url,
                "json_mode": self.json_mode}

    def complete(self, system: str, user: str) -> str:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - environment-specific
                raise RuntimeError(
                    "The 'openai' package is required for an OpenAI-compatible "
                    "endpoint. Install it (uv add openai) and set "
                    "CODING_JUDGE_API_KEY.") from exc
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=os.environ.get("CODING_JUDGE_API_KEY") or None)
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            **({"response_format": {"type": "json_object"}}
               if self.json_mode else {}),
        )
        return response.choices[0].message.content or ""


def build_client(model: str = DEFAULT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS) -> LLMClient:
    if DEFAULT_BASE_URL:
        return OpenAICompatibleClient(model=model, max_tokens=max_tokens)
    return LLMClient(model=model, max_tokens=max_tokens)


def probe_endpoint(client: LLMClient) -> dict:
    """Pre-flight check: confirm the endpoint answers, and — where JSON mode
    is configurable — report whether it accepts the constrained request. Run
    before freezing so the decoding config is settled rather than discovered
    partway through a coding run."""
    system = "Reply with a single JSON object and no other text."
    user = 'Return exactly {"ok": true}.'

    def attempt() -> dict:
        try:
            return {"ok": True, "reply": _parse_json_object(client.complete(system, user))}
        except Exception as exc:  # provider errors vary; report, never raise
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if not hasattr(client, "json_mode"):
        return {"decoding": client.decoding(), "attempts": {"default": attempt()}}
    configured = client.json_mode
    attempts = {}
    for mode in (True, False):
        client.json_mode = mode
        attempts[f"json_mode={str(mode).lower()}"] = attempt()
    client.json_mode = configured
    return {"decoding": client.decoding(), "attempts": attempts}


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start:end + 1])


def _retry_prompt(user: str, raw: str, errors: list[str]) -> str:
    """Re-ask with the rejected output quoted back: naming the errors alone
    leaves the model guessing which part of its own output they refer to."""
    return (user + "\n\nYour previous output was rejected:\n- "
            + "\n- ".join(errors[:20])
            + "\n\nThat output began:\n"
            + raw.strip()[:RETRY_EXCERPT_CHARS]
            + "\n\nReturn a corrected JSON object, and nothing else.")


def _provenance(client: LLMClient, root: Path, attempts: int, role: str) -> dict:
    return {
        "role": role,
        "schema_version": SCHEMA_VERSION,
        "decoding": client.decoding(),
        "materials": materials_fingerprint(),
        "freeze": manifest_status(root),
        "attempts": attempts,
        "created_at_unix_s": time.time(),
    }


def judge_packet(packet: dict, client: LLMClient, root: Path) -> dict:
    system = prompts.judge_system_prompt()
    user = prompts.judge_user_prompt(packet)
    errors: list[str] = []
    labels: dict | None = None
    attempts = 0
    prompt = user
    for attempts in range(1, MAX_ATTEMPTS + 1):
        raw = client.complete(system, prompt)
        try:
            candidate = _parse_json_object(raw)
        except ValueError as exc:
            errors = [str(exc)]
        else:
            errors = validate_labels(candidate, packet)
            labels = candidate
            if not errors:
                break
        prompt = _retry_prompt(user, raw, errors)
    return {
        "packet_id": packet["packet_id"],
        "labels": labels,
        "schema_errors": errors,
        "provenance": _provenance(client, root, attempts, "judge"),
    }


def verify_packet(packet: dict, labels: dict, client: LLMClient, root: Path) -> dict:
    system = prompts.verifier_system_prompt()
    user = prompts.verifier_user_prompt(packet, labels)
    checks: list[dict] = []
    errors: list[str] = []
    attempts = 0
    prompt = user
    for attempts in range(1, MAX_ATTEMPTS + 1):
        raw = client.complete(system, prompt)
        try:
            parsed = _parse_json_object(raw)
        except ValueError as exc:
            errors = [str(exc)]
        else:
            errors = validate_checks(parsed)
            checks = [row for row in (parsed.get("checks") or [])
                      if isinstance(row, dict)]
            if not errors:
                break
        prompt = _retry_prompt(user, raw, errors)
    return {
        "packet_id": packet["packet_id"],
        "checks": checks,
        "schema_errors": errors,
        "disagreements": [row for row in checks
                          if row.get("verdict") == "disagree"],
        "uncertain": [row for row in checks if row.get("verdict") == "uncertain"],
        "provenance": _provenance(client, root, attempts, "verifier"),
    }


def run_judging(root: Path, client: LLMClient, pilot: bool = False,
                only_packet_ids: set[str] | None = None,
                skip_existing: bool = True) -> dict:
    status = manifest_status(root)
    if not pilot and not status.get("valid"):
        raise RuntimeError(
            "Coding materials are not frozen (or drifted since freezing): "
            f"{status}. Run `python -m study.coding freeze` first, or pass "
            "--pilot for pilot-only runs.")
    judge_dir = root / "labels" / "judge"
    verifier_dir = root / "labels" / "verifier"
    judge_dir.mkdir(parents=True, exist_ok=True)
    verifier_dir.mkdir(parents=True, exist_ok=True)

    done, skipped, failed = [], [], []
    for row in read_index(root):
        pid = row["packet_id"]
        if only_packet_ids and pid not in only_packet_ids:
            continue
        judge_path = judge_dir / f"{pid}.json"
        if skip_existing and judge_path.exists():
            skipped.append(pid)
            continue
        packet = json.loads((root / "packets" / f"{pid}.json").read_text())
        result = judge_packet(packet, client, root)
        result["pilot"] = pilot
        judge_path.write_text(json.dumps(result, indent=2, sort_keys=True))
        if result["labels"] is not None and not result["schema_errors"]:
            verdict = verify_packet(packet, result["labels"], client, root)
            verdict["pilot"] = pilot
            (verifier_dir / f"{pid}.json").write_text(
                json.dumps(verdict, indent=2, sort_keys=True))
            done.append(pid)
        else:
            failed.append(pid)
    return {"judged": done, "skipped_existing": skipped,
            "schema_failed": failed}
