"""X-VC streaming voice-conversion server (alternative to meanvc_server.py).

Runs in its OWN venv (X-VC pins torch==2.5.1 / transformers==4.44.1, incompatible
with the shared hearmeout-venv) and on the SAME port + endpoint contract as
meanvc_server.py, so it's a drop-in swap selected by run_all.sh (VC_ENGINE=xvc):

    GET/POST /api/meanvc/load-target   - register a target voice (precompute conditions)
    GET      /api/meanvc/stream        - browser-mediated VC (legacy/fallback)
    GET      /api/meanvc/chat-proxy    - server-side VC bridge to PersonaPlex (the live path)

It reuses X-VC's OFFICIAL inference code verbatim (bins.infer_utils:
load_xvc / precompute_conditions / run_stream_chunk_forward and the run_streaming
window math, plus models.codec.sac.utils.process_audio). The only thing added on
top is feeding each window from a live incoming buffer instead of a complete file
(X-VC did not publish a live-server entrypoint).

Env:
  XVC_DIR              path to the cloned X-VC repo (added to sys.path; also the cwd)
  XVC_CONFIG           default $XVC_DIR/configs/xvc.yaml
  XVC_CKPT             default $XVC_DIR/ckpts/xvc.pt
  XVC_DEVICE           CUDA device index (default 0)
  XVC_EMA_LOAD         load EMA weights (default 1)
  XVC_CHUNK_MS/CURRENT_MS/SMOOTH_MS/FUTURE_MS  streaming window (default 2400/120/20/100)
  XVC_SILENCE_GATE_RMS / XVC_SILENCE_HANGOVER_MS  quiet-window GPU bypass (0.008 / 360)
  MEANVC_PORT          listen port (default 5002)
  SSL_DIR              dir with cert.pem/key.pem
  PERSONAPLEX_PROXY_HOST / PERSONAPLEX_PROXY_PORT   default 127.0.0.1 / 8000
  XVC_PROXY_DEBUG_DIR  optional: dump exactly-what-PersonaPlex-hears WAVs
"""
import asyncio
import contextlib
import io
import json
import logging
import os
import shutil
import struct
import sys
import tempfile
import time
import uuid
import wave
from urllib.parse import urlencode

import numpy as np
import torch
from aiohttp import web
import aiohttp
import sphn
import torchaudio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xvc_server")

# --- make the X-VC repo importable -----------------------------------------
XVC_DIR = os.environ.get("XVC_DIR", os.getcwd())
if XVC_DIR not in sys.path:
    sys.path.insert(0, XVC_DIR)

# Shared OpenTelemetry helper lives at <repo>/services/common (this file is
# <repo>/services/xvc/server.py). No-op unless OTEL_* is configured.
_SERVICES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)
try:
    from common import otel
    from common import logging_setup
    from common.study_events import EnergySpeechTracker, StudyEventBuffer
    logging_setup.init_logging("xvc")  # export logs over OTLP (trace-correlated) when observability is enabled
except Exception:  # noqa: BLE001
    otel = None
    logging_setup = None
    EnergySpeechTracker = None
    StudyEventBuffer = None

from bins.infer_utils import (  # noqa: E402  (import after sys.path setup)
    load_xvc,
    precompute_conditions,
    run_stream_chunk_forward,
)
from models.codec.sac.utils import process_audio  # noqa: E402
from utils.audio import audio_highpass_filter  # noqa: E402

TAG_AUDIO = b"\x01"      # converted audio -> PersonaPlex (Opus)
TAG_VC_USER = b"\x03"    # converted user PCM (float32 16k) -> browser
STUDY_FRAME = struct.Struct("<4sIIII")
STUDY_FRAME_MAGIC = b"HMO1"

XVC_CONFIG = os.environ.get("XVC_CONFIG", os.path.join(XVC_DIR, "configs/xvc.yaml"))
XVC_CKPT = os.environ.get("XVC_CKPT", os.path.join(XVC_DIR, "ckpts/xvc.pt"))
XVC_DEVICE = int(os.environ.get("XVC_DEVICE", 0))
XVC_EMA_LOAD = os.environ.get("XVC_EMA_LOAD", "1") not in ("0", "false", "False")

CHUNK_MS = int(os.environ.get("XVC_CHUNK_MS", 2400))
CURRENT_MS = int(os.environ.get("XVC_CURRENT_MS", 120))
SMOOTH_MS = int(os.environ.get("XVC_SMOOTH_MS", 20))
FUTURE_MS = int(os.environ.get("XVC_FUTURE_MS", 100))
SILENCE_GATE_RMS = float(os.environ.get("XVC_SILENCE_GATE_RMS", "0.008"))
SILENCE_HANGOVER_MS = int(os.environ.get("XVC_SILENCE_HANGOVER_MS", "360"))

PERSONAPLEX_HOST = os.environ.get("PERSONAPLEX_PROXY_HOST", "127.0.0.1")
PERSONAPLEX_PORT = os.environ.get("PERSONAPLEX_PROXY_PORT", "8000")
PERSONAPLEX_INPUT_SR = int(os.environ.get("PERSONAPLEX_INPUT_SR", "24000"))

# Globals populated on startup.
cfg: dict | None = None
model = None
device: torch.device | None = None
SR = 16000
HP_CUT = 0.0
MASK_TARGET_COND = True
CONTENT_PATH_WARMED = False

# target_id -> (speaker_condition, frame_condition)
targets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _save_wav(path: str, pcm: np.ndarray, sr: int) -> None:
    pcm = np.clip(pcm, -1.0, 1.0)
    ints = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(ints.tobytes())


def _wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    output = io.BytesIO()
    pcm = np.clip(np.asarray(pcm).reshape(-1), -1.0, 1.0)
    ints = (pcm * 32767.0).astype("<i2")
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(ints.tobytes())
    return output.getvalue()


def _target_conditions(target_path: str):
    """Build target conditions using the same path as live X-VC."""
    target_np = process_audio(target_path, cfg, int(cfg["latent_hop_length"]))
    target_wav = torch.from_numpy(target_np)[None, None].float().to(device)
    if MASK_TARGET_COND:
        pad = torch.zeros((1, 1, int(CHUNK_MS * SR / 1000)), device=device)
        target_wav_cond = torch.cat([target_wav, pad], dim=-1)
    else:
        target_wav_cond = target_wav
    spk, frame = precompute_conditions(model, target_wav, target_wav_cond)
    return target_np, spk, frame


def _fit_length(pcm: np.ndarray, samples: int) -> np.ndarray:
    pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if len(pcm) >= samples:
        return pcm[:samples]
    return np.pad(pcm, (0, samples - len(pcm)), mode="constant")


@torch.inference_mode()
def _convert_soundboard_file(source_path: str, target_path: str, output_sr: int):
    """Convert a complete take through the same session used for live X-VC.

    Right padding supplies the live session's final look-ahead window and is
    removed before the single 16 kHz -> PersonaPlex resample. The silence gate
    outputs digital silence rather than deleting pauses, so the final WAV keeps
    the exact source timeline and sample count.
    """
    source_wave, source_sr = torchaudio.load(source_path)
    if source_wave.numel() == 0 or source_wave.shape[-1] == 0:
        raise ValueError("source audio is empty")
    source_samples = int(source_wave.shape[-1])
    source_sr = int(source_sr)
    exact_xvc_samples = round(source_samples * SR / source_sr)
    exact_output_samples = round(source_samples * output_sr / source_sr)

    source_mono = source_wave.mean(dim=0)
    if source_sr != SR:
        source_mono = torchaudio.functional.resample(source_mono, source_sr, SR)
    source_16k = _fit_length(source_mono.numpy(), exact_xvc_samples)
    _, spk, frame = _target_conditions(target_path)

    session = XVCStreamSession(spk, frame)
    current_samples = CURRENT_MS * SR // 1000
    window_count = max(1, int(np.ceil(exact_xvc_samples / current_samples)))
    required_samples = (
        (window_count - 1) * CURRENT_MS
        + CURRENT_MS
        + SMOOTH_MS
        + FUTURE_MS
    ) * SR // 1000
    padded_source = _fit_length(source_16k, required_samples)
    chunks = session.feed(padded_source)
    if not chunks:
        raise RuntimeError("X-VC produced no output windows")
    converted_16k = _fit_length(
        np.concatenate(chunks), exact_xvc_samples
    )
    converted_out = torchaudio.functional.resample(
        torch.from_numpy(converted_16k), SR, output_sr).numpy()
    converted_out = _fit_length(converted_out, exact_output_samples)

    metadata = {
        "engine": "xvc",
        "input_sample_rate": source_sr,
        "input_samples": source_samples,
        "xvc_sample_rate": SR,
        "xvc_samples": exact_xvc_samples,
        "output_sample_rate": output_sr,
        "output_samples": exact_output_samples,
        "output_windows": len(chunks),
        "inference_windows": session.inference_windows,
        "silence_bypassed_windows": session.silence_bypassed_windows,
    }
    return _wav_bytes(converted_out, output_sr), metadata


def _parse_study_frame(data: bytes) -> tuple[bytes, dict]:
    if len(data) < STUDY_FRAME.size or data[:4] != STUDY_FRAME_MAGIC:
        return data, {}
    magic, sequence, capture_start, sample_count, capture_rate = STUDY_FRAME.unpack_from(data)
    payload = data[STUDY_FRAME.size:]
    if magic != STUDY_FRAME_MAGIC or len(payload) != sample_count * 4:
        raise ValueError("invalid HMO study audio frame")
    return payload, {
        "browser_chunk_sequence": sequence,
        "capture_start_sample": capture_start,
        "capture_sample_count": sample_count,
        "capture_sample_rate_hz": capture_rate,
    }


def _decode_opus_stream(encoded: bytes, sample_rate: int) -> bytes | None:
    if not encoded:
        return None
    reader = sphn.OpusStreamReader(sample_rate)
    reader.append_bytes(encoded)
    chunks = []
    while True:
        pcm = reader.read_pcm()
        if pcm.shape[-1] == 0:
            break
        chunks.append(np.asarray(pcm, dtype=np.float32).reshape(-1))
    return _wav_bytes(np.concatenate(chunks), sample_rate) if chunks else None


class XVCStreamSession:
    """Online driver around X-VC's official per-window forward.

    Mirrors bins.infer_utils.run_streaming exactly, but pulls each window from a
    growing buffer (live mic) instead of a complete source array. Per-window work
    is stateless except the overlap cross-fade tail_buffer.
    """

    def __init__(self, speaker_condition, frame_condition):
        self.spk = speaker_condition
        self.frame = frame_condition
        self.sr = SR
        self.current_ms = CURRENT_MS
        self.smooth_ms = SMOOTH_MS
        self.future_ms = FUTURE_MS
        self.history_ms = CHUNK_MS - CURRENT_MS - SMOOTH_MS - FUTURE_MS
        if self.history_ms < 0:
            raise ValueError("CHUNK_MS - CURRENT_MS - SMOOTH_MS - FUTURE_MS must be >= 0")
        self.overlap_len = SMOOTH_MS * SR // 1000
        if self.overlap_len > 0:
            self.fade_in = 0.5 * (
                1 - torch.cos(torch.pi * torch.linspace(0, 1, self.overlap_len, device=device))
            )
            self.fade_out = 1 - self.fade_in
            self.tail_buffer = torch.zeros(1, 1, self.overlap_len, device=device)
        else:
            self.fade_in = self.fade_out = self.tail_buffer = None
        self.buf = np.zeros(0, dtype=np.float32)
        self.i = 0
        self.inference_windows = 0
        self.silence_bypassed_windows = 0
        self.silence_hangover_windows = max(
            0, int(np.ceil(SILENCE_HANGOVER_MS / self.current_ms)))
        self.silence_hangover_remaining = 0

    @torch.inference_mode()
    def feed(self, pcm: np.ndarray) -> list[np.ndarray]:
        """Append incoming 16 kHz PCM, return any completed current-region chunks."""
        self.buf = np.concatenate([self.buf, pcm.astype(np.float32)])
        outs: list[np.ndarray] = []
        while True:
            start = (self.i * self.current_ms - self.history_ms) * self.sr // 1000
            end = (
                self.i * self.current_ms + self.current_ms + self.smooth_ms + self.future_ms
            ) * self.sr // 1000
            if len(self.buf) < end:
                break  # need more look-ahead audio before this window is ready
            left_pad = max(0, -start)
            seg = self.buf[max(0, start):end]
            if left_pad:
                seg = np.concatenate([np.zeros(left_pad, dtype=np.float32), seg])
            if HP_CUT:
                seg = audio_highpass_filter(seg, self.sr, HP_CUT).astype(np.float32)
            cur_start = self.history_ms * self.sr // 1000
            activity_end = (
                self.history_ms + self.current_ms + self.smooth_ms + self.future_ms
            ) * self.sr // 1000
            activity = seg[cur_start:activity_end]
            rms = float(np.sqrt(np.mean(activity * activity))) if len(activity) else 0.0
            active = SILENCE_GATE_RMS <= 0 or rms >= SILENCE_GATE_RMS
            if active:
                self.silence_hangover_remaining = self.silence_hangover_windows
            elif self.silence_hangover_remaining > 0:
                self.silence_hangover_remaining -= 1
                active = True

            if active:
                outs.append(self._forward(seg))
                self.inference_windows += 1
            else:
                # Preserve the sample-exact stream with digital silence. Passing
                # raw mic audio here would leak natural speech into a VC condition
                # whenever a quiet syllable fell below the gate.
                cur_end = (self.history_ms + self.current_ms) * self.sr // 1000
                outs.append(np.zeros(cur_end - cur_start, dtype=np.float32))
                self.silence_bypassed_windows += 1
                if self.tail_buffer is not None:
                    self.tail_buffer.zero_()
            self.i += 1
        return outs

    @torch.inference_mode()
    def warm(self) -> None:
        """Warm content-path CUDA kernels without consuming participant audio."""
        segment_samples = (
            self.history_ms + self.current_ms + self.smooth_ms + self.future_ms
        ) * self.sr // 1000
        self._forward(np.zeros(segment_samples, dtype=np.float32))
        if self.tail_buffer is not None:
            self.tail_buffer.zero_()

    @torch.inference_mode()
    def _forward(self, seg_np: np.ndarray) -> np.ndarray:
        win = torch.from_numpy(seg_np)[None, None].float().to(device)
        out = run_stream_chunk_forward(model, win, self.spk, self.frame)
        cur_start = self.history_ms * self.sr // 1000
        cur_end = (self.history_ms + self.current_ms) * self.sr // 1000
        cur = out[:, :, cur_start:cur_end]
        if self.overlap_len > 0:
            if self.i > 0:
                head = cur[..., : self.overlap_len]
                head_sm = self.tail_buffer * self.fade_out + head * self.fade_in
                cur = torch.cat([head_sm, cur[..., self.overlap_len:]], dim=-1)
            tail_start = (self.history_ms + self.current_ms) * self.sr // 1000
            self.tail_buffer = out[:, :, tail_start: tail_start + self.overlap_len]
        return cur.squeeze().detach().cpu().numpy().astype(np.float32)


async def handle_load_target(request: web.Request) -> web.Response:
    """POST /api/meanvc/load-target - upload target WAV, precompute conditions."""
    global CONTENT_PATH_WARMED
    post = await request.post()
    field = post.get("wav")
    if field is None:
        return web.json_response({"error": "missing 'wav' file"}, status=400)

    target_id = uuid.uuid4().hex
    tmp_path = os.path.join("/tmp", f"xvc_target_{target_id}.wav")
    with open(tmp_path, "wb") as f:
        f.write(field.file.read())

    try:
        target_np, spk, frame = _target_conditions(tmp_path)
        if not CONTENT_PATH_WARMED:
            warmup_started = time.perf_counter()
            XVCStreamSession(spk, frame).warm()
            CONTENT_PATH_WARMED = True
            logger.info("[xvc] content path warmed in %.1f ms",
                        (time.perf_counter() - warmup_started) * 1000.0)
        targets[target_id] = (spk, frame)
        duration = round(len(target_np) / SR, 2)
        logger.info(f"[xvc] loaded target {target_id} ({duration}s)")
        return web.json_response({"target_id": target_id, "duration_seconds": duration})
    except Exception as e:
        logger.exception("[xvc] load-target failed")
        return web.json_response({"error": str(e)}, status=500)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


async def handle_file_conversion(request: web.Request) -> web.Response:
    """POST /api/xvc/file-conversion - bake a complete clip with X-VC.

    Accepts source_audio and target_audio WAV uploads. Unlike the legacy
    app-api Seed-VC route, this does not run VAD or concatenate speech regions.
    """
    post = await request.post()
    source = post.get("source_audio")
    target = post.get("target_audio")
    if (
        source is None
        or target is None
        or not hasattr(source, "file")
        or not hasattr(target, "file")
    ):
        return web.json_response(
            {"error": "missing source_audio or target_audio file"}, status=400)
    try:
        output_sr = int(post.get("output_sr", PERSONAPLEX_INPUT_SR))
    except (TypeError, ValueError):
        return web.json_response({"error": "output_sr must be an integer"}, status=400)
    if output_sr != PERSONAPLEX_INPUT_SR:
        return web.json_response(
            {"error": f"output_sr must be {PERSONAPLEX_INPUT_SR} for PersonaPlex"},
            status=400,
        )

    work_dir = tempfile.mkdtemp(prefix="xvc_file_")
    source_path = os.path.join(work_dir, "source.wav")
    target_path = os.path.join(work_dir, "target.wav")
    try:
        with open(source_path, "wb") as stream:
            stream.write(source.file.read())
        with open(target_path, "wb") as stream:
            stream.write(target.file.read())

        loop = asyncio.get_running_loop()
        async with request.app["file_conversion_lock"]:
            wav_data, metadata = await loop.run_in_executor(
                None, _convert_soundboard_file, source_path, target_path, output_sr)
        input_ms = metadata["input_samples"] * 1000.0 / metadata["input_sample_rate"]
        output_ms = metadata["output_samples"] * 1000.0 / output_sr
        logger.info(
            "[xvc] file conversion %.1f ms -> %.1f ms in %d windows "
            "(%d inferred, %d silence)",
            input_ms,
            output_ms,
            metadata["output_windows"],
            metadata["inference_windows"],
            metadata["silence_bypassed_windows"],
        )
        return web.Response(
            body=wav_data,
            content_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="xvc_baked.wav"',
                "X-VC-Engine": "xvc",
                "X-Input-Duration-Ms": f"{input_ms:.6f}",
                "X-Output-Duration-Ms": f"{output_ms:.6f}",
                "X-Input-Samples": str(metadata["input_samples"]),
                "X-Output-Samples": str(metadata["output_samples"]),
                "X-Sample-Rate": str(output_sr),
                "X-XVC-Inference-Windows": str(metadata["inference_windows"]),
                "X-XVC-Silence-Windows": str(metadata["silence_bypassed_windows"]),
            },
        )
    except (ValueError, RuntimeError) as exc:
        logger.exception("[xvc] file conversion failed")
        return web.json_response({"error": str(exc)}, status=400)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[xvc] file conversion failed")
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _maybe_resampler(source_sr: int):
    if source_sr == SR:
        return None
    return torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=SR).to("cpu")


async def handle_stream(request: web.Request) -> web.WebSocketResponse:
    """GET /api/meanvc/stream - browser-mediated VC (legacy fallback).

    Browser sends raw float32 PCM; we return converted float32 PCM (16 kHz).
    """
    target_id = request.query.get("target_id", "")
    source_sr = int(request.query.get("source_sr", SR))
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    if target_id not in targets:
        await ws.send_json({"error": f"Unknown target_id: {target_id}"})
        await ws.close()
        return ws

    spk, frame = targets[target_id]
    session = XVCStreamSession(spk, frame)
    resampler = _maybe_resampler(source_sr)
    loop = asyncio.get_event_loop()
    await ws.send_json({"status": "ready"})

    async for msg in ws:
        if msg.type == web.WSMsgType.BINARY:
            incoming = np.frombuffer(msg.data, dtype=np.float32).copy()
            if resampler is not None:
                incoming = resampler(torch.from_numpy(incoming).unsqueeze(0)).squeeze(0).numpy()
            curs = await loop.run_in_executor(None, session.feed, incoming)
            for cur in curs:
                if not ws.closed:
                    await ws.send_bytes(cur.tobytes())
        elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
            break
    return ws


STUDY_APP_API_URL = os.environ.get("STUDY_APP_API_URL", "https://127.0.0.1:5001")


async def upload_study_proxy_artifacts(session_id: str, artifacts: dict[str, bytes],
                                       metadata: dict) -> tuple[bool, str | None]:
    if not session_id or session_id.endswith("_CHECK"):
        return True, None
    form = aiohttp.FormData()
    for name, value in artifacts.items():
        content_type = "audio/wav" if name.endswith(".wav") else "audio/ogg"
        form.add_field(name.replace(".", "_"), value, filename=name,
                       content_type=content_type)
    form.add_field("metadata", json.dumps(metadata), content_type="application/json")
    url = f"{STUDY_APP_API_URL}/api/study/internal/session/{session_id}/proxy-artifacts"
    try:
        async with aiohttp.ClientSession() as client:
            async with client.post(
                url, data=form, ssl=False,
                headers={"X-Study-Event-Token": os.environ.get(
                    "STUDY_EVENT_TOKEN", "local-study-events")},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status != 200:
                    return False, f"HTTP {response.status}: {(await response.text())[-500:]}"
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


async def resolve_study_condition(session_id: str):
    """Study mode: resolve an opaque session_id to its hidden condition via app-api
    over localhost. Returns None if unavailable. (Same contract as MeanVC.)"""
    url = f"{STUDY_APP_API_URL}/api/study/condition/{session_id}"
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(url, ssl=False, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    logger.error(f"[xvc proxy] condition resolve HTTP {r.status} for {session_id}")
                    return None
                return await r.json()
    except Exception as e:  # noqa: BLE001
        logger.error(f"[xvc proxy] condition resolve failed: {e}")
        return None


async def handle_chat_proxy(request: web.Request) -> web.WebSocketResponse:
    """GET /api/meanvc/chat-proxy - server-side VC bridge to PersonaPlex.

    Browser sends raw float32 mic PCM; we convert each window with X-VC, Opus-encode
    at 24 kHz, and forward to PersonaPlex over localhost. PersonaPlex's framed replies
    (0x00/0x01/0x02) are relayed back verbatim; the converted user PCM (16 kHz) is also
    sent back tagged 0x03 for the browser's downloads/monitor.
    """
    target_id = request.query.get("target_id", "default")
    source_sr = int(request.query.get("source_sr", SR))
    voice_prompt = request.query.get("voice_prompt", "")
    text_prompt = request.query.get("text_prompt", "")
    session_id = request.query.get("session_id", "")
    resampler = _maybe_resampler(source_sr)

    browser_ws = web.WebSocketResponse()
    await browser_ws.prepare(request)

    study_id = None
    # Study mode: resolve the hidden prompt + timed voice schedule via session_id.
    if session_id:
        cond = await resolve_study_condition(session_id)
        if cond is None:
            await browser_ws.send_json({"error": "Unknown or unresolved session"})
            await browser_ws.close()
            return browser_ws
        text_prompt = cond.get("text_prompt", "")
        voice_prompt = cond.get("voice_prompt") or voice_prompt
        schedule = cond.get("schedule") or [{"mode": "natural", "start_s": 0, "end_s": None}]
        study_id = cond.get("study_id")
    else:
        schedule = [{"mode": "vc", "start_s": 0, "end_s": None, "engine_target_id": target_id}]

    # One X-VC session per distinct VC target in the schedule; natural = pass-through.
    vc_sessions = {}
    unavailable_targets = []
    for seg in schedule:
        if seg.get("mode") == "vc":
            tid = seg.get("engine_target_id")
            if not tid or tid not in targets:
                unavailable_targets.append(tid or "missing")
            elif tid not in vc_sessions:
                spk, frame = targets[tid]
                vc_sessions[tid] = XVCStreamSession(spk, frame)
    if unavailable_targets:
        logger.error("[xvc proxy] VC schedule has unavailable target(s) for %s: %s",
                     session_id or "legacy session", unavailable_targets)
        await browser_ws.send_json({"error": "VC target is unavailable for this session"})
        await browser_ws.close()
        return browser_ws

    def active_segment(elapsed_s):
        for seg in schedule:
            start = seg.get("start_s") or 0
            end = seg.get("end_s")
            if elapsed_s >= start and (end is None or elapsed_s < end):
                return seg
        return schedule[-1]

    loop = asyncio.get_event_loop()
    # X-VC outputs 16 kHz; sphn's Opus encoder only accepts 24/48 kHz (PersonaPlex
    # uses 24 kHz = its mimi rate). Encode at 24 kHz and upsample before encoding.
    opus_writer = sphn.OpusStreamWriter(24000)
    out_resampler = torchaudio.transforms.Resample(16000, 24000).to("cpu")
    OPUS_FRAME = 1920

    debug_dir = os.environ.get("XVC_PROXY_DEBUG_DIR")
    opus_reader_dbg = sphn.OpusStreamReader(24000) if debug_dir else None
    debug_pcm: list[np.ndarray] = []

    if logging_setup:
        logging_setup.set_log_session(session_id or target_id, study_id)
    if otel:
        otel.set_session_attributes(session_id=session_id or target_id, engine="xvc",
                                    study_id=study_id,
                                    schedule=str([s.get("mode") for s in schedule]))

    qs = urlencode({"voice_prompt": voice_prompt, "text_prompt": text_prompt})
    pplx_url = f"wss://{PERSONAPLEX_HOST}:{PERSONAPLEX_PORT}/api/chat?{qs}"
    logger.info(f"[xvc proxy] connecting to PersonaPlex: {pplx_url}")

    _tracer = otel.get_tracer("xvc") if otel else None
    client = aiohttp.ClientSession()
    try:
        with (otel.start_span(_tracer, "personaplex.connect", kind="client") if otel
              else contextlib.nullcontext()):
            pplx_ws = await client.ws_connect(pplx_url, ssl=False, max_msg_size=0)
    except Exception as e:
        logger.error(f"[xvc proxy] PersonaPlex connect failed: {e}")
        await browser_ws.send_json({"error": f"PersonaPlex unavailable: {e}"})
        await browser_ws.close()
        await client.close()
        return browser_ws

    chunk_count = 0
    processed_samples = 0            # SR-rate samples consumed → elapsed time for the schedule
    transmitted_samples = 0          # 16 kHz PCM returned to browser / represented in participant.wav
    model_bound_samples = 0          # 24 kHz PCM framed for PersonaPlex
    input_chunk_sequence = 0
    model_input_packet_sequence = 0
    model_packet_sequence = 0
    current_mode = None
    current_transmitted_mode = None
    assistant_packet_active = False
    last_assistant_packet_ns = 0
    opus_pcm_buf = np.zeros(0, dtype=np.float32)
    # latency tracking
    first_send_ts = None             # when the first converted audio was sent to PersonaPlex
    first_model_done = False         # first PersonaPlex audio frame seen
    vc_ms_total = 0.0                # sum of per-window X-VC inference time
    vc_windows = 0
    vc_silence_bypassed_windows = 0
    capture_study_artifacts = bool(session_id and not session_id.endswith("_CHECK"))
    proxy_received_pcm: list[np.ndarray] = []
    proxy_transmitted_pcm: list[np.ndarray] = []
    personaplex_input_opus = bytearray()
    personaplex_output_opus = bytearray()
    events = StudyEventBuffer(session_id, "xvc", STUDY_APP_API_URL) if StudyEventBuffer else None
    speech = EnergySpeechTracker(SR) if EnergySpeechTracker else None
    if events:
        events.add("stream_start", input_sample_rate_hz=SR,
                   transmitted_sample_rate_hz=SR, model_bound_sample_rate_hz=24000,
                   schedule=schedule, xvc_content_path_warmed=CONTENT_PATH_WARMED)
        for segment in schedule[1:]:
            events.add("route_switch_requested", requested_start_s=segment.get("start_s"),
                       requested_input_sample=int(float(segment.get("start_s") or 0) * SR),
                       to_mode=segment.get("mode"))

    async def browser_to_pplx():
        nonlocal chunk_count, processed_samples, transmitted_samples, model_bound_samples
        nonlocal input_chunk_sequence, model_input_packet_sequence
        nonlocal current_mode, current_transmitted_mode
        nonlocal opus_pcm_buf, first_send_ts, vc_ms_total, vc_windows
        nonlocal vc_silence_bypassed_windows
        async for msg in browser_ws:
            if msg.type == web.WSMsgType.BINARY:
                try:
                    pcm_payload, browser_frame = _parse_study_frame(msg.data)
                except ValueError as exc:
                    if events:
                        events.add("invalid_input_frame", error=str(exc), bytes=len(msg.data))
                    continue
                incoming = np.frombuffer(pcm_payload, dtype=np.float32).copy()
                if resampler is not None:
                    incoming = resampler(torch.from_numpy(incoming).unsqueeze(0)).squeeze(0).numpy()
                if capture_study_artifacts:
                    proxy_received_pcm.append(incoming.copy())
                input_chunk_sequence += 1
                input_start = processed_samples
                input_end = input_start + len(incoming)
                seg = active_segment(processed_samples / float(SR))
                processed_samples += len(incoming)
                mode = seg.get("mode", "natural")
                if events:
                    events.add("input_chunk", chunk_sequence=input_chunk_sequence,
                               input_start_sample=input_start, input_end_sample=input_end,
                               samples=len(incoming), source_sample_rate_hz=source_sr,
                               input_sample_rate_hz=SR,
                               **browser_frame)
                    if current_mode != mode:
                        events.add("route_activated", from_mode=current_mode, to_mode=mode,
                                   input_sample=input_start,
                                   transmitted_sample=transmitted_samples,
                                   model_bound_sample=model_bound_samples,
                                   requested_start_s=seg.get("start_s"))
                        current_mode = mode
                    if speech:
                        for row in speech.update(incoming, input_start, input_end):
                            event = row.pop("event")
                            events.add(event, **row)
                tid = seg.get("engine_target_id") if seg.get("mode") == "vc" else None
                sess = vc_sessions.get(tid) if tid else None
                if sess is None:
                    # Natural (or missing target): forward the raw mic window unmodified.
                    curs = [incoming]
                else:
                    _t0 = time.perf_counter()
                    inference_before = sess.inference_windows
                    bypassed_before = sess.silence_bypassed_windows
                    try:
                        curs = await loop.run_in_executor(None, sess.feed, incoming)
                    except Exception as e:
                        logger.error(f"[xvc proxy] inference error: {e}")
                        if events:
                            events.add("inference_failure", input_start_sample=input_start,
                                       input_end_sample=input_end, error=str(e))
                        continue
                    elapsed_ms = (time.perf_counter() - _t0) * 1000.0
                    inferred = sess.inference_windows - inference_before
                    bypassed = sess.silence_bypassed_windows - bypassed_before
                    vc_silence_bypassed_windows += bypassed
                    if inferred:  # X-VC GPU forward latency, per inferred window
                        dt = elapsed_ms / inferred
                        vc_ms_total += elapsed_ms
                        vc_windows += inferred
                        if otel:
                            otel.record_latency("vc.inference_ms", dt, engine="xvc")
                    if events and curs:
                        events.add(
                            "xvc_inference_batch",
                            input_start_sample=input_start,
                            input_end_sample=input_end,
                            output_windows=len(curs),
                            inference_windows=inferred,
                            silence_bypassed_windows=bypassed,
                            elapsed_ms=round(elapsed_ms, 3),
                            silence_gate_rms=SILENCE_GATE_RMS,
                        )

                for cur in curs:
                    if capture_study_artifacts:
                        proxy_transmitted_pcm.append(np.asarray(cur, dtype=np.float32).copy())
                    chunk_count += 1
                    output_start = transmitted_samples
                    transmitted_samples += len(cur)
                    if events:
                        if current_transmitted_mode != mode:
                            events.add("transmitted_route_activated",
                                       from_mode=current_transmitted_mode, to_mode=mode,
                                       transmitted_sample=output_start,
                                       input_start_sample=input_start,
                                       input_end_sample=input_end)
                            current_transmitted_mode = mode
                        events.add("transmitted_window", output_sequence=chunk_count,
                                   route_mode=mode, input_start_sample=input_start,
                                   input_end_sample=input_end,
                                   transmitted_start_sample=output_start,
                                   transmitted_end_sample=transmitted_samples)
                    cur_24k = (
                        out_resampler(torch.from_numpy(cur).unsqueeze(0)).squeeze(0).numpy()
                    )
                    opus_pcm_buf = np.concatenate([opus_pcm_buf, cur_24k])
                    while len(opus_pcm_buf) >= OPUS_FRAME:
                        frame_pcm = np.ascontiguousarray(opus_pcm_buf[:OPUS_FRAME])
                        opus_pcm_buf = opus_pcm_buf[OPUS_FRAME:]
                        frame_start = model_bound_samples
                        model_bound_samples += OPUS_FRAME
                        opus_writer.append_pcm(frame_pcm)
                        while True:
                            encoded = opus_writer.read_bytes()
                            if len(encoded) == 0:
                                break
                            if first_send_ts is None:
                                first_send_ts = time.perf_counter()
                            if capture_study_artifacts:
                                personaplex_input_opus.extend(encoded)
                            await pplx_ws.send_bytes(TAG_AUDIO + encoded)
                            model_input_packet_sequence += 1
                            if events:
                                events.add("personaplex_input_packet",
                                           packet_sequence=model_input_packet_sequence,
                                           route_mode=mode,
                                           model_bound_start_sample=frame_start,
                                           model_bound_end_sample=model_bound_samples,
                                           encoded_bytes=len(encoded))
                            if opus_reader_dbg is not None:
                                opus_reader_dbg.append_bytes(encoded)
                                pcm = opus_reader_dbg.read_pcm()
                                if pcm.shape[-1] > 0:
                                    debug_pcm.append(pcm.astype(np.float32))
                    if not browser_ws.closed:
                        await browser_ws.send_bytes(TAG_VC_USER + cur.tobytes())
                if events:
                    events.flush_nowait()
            elif msg.type == web.WSMsgType.TEXT:
                if events:
                    try:
                        control = json.loads(msg.data)
                    except json.JSONDecodeError:
                        control = {}
                    if control.get("type") == "capture_summary":
                        events.add("client_capture_summary", reported_by="browser",
                                   callbacks=control.get("callbacks"), samples=control.get("samples"),
                                   sample_rate_hz=control.get("sampleRateHz"),
                                   estimated_dropped_samples=control.get("estimatedDroppedSamples"),
                                   detector=control.get("detector"))
            elif msg.type in (web.WSMsgType.CLOSE, web.WSMsgType.ERROR):
                break

    async def pplx_to_browser():
        nonlocal first_model_done, model_packet_sequence
        nonlocal assistant_packet_active, last_assistant_packet_ns
        async for msg in pplx_ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                tag = msg.data[:1]
                model_packet_sequence += 1
                if capture_study_artifacts and tag == b"\x01":
                    personaplex_output_opus.extend(msg.data[1:])
                if events:
                    now_ns = time.monotonic_ns()
                    events.add("personaplex_output_packet", packet_sequence=model_packet_sequence,
                               tag=int(tag[0]) if tag else None,
                               payload_bytes=max(0, len(msg.data) - 1))
                    if tag == b"\x01":
                        if (not assistant_packet_active or
                                now_ns - last_assistant_packet_ns > 400_000_000):
                            if assistant_packet_active:
                                events.add("assistant_speech_end", packet_sequence=model_packet_sequence - 1,
                                           detector="packet_gap")
                            assistant_packet_active = True
                            events.add("assistant_speech_start", packet_sequence=model_packet_sequence,
                                       detector="packet_gap")
                        last_assistant_packet_ns = now_ns
                # 0x01 = model Opus audio: measure time-to-first-response after we
                # started sending converted user audio to PersonaPlex.
                if not first_model_done and msg.data[:1] == b"\x01" and first_send_ts is not None:
                    first_model_done = True
                    _lat = (time.perf_counter() - first_send_ts) * 1000.0
                    if otel:
                        otel.record_latency("personaplex.first_response_ms", _lat, engine="xvc")
                        otel.set_session_attributes(first_response_ms=round(_lat))
                    logger.info(f"[xvc proxy] first PersonaPlex audio {_lat:.0f} ms after first send")
                if not browser_ws.closed:
                    await browser_ws.send_bytes(msg.data)
                if events:
                    events.flush_nowait()
            elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                break

    tasks = [
        asyncio.create_task(browser_to_pplx()),
        asyncio.create_task(pplx_to_browser()),
    ]
    try:
        _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
    finally:
        await pplx_ws.close()
        await client.close()
        if not browser_ws.closed:
            await browser_ws.close()
        # Release GPU memory held by this session's X-VC buffers so it doesn't
        # accumulate across scenarios / participants on the shared GPU.
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    if capture_study_artifacts:
        artifact_bytes: dict[str, bytes] = {}
        uploaded = False
        upload_error = None
        try:
            if proxy_received_pcm:
                received = np.concatenate(proxy_received_pcm)
                artifact_bytes["proxy_received.wav"] = await loop.run_in_executor(
                    None, _wav_bytes, received, SR)
            if proxy_transmitted_pcm:
                transmitted = np.concatenate(proxy_transmitted_pcm)
                artifact_bytes["participant_proxy.wav"] = await loop.run_in_executor(
                    None, _wav_bytes, transmitted, SR)
            if personaplex_input_opus:
                exact_input = bytes(personaplex_input_opus)
                artifact_bytes["personaplex_input.opus"] = exact_input
                try:
                    decoded = await loop.run_in_executor(
                        None, _decode_opus_stream, exact_input, 24000)
                    if decoded:
                        artifact_bytes["personaplex_input_decoded.wav"] = decoded
                except Exception as exc:  # exact Opus remains the authoritative artifact
                    logger.warning(f"[xvc proxy] post-hoc Opus decode failed: {exc}")
                    if events:
                        events.add("personaplex_input_decode_failed", error=str(exc))
            if personaplex_output_opus:
                artifact_bytes["personaplex_output.opus"] = bytes(personaplex_output_opus)
            uploaded, upload_error = await upload_study_proxy_artifacts(
                session_id,
                artifact_bytes,
                {
                    "schema": "hmo.proxy-artifacts.v1",
                    "engine": "xvc",
                    "input_sample_rate_hz": SR,
                    "transmitted_sample_rate_hz": SR,
                    "model_bound_sample_rate_hz": 24000,
                    "input_samples": processed_samples,
                    "transmitted_samples": transmitted_samples,
                    "model_bound_samples": model_bound_samples,
                    "input_chunks": input_chunk_sequence,
                    "model_input_packets": model_input_packet_sequence,
                    "model_packets": model_packet_sequence,
                    "xvc_inference_windows": vc_windows,
                    "xvc_silence_bypassed_windows": vc_silence_bypassed_windows,
                    "xvc_silence_gate_rms": SILENCE_GATE_RMS,
                    "xvc_silence_hangover_ms": SILENCE_HANGOVER_MS,
                    "xvc_content_path_warmed": CONTENT_PATH_WARMED,
                    "frame_header": "HMO1/<4sIIII>",
                },
            )
        except Exception as exc:  # teardown must still persist stream_stop events
            upload_error = str(exc)
            logger.exception("[xvc proxy] failed to build study proxy artifacts")
        if events:
            events.add(
                "proxy_artifacts_complete" if uploaded else "proxy_artifacts_failed",
                artifacts=sorted(artifact_bytes), error=upload_error,
            )

    if events:
        if speech:
            for row in speech.close():
                event = row.pop("event")
                events.add(event, **row)
        if assistant_packet_active:
            events.add("assistant_speech_end", packet_sequence=model_packet_sequence,
                       detector="packet_gap")
        events.add("stream_stop", input_samples=processed_samples,
                   transmitted_samples=transmitted_samples,
                   model_bound_samples=model_bound_samples,
                   input_chunks=input_chunk_sequence,
                   model_input_packets=model_input_packet_sequence,
                   model_output_packets=model_packet_sequence,
                   output_windows=chunk_count,
                   xvc_inference_windows=vc_windows,
                   xvc_silence_bypassed_windows=vc_silence_bypassed_windows)
        await events.flush(force=True)

    if debug_dir and debug_pcm:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            out_path = os.path.join(debug_dir, f"pplx_input_{target_id}_{int(time.time())}.wav")
            _save_wav(out_path, np.concatenate(debug_pcm), 24000)
            logger.info(f"[xvc proxy] saved PersonaPlex-input audio to {out_path}")
        except Exception as e:
            logger.error(f"[xvc proxy] failed to save debug WAV: {e}")

    if otel:
        otel.set_session_attributes(chunks=chunk_count)
        if vc_windows:
            otel.set_session_attributes(vc_inference_avg_ms=round(vc_ms_total / vc_windows, 1))
    logger.info(
        "[xvc proxy] closed after %d chunks (%d inferred, %d silence-bypassed)",
        chunk_count, vc_windows, vc_silence_bypassed_windows)
    return browser_ws


@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        resp = web.Response()
    else:
        resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


async def handle_info(_request: web.Request) -> web.Response:
    """Lets the frontend show the correct engine label. Both engines mount the
    same /api/meanvc/* routes, so this is the only way to tell them apart from
    the client side."""
    return web.json_response({"engine": "xvc"})


def create_app() -> web.Application:
    mws = [cors_middleware]
    # Tracing middleware opens a SERVER span per request and reads the W3C
    # traceparent from headers or the ?traceparent= query (WebSocket handshakes),
    # so a browser session links to app-api's /condition span. No-op if disabled.
    if otel and otel.init_tracing("xvc"):
        otel.init_metrics("xvc")          # latency histograms (vc.inference_ms, personaplex.first_response_ms)
        otel.instrument_aiohttp_client()  # traces the /condition GET to app-api
        mws.append(otel.aiohttp_middleware("xvc"))
        logger.info("OpenTelemetry tracing enabled (xvc)")
    app = web.Application(middlewares=mws, client_max_size=10 * 1024 * 1024)
    app["file_conversion_lock"] = asyncio.Lock()
    app.router.add_get("/api/meanvc/info", handle_info)
    app.router.add_post("/api/meanvc/load-target", handle_load_target)
    app.router.add_post("/api/xvc/file-conversion", handle_file_conversion)
    app.router.add_get("/api/meanvc/stream", handle_stream)
    app.router.add_get("/api/meanvc/chat-proxy", handle_chat_proxy)
    return app


async def on_startup(app: web.Application):
    global cfg, model, device, SR, HP_CUT, MASK_TARGET_COND
    logger.info(f"[xvc] loading model: config={XVC_CONFIG} ckpt={XVC_CKPT} device={XVC_DEVICE}")
    cfg, model, device = load_xvc(XVC_CONFIG, XVC_CKPT, XVC_DEVICE, XVC_EMA_LOAD)
    SR = int(cfg["sample_rate"])
    HP_CUT = float(cfg.get("highpass_cutoff_freq", 0.0))
    MASK_TARGET_COND = bool(cfg.get("dataloader", {}).get("mask_target_condition", True))
    logger.info(
        f"[xvc] ready: sr={SR} hp_cut={HP_CUT} window(ms) chunk={CHUNK_MS} "
        f"current={CURRENT_MS} smooth={SMOOTH_MS} future={FUTURE_MS}"
    )


def main():
    import ssl

    port = int(os.environ.get("MEANVC_PORT", 5002))
    app = create_app()
    app.on_startup.append(on_startup)
    ssl_dir = os.environ.get("SSL_DIR", "/app/ssl")
    ssl_context = None
    cert_file = os.path.join(ssl_dir, "cert.pem")
    key_file = os.path.join(ssl_dir, "key.pem")
    if os.path.exists(cert_file) and os.path.exists(key_file):
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(cert_file, key_file)
        logger.info(f"SSL enabled from {ssl_dir}")
    logger.info(f"X-VC server starting on port {port} (ssl={ssl_context is not None})")
    web.run_app(app, port=port, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
