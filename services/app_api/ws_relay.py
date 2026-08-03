"""Same-origin WebSocket relay for the VC proxy.

The browser's audio socket historically connected straight to the VC engine on
:5002. That cross-origin, raw-IP, odd-port socket is exactly the shape ad
blockers filter, and Firefox requires a second certificate acceptance for the
extra port. Serving the same endpoint on the page's own origin (:5001) removes
both failure classes; frames are relayed verbatim in both directions, so the
engine's framing contract (tagged binary + JSON text) is untouched.

The upstream defaults to the local engine and can be overridden with
STUDY_CHAT_UPSTREAM (e.g. ws://127.0.0.1:<port> in tests).
"""

import asyncio
import contextlib
import logging
import os
import ssl
from urllib.parse import parse_qs

import websockets
from fastapi import FastAPI, WebSocket

logger = logging.getLogger(__name__)

RELAY_PATH = "/api/meanvc/chat-proxy"
DEFAULT_UPSTREAM = "wss://127.0.0.1:5002"


def _sendable_code(code) -> int:
    # 1005/1006/1015 are synthesized locally and must never appear in a close
    # frame; report those (and unknown) endings as 1011 so the peer sees an
    # abnormal end rather than a protocol error.
    if code is None or code in (1005, 1006, 1015):
        return 1011
    return int(code)


async def _pump_browser_to_upstream(browser: WebSocket, upstream):
    """Forward client frames until the browser disconnects; returns its close code."""
    while True:
        message = await browser.receive()
        if message["type"] == "websocket.disconnect":
            return message.get("code")
        data = message.get("bytes")
        if data is not None:
            await upstream.send(data)
            continue
        text = message.get("text")
        if text is not None:
            await upstream.send(text)


async def _pump_upstream_to_browser(upstream, browser: WebSocket):
    async for message in upstream:
        if isinstance(message, (bytes, bytearray, memoryview)):
            await browser.send_bytes(bytes(message))
        else:
            await browser.send_text(message)


async def _relay(browser: WebSocket) -> None:
    await browser.accept()
    query = browser.url.query or ""
    sid = (parse_qs(query).get("session_id") or ["-"])[0]
    upstream_base = os.environ.get("STUDY_CHAT_UPSTREAM", DEFAULT_UPSTREAM).rstrip("/")
    url = f"{upstream_base}{RELAY_PATH}" + (f"?{query}" if query else "")

    ssl_ctx = None
    if url.startswith("wss:"):
        # The upstream is this host's own engine behind the self-signed cert.
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

    try:
        upstream = await websockets.connect(url, ssl=ssl_ctx, max_size=None, open_timeout=10)
    except Exception as exc:  # noqa: BLE001 - any dial failure ends the same way
        logger.warning(f"[relay] upstream connect failed session={sid}: {exc}")
        with contextlib.suppress(Exception):
            await browser.close(code=1011, reason="VC engine unreachable")
        return

    logger.info(f"[relay] chat-proxy open session={sid} -> {upstream_base}")
    browser_code = None
    up_task = asyncio.create_task(_pump_browser_to_upstream(browser, upstream))
    down_task = asyncio.create_task(_pump_upstream_to_browser(upstream, browser))
    try:
        done, pending = await asyncio.wait(
            {up_task, down_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        if up_task in done:
            # Browser went away (or its pump failed): pass its close code upstream.
            with contextlib.suppress(Exception):
                browser_code = up_task.result()
            with contextlib.suppress(Exception):
                await upstream.close(code=_sendable_code(browser_code))
        else:
            # Upstream ended (or its pump failed): mirror its close to the browser.
            with contextlib.suppress(Exception):
                down_task.result()
            with contextlib.suppress(Exception):
                await browser.close(code=_sendable_code(getattr(upstream, "close_code", None)))
    finally:
        with contextlib.suppress(Exception):
            await upstream.close()
        logger.info(
            f"[relay] chat-proxy closed session={sid} "
            f"browser_code={browser_code} upstream_code={getattr(upstream, 'close_code', None)}")


def register_chat_relay(app: FastAPI) -> None:
    @app.websocket(RELAY_PATH)
    async def chat_proxy_relay(browser: WebSocket):
        await _relay(browser)
