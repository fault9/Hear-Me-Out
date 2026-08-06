from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import unittest

import websockets
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ws_relay import AUDIT_RELAY_PATH, RELAY_PATH, register_chat_relay


class FakeUpstream:
    """Plain-ws stand-in for the VC proxy: greets with the request target, echoes
    binary verbatim, echoes text with a prefix, and closes cleanly on 'CLOSE'."""

    def __init__(self):
        self.port = None
        self._ready = threading.Event()
        self._stop = None
        self._loop = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        assert self._ready.wait(5), "fake upstream failed to start"
        return self

    def stop(self):
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set_result, None)
        self._thread.join(5)

    def _run(self):
        asyncio.run(self._main())

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop = self._loop.create_future()

        async def handler(ws):
            request = getattr(ws, "request", None)
            target = request.path if request is not None else ws.path
            await ws.send(f"target:{target}")
            async for message in ws:
                if isinstance(message, bytes):
                    await ws.send(message)
                elif message == "CLOSE":
                    await ws.close(code=1000, reason="done")
                    return
                else:
                    await ws.send(f"echo:{message}")

        server = await websockets.serve(handler, "127.0.0.1", 0)
        self.port = server.sockets[0].getsockname()[1]
        self._ready.set()
        try:
            await self._stop
        finally:
            server.close()
            await server.wait_closed()


def make_client() -> TestClient:
    app = FastAPI()
    register_chat_relay(app)
    return TestClient(app)


class WsRelayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = FakeUpstream().start()
        cls.upstream_url = f"ws://127.0.0.1:{cls.upstream.port}"
        os.environ["STUDY_CHAT_UPSTREAM"] = cls.upstream_url
        os.environ["PERSONAPLEX_AUDIT_UPSTREAM"] = cls.upstream_url

    @classmethod
    def tearDownClass(cls):
        cls.upstream.stop()
        os.environ.pop("STUDY_CHAT_UPSTREAM", None)
        os.environ.pop("PERSONAPLEX_AUDIT_UPSTREAM", None)

    def test_relays_query_binary_and_text_verbatim(self):
        client = make_client()
        with client.websocket_connect(
                f"{RELAY_PATH}?session_id=S1&source_sr=48000") as ws:
            greeting = ws.receive_text()
            self.assertEqual(
                greeting, f"target:{RELAY_PATH}?session_id=S1&source_sr=48000")
            payload = b"\x01" + bytes(range(32))
            ws.send_bytes(payload)
            self.assertEqual(ws.receive_bytes(), payload)
            ws.send_text('{"type":"capture_summary"}')
            self.assertEqual(ws.receive_text(), 'echo:{"type":"capture_summary"}')

    def test_upstream_close_reaches_browser(self):
        client = make_client()
        with client.websocket_connect(RELAY_PATH) as ws:
            ws.receive_text()
            ws.send_text("CLOSE")
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_text()
            self.assertEqual(ctx.exception.code, 1000)

    def test_audit_relay_returns_independent_delivery_receipt(self):
        client = make_client()
        with client.websocket_connect(
                f"{AUDIT_RELAY_PATH}?text_prompt=test") as ws:
            self.assertEqual(
                ws.receive_text(), "target:/api/chat?text_prompt=test"
            )
            ws.send_text(json.dumps({
                "type": "hmo.audit.delivery_reset",
                "request_id": "reset-1",
            }))
            reset = json.loads(ws.receive_text())
            self.assertEqual(reset["type"], "hmo.audit.delivery_reset_ack")

            payloads = [b"\x01first-page", b"\x01second-page"]
            for payload in payloads:
                ws.send_bytes(payload)
                self.assertEqual(ws.receive_bytes(), payload)

            ws.send_text(json.dumps({
                "type": "hmo.audit.delivery_receipt_request",
                "request_id": "receipt-1",
            }))
            receipt = json.loads(ws.receive_text())
            expected_hash = hashlib.sha256(b"".join(payloads)).hexdigest()
            self.assertEqual(receipt["type"], "hmo.audit.delivery_receipt")
            self.assertEqual(receipt["relay_received"]["frames"], 2)
            self.assertEqual(
                receipt["relay_received"]["bytes"],
                sum(len(payload) for payload in payloads),
            )
            self.assertEqual(receipt["relay_received"]["sha256"], expected_hash)
            self.assertEqual(
                receipt["relay_forwarded"], receipt["relay_received"]
            )

    def test_unreachable_upstream_closes_1011(self):
        os.environ["STUDY_CHAT_UPSTREAM"] = "ws://127.0.0.1:1"
        try:
            client = make_client()
            with client.websocket_connect(RELAY_PATH) as ws:
                with self.assertRaises(WebSocketDisconnect) as ctx:
                    ws.receive_text()
                self.assertEqual(ctx.exception.code, 1011)
        finally:
            os.environ["STUDY_CHAT_UPSTREAM"] = self.upstream_url


if __name__ == "__main__":
    unittest.main()
