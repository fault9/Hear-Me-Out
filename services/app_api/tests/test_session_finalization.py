from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from study import router as study_router
from study.storage import SqliteBackend


class SessionFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.media = self.root / "media"
        self.sessions = self.media / "sessions"
        self.targets = self.media / "targets"
        self.backend = SqliteBackend(str(self.root / "study.db"))

        study = self.backend.create_study("finalization")
        scenario = self.backend.add_scenario(study["id"], {
            "order_idx": 0,
            "title": "scenario",
            "scenario_card": {},
            "system_prompt": "prompt",
            "voice_schedule": [],
        })
        participant = self.backend.generate_participants(
            study["id"], 1, [scenario["id"]])[0]
        run = self.backend.start_run(participant["participant_id"], "restart")
        self.session_id = "P01001_R01_S01_A01"
        session = self.backend.create_session(
            self.session_id,
            participant["participant_id"],
            scenario["id"],
            1,
            "stable_converted",
            "target",
            run["id"],
            run["attempt"],
            1,
            [{"mode": "vc", "start_s": 0, "end_s": None}],
            {"engine": "xvc"},
        )

        patches = [
            patch.object(study_router, "STUDY_DATA_DIR", self.media),
            patch.object(study_router, "SESSIONS_DIR", self.sessions),
            patch.object(study_router, "TARGETS_DIR", self.targets),
            patch.object(study_router, "get_backend", return_value=self.backend),
            patch.object(study_router, "get_manager", return_value=Mock()),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

        self.session_dir = study_router._session_dir(session)
        self.session_dir.mkdir(parents=True)
        (self.session_dir / "events.jsonl").touch()
        self.backend.update_session_artifacts(self.session_id, {"artifacts": {}})

        app = FastAPI()
        app.include_router(study_router.build_study_router())
        self.client = TestClient(app)

    def tearDown(self):
        self.temp.cleanup()

    def _post_events(self, rows: list[dict]):
        return self.client.post(
            f"/api/study/internal/session/{self.session_id}/events",
            json={"events": rows},
            headers={"X-Study-Event-Token": study_router.EVENT_TOKEN},
        )

    def _post_proxy(self, received: bytes = b"received"):
        return self.client.post(
            f"/api/study/internal/session/{self.session_id}/proxy-artifacts",
            files={
                "proxy_received_wav": ("proxy_received.wav", received, "audio/wav"),
                "participant_proxy_wav": (
                    "participant_proxy.wav", b"participant", "audio/wav"),
                "personaplex_input_opus": (
                    "personaplex_input.opus", b"opus-input", "audio/ogg"),
            },
            data={"metadata": json.dumps({
                "schema": "hmo.proxy-artifacts.v1",
                "input_chunks": 4,
            })},
            headers={"X-Study-Event-Token": study_router.EVENT_TOKEN},
        )

    def _save_browser(self, participant: bytes = b"participant"):
        return self.client.post(
            f"/api/study/session/{self.session_id}/save",
            files={
                "participant": ("participant.wav", participant, "audio/wav"),
                "participant_raw": (
                    "participant_raw.wav", b"participant-raw", "audio/wav"),
                "model": ("model.wav", b"model", "audio/wav"),
                "merged": ("merged.wav", b"merged", "audio/wav"),
                "model_transcript_file": (
                    "model_transcript.json", b"[]", "application/json"),
                "client_timeline_file": (
                    "client_timeline.json",
                    json.dumps({"schema": "hmo.client-timeline.v1"}).encode(),
                    "application/json",
                ),
            },
        )

    def test_browser_artifact_save_is_hash_idempotent(self):
        first = self._save_browser()
        self.assertEqual(first.status_code, 200)
        duplicate = self._save_browser()
        self.assertEqual(duplicate.status_code, 200)

        conflicting = self._save_browser(participant=b"different")
        self.assertEqual(conflicting.status_code, 409)
        self.assertEqual(
            (self.session_dir / "participant.wav").read_bytes(), b"participant")

    def test_incomplete_end_stays_retryable_until_all_proxy_evidence_arrives(self):
        ended = self.client.post(
            f"/api/study/session/{self.session_id}/end",
            json={"reason": "goal_reached"},
        )
        self.assertEqual(ended.status_code, 200)
        self.assertFalse(ended.json()["sealed"])
        self.assertEqual(ended.json()["finalization"]["status"], "pending")
        self.assertFalse((self.session_dir / "manifest.final.json").exists())
        self.assertIsNotNone(self.backend.get_session(self.session_id)["ended_at"])

        stopped = self._post_events([
            {"event": "stream_stop", "event_sequence": 1},
        ])
        self.assertEqual(stopped.status_code, 200)
        self.assertFalse(stopped.json()["sealed"])

        prepared = self._post_events([
            {"event": "proxy_artifacts_prepared", "event_sequence": 2},
        ])
        self.assertEqual(prepared.status_code, 200)
        self.assertFalse(prepared.json()["sealed"])

        artifacts = self._post_proxy()
        self.assertEqual(artifacts.status_code, 200)
        self.assertTrue(artifacts.json()["sealed"])
        self.assertTrue((self.session_dir / "manifest.final.json").exists())

        # Lost HTTP responses are safe to retry after sealing.
        duplicate_event = self._post_events([
            {"event": "proxy_artifacts_prepared", "event_sequence": 2},
        ])
        self.assertEqual(duplicate_event.status_code, 200)
        self.assertTrue(duplicate_event.json()["idempotent"])
        conflicting_event = self._post_events([
            {"event": "proxy_artifacts_failed", "event_sequence": 2},
        ])
        self.assertEqual(conflicting_event.status_code, 409)
        duplicate_proxy = self._post_proxy()
        self.assertEqual(duplicate_proxy.status_code, 200)
        self.assertTrue(duplicate_proxy.json()["idempotent"])

        conflicting_proxy = self._post_proxy(received=b"different")
        self.assertEqual(conflicting_proxy.status_code, 409)
        self.assertEqual(
            (self.session_dir / "proxy_received.wav").read_bytes(), b"received")


if __name__ == "__main__":
    unittest.main()
