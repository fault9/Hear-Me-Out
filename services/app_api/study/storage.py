"""Persistence for the study platform (v2 — multi-study).

`StorageBackend` is the interface; `SqliteBackend` is the pilot implementation
(one SQLite file, WAV/JSON artifacts on disk). A future `MongoBackend` can
implement the same interface. Rows are returned as plain dicts (JSON columns
decoded, with the `_json` suffix stripped) to stay backend-neutral.

Studies are independent (no "single active study"); a participant code maps to
exactly one study. Scenarios are first-class rows with a per-scenario voice
schedule; targets carry the VC engine they belong to.
"""

from __future__ import annotations

import abc
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

RUN_WINDOW_SECONDS = 3600
SCHEMA_VERSION = 6


def _now() -> float:
    return time.time()


class StorageBackend(abc.ABC):
    # studies
    @abc.abstractmethod
    def create_study(self, name: str, description: str = "") -> dict: ...
    @abc.abstractmethod
    def list_studies(self) -> list[dict]: ...
    @abc.abstractmethod
    def get_study(self, study_id: int) -> Optional[dict]: ...
    @abc.abstractmethod
    def update_study(self, study_id: int, name: Optional[str] = None, description: Optional[str] = None,
                     questionnaires: Optional[dict] = None, settings: Optional[dict] = None) -> Optional[dict]: ...
    @abc.abstractmethod
    def archive_study(self, study_id: int, archived: bool = True) -> None: ...
    # scenarios
    @abc.abstractmethod
    def list_scenarios(self, study_id: int) -> list[dict]: ...
    @abc.abstractmethod
    def get_scenario(self, scenario_id: int) -> Optional[dict]: ...
    @abc.abstractmethod
    def add_scenario(self, study_id: int, data: dict) -> dict: ...
    @abc.abstractmethod
    def update_scenario(self, scenario_id: int, data: dict) -> Optional[dict]: ...
    @abc.abstractmethod
    def delete_scenario(self, scenario_id: int) -> None: ...
    # targets
    @abc.abstractmethod
    def add_target(self, study_id: int, ref: str, speaker_id: str, label: str,
                   wav_path: str, engine: str) -> dict: ...
    @abc.abstractmethod
    def list_targets(self, study_id: int) -> list[dict]: ...
    @abc.abstractmethod
    def delete_target(self, target_id: int) -> None: ...
    @abc.abstractmethod
    def set_engine_target_id(self, target_id: int, engine_target_id: Optional[str]) -> None: ...
    @abc.abstractmethod
    def clear_engine_target_ids(self, study_id: int, engine: str) -> None: ...
    # participants
    @abc.abstractmethod
    def generate_participants(self, study_id: int, count: int, scenario_order: list[int],
                              allocations: Optional[list[dict]] = None) -> list[dict]: ...
    @abc.abstractmethod
    def get_participant_by_code(self, code: str) -> Optional[dict]: ...
    @abc.abstractmethod
    def list_participants(self, study_id: int) -> list[dict]: ...
    @abc.abstractmethod
    def assign_participant(self, participant_id: str, allocation: dict,
                           allocation_stratum: str) -> dict: ...
    # runs
    @abc.abstractmethod
    def get_latest_run(self, participant_id: str) -> Optional[dict]: ...
    @abc.abstractmethod
    def start_run(self, participant_id: str, mode: str) -> dict: ...
    @abc.abstractmethod
    def update_run_progress(self, run_id: int, current_step: dict, completed: dict) -> None: ...
    @abc.abstractmethod
    def submit_run(self, run_id: int) -> None: ...
    @abc.abstractmethod
    def get_live_run(self, study_id: int) -> Optional[dict]: ...
    @abc.abstractmethod
    def list_runs(self, study_id: int) -> list[dict]: ...
    # sessions
    @abc.abstractmethod
    def create_session(self, session_id: str, participant_id: str, scenario_id: str,
                       scenario_order: int, voice_condition: str, target_speaker_id: str,
                       run_id: int, run_attempt: int, scenario_attempt: int,
                       schedule: list, config_snapshot: dict) -> dict: ...
    @abc.abstractmethod
    def next_session_attempt(self, participant_id: str, run_id: int, scenario_order: int) -> int: ...
    @abc.abstractmethod
    def save_session(self, session_id: str, files: dict, transcript: Any, metrics: Any,
                     audiobox_available: bool) -> None: ...
    @abc.abstractmethod
    def update_session_analysis(self, session_id: str, transcript: Any, metrics: Any,
                                audiobox_available: bool) -> None: ...
    @abc.abstractmethod
    def update_session_artifacts(self, session_id: str, manifest: dict) -> None: ...
    @abc.abstractmethod
    def update_session_vc_quality(self, session_id: str, status: str, result: Any) -> None: ...
    @abc.abstractmethod
    def end_session(self, session_id: str, end_reason: str) -> None: ...
    @abc.abstractmethod
    def get_session(self, session_id: str) -> Optional[dict]: ...
    @abc.abstractmethod
    def list_sessions(self, study_id: int) -> list[dict]: ...
    @abc.abstractmethod
    def get_latest_session(self, participant_id: str, scenario_order: int,
                           run_id: Optional[int] = None) -> Optional[dict]: ...
    # answers
    @abc.abstractmethod
    def save_answer(self, participant_id: str, session_id: Optional[str], kind: str, payload: Any) -> None: ...
    @abc.abstractmethod
    def list_answers(self, study_id: int) -> list[dict]: ...
    @abc.abstractmethod
    def has_answer(self, participant_id: str, run_id: int, kind: str) -> bool: ...


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS study (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL DEFAULT '',
  description TEXT NOT NULL DEFAULT '',
  questionnaires_json TEXT NOT NULL DEFAULT '{}',
  settings_json TEXT NOT NULL DEFAULT '{}',
  archived INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS scenario (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  study_id INTEGER NOT NULL,
  order_idx INTEGER NOT NULL DEFAULT 0,
  title TEXT NOT NULL DEFAULT '',
  scenario_card_json TEXT NOT NULL DEFAULT '{}',
  system_prompt TEXT NOT NULL DEFAULT '',
  voice_prompt TEXT NOT NULL DEFAULT '',
  voice_schedule_json TEXT NOT NULL DEFAULT '[]',
  post_items_json TEXT NOT NULL DEFAULT '[]',
  time_limit_s INTEGER NOT NULL DEFAULT 300,
  is_test INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS target (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  study_id INTEGER NOT NULL,
  ref TEXT NOT NULL,
  speaker_id TEXT NOT NULL DEFAULT '',
  label TEXT NOT NULL DEFAULT '',
  wav_path TEXT NOT NULL,
  engine TEXT NOT NULL DEFAULT 'meanvc',
  engine_target_id TEXT,
  UNIQUE(study_id, ref)
);
CREATE TABLE IF NOT EXISTS participant (
  participant_id TEXT PRIMARY KEY,
  code TEXT NOT NULL UNIQUE,
  study_id INTEGER NOT NULL,
  scenario_order_json TEXT NOT NULL DEFAULT '[]',
  variant_id TEXT,
  target_ref TEXT,
  assignment_json TEXT NOT NULL DEFAULT '{}',
  allocation_status TEXT NOT NULL DEFAULT 'issued',
  allocation_stratum TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS run (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  participant_id TEXT NOT NULL,
  study_id INTEGER NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 1,
  status TEXT NOT NULL DEFAULT 'in_progress',
  current_step_json TEXT NOT NULL DEFAULT '{}',
  completed_json TEXT NOT NULL DEFAULT '{}',
  started_at REAL NOT NULL,
  expires_at REAL NOT NULL,
  submitted_at REAL
);
CREATE TABLE IF NOT EXISTS session (
  session_id TEXT PRIMARY KEY,
  participant_id TEXT NOT NULL,
  study_id INTEGER NOT NULL,
  scenario_id TEXT NOT NULL,
  scenario_order INTEGER NOT NULL,
  voice_condition TEXT NOT NULL DEFAULT '',
  target_speaker_id TEXT NOT NULL DEFAULT '',
  run_id INTEGER,
  run_attempt INTEGER NOT NULL DEFAULT 1,
  scenario_attempt INTEGER NOT NULL DEFAULT 1,
  schedule_json TEXT NOT NULL DEFAULT '[]',
  config_snapshot_json TEXT NOT NULL DEFAULT '{}',
  started_at REAL NOT NULL,
  ended_at REAL,
  end_reason TEXT,
  files_json TEXT,
  transcript_json TEXT,
  metrics_json TEXT,
  audiobox_available INTEGER,
  artifact_manifest_json TEXT,
  vc_quality_json TEXT,
  vc_quality_status TEXT NOT NULL DEFAULT 'pending'
);
CREATE TABLE IF NOT EXISTS answer (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  participant_id TEXT NOT NULL,
  study_id INTEGER NOT NULL,
  run_id INTEGER,
  session_id TEXT,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at REAL NOT NULL
);
"""


def _loads(row: sqlite3.Row, json_cols: tuple[str, ...]) -> dict:
    d = dict(row)
    for c in json_cols:
        if c in d and d[c] is not None:
            key = c[:-5] if c.endswith("_json") else c
            try:
                d[key] = json.loads(d[c])
            except (json.JSONDecodeError, TypeError):
                d[key] = None
    return d


class SqliteBackend(StorageBackend):
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @staticmethod
    def _add_column(c, table, col, decl):
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass  # column already exists

    def _init_schema(self):
        with self._conn() as c:
            c.executescript(_SCHEMA)
            row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            ver = int(row["value"]) if row else 0
            # Additive migrations preserve existing studies (no rebuild).
            if ver < 3:
                self._add_column(c, "study", "settings_json", "TEXT NOT NULL DEFAULT '{}'")
                self._add_column(c, "scenario", "post_items_json", "TEXT NOT NULL DEFAULT '[]'")
            if ver < 4:
                self._add_column(c, "participant", "variant_id", "TEXT")
                self._add_column(c, "participant", "target_ref", "TEXT")
                self._add_column(c, "participant", "assignment_json", "TEXT NOT NULL DEFAULT '{}'")
                self._add_column(c, "participant", "allocation_status", "TEXT NOT NULL DEFAULT 'issued'")
                self._add_column(c, "session", "run_id", "INTEGER")
                self._add_column(c, "session", "run_attempt", "INTEGER NOT NULL DEFAULT 1")
                self._add_column(c, "session", "scenario_attempt", "INTEGER NOT NULL DEFAULT 1")
                self._add_column(c, "session", "schedule_json", "TEXT NOT NULL DEFAULT '[]'")
                self._add_column(c, "session", "config_snapshot_json", "TEXT NOT NULL DEFAULT '{}'")
                self._add_column(c, "answer", "run_id", "INTEGER")
            if ver < 5:
                self._add_column(c, "session", "artifact_manifest_json", "TEXT")
                self._add_column(c, "session", "vc_quality_json", "TEXT")
                self._add_column(c, "session", "vc_quality_status", "TEXT NOT NULL DEFAULT 'pending'")
            if ver < 6:
                self._add_column(c, "participant", "allocation_stratum", "TEXT")
            c.execute("CREATE UNIQUE INDEX IF NOT EXISTS session_attempt_unique ON session("
                      "participant_id, run_id, scenario_order, scenario_attempt)")
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
                      (str(SCHEMA_VERSION),))

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---------- studies ----------
    def create_study(self, name, description="") -> dict:
        with self._conn() as c:
            cur = c.execute("INSERT INTO study(name, description, questionnaires_json, created_at) "
                            "VALUES(?,?,?,?)", (name, description, "{}", _now()))
            sid = cur.lastrowid
        return self.get_study(sid)

    def list_studies(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM study ORDER BY archived, id DESC").fetchall()
        return [_loads(r, ("questionnaires_json", "settings_json")) for r in rows]

    def get_study(self, study_id) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM study WHERE id=?", (study_id,)).fetchone()
        return _loads(row, ("questionnaires_json", "settings_json")) if row else None

    def update_study(self, study_id, name=None, description=None, questionnaires=None,
                     settings=None) -> Optional[dict]:
        with self._conn() as c:
            if name is not None:
                c.execute("UPDATE study SET name=? WHERE id=?", (name, study_id))
            if description is not None:
                c.execute("UPDATE study SET description=? WHERE id=?", (description, study_id))
            if questionnaires is not None:
                c.execute("UPDATE study SET questionnaires_json=? WHERE id=?",
                          (json.dumps(questionnaires), study_id))
            if settings is not None:
                c.execute("UPDATE study SET settings_json=? WHERE id=?",
                          (json.dumps(settings), study_id))
        return self.get_study(study_id)

    def archive_study(self, study_id, archived=True) -> None:
        with self._conn() as c:
            c.execute("UPDATE study SET archived=? WHERE id=?", (1 if archived else 0, study_id))

    # ---------- scenarios ----------
    def list_scenarios(self, study_id) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM scenario WHERE study_id=? ORDER BY order_idx, id",
                             (study_id,)).fetchall()
        return [_loads(r, ("scenario_card_json", "voice_schedule_json", "post_items_json")) for r in rows]

    def get_scenario(self, scenario_id) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM scenario WHERE id=?", (scenario_id,)).fetchone()
        return _loads(row, ("scenario_card_json", "voice_schedule_json", "post_items_json")) if row else None

    def _scenario_cols(self, data: dict) -> dict:
        return {
            "order_idx": data.get("order_idx", 0),
            "title": data.get("title", ""),
            "scenario_card_json": json.dumps(data.get("scenario_card", {})),
            "system_prompt": data.get("system_prompt", ""),
            "voice_prompt": data.get("voice_prompt", ""),
            "voice_schedule_json": json.dumps(data.get("voice_schedule", [])),
            "post_items_json": json.dumps(data.get("post_items", [])),
            "time_limit_s": int(data.get("time_limit_s", 300)),
            "is_test": 1 if data.get("is_test") else 0,
        }

    def add_scenario(self, study_id, data) -> dict:
        cols = self._scenario_cols(data)
        with self._conn() as c:
            if "order_idx" not in data:
                n = c.execute("SELECT COUNT(*) AS n FROM scenario WHERE study_id=?", (study_id,)).fetchone()["n"]
                cols["order_idx"] = n
            cur = c.execute(
                "INSERT INTO scenario(study_id, order_idx, title, scenario_card_json, system_prompt, "
                "voice_prompt, voice_schedule_json, post_items_json, time_limit_s, is_test) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (study_id, cols["order_idx"], cols["title"], cols["scenario_card_json"],
                 cols["system_prompt"], cols["voice_prompt"], cols["voice_schedule_json"],
                 cols["post_items_json"], cols["time_limit_s"], cols["is_test"]))
            sid = cur.lastrowid
        return self.get_scenario(sid)

    def update_scenario(self, scenario_id, data) -> Optional[dict]:
        cols = self._scenario_cols(data)
        with self._conn() as c:
            c.execute(
                "UPDATE scenario SET order_idx=?, title=?, scenario_card_json=?, system_prompt=?, "
                "voice_prompt=?, voice_schedule_json=?, post_items_json=?, time_limit_s=?, is_test=? WHERE id=?",
                (cols["order_idx"], cols["title"], cols["scenario_card_json"], cols["system_prompt"],
                 cols["voice_prompt"], cols["voice_schedule_json"], cols["post_items_json"],
                 cols["time_limit_s"], cols["is_test"], scenario_id))
        return self.get_scenario(scenario_id)

    def delete_scenario(self, scenario_id) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM scenario WHERE id=?", (scenario_id,))

    # ---------- targets ----------
    def add_target(self, study_id, ref, speaker_id, label, wav_path, engine) -> dict:
        with self._conn() as c:
            c.execute(
                "INSERT INTO target(study_id, ref, speaker_id, label, wav_path, engine) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(study_id, ref) DO UPDATE SET speaker_id=excluded.speaker_id, "
                "label=excluded.label, wav_path=excluded.wav_path, engine=excluded.engine, engine_target_id=NULL",
                (study_id, ref, speaker_id, label, wav_path, engine))
            row = c.execute("SELECT * FROM target WHERE study_id=? AND ref=?", (study_id, ref)).fetchone()
        return dict(row)

    def list_targets(self, study_id) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM target WHERE study_id=? ORDER BY id", (study_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_target(self, target_id) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM target WHERE id=?", (target_id,))

    def set_engine_target_id(self, target_id, engine_target_id) -> None:
        with self._conn() as c:
            c.execute("UPDATE target SET engine_target_id=? WHERE id=?", (engine_target_id, target_id))

    def clear_engine_target_ids(self, study_id, engine) -> None:
        with self._conn() as c:
            c.execute("UPDATE target SET engine_target_id=NULL WHERE study_id=? AND engine=?",
                      (study_id, engine))

    # ---------- participants ----------
    def generate_participants(self, study_id, count, scenario_order, allocations=None) -> list[dict]:
        created = []
        with self._conn() as c:
            n = c.execute("SELECT COUNT(*) AS n FROM participant WHERE study_id=?", (study_id,)).fetchone()["n"]
            for i in range(count):
                pid = f"P{study_id:02d}{n + i + 1:03d}"
                code = _gen_code()
                while c.execute("SELECT 1 FROM participant WHERE code=?", (code,)).fetchone():
                    code = _gen_code()
                allocation = (allocations or [{}])[i] if allocations else {}
                assigned_order = allocation.get("scenario_order") or scenario_order
                c.execute("INSERT INTO participant(participant_id, code, study_id, scenario_order_json, "
                          "variant_id, target_ref, assignment_json, allocation_status, "
                          "allocation_stratum, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                          (pid, code, study_id, json.dumps(assigned_order),
                           allocation.get("variant_id"), allocation.get("target_ref"),
                           json.dumps(allocation.get("assignment") or {}),
                           allocation.get("allocation_status") or "issued",
                           allocation.get("allocation_stratum"), _now()))
                created.append({"participant_id": pid, "code": code,
                                "scenario_order": assigned_order,
                                "variant_id": allocation.get("variant_id"),
                                "target_ref": allocation.get("target_ref"),
                                "allocation_status": allocation.get("allocation_status") or "issued"})
        return created

    def get_participant_by_code(self, code) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM participant WHERE code=?", (code.strip(),)).fetchone()
        return _loads(row, ("scenario_order_json", "assignment_json")) if row else None

    def _participant_by_id(self, c, participant_id) -> Optional[dict]:
        row = c.execute("SELECT * FROM participant WHERE participant_id=?", (participant_id,)).fetchone()
        return _loads(row, ("scenario_order_json", "assignment_json")) if row else None

    def list_participants(self, study_id) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM participant WHERE study_id=? ORDER BY participant_id",
                             (study_id,)).fetchall()
        return [_loads(r, ("scenario_order_json", "assignment_json")) for r in rows]

    def assign_participant(self, participant_id, allocation, allocation_stratum) -> dict:
        with self._conn() as c:
            cur = c.execute(
                "UPDATE participant SET scenario_order_json=?, variant_id=?, target_ref=?, "
                "assignment_json=?, allocation_status='assigned', allocation_stratum=? "
                "WHERE participant_id=? AND allocation_status='awaiting_profile'",
                (json.dumps(allocation.get("scenario_order") or []),
                 allocation.get("variant_id"), allocation.get("target_ref"),
                 json.dumps(allocation.get("assignment") or {}), allocation_stratum,
                 participant_id))
            if cur.rowcount != 1:
                raise ValueError("participant is not awaiting target assignment")
            row = c.execute("SELECT * FROM participant WHERE participant_id=?",
                            (participant_id,)).fetchone()
        return _loads(row, ("scenario_order_json", "assignment_json"))

    # ---------- runs ----------
    def get_latest_run(self, participant_id) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM run WHERE participant_id=? ORDER BY attempt DESC LIMIT 1",
                            (participant_id,)).fetchone()
        return self._run_dict(row) if row else None

    def start_run(self, participant_id, mode) -> dict:
        now = _now()
        expires = now + RUN_WINDOW_SECONDS
        with self._conn() as c:
            prow = c.execute("SELECT study_id FROM participant WHERE participant_id=?",
                             (participant_id,)).fetchone()
            study_id = prow["study_id"]
            latest = c.execute("SELECT * FROM run WHERE participant_id=? ORDER BY attempt DESC LIMIT 1",
                               (participant_id,)).fetchone()
            resumable = latest and mode == "resume" and latest["status"] in ("in_progress", "expired")
            if resumable:
                c.execute("UPDATE run SET status='in_progress', started_at=?, expires_at=?, submitted_at=NULL "
                          "WHERE id=?", (now, expires, latest["id"]))
                rid = latest["id"]
            else:
                attempt = (latest["attempt"] + 1) if latest else 1
                cur = c.execute("INSERT INTO run(participant_id, study_id, attempt, status, current_step_json, "
                                "completed_json, started_at, expires_at) VALUES(?,?,?,?,?,?,?,?)",
                                (participant_id, study_id, attempt, "in_progress", "{}", "{}", now, expires))
                rid = cur.lastrowid
            row = c.execute("SELECT * FROM run WHERE id=?", (rid,)).fetchone()
        return self._run_dict(row)

    def update_run_progress(self, run_id, current_step, completed) -> None:
        with self._conn() as c:
            c.execute("UPDATE run SET current_step_json=?, completed_json=? WHERE id=?",
                      (json.dumps(current_step), json.dumps(completed), run_id))

    def submit_run(self, run_id) -> None:
        with self._conn() as c:
            c.execute("UPDATE run SET status='submitted', submitted_at=? WHERE id=?", (_now(), run_id))

    def get_live_run(self, study_id) -> Optional[dict]:
        # A live run holds the single-GPU lock regardless of which study it belongs to.
        with self._conn() as c:
            row = c.execute("SELECT * FROM run WHERE status='in_progress' AND expires_at > ? "
                            "ORDER BY started_at DESC LIMIT 1", (_now(),)).fetchone()
        return self._run_dict(row) if row else None

    def list_runs(self, study_id) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM run WHERE study_id=? ORDER BY started_at DESC", (study_id,)).fetchall()
        return [self._run_dict(r) for r in rows]

    def _run_dict(self, row) -> dict:
        d = _loads(row, ("current_step_json", "completed_json"))
        if d["status"] == "in_progress" and d["expires_at"] is not None and d["expires_at"] < _now():
            d["status"] = "expired"
        d["remaining_seconds"] = max(0, int(d["expires_at"] - _now())) if d.get("expires_at") else 0
        return d

    # ---------- sessions ----------
    def next_session_attempt(self, participant_id, run_id, scenario_order) -> int:
        with self._conn() as c:
            row = c.execute("SELECT MAX(scenario_attempt) AS n FROM session WHERE participant_id=? "
                            "AND run_id=? AND scenario_order=?",
                            (participant_id, run_id, scenario_order)).fetchone()
        return int(row["n"] or 0) + 1

    def create_session(self, session_id, participant_id, scenario_id, scenario_order,
                       voice_condition, target_speaker_id, run_id, run_attempt,
                       scenario_attempt, schedule, config_snapshot) -> dict:
        with self._conn() as c:
            prow = c.execute("SELECT study_id FROM participant WHERE participant_id=?",
                             (participant_id,)).fetchone()
            study_id = prow["study_id"]
            c.execute("INSERT INTO session(session_id, participant_id, study_id, scenario_id, "
                      "scenario_order, voice_condition, target_speaker_id, run_id, run_attempt, "
                      "scenario_attempt, schedule_json, config_snapshot_json, started_at) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (session_id, participant_id, study_id, scenario_id, scenario_order,
                       voice_condition, target_speaker_id, run_id, run_attempt,
                       scenario_attempt, json.dumps(schedule), json.dumps(config_snapshot), _now()))
            row = c.execute("SELECT * FROM session WHERE session_id=?", (session_id,)).fetchone()
        return dict(row)

    def save_session(self, session_id, files, transcript, metrics, audiobox_available) -> None:
        with self._conn() as c:
            c.execute("UPDATE session SET files_json=?, transcript_json=?, metrics_json=?, "
                      "audiobox_available=? WHERE session_id=?",
                      (json.dumps(files), json.dumps(transcript), json.dumps(metrics),
                       1 if audiobox_available else 0, session_id))

    def update_session_analysis(self, session_id, transcript, metrics, audiobox_available) -> None:
        with self._conn() as c:
            c.execute("UPDATE session SET transcript_json=?, metrics_json=?, audiobox_available=? "
                      "WHERE session_id=?",
                      (json.dumps(transcript), json.dumps(metrics),
                       1 if audiobox_available else 0, session_id))

    def update_session_artifacts(self, session_id, manifest) -> None:
        with self._conn() as c:
            c.execute("UPDATE session SET artifact_manifest_json=? WHERE session_id=?",
                      (json.dumps(manifest), session_id))

    def update_session_vc_quality(self, session_id, status, result) -> None:
        with self._conn() as c:
            c.execute("UPDATE session SET vc_quality_status=?, vc_quality_json=? WHERE session_id=?",
                      (status, json.dumps(result), session_id))

    def end_session(self, session_id, end_reason) -> None:
        with self._conn() as c:
            c.execute("UPDATE session SET ended_at=?, end_reason=? WHERE session_id=?",
                      (_now(), end_reason, session_id))

    def get_session(self, session_id) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM session WHERE session_id=?", (session_id,)).fetchone()
        return _loads(row, ("files_json", "transcript_json", "metrics_json", "schedule_json",
                            "config_snapshot_json", "artifact_manifest_json", "vc_quality_json")) if row else None

    def list_sessions(self, study_id) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM session WHERE study_id=? ORDER BY started_at", (study_id,)).fetchall()
        return [_loads(r, ("files_json", "transcript_json", "metrics_json", "schedule_json",
                           "config_snapshot_json", "artifact_manifest_json", "vc_quality_json")) for r in rows]

    def get_latest_session(self, participant_id, scenario_order, run_id=None) -> Optional[dict]:
        query = "SELECT * FROM session WHERE participant_id=? AND scenario_order=?"
        args: list[Any] = [participant_id, scenario_order]
        if run_id is not None:
            query += " AND run_id=?"
            args.append(run_id)
        query += " ORDER BY run_attempt DESC, scenario_attempt DESC LIMIT 1"
        with self._conn() as c:
            row = c.execute(query, args).fetchone()
        return _loads(row, ("files_json", "transcript_json", "metrics_json", "schedule_json",
                            "config_snapshot_json", "artifact_manifest_json", "vc_quality_json")) if row else None

    # ---------- answers ----------
    def save_answer(self, participant_id, session_id, kind, payload) -> None:
        with self._conn() as c:
            prow = c.execute("SELECT study_id FROM participant WHERE participant_id=?",
                             (participant_id,)).fetchone()
            study_id = prow["study_id"] if prow else 0
            run = c.execute("SELECT id FROM run WHERE participant_id=? ORDER BY attempt DESC LIMIT 1",
                            (participant_id,)).fetchone()
            c.execute("INSERT INTO answer(participant_id, study_id, run_id, session_id, kind, payload_json, created_at) "
                      "VALUES(?,?,?,?,?,?,?)",
                      (participant_id, study_id, run["id"] if run else None,
                       session_id, kind, json.dumps(payload), _now()))

    def list_answers(self, study_id) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM answer WHERE study_id=? ORDER BY created_at", (study_id,)).fetchall()
        return [_loads(r, ("payload_json",)) for r in rows]

    def has_answer(self, participant_id, run_id, kind) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM answer WHERE participant_id=? AND run_id=? AND kind=? LIMIT 1",
                (participant_id, run_id, kind)).fetchone()
        return row is not None


def _gen_code() -> str:
    alphabet = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(6))


_backend: Optional[StorageBackend] = None


def get_backend() -> StorageBackend:
    global _backend
    if _backend is None:
        kind = os.environ.get("STUDY_STORAGE", "sqlite").lower()
        if kind == "sqlite":
            # DB lives under STUDY_DATA_ROOT (a mountable volume) unless overridden.
            root = os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data"))
            db_path = os.path.expanduser(os.environ.get("STUDY_DB_PATH", str(Path(root) / "study.db")))
            _backend = SqliteBackend(db_path)
        else:
            raise RuntimeError(f"Unknown STUDY_STORAGE backend: {kind}")
    return _backend
