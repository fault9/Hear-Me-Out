from __future__ import annotations

import csv
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_prep():
    path = Path(__file__).with_name("prep.py")
    spec = importlib.util.spec_from_file_location("hmo_analysis_prep", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class PrepTests(unittest.TestCase):
    def test_sensitivity_excludes_participant_with_any_invalid_attempt(self):
        prep = _load_prep()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, output = root / "data", root / "output"
            data.mkdir()
            scenario_fields = sorted(set(prep.SCENARIO_KEEP) | {
                "analysis_included", "session_id", "participant_id",
                "technical_status",
            })
            scenarios = [
                {"session_id": "S1", "participant_id": "P1",
                 "analysis_included": "1", "valid_for_condition_analysis": "1",
                 "technical_status": "valid", "demonstrated_grounding": ""},
                {"session_id": "S2", "participant_id": "P2",
                 "analysis_included": "1", "valid_for_condition_analysis": "1",
                 "technical_status": "valid", "demonstrated_grounding": "1"},
                {"session_id": "S2_failed", "participant_id": "P2",
                 "analysis_included": "0", "valid_for_condition_analysis": "0",
                 "technical_status": "invalid", "demonstrated_grounding": ""},
                # Abandoned capture: never evaluated, so it is neither valid nor
                # invalid and must not remove P3 from the sensitivity frame.
                {"session_id": "S3", "participant_id": "P3",
                 "analysis_included": "1", "valid_for_condition_analysis": "1",
                 "technical_status": "valid", "demonstrated_grounding": ""},
                {"session_id": "S3_abandoned", "participant_id": "P3",
                 "analysis_included": "0", "valid_for_condition_analysis": "0",
                 "technical_status": "pending", "demonstrated_grounding": ""},
            ]
            with (data / "scenarios.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=scenario_fields)
                writer.writeheader()
                writer.writerows(scenarios)
            with (data / "units.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=prep.UNIT_KEEP)
                writer.writeheader()
            prep.DATA, prep.OUT = str(data), str(output)

            prep.main()

            with (output / "scenario_level.csv").open() as handle:
                primary = list(csv.DictReader(handle))
            with (output / "scenario_level_sensitivity_complete_technical.csv").open() as handle:
                sensitivity = list(csv.DictReader(handle))
            self.assertEqual([row["participant_id"] for row in primary],
                             ["P1", "P2", "P3"])
            self.assertEqual([row["participant_id"] for row in sensitivity],
                             ["P1", "P3"])
            self.assertEqual(primary[0]["demonstrated_grounding"], "")
            with (output / "sensitivity_complete_technical.json").open() as handle:
                report = json.load(handle)
            self.assertEqual(report["excluded_participant_ids"], ["P2"])
            self.assertEqual(report["unevaluated_session_ids"], ["S3_abandoned"])

    def test_turn_frames_keep_only_certified_synchronization(self):
        prep = _load_prep()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, output = root / "data", root / "output"
            data.mkdir()
            with (data / "scenarios.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(
                    set(prep.SCENARIO_KEEP) | {"analysis_included"}))
                writer.writeheader()
                writer.writerow({"session_id": "S1", "participant_id": "P1",
                                 "analysis_included": "1",
                                 "valid_for_condition_analysis": "1"})
            with (data / "units.csv").open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=prep.UNIT_KEEP).writeheader()
            event_fields = sorted(set(prep.TURN_EVENT_KEEP) | {
                "analysis_included", "valid_for_manual_turn_verification",
                "crosswalk_complete"})
            events = [
                # Certified session: eligible for the turn-taking models.
                {"session_id": "S1", "participant_id": "P1", "episode_id": "e1",
                 "analysis_included": "1",
                 "valid_for_manual_turn_verification": "1",
                 "crosswalk_complete": "1"},
                # Included for content measures, but synchronization never
                # certified (delivery failure) — must not reach a turn model.
                {"session_id": "S2", "participant_id": "P2", "episode_id": "e2",
                 "analysis_included": "1",
                 "valid_for_manual_turn_verification": "0",
                 "crosswalk_complete": "0"},
            ]
            with (data / "turn_events.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=event_fields)
                writer.writeheader()
                writer.writerows(events)
            prep.DATA, prep.OUT = str(data), str(output)

            prep.main()

            with (output / "turn_events_certified.csv").open() as handle:
                certified = list(csv.DictReader(handle))
            self.assertEqual([row["session_id"] for row in certified], ["S1"])
            # turn_gaps.csv absent from this export: the frame is still written.
            self.assertTrue((output / "turn_gaps_certified.csv").exists())


if __name__ == "__main__":
    unittest.main()
