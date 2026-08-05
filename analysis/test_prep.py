from __future__ import annotations

import csv
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
            })
            scenarios = [
                {"session_id": "S1", "participant_id": "P1",
                 "analysis_included": "1", "valid_for_condition_analysis": "1",
                 "demonstrated_grounding": ""},
                {"session_id": "S2", "participant_id": "P2",
                 "analysis_included": "1", "valid_for_condition_analysis": "1",
                 "demonstrated_grounding": "1"},
                {"session_id": "S2_failed", "participant_id": "P2",
                 "analysis_included": "0", "valid_for_condition_analysis": "0",
                 "demonstrated_grounding": ""},
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
            self.assertEqual([row["participant_id"] for row in primary], ["P1", "P2"])
            self.assertEqual([row["participant_id"] for row in sensitivity], ["P1"])
            self.assertEqual(primary[0]["demonstrated_grounding"], "")


if __name__ == "__main__":
    unittest.main()
