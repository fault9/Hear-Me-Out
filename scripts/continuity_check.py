"""Verify a session's transmitted-audio continuity from surviving artifacts.

Thin CLI over services/app_api/study/continuity.py (the analysis worker runs
the same check automatically when a session's proxy delivery failed).

Usage: python3 scripts/continuity_check.py <session_dir>
       python3 scripts/continuity_check.py --events <events.jsonl>   (diary only)
"""

import importlib.util
import os
import sys

_MODULE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "services", "app_api", "study", "continuity.py")
_spec = importlib.util.spec_from_file_location("continuity", _MODULE_PATH)
continuity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(continuity)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args[0] == "--events":
        failures, report = continuity.check(continuity.load_events(args[1]), None)
        result = {"verdict": "pass_diary_only" if not failures else "fail",
                  "failures": failures, **report}
    else:
        result = continuity.run_for_session_dir(args[0])
    for key, value in result.items():
        if key != "failures":
            print(f"{key}: {value}")
    print()
    print(result["verdict"].upper())
    for failure in result["failures"]:
        print(" -", failure)
    sys.exit(1 if result["verdict"] == "fail" else 0)


if __name__ == "__main__":
    main()
