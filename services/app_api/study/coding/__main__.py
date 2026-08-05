from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from study.coding import agreement, freeze, packets, review, runner  # noqa: E402


def _data_root() -> Path:
    return Path(os.path.expanduser(os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))


def _root(args) -> Path:
    return packets.coding_root(_data_root(), args.study_id)


def _print(value) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m study.coding")
    parser.add_argument("--study-id", type=int, default=int(os.environ.get("CODING_STUDY_ID", "1")))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("packets")
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--note", default="")
    p_freeze.add_argument("--model", default=runner.DEFAULT_MODEL)
    sub.add_parser("status")
    p_judge = sub.add_parser("judge")
    p_judge.add_argument("--pilot", action="store_true")
    p_judge.add_argument("--model", default=runner.DEFAULT_MODEL)
    p_judge.add_argument("--packet", action="append", default=[],
                         help="restrict to specific packet id(s)")
    p_judge.add_argument("--redo", action="store_true",
                         help="re-judge packets that already have labels")
    p_review = sub.add_parser("review")
    p_review.add_argument("--seed", type=int, required=True)
    p_review.add_argument("--fraction", type=float, default=review.SAMPLE_FRACTION)
    sub.add_parser("import-human")
    sub.add_parser("finalize")
    sub.add_parser("agreement")
    sub.add_parser("reliability-expand")

    args = parser.parse_args()
    root = _root(args)

    if args.command == "packets":
        from study.session_scope import annotate_analysis_scopes
        from study.storage import get_backend
        backend = get_backend()
        sessions = annotate_analysis_scopes(
            backend.list_sessions(args.study_id), backend.list_runs(args.study_id))
        scenarios = {str(row["id"]): row for row in backend.list_scenarios(args.study_id)}
        _print(packets.write_packets(sessions, _data_root(), args.study_id, scenarios))
    elif args.command == "freeze":
        client = runner.LLMClient(model=args.model)
        _print(freeze.freeze(root, client.decoding(), note=args.note))
    elif args.command == "status":
        _print(freeze.manifest_status(root))
    elif args.command == "judge":
        client = runner.LLMClient(model=args.model)
        _print(runner.run_judging(
            root, client, pilot=args.pilot,
            only_packet_ids=set(args.packet) or None,
            skip_existing=not args.redo))
    elif args.command == "review":
        flags = review.compute_flags(root)
        sample = review.stratified_sample(root, seed=args.seed, fraction=args.fraction)
        exported = review.export_review(root)
        _print({"flags": flags, "sample": sample, "export": exported})
    elif args.command == "import-human":
        _print(review.import_human(root))
    elif args.command == "finalize":
        _print(review.finalize(root))
    elif args.command == "agreement":
        _print(agreement.agreement_report(root))
    elif args.command == "reliability-expand":
        expansion = review.expand_for_reliability(root)
        exported = review.export_review(root)
        _print({"expansion": expansion, "export": exported})


if __name__ == "__main__":
    main()
