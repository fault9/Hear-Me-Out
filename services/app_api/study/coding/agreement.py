"""Judge-vs-human agreement statistics (method §LLM-Assisted Coding).

Raw agreement and Cohen's kappa for categorical labels, linear-weighted kappa
for the ordinal outcome, and ICC(2,1) (two-way random effects, absolute
agreement, single rater) for repair counts. Pure Python so it runs anywhere
the data root does; formulas are standard and unit-tested against known
examples.
"""

from __future__ import annotations

import json
from pathlib import Path

from .packets import read_index, repair_is_post_boundary

BINARY_UNIT_FIELDS = ("attempted", "complete_raw", "acknowledgement",
                      "update_claim", "incorporation", "retention")
RELIABILITY_THRESHOLDS = {
    "minimum_raw_agreement": 0.80,
    "minimum_kappa": 0.60,
    "minimum_raw_agreement_when_kappa_undefined": 0.90,
    "minimum_weighted_kappa": 0.60,
    "minimum_icc_2_1": 0.75,
}


def raw_agreement(pairs: list[tuple]) -> float | None:
    if not pairs:
        return None
    return sum(1 for a, b in pairs if a == b) / len(pairs)


def cohen_kappa(pairs: list[tuple]) -> float | None:
    if not pairs:
        return None
    categories = sorted({v for pair in pairs for v in pair}, key=repr)
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = 0.0
    for category in categories:
        pa = sum(1 for a, _ in pairs if a == category) / n
        pb = sum(1 for _, b in pairs if b == category) / n
        pe += pa * pb
    if pe >= 1.0:
        return None  # no chance-corrected information (single category)
    return (po - pe) / (1 - pe)


def weighted_kappa(pairs: list[tuple], categories: list) -> float | None:
    """Linear disagreement weights over an ordered category list."""
    if not pairs or len(categories) < 2:
        return None
    index = {c: i for i, c in enumerate(categories)}
    if any(a not in index or b not in index for a, b in pairs):
        return None
    n = len(pairs)
    k = len(categories) - 1
    observed = 0.0
    for a, b in pairs:
        observed += abs(index[a] - index[b]) / k
    observed /= n
    marginal_a = [sum(1 for a, _ in pairs if a == c) / n for c in categories]
    marginal_b = [sum(1 for _, b in pairs if b == c) / n for c in categories]
    expected = 0.0
    for i, pa in enumerate(marginal_a):
        for j, pb in enumerate(marginal_b):
            expected += pa * pb * abs(i - j) / k
    if expected == 0:
        return None
    return 1 - observed / expected


def icc_2_1(pairs: list[tuple]) -> float | None:
    """ICC(2,1): two-way random effects, absolute agreement, single rater."""
    n = len(pairs)
    if n < 2:
        return None
    k = 2
    grand = sum(a + b for a, b in pairs) / (n * k)
    subject_means = [(a + b) / k for a, b in pairs]
    rater_means = [sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n]
    ss_rows = k * sum((m - grand) ** 2 for m in subject_means)
    ss_cols = n * sum((m - grand) ** 2 for m in rater_means)
    ss_total = sum((v - grand) ** 2 for pair in pairs for v in pair)
    ss_error = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = max(0.0, ss_error) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + k * (msc - mse) / n
    if denominator == 0:
        return None
    return (msr - mse) / denominator


def _read_meta(root: Path, packet_id: str) -> dict | None:
    try:
        return json.loads((root / "meta" / f"{packet_id}.json").read_text())
    except (OSError, ValueError):
        return None


def _post_repairs(labels: dict, meta: dict) -> int:
    return sum(1 for move in labels.get("repairs") or []
               if repair_is_post_boundary(move.get("utterance_id"), meta))


def _load_labels(root: Path, role: str, packet_id: str) -> dict | None:
    try:
        record = json.loads((root / "labels" / role / f"{packet_id}.json").read_text())
    except (OSError, ValueError):
        return None
    # Adjudicated human labels start from the judge's own output, so counting
    # them would measure agreement of a label set with its own source.
    if role == "human" and (record.get("provenance") or {}).get(
            "mode", "independent") != "independent":
        return None
    return record.get("labels")


def _categorical_reliability(metrics: dict) -> dict:
    raw = metrics.get("raw_agreement")
    kappa = metrics.get("kappa")
    reasons = []
    if not metrics.get("n"):
        reasons.append("no_double_coded_observations")
    elif raw is None or raw < RELIABILITY_THRESHOLDS["minimum_raw_agreement"]:
        reasons.append("raw_agreement_below_threshold")
    if metrics.get("n"):
        if kappa is None:
            if (raw is None or raw < RELIABILITY_THRESHOLDS[
                    "minimum_raw_agreement_when_kappa_undefined"]):
                reasons.append("kappa_undefined_without_high_raw_agreement")
        elif kappa < RELIABILITY_THRESHOLDS["minimum_kappa"]:
            reasons.append("kappa_below_threshold")
    return {**metrics, "reliable": not reasons, "reasons": reasons}


def assess_reliability(report: dict) -> dict:
    """Apply the frozen expansion thresholds before adjudication."""
    fields: dict[str, dict] = {}
    for field, metrics in (report.get("unit_labels") or {}).items():
        fields[f"units.{field}"] = _categorical_reliability(metrics)
    fields["final_account_accuracy"] = _categorical_reliability(
        report.get("final_account_accuracy") or {})
    fields["any_repair"] = _categorical_reliability(
        report.get("any_repair") or {})

    outcome = dict(report.get("outcome_level") or {})
    outcome_reasons = []
    if not outcome.get("n"):
        outcome_reasons.append("no_double_coded_observations")
    elif (outcome.get("weighted_kappa_linear") is None
          or outcome["weighted_kappa_linear"] <
          RELIABILITY_THRESHOLDS["minimum_weighted_kappa"]):
        outcome_reasons.append("weighted_kappa_below_threshold")
    fields["outcome_level"] = {
        **outcome, "reliable": not outcome_reasons,
        "reasons": outcome_reasons,
    }

    # The co-primary is the post-transition count, so it is gated on the same
    # threshold as the total rather than inheriting the total's verdict.
    for name in ("repair_total", "repair_post_transition"):
        repair = dict(report.get(name) or {})
        repair_reasons = []
        if not repair.get("n"):
            repair_reasons.append("no_double_coded_observations")
        elif (repair.get("icc_2_1") is None
              or repair["icc_2_1"] < RELIABILITY_THRESHOLDS["minimum_icc_2_1"]):
            repair_reasons.append("icc_below_threshold")
        fields[name] = {
            **repair, "reliable": not repair_reasons, "reasons": repair_reasons,
        }
    failed = sorted(field for field, value in fields.items()
                    if not value["reliable"])
    return {
        "thresholds": RELIABILITY_THRESHOLDS,
        "fields": fields,
        "failed_fields": failed,
        "requires_full_human_review": bool(failed),
    }


def agreement_report(root: Path) -> dict:
    """Compare judge and human labels on every packet that has both.
    Reported BEFORE adjudication (final labels are not consulted)."""
    unit_pairs: dict[str, list[tuple]] = {f: [] for f in BINARY_UNIT_FIELDS}
    outcome_pairs: list[tuple] = []
    account_pairs: list[tuple] = []
    repair_count_pairs: list[tuple] = []
    post_repair_pairs: list[tuple] = []
    any_repair_pairs: list[tuple] = []
    compared = []
    for row in read_index(root):
        pid = row["packet_id"]
        judge = _load_labels(root, "judge", pid)
        human = _load_labels(root, "human", pid)
        if judge is None or human is None:
            continue
        compared.append(pid)
        meta = _read_meta(root, pid)
        for j_unit, h_unit in zip(judge.get("units") or [], human.get("units") or []):
            for field in BINARY_UNIT_FIELDS:
                a, b = j_unit.get(field), h_unit.get(field)
                if a is not None and b is not None:
                    unit_pairs[field].append((a, b))
        j_level = (judge.get("outcome") or {}).get("level")
        h_level = (human.get("outcome") or {}).get("level")
        if j_level is not None and h_level is not None:
            outcome_pairs.append((j_level, h_level))
        j_account = (judge.get("final_account_accuracy") or {}).get("value")
        h_account = (human.get("final_account_accuracy") or {}).get("value")
        if j_account and h_account:
            account_pairs.append((j_account, h_account))
        j_repairs = len(judge.get("repairs") or [])
        h_repairs = len(human.get("repairs") or [])
        repair_count_pairs.append((j_repairs, h_repairs))
        any_repair_pairs.append((int(j_repairs > 0), int(h_repairs > 0)))
        # Two coders can record the same number of moves and anchor them to
        # different utterances, so the total's reliability does not carry to
        # the co-primary. Score the measure actually modelled.
        if meta is not None:
            post_repair_pairs.append((_post_repairs(judge, meta),
                                      _post_repairs(human, meta)))

    report = {
        "packets_compared": len(compared),
        "packet_ids": sorted(compared),
        "unit_labels": {
            field: {"n": len(pairs),
                    "raw_agreement": raw_agreement(pairs),
                    "kappa": cohen_kappa(pairs)}
            for field, pairs in unit_pairs.items()
        },
        "outcome_level": {
            "n": len(outcome_pairs),
            "raw_agreement": raw_agreement(outcome_pairs),
            "weighted_kappa_linear": weighted_kappa(outcome_pairs, [1, 2, 3, 4]),
        },
        "final_account_accuracy": {
            "n": len(account_pairs),
            "raw_agreement": raw_agreement(account_pairs),
            "kappa": cohen_kappa(account_pairs),
        },
        "repair_total": {
            "n": len(repair_count_pairs),
            "icc_2_1": icc_2_1(repair_count_pairs),
        },
        "repair_post_transition": {
            "n": len(post_repair_pairs),
            "icc_2_1": icc_2_1(post_repair_pairs),
        },
        "any_repair": {
            "n": len(any_repair_pairs),
            "raw_agreement": raw_agreement(any_repair_pairs),
            "kappa": cohen_kappa(any_repair_pairs),
        },
    }
    report["reliability"] = assess_reliability(report)
    out_dir = root / "agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
