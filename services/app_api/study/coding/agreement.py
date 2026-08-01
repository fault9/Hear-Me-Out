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

from .packets import read_index

BINARY_UNIT_FIELDS = ("attempted", "complete_raw", "acknowledgement",
                      "update_claim", "incorporation", "retention")


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


def _load_labels(root: Path, role: str, packet_id: str) -> dict | None:
    try:
        record = json.loads((root / "labels" / role / f"{packet_id}.json").read_text())
    except (OSError, ValueError):
        return None
    return record.get("labels")


def agreement_report(root: Path) -> dict:
    """Compare judge and human labels on every packet that has both.
    Reported BEFORE adjudication (final labels are not consulted)."""
    unit_pairs: dict[str, list[tuple]] = {f: [] for f in BINARY_UNIT_FIELDS}
    outcome_pairs: list[tuple] = []
    account_pairs: list[tuple] = []
    repair_count_pairs: list[tuple] = []
    any_repair_pairs: list[tuple] = []
    compared = []
    for row in read_index(root):
        pid = row["packet_id"]
        judge = _load_labels(root, "judge", pid)
        human = _load_labels(root, "human", pid)
        if judge is None or human is None:
            continue
        compared.append(pid)
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
        "any_repair": {
            "n": len(any_repair_pairs),
            "raw_agreement": raw_agreement(any_repair_pairs),
            "kappa": cohen_kappa(any_repair_pairs),
        },
    }
    out_dir = root / "agreement"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    return report
