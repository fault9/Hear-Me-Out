"""Notebook coding form for the human review sheets.

The form reads and writes the same sheet files the CLI produces and validates
with the frozen validators, so it adds an interface and not a second
definition of the contract. ipywidgets is imported inside `coder` for the
same reason the runner constructs its API client lazily: the rest of the
pipeline runs where it is not installed.

    from study.coding.review_form import coder
    coder(study_id=1, initials="FO")
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .packets import coding_root
from .schema import (ACCESS_FLAG_TYPES, FINAL_ACCOUNT_VALUES,
                     GROUNDING_STAGES, REPAIR_CATEGORIES, consistency_issues,
                     validate_labels)

UNIT_LABELS = ("attempted", "complete_raw") + GROUNDING_STAGES
# attempted is the one unit label the schema will not accept as null.
BINARY_ONLY = ("attempted",)


def _data_root() -> Path:
    return Path(os.path.expanduser(
        os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))


def _utterances(packet: dict, speaker: str | None = None) -> list[tuple[str, str]]:
    return [(f"{row.get('id')}  {str(row.get('text') or '')[:70]}", row.get("id"))
            for row in packet.get("utterances") or []
            if speaker is None or row.get("speaker") == speaker]


def _scenario_html(scenario: dict) -> str:
    units = "".join(f"<li>{u}</li>" for u in scenario.get("unit_definitions") or [])
    levels = "".join(
        f"<li><b>{row.get('score')}</b> — {row.get('criteria') or ''}</li>"
        for row in scenario.get("outcome_levels") or [] if isinstance(row, dict))
    return (f"<h4>{scenario.get('title') or ''}</h4>"
            f"<b>Critical units</b><ol>{units}</ol>"
            f"<b>Bounded action</b>: {scenario.get('bounded_action') or ''}<br>"
            f"<b>Required final account</b>: "
            f"{scenario.get('required_final_account') or ''}"
            f"<b>Outcome levels</b><ul>{levels}</ul>")


def _transcript_html(packet: dict) -> str:
    rows = []
    for row in packet.get("utterances") or []:
        flag = ("&nbsp;<i>[asr incomplete]</i>" if row.get("asr_ok") is False
                else "&nbsp;<i>[low confidence]</i>"
                if row.get("asr_low_confidence") else "")
        weight = "bold" if row.get("speaker") == "participant" else "normal"
        rows.append(f"<div style='margin-bottom:4px'><code>{row.get('id')}</code> "
                    f"<span style='font-weight:{weight}'>{row.get('text')}</span>"
                    f"{flag}</div>")
    return ("<div style='max-height:420px;overflow:auto;padding:6px;"
            "border:1px solid #ccc'>" + "".join(rows) + "</div>")


def coder(study_id: int = 1, initials: str = "", data_root: Path | None = None) -> None:
    """Render the coding form for every sheet still awaiting a coder."""
    import ipywidgets as widgets
    from IPython.display import display

    root = coding_root(data_root or _data_root(), study_id)
    sheets_dir = root / "review" / "sheets"

    def _sheet_paths() -> list[Path]:
        return sorted(sheets_dir.glob("*.json"))

    def _label(path: Path) -> str:
        sheet = json.loads(path.read_text())
        done = "done" if sheet.get("coder") else "todo"
        return f"{path.stem[:8]}  [{sheet.get('mode')}]  {done}"

    picker = widgets.Dropdown(description="Packet",
                              layout=widgets.Layout(width="440px"))
    coder_box = widgets.Text(value=initials, description="Coder",
                             layout=widgets.Layout(width="220px"))
    scenario = widgets.HTML()
    transcript = widgets.HTML()
    status = widgets.Output()
    save = widgets.Button(description="Save sheet", button_style="primary")

    # Rebuilt for each packet, since the unit count and the utterance options
    # are packet-specific.
    unit_boxes: list[dict] = []
    repair_rows: list[dict] = []
    flag_rows: list[dict] = []
    units_area = widgets.VBox()
    repairs_area = widgets.VBox()
    flags_area = widgets.VBox()
    add_repair = widgets.Button(description="Add repair", icon="plus")
    add_flag = widgets.Button(description="Add access flag", icon="plus")

    probe = widgets.Dropdown(description="Final probe",
                             layout=widgets.Layout(width="620px"))
    spontaneous = widgets.Dropdown(description="Spontaneous",
                                   layout=widgets.Layout(width="620px"))
    level = widgets.RadioButtons(description="Outcome")
    level_evidence = widgets.SelectMultiple(description="Evidence", rows=4,
                                            layout=widgets.Layout(width="620px"))
    rationale = widgets.Textarea(description="Rationale",
                                 layout=widgets.Layout(width="620px", height="60px"))
    level_conf = widgets.BoundedFloatText(min=0, max=1, step=0.01, value=0.9,
                                          description="Confidence")
    account = widgets.RadioButtons(options=list(FINAL_ACCOUNT_VALUES),
                                   description="Final account")
    account_evidence = widgets.SelectMultiple(description="Evidence", rows=4,
                                              layout=widgets.Layout(width="620px"))
    account_conf = widgets.BoundedFloatText(min=0, max=1, step=0.01, value=0.9,
                                            description="Confidence")
    notes = widgets.Textarea(description="Notes",
                             layout=widgets.Layout(width="620px", height="50px"))

    def _tri(name: str, value):
        options = [("0", 0), ("1", 1)]
        if name not in BINARY_ONLY:
            options.append(("null", None))
        if value not in [v for _, v in options]:
            value = options[0][1]
        return widgets.ToggleButtons(options=options, value=value,
                                     layout=widgets.Layout(width="230px"))

    def _unit_box(unit: dict, packet: dict):
        rows, fields = [], {}
        for name in UNIT_LABELS:
            toggle = _tri(name, unit.get(name))
            conf = widgets.BoundedFloatText(
                min=0, max=1, step=0.01, description="conf",
                value=float((unit.get("confidence") or {}).get(name) or 0.9),
                layout=widgets.Layout(width="150px"))
            evidence = widgets.SelectMultiple(
                options=_utterances(packet), rows=3,
                value=tuple((unit.get("evidence_utterance_ids") or {}).get(name) or ()),
                layout=widgets.Layout(width="520px"))
            fields[name] = {"value": toggle, "confidence": conf,
                            "evidence": evidence}
            rows.append(widgets.HBox([
                widgets.HTML(f"<code style='width:130px;display:inline-block'>"
                             f"{name}</code>"), toggle, conf, evidence]))
        delivery = widgets.SelectMultiple(
            options=_utterances(packet, "participant"), rows=4,
            value=tuple(unit.get("delivery_utterance_ids") or ()),
            layout=widgets.Layout(width="620px"))
        rows.insert(0, widgets.HBox([
            widgets.HTML("<code style='width:130px;display:inline-block'>"
                         "delivery ids</code>"), delivery]))
        fields["delivery"] = delivery
        fields["unit_index"] = unit.get("unit_index")
        return fields, widgets.VBox(rows)

    def _repair_row(packet: dict, move: dict | None = None):
        move = move or {}
        row = {
            "utterance_id": widgets.Dropdown(
                options=_utterances(packet, "participant"),
                value=move.get("utterance_id") or None,
                layout=widgets.Layout(width="420px")),
            "category": widgets.Dropdown(
                options=list(REPAIR_CATEGORIES),
                value=move.get("category") or REPAIR_CATEGORIES[0],
                layout=widgets.Layout(width="200px")),
            "trouble": widgets.Text(value=move.get("trouble") or "",
                                    placeholder="the trouble this move addresses",
                                    layout=widgets.Layout(width="320px")),
            "confidence": widgets.BoundedFloatText(
                min=0, max=1, step=0.01, value=float(move.get("confidence") or 0.9),
                layout=widgets.Layout(width="110px")),
        }
        drop = widgets.Button(description="", icon="trash",
                              layout=widgets.Layout(width="45px"))
        box = widgets.HBox([row["utterance_id"], row["category"],
                            row["trouble"], row["confidence"], drop])
        row["box"] = box

        def _remove(_):
            repair_rows.remove(row)
            repairs_area.children = tuple(r["box"] for r in repair_rows)

        drop.on_click(_remove)
        return row

    def _flag_row(packet: dict, flag: dict | None = None):
        flag = flag or {}
        row = {
            "type": widgets.Dropdown(options=list(ACCESS_FLAG_TYPES),
                                     value=flag.get("type") or ACCESS_FLAG_TYPES[0],
                                     layout=widgets.Layout(width="250px")),
            "utterance_id": widgets.Dropdown(
                options=_utterances(packet), value=flag.get("utterance_id") or None,
                layout=widgets.Layout(width="420px")),
            "confidence": widgets.BoundedFloatText(
                min=0, max=1, step=0.01, value=float(flag.get("confidence") or 0.9),
                layout=widgets.Layout(width="110px")),
        }
        drop = widgets.Button(description="", icon="trash",
                              layout=widgets.Layout(width="45px"))
        row["box"] = widgets.HBox([row["type"], row["utterance_id"],
                                   row["confidence"], drop])

        def _remove(_):
            flag_rows.remove(row)
            flags_area.children = tuple(r["box"] for r in flag_rows)

        drop.on_click(_remove)
        return row

    def _load(_=None):
        path = picker.value
        if path is None:
            return
        sheet = json.loads(Path(path).read_text())
        packet = sheet.get("packet") or {}
        labels = sheet.get("labels") or {}
        scenario.value = _scenario_html(packet.get("scenario") or {})
        transcript.value = _transcript_html(packet)
        if sheet.get("coder"):
            coder_box.value = sheet["coder"]

        unit_boxes.clear()
        boxes = []
        for unit in labels.get("units") or []:
            fields, box = _unit_box(unit, packet)
            unit_boxes.append(fields)
            boxes.append(widgets.VBox(
                [widgets.HTML(f"<b>Unit {fields['unit_index']}</b>"), box]))
        units_area.children = tuple(boxes)

        repair_rows.clear()
        repair_rows.extend(_repair_row(packet, m) for m in labels.get("repairs") or [])
        repairs_area.children = tuple(r["box"] for r in repair_rows)
        flag_rows.clear()
        flag_rows.extend(_flag_row(packet, f) for f in labels.get("access_flags") or [])
        flags_area.children = tuple(r["box"] for r in flag_rows)

        none_option = [("(none)", None)]
        probe.options = none_option + _utterances(packet, "participant")
        probe.value = (labels.get("final_probe") or {}).get("utterance_id")
        spontaneous.options = none_option + _utterances(packet, "assistant")
        spontaneous.value = (labels.get("final_probe") or {}).get(
            "spontaneous_final_account_utterance_id")

        outcome = labels.get("outcome") or {}
        scores = [row.get("score") for row
                  in (packet.get("scenario") or {}).get("outcome_levels") or []
                  if isinstance(row, dict)]
        level.options = [(str(s), s) for s in scores]
        level.value = outcome.get("level") if outcome.get("level") in scores else None
        level_evidence.options = _utterances(packet)
        level_evidence.value = tuple(outcome.get("evidence_utterance_ids") or ())
        rationale.value = outcome.get("rationale") or ""
        level_conf.value = float(outcome.get("confidence") or 0.9)

        acc = labels.get("final_account_accuracy") or {}
        account.value = acc.get("value") or FINAL_ACCOUNT_VALUES[0]
        account_evidence.options = _utterances(packet)
        account_evidence.value = tuple(acc.get("evidence_utterance_ids") or ())
        account_conf.value = float(acc.get("confidence") or 0.9)
        notes.value = labels.get("notes") or ""
        with status:
            status.clear_output()

    def _collect() -> dict:
        units = []
        for fields in unit_boxes:
            unit = {"unit_index": fields["unit_index"],
                    "delivery_utterance_ids": list(fields["delivery"].value)}
            evidence, confidence = {}, {}
            for name in UNIT_LABELS:
                value = fields[name]["value"].value
                unit[name] = value
                if value is not None:
                    evidence[name] = list(fields[name]["evidence"].value)
                    confidence[name] = fields[name]["confidence"].value
            unit["evidence_utterance_ids"] = evidence
            unit["confidence"] = confidence
            units.append(unit)
        return {
            "units": units,
            "repairs": [{"utterance_id": r["utterance_id"].value,
                         "category": r["category"].value,
                         "trouble": r["trouble"].value,
                         "confidence": r["confidence"].value}
                        for r in repair_rows],
            "final_probe": {
                "utterance_id": probe.value,
                "spontaneous_final_account_utterance_id": spontaneous.value},
            "outcome": {"level": level.value,
                        "evidence_utterance_ids": list(level_evidence.value),
                        "rationale": rationale.value,
                        "confidence": level_conf.value},
            "final_account_accuracy": {
                "value": account.value,
                "evidence_utterance_ids": list(account_evidence.value),
                "confidence": account_conf.value},
            "access_flags": [{"type": r["type"].value,
                              "utterance_id": r["utterance_id"].value,
                              "confidence": r["confidence"].value}
                             for r in flag_rows],
            "notes": notes.value,
        }

    def _save(_):
        path = Path(picker.value)
        sheet = json.loads(path.read_text())
        labels = _collect()
        errors = validate_labels(labels, sheet.get("packet") or {})
        sheet["labels"] = labels
        sheet["coder"] = coder_box.value
        path.write_text(json.dumps(sheet, indent=2, sort_keys=True))
        with status:
            status.clear_output()
            print(f"saved {path.name}")
            for line in errors:
                print("  schema:", line)
            for line in consistency_issues(labels, sheet.get("packet") or {}):
                print("  consistency:", line)
            if not errors:
                print("  schema valid")
        picker.options = [(_label(p), str(p)) for p in _sheet_paths()]
        picker.value = str(path)

    picker.options = [(_label(p), str(p)) for p in _sheet_paths()]
    picker.observe(_load, names="value")
    add_repair.on_click(lambda _: (
        repair_rows.append(_repair_row(json.loads(Path(picker.value).read_text())
                                       .get("packet") or {})),
        repairs_area.__setattr__("children", tuple(r["box"] for r in repair_rows))))
    add_flag.on_click(lambda _: (
        flag_rows.append(_flag_row(json.loads(Path(picker.value).read_text())
                                   .get("packet") or {})),
        flags_area.__setattr__("children", tuple(r["box"] for r in flag_rows))))
    save.on_click(_save)

    display(widgets.VBox([
        widgets.HBox([picker, coder_box]),
        scenario, transcript,
        widgets.HTML("<h4>Units</h4>"), units_area,
        widgets.HTML("<h4>Repairs</h4>"), repairs_area, add_repair,
        widgets.HTML("<h4>Final account</h4>"),
        probe, spontaneous, level, level_evidence, rationale, level_conf,
        account, account_evidence, account_conf,
        widgets.HTML("<h4>Access flags</h4>"), flags_area, add_flag,
        notes, save, status,
    ]))
    _load()
