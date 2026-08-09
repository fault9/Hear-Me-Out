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
# Stands in for None inside a toggle group. See _tri.
_NULL = "\x00null"
# Hover text on the label, so the codebook rule is one pointer away rather
# than one scroll away. Abbreviated from sections 3 and 4.
LABEL_HELP = {
    "attempted": "Any identifiable attempt to convey the unit, however partial.",
    "complete_raw": "The unit's full content appears in the participant's raw "
                    "speech, in one utterance or across several. Paraphrase "
                    "counts; specific content must be there.",
    "acknowledgement": "The assistant verbally registers the unit: repetition, "
                       "confirmation, explicit acceptance.",
    "update_claim": "The assistant explicitly claims to have recorded, updated, "
                    "corrected or accepted the unit or its record.",
    "incorporation": "Later reasoning or decisions operationally use the unit "
                     "correctly. Contradicted content is not incorporation.",
    "retention": "The unit is correctly present in the final-probe response. "
                 "Null when there is no probe and no spontaneous account.",
    "delivery ids": "The minimal ordered participant utterances that jointly "
                    "complete the unit. Empty when complete_raw is 0.",
}
# attempted is the one unit label the schema will not accept as null.
BINARY_ONLY = ("attempted",)


def _data_root() -> Path:
    return Path(os.path.expanduser(
        os.environ.get("STUDY_DATA_ROOT", "/workspace/data")))


def _utterances(packet: dict, speaker: str | None = None) -> list[tuple[str, str]]:
    return [(f"{row.get('id')}  {str(row.get('text') or '')[:70]}", row.get("id"))
            for row in packet.get("utterances") or []
            if speaker is None or row.get("speaker") == speaker]


def _ids(packet: dict, speaker: str | None = None) -> list[str]:
    """Bare ids for the tag pickers. The transcript sits above the form, so a
    row needs the id rather than a second copy of the text."""
    return [row.get("id") for row in packet.get("utterances") or []
            if speaker is None or row.get("speaker") == speaker]


def _head(title: str, hint: str):
    """A section heading with the codebook rule that decides it. The codebook
    is the definition; this is only enough to stop the coder leaving it."""
    import ipywidgets as widgets

    return widgets.HTML(
        f"<h4 style='margin-bottom:2px'>{title}</h4>"
        f"<div style='color:#666;font-size:90%;margin-bottom:6px'>{hint}</div>")


def _pane(child, width: str):
    """A column that scrolls on its own.

    The height belongs on a block container wrapping the box, never on the box
    itself: a flex column told to fit a height shrinks its rows to reach it,
    and the widgets carrying no width of their own - the radio buttons, the
    tag pickers, the buttons - collapse to a line.
    """
    import ipywidgets as widgets

    return widgets.Box([child], layout=widgets.Layout(
        width=width, height="840px", overflow="auto", display="block"))


def _labels_legend():
    """The label definitions, once above the units rather than on each row.

    Repeated per unit they are longer than the unit; on hover alone they are
    only found by someone who already knows they are there.
    """
    import ipywidgets as widgets

    items = "".join(
        f"<div style='margin-bottom:2px'><code>{name}</code> &mdash; "
        f"{LABEL_HELP[name]}</div>"
        for name in ("delivery ids",) + UNIT_LABELS)
    return widgets.HTML(
        f"<div style='color:#666;font-size:90%;margin-bottom:8px'>{items}"
        f"<div style='margin-top:4px'><i>The four grounding stages are null "
        f"unless complete_raw is 1.</i></div></div>")


def _field_label(name: str):
    """A unit label carrying its codebook rule as hover text."""
    import ipywidgets as widgets

    return widgets.HTML(
        f"<code title='{LABEL_HELP.get(name, '')}' style='width:120px;"
        f"display:inline-block;cursor:help'>{name}</code>")


def _levels_html(rows: list[dict]) -> str:
    """The scenario's levels, listed beside the buttons that choose them.

    Not on the buttons: a radio option gets one row of fixed height, and a
    criterion long enough to wrap prints over the option beneath it.
    """
    items = "".join(
        f"<li><b>{row.get('score')}</b> — {row.get('criteria') or ''}</li>"
        for row in rows)
    return (f"<ul style='color:#444;font-size:90%;margin:0 0 4px 0;"
            f"padding-left:20px'>{items}</ul>")


def _scenario_html(scenario: dict) -> str:
    units = "".join(f"<li>{u}</li>" for u in scenario.get("unit_definitions") or [])
    levels = "".join(
        f"<li><b>{row.get('score')}</b> — {row.get('criteria') or ''}</li>"
        for row in scenario.get("outcome_levels") or [] if isinstance(row, dict))
    return (f"<div style='max-height:250px;overflow:auto;padding:6px;"
            f"border:1px solid #ccc;margin-bottom:6px'>"
            f"<h4 style='margin-top:0'>{scenario.get('title') or ''}</h4>"
            f"<b>Critical units</b><ol>{units}</ol>"
            f"<b>Bounded action</b>: {scenario.get('bounded_action') or ''}<br>"
            f"<b>Required final account</b>: "
            f"{scenario.get('required_final_account') or ''}<br>"
            f"<b>Outcome levels</b><ul>{levels}</ul></div>")


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
    return ("<div style='max-height:560px;overflow:auto;padding:6px;"
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

    probe = widgets.Dropdown(description="Request",
                             layout=widgets.Layout(width="620px"))
    spontaneous = widgets.Dropdown(description="Unprompted",
                                   layout=widgets.Layout(width="620px"))
    level_criteria = widgets.HTML()
    level = widgets.RadioButtons(description="Outcome")
    level_evidence = widgets.TagsInput(allow_duplicates=False,
                                       layout=widgets.Layout(width="620px"))
    rationale = widgets.Textarea(description="Rationale",
                                 layout=widgets.Layout(width="620px", height="60px"))
    level_conf = widgets.BoundedFloatText(min=0, max=1, step=0.01, value=0.9,
                                          description="Confidence")
    account = widgets.RadioButtons(options=list(FINAL_ACCOUNT_VALUES),
                                   description="Final account")
    account_evidence = widgets.TagsInput(allow_duplicates=False,
                                         layout=widgets.Layout(width="620px"))
    account_conf = widgets.BoundedFloatText(min=0, max=1, step=0.01, value=0.9,
                                            description="Confidence")
    notes = widgets.Textarea(description="Notes",
                             layout=widgets.Layout(width="620px", height="50px"))

    def _tri(name: str, value):
        # Null is carried as a string, not as None: ToggleButtons reads None as
        # nothing selected, so an option valued None can never light up and a
        # deliberate null looks exactly like an untouched row.
        options = [("0", 0), ("1", 1)]
        if name not in BINARY_ONLY:
            options.append(("null", _NULL))
        if value is None and name not in BINARY_ONLY:
            value = _NULL
        if value not in [v for _, v in options]:
            value = options[0][1]
        # Wide enough for three buttons whatever this label offers: too narrow
        # and they wrap one per line, so the row reads as a column. Fixed
        # rather than fitted, to keep the columns after it in line.
        return widgets.ToggleButtons(
            options=options, value=value, style={"button_width": "52px"},
            layout=widgets.Layout(width="184px"))

    def _tags(allowed: list[str], value, width: str = "520px"):
        return widgets.TagsInput(
            value=[v for v in (value or []) if v in allowed],
            allowed_tags=allowed, allow_duplicates=False,
            layout=widgets.Layout(width=width))

    def _unit_box(unit: dict, packet: dict):
        rows, fields = [], {}
        all_ids, participant_ids = _ids(packet), _ids(packet, "participant")
        for name in UNIT_LABELS:
            toggle = _tri(name, unit.get(name))
            conf = widgets.BoundedFloatText(
                min=0, max=1, step=0.01,
                value=float((unit.get("confidence") or {}).get(name) or 0.9),
                layout=widgets.Layout(width="80px"))
            # Narrower than the row it shares: label, toggles and confidence
            # already take half the column, and an overflowing row breaks.
            evidence = _tags(
                all_ids, (unit.get("evidence_utterance_ids") or {}).get(name),
                width="330px")
            fields[name] = {"value": toggle, "confidence": conf,
                            "evidence": evidence}
            rows.append(widgets.HBox(
                [_field_label(name), toggle, conf, evidence],
                layout=widgets.Layout(align_items="center")))
        delivery = _tags(participant_ids, unit.get("delivery_utterance_ids"))
        rows.insert(0, widgets.HBox(
            [_field_label("delivery ids"), delivery],
            layout=widgets.Layout(align_items="center")))
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
        # A unit is one of the scenario's critical facts, and its number means
        # nothing without it: numbered alone, the coder reads the brief again
        # for every row.
        definitions = (packet.get("scenario") or {}).get("unit_definitions") or []
        for unit in labels.get("units") or []:
            fields, box = _unit_box(unit, packet)
            unit_boxes.append(fields)
            index = fields["unit_index"]
            defined = (definitions[index - 1]
                       if isinstance(index, int) and 1 <= index <= len(definitions)
                       else "")
            boxes.append(widgets.VBox([
                widgets.HTML(f"<b>Unit {index}</b> &mdash; {defined}"), box]))
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
        rows = [row for row
                in (packet.get("scenario") or {}).get("outcome_levels") or []
                if isinstance(row, dict)]
        scores = [row.get("score") for row in rows]
        # The levels are the scenario's, not a fixed scale, and choosing one
        # off a bare 1-4 means scrolling back to the brief for every packet.
        level_criteria.value = _levels_html(rows)
        level.options = [(str(s), s) for s in scores]
        level.value = outcome.get("level") if outcome.get("level") in scores else None
        level_evidence.allowed_tags = _ids(packet)
        level_evidence.value = [v for v in outcome.get("evidence_utterance_ids") or []
                                if v in level_evidence.allowed_tags]
        rationale.value = outcome.get("rationale") or ""
        level_conf.value = float(outcome.get("confidence") or 0.9)

        acc = labels.get("final_account_accuracy") or {}
        account.value = acc.get("value") or FINAL_ACCOUNT_VALUES[0]
        account_evidence.allowed_tags = _ids(packet)
        account_evidence.value = [v for v in acc.get("evidence_utterance_ids") or []
                                  if v in account_evidence.allowed_tags]
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
                value = None if value == _NULL else value
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

    # The transcript is what every field is read off, so it scrolls in its own
    # column rather than above the form, where the first unit pushed it away.
    fields = widgets.VBox([
        _head("Units", "One row per critical unit. Evidence cites the "
                       "utterance ids that ground each label."),
        _labels_legend(), units_area,
        _head("Repairs", "One move per utterance, in the codebook's priority "
                         "order. First deliveries and plain agreement are not "
                         "repairs."),
        repairs_area, add_repair,
        # Three codebook sections, not one: the probe anchors retention (4),
        # the level is the ordinal outcome (6), and accuracy is scored on the
        # probe response (7). Under one heading a coder cites the readback as
        # evidence for the level.
        _head("Final probe", "The participant turn asking for a summary, or "
                             "an assistant turn that gives the whole account "
                             "unprompted. Retention is scored on the "
                             "assistant's reply; neither turn means retention "
                             "stays null."),
        probe, spontaneous,
        _head("Task outcome", "The scenario's own levels. The rationale "
                              "argues the level against the one above it."),
        level_criteria, level, level_evidence, rationale, level_conf,
        _head("Final-account accuracy", "Whether the probe response states "
                                        "what actually happened. Scored apart "
                                        "from the level, never folded into it."),
        account, account_evidence, account_conf,
        _head("Access flags", "Candidate nominations only: what looks cut "
                              "off, abandoned, or closed early. Timing "
                              "decides these outside the form."),
        flags_area, add_flag,
        notes, save, status,
    ], layout=widgets.Layout(padding="0 14px 0 0"))
    brief = widgets.VBox([scenario, transcript])
    display(widgets.VBox([
        widgets.HBox([picker, coder_box]),
        widgets.HBox([_pane(fields, "56%"), _pane(brief, "44%")]),
    ]))
    _load()
