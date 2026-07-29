"""Pydantic models for the study API (v2 — multi-study).

Questionnaire items are typed and built by the admin UI; the renderer and the
builder share this schema. Models allow extra fields so the protocol can evolve
without breaking older stored data.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

FieldType = Literal["text", "textarea", "number", "radio", "select", "checkbox",
                    "switch", "scale", "audio_playback"]


class ShowIf(BaseModel):
    field: str
    in_: list = Field(default_factory=list, alias="in")

    class Config:
        populate_by_name = True
        extra = "allow"


class QuestionnaireItem(BaseModel):
    id: str
    type: FieldType = "text"
    label: str = ""
    required: bool = False
    options: Optional[list[str]] = None       # radio / select / checkbox
    options_source: Optional[str] = None      # "scenarios" auto-fills options with scenario titles
    allow_other: bool = False                 # radio/checkbox: add an "other"/self-describe text box
    other_label: Optional[str] = None
    extra_options: Optional[list[str]] = None  # scale: non-numeric choices (e.g. "Not applicable")
    show_if: Optional[dict] = None            # {field, in:[...]} conditional display
    min: Optional[float] = None               # number / scale
    max: Optional[float] = None
    min_label: Optional[str] = None           # scale endpoints
    max_label: Optional[str] = None
    placeholder: Optional[str] = None

    class Config:
        extra = "allow"


class Questionnaires(BaseModel):
    consent: list[QuestionnaireItem] = Field(default_factory=list)
    background: list[QuestionnaireItem] = Field(default_factory=list)
    post: list[QuestionnaireItem] = Field(default_factory=list)
    final: list[QuestionnaireItem] = Field(default_factory=list)

    class Config:
        extra = "allow"


class ExtraField(BaseModel):
    label: str = ""
    value: str = ""

    class Config:
        extra = "allow"


class ScenarioCard(BaseModel):
    role: str = ""
    current_situation: str = ""
    goal: str = ""
    how_to_interact: str = ""
    suggested_first_line: str = ""
    additional_details: str = ""
    # Researcher-defined additional labeled fields shown on the scenario card.
    extra_fields: list[ExtraField] = Field(default_factory=list)

    class Config:
        extra = "allow"


# Participant-facing card fields that must be filled in (in display order).
REQUIRED_CARD_FIELDS = [
    ("role", "Your role"),
    ("current_situation", "Current situation"),
    ("goal", "Your goal"),
    ("how_to_interact", "How to interact"),
    ("suggested_first_line", "Suggested first line"),
    ("additional_details", "Additional details"),
]


class VoiceSegment(BaseModel):
    mode: Literal["natural", "vc"] = "natural"
    engine: Optional[str] = None              # meanvc | xvc (vc segments)
    target_ref: Optional[str] = None          # references a target row's ref
    start_s: float = 0
    end_s: Optional[float] = None             # null => until the time limit

    class Config:
        extra = "allow"


class Scenario(BaseModel):
    id: Optional[int] = None
    order_idx: int = 0
    title: str = ""
    scenario_card: ScenarioCard = Field(default_factory=ScenarioCard)
    system_prompt: str = ""
    voice_prompt: str = ""
    voice_schedule: list[VoiceSegment] = Field(default_factory=list)
    time_limit_s: int = 300
    # Scenario-specific post-questionnaire items (appended after the shared post items).
    post_items: list[QuestionnaireItem] = Field(default_factory=list)
    # Practice scenario: always runs first, shown to the participant as practice, still
    # recorded but flagged so it doesn't count toward the study. At most one per study.
    is_test: bool = False

    class Config:
        extra = "allow"


class StudySettings(BaseModel):
    welcome_text: str = ""
    estimated_duration: str = ""
    playback: dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"


# ---- request bodies ----
class CreateStudyRequest(BaseModel):
    name: str
    description: str = ""


class UpdateStudyRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[dict[str, Any]] = None


class EnterRequest(BaseModel):
    code: str


class RunStartRequest(BaseModel):
    code: str
    mode: Literal["resume", "restart"] = "resume"


class ProgressRequest(BaseModel):
    code: str
    current_step: dict[str, Any] = Field(default_factory=dict)
    completed: dict[str, Any] = Field(default_factory=dict)


class SessionStartRequest(BaseModel):
    code: str
    scenario_order: int


class QuestionnaireRequest(BaseModel):
    code: str
    kind: Literal["consent", "background", "post", "final"]
    payload: dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None


class SubmitRequest(BaseModel):
    code: str


class GenerateRequest(BaseModel):
    count: int = 1


def default_questionnaires() -> dict:
    """Placeholder questionnaires so a new study is immediately runnable.
    Researchers edit these in the builder."""
    consent = [
        QuestionnaireItem(id="consent_participation", type="switch",
                          label="I consent to participate in this study.", required=True),
        QuestionnaireItem(id="consent_audio", type="switch",
                          label="I consent to my audio being recorded and saved.", required=True),
        QuestionnaireItem(id="consent_logs", type="switch",
                          label="I consent to transcripts and interaction logs being saved.", required=True),
    ]
    background = [
        QuestionnaireItem(id="age", type="number", label="Age", min=0, max=120),
        QuestionnaireItem(id="native_language", type="text", label="Native language"),
        QuestionnaireItem(id="assistant_use", type="radio", label="How often do you use voice assistants?",
                          options=["Never", "Rarely", "Monthly", "Weekly", "Daily"]),
    ]
    post = [
        QuestionnaireItem(id="understanding", type="scale", label="The system understood me.",
                          min=1, max=7, min_label="Strongly disagree", max_label="Strongly agree"),
        QuestionnaireItem(id="turn_taking", type="scale", label="Turn-taking felt natural.",
                          min=1, max=7, min_label="Strongly disagree", max_label="Strongly agree"),
        QuestionnaireItem(id="trust", type="scale", label="I trusted the system.",
                          min=1, max=7, min_label="Strongly disagree", max_label="Strongly agree"),
        QuestionnaireItem(id="comments", type="textarea", label="Any comments about this interaction?"),
    ]
    final = [
        QuestionnaireItem(id="usability", type="scale", label="The system was easy to use.",
                          min=1, max=7, min_label="Strongly disagree", max_label="Strongly agree"),
        QuestionnaireItem(id="voice_differences", type="textarea",
                          label="Did you notice differences between the voices? Describe."),
        QuestionnaireItem(id="overall_comments", type="textarea", label="Overall comments"),
    ]
    return Questionnaires(consent=consent, background=background, post=post, final=final).model_dump()
