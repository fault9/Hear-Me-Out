"""Study platform: participant-experiment backend mounted on app-api when
APP_MODE=study. Provides study configuration, participant runs (resumable, with
a 1-hour window), per-scenario session saving, questionnaires, the VC-engine
prepare lifecycle, and data export.

The larger participant/voice pipeline (PersonaPlex + the active VC engine) is
unchanged; this package only adds the study wrapper and reliable persistence.
"""

from .router import build_study_router

__all__ = ["build_study_router"]
