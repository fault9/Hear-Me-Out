"""LLM-assisted coding pipeline for the participant study.

Subcommands (run from services/app_api, under the app-api venv):

    python -m study.coding packets  --study-id N        build blinded packets
    python -m study.coding freeze   [--note TEXT]        freeze materials
    python -m study.coding status                        freeze/drift status
    python -m study.coding judge    [--pilot] [--model M]  judge + verify
    python -m study.coding review   --seed S [--fraction F] flags + sample + sheets
    python -m study.coding import-human                  ingest filled sheets
    python -m study.coding finalize                      merge to final labels
    python -m study.coding agreement                     judge-vs-human stats
    python -m study.coding reliability-expand            queue all remaining
                                                         packets if a frozen
                                                         threshold failed

All state lives under STUDY_DATA_ROOT/coding/study<id>/ so the pipeline runs
against a copied data root on any machine.
"""
