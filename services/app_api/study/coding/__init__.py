"""LLM-assisted coding pipeline for the participant study.

Subcommands (run from services/app_api, under the app-api venv):

    python -m study.coding --study-id N packets        build blinded packets
    python -m study.coding --study-id N freeze         freeze materials
    python -m study.coding --study-id N status         freeze/drift status
    python -m study.coding --study-id N judge          judge + verify
    python -m study.coding --study-id N review         flags + sample + sheets
    python -m study.coding --study-id N import-human   ingest filled sheets
    python -m study.coding --study-id N finalize       merge to final labels
    python -m study.coding --study-id N agreement      judge-vs-human stats
    python -m study.coding --study-id N reliability-expand
                                                       queue remaining packets

All state lives under STUDY_DATA_ROOT/coding/study<id>/ so the pipeline runs
against a copied data root on any machine.
"""
