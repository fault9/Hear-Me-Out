# Confirmatory analysis

Implements the Analysis section of the method chapter: two primary contrasts
(VC activation vs stable converted; VC deactivation vs stable natural), two
co-primary outcomes (scenario-level demonstrated grounding; post-transition
repair count), Holm-adjusted as one four-test family. Mixed-effects models
with participant random intercepts throughout.

## Iron rules

- `data/` and `output/` are gitignored. No participant data ever enters git.
- The confirmatory run on real labels happens ONCE, after collection ends and
  coding is finalized. Until then, run models only with `--permute` (condition
  labels shuffled within participant) to validate the pipeline without
  inspecting condition-level results.

## Workflow

1. On the container, after coding is finalized:
   `python -m study.dataset_export --study-id 1` and copy the export
   directory's CSVs into `analysis/data/`.
2. `python3 prep.py` — builds model frames in `output/frames/`
   (included sessions only).
3. Smoke test (any time): `Rscript models.R --permute`
4. Final confirmatory run (once): `Rscript models.R --confirmatory`
   Results land in `output/confirmatory_results.csv`.
5. Prespecified technical-completeness sensitivity:
   `Rscript models.R --confirmatory --complete-technical-sensitivity`.

Blank coded outcomes remain missing throughout frame preparation and model
fitting. In particular, a missing `demonstrated_grounding` value is never
converted to a grounding failure. `prep.py` also writes the participant IDs
excluded by the sensitivity rule to
`output/frames/sensitivity_complete_technical.json`.

Requires R with: lme4, glmmTMB, ordinal. Install once in R:
`install.packages(c("lme4", "glmmTMB", "ordinal"))`
