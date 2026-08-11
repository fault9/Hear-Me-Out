#!/bin/bash
# One analysis run, end to end: take a dataset export, build the frames, fit
# the models, archive the output with a record of what produced it.
#
#   analysis/run.sh <export-dir>                  # permuted (default, safe)
#   analysis/run.sh <export-dir> --exploratory    # real labels, preliminary
#   CONFIRMATORY=1 analysis/run.sh <export-dir> --confirmatory
#
# Only analysis/data and analysis/output are cleared, and both are rebuilt from
# the export on every run. The export itself, study-data, and the coding
# directory are never written to.
#
# Archives land in $PRE_ANALYSIS_DIR (default ~/study-data/pre_analysis) as
# run_<timestamp>_<mode>/, each with a PROVENANCE.txt naming the export, the
# mode, the analysis commit, and the human/judge split behind the numbers.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ARCHIVE_ROOT="${PRE_ANALYSIS_DIR:-$HOME/study-data/pre_analysis}"
SRC="${1:-}"
MODE="${2:---permute}"

if [ -z "$SRC" ]; then
  echo "usage: $0 <export-dir> [--permute|--exploratory|--confirmatory]" >&2
  exit 2
fi

case "$MODE" in
  --permute|--exploratory|--confirmatory) ;;
  *) echo "unknown mode: $MODE" >&2; exit 2 ;;
esac

SRC="$(cd "$SRC" 2>/dev/null && pwd)" || { echo "no such directory: ${1}" >&2; exit 2; }

if [ ! -f "$SRC/scenarios.csv" ]; then
  echo "not a dataset export (no scenarios.csv): $SRC" >&2
  exit 2
fi

# The export is the only copy of what was exported; refuse to treat the working
# copy as a source, which would delete it in the clear step below.
if [ "$SRC" = "$HERE/data" ]; then
  echo "source is analysis/data itself — pass the export directory instead" >&2
  exit 2
fi

# The confirmatory run happens ONCE, after collection ends and coding is
# finalized (see README). Making it this easy to run is exactly why it needs a
# second key.
if [ "$MODE" = "--confirmatory" ] && [ "${CONFIRMATORY:-}" != "1" ]; then
  echo "refusing --confirmatory without CONFIRMATORY=1." >&2
  echo "This is the one-shot run. Collection must be closed and coding final." >&2
  exit 2
fi

STAMP="$(date +%Y%m%dT%H%M%S)"
RUN="$ARCHIVE_ROOT/run_${STAMP}_${MODE#--}"

echo "=== source   $SRC"
echo "=== mode     $MODE"
echo "=== archive  $RUN"

# Both are derived: data from the export, output from prep + models.
rm -rf "$HERE/output"
mkdir -p "$HERE/data"
rm -f "$HERE/data"/*.csv
cp "$SRC"/*.csv "$HERE/data/"
echo "=== copied   $(ls "$HERE/data"/*.csv | wc -l) csv files"

cd "$HERE"
python3 prep.py

# Descriptives come from the same frames the models use, so the tables beside
# an estimate cannot drift from it. The blind pass runs whatever the mode, and
# is written separately: it stays quotable on its own while collection is open.
# The stratified pass is added only for a run that is already unblinded.
python3 descriptives.py > "output/descriptives_blind.txt"
cat "output/descriptives_blind.txt"
if [ "$MODE" != "--permute" ]; then
  python3 descriptives.py --by-condition \
    > "output/descriptives_by_condition.txt"
  cat "output/descriptives_by_condition.txt"
fi

Rscript models.R "$MODE"

mkdir -p "$RUN"
cp -r "$HERE/output/"* "$RUN/"

{
  echo "run:        $STAMP"
  echo "mode:       $MODE"
  echo "export:     $(basename "$SRC")"
  echo "export_dir: $SRC"
  echo "analysis:   $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$HERE" diff --quiet -- "$HERE" 2>/dev/null || echo ' (uncommitted changes)')"
  python3 - "$HERE/data/scenarios.csv" <<'PY'
import collections, csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
counts = collections.Counter(r.get("coding_label_source") or "none" for r in rows)
print("scenarios:  %d rows" % len(rows))
print("labels:     " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
PY
} > "$RUN/PROVENANCE.txt"

echo
cat "$RUN/PROVENANCE.txt"
echo
echo "=== done: $RUN"
