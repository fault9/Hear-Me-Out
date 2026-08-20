#!/usr/bin/env bash
# Small guided wrapper around preprocessing, coding, export, and analysis.
set -euo pipefail

ANALYSIS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$ANALYSIS_DIR")"
APP_API_DIR="$REPO_DIR/services/app_api"
APP_API_PYTHON="$APP_API_DIR/.venv/bin/python"

clear_keys() { unset ANTHROPIC_API_KEY CODING_JUDGE_API_KEY; }
trap clear_keys EXIT
trap 'clear_keys; exit 130' HUP INT TERM

ask() {
  local name="$1" label="$2" default="$3" answer=""
  if [ -t 0 ]; then read -r -p "$label [$default] " answer; fi
  printf -v "$name" '%s' "${answer:-$default}"
}

secret() {
  local name="$1" label="$2" answer=""
  [ -n "${!name:-}" ] && return
  [ -t 0 ] || { echo "$name is required." >&2; exit 2; }
  read -r -s -p "$label: " answer; echo
  [ -n "$answer" ] || { echo "No key entered." >&2; exit 2; }
  printf -v "$name" '%s' "$answer"
  export "$name"
}

run_module() {
  if [ -x "$APP_API_PYTHON" ]; then
    (cd "$APP_API_DIR" && PYTHONDONTWRITEBYTECODE=1 "$APP_API_PYTHON" -m "$@")
  elif command -v uv >/dev/null 2>&1; then
    (cd "$APP_API_DIR" && PYTHONDONTWRITEBYTECODE=1 uv run python -m "$@")
  else
    echo "Run infra/setup.sh first; the app-api environment is missing." >&2
    exit 2
  fi
}

configure_judge() {
  local provider
  ask provider "Provider (openai-compatible or anthropic)" \
    "${CODING_PROVIDER:-openai-compatible}"
  if [ "$provider" = "anthropic" ]; then
    unset CODING_JUDGE_BASE_URL CODING_JUDGE_API_KEY
    ask CODING_JUDGE_MODEL "Model" "${CODING_JUDGE_MODEL:-claude-sonnet-5}"
    export CODING_JUDGE_MODEL
    secret ANTHROPIC_API_KEY "Anthropic key (hidden, not saved)"
  elif [ "$provider" = "openai-compatible" ]; then
    unset ANTHROPIC_API_KEY
    ask CODING_JUDGE_BASE_URL "Base URL" \
      "${CODING_JUDGE_BASE_URL:-https://api.berget.ai/v1}"
    ask CODING_JUDGE_MODEL "Model" \
      "${CODING_JUDGE_MODEL:-openai/gpt-oss-120b}"
    export CODING_JUDGE_BASE_URL CODING_JUDGE_MODEL
    secret CODING_JUDGE_API_KEY "API key (hidden, not saved)"
  else
    echo "Unknown provider: $provider" >&2
    exit 2
  fi
  CODING_MAX_TOKENS="${CODING_MAX_TOKENS:-16384}"
  export CODING_MAX_TOKENS
}

export_dataset() {
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  EXPORT_DIR="${STUDY_EXPORT_DIR:-$STUDY_DATA_ROOT/exports/study${STUDY_ID}/dataset_$stamp}"
  run_module study.dataset_export --study-id "$STUDY_ID" --out "$EXPORT_DIR"
  echo "dataset: $EXPORT_DIR"
}

usage() {
  echo "usage: analysis/study.sh [preprocess|judge|freeze|finalize|export|analyze|status]"
}

ACTION="${1:-${STUDY_ACTION:-}}"
if [ "$ACTION" = "-h" ] || [ "$ACTION" = "--help" ]; then usage; exit 0; fi
if [ -z "$ACTION" ]; then
  echo "1 preprocess  2 judge  3 freeze  4 finalize  5 export  6 analyze  7 status"
  ask choice "Choice" "7"
  case "$choice" in
    1) ACTION=preprocess ;; 2) ACTION=judge ;; 3) ACTION=freeze ;;
    4) ACTION=finalize ;; 5) ACTION=export ;; 6) ACTION=analyze ;;
    7) ACTION=status ;; *) usage; exit 2 ;;
  esac
fi

ask STUDY_ID "Study ID" "${STUDY_ID:-1}"
ask STUDY_DATA_ROOT "Study data root" "${STUDY_DATA_ROOT:-/workspace/data}"
export STUDY_DATA_ROOT CODING_STUDY_ID="$STUDY_ID"

case "$ACTION" in
  preprocess)
    run_module study.analysis_worker "$STUDY_ID"
    ;;
  judge)
    configure_judge
    run_module study.coding --study-id "$STUDY_ID" packets
    run_module study.coding --study-id "$STUDY_ID" probe --model "$CODING_JUDGE_MODEL"
    ask kind "Run kind (pilot or full)" "${CODING_RUN_KIND:-pilot}"
    if [ "$kind" = "pilot" ]; then
      run_module study.coding --study-id "$STUDY_ID" judge --pilot \
        --model "$CODING_JUDGE_MODEL"
    elif [ "$kind" = "full" ]; then
      run_module study.coding --study-id "$STUDY_ID" status
      ask approval "Type FULL to use the frozen full-dataset pass" "cancel"
      [ "$approval" = "FULL" ] || exit 2
      run_module study.coding --study-id "$STUDY_ID" judge \
        --model "$CODING_JUDGE_MODEL"
      run_module study.coding --study-id "$STUDY_ID" review \
        --seed "${CODING_REVIEW_SEED:-20260801}"
    else
      echo "Run kind must be pilot or full." >&2; exit 2
    fi
    ;;
  freeze)
    configure_judge
    run_module study.coding --study-id "$STUDY_ID" probe --model "$CODING_JUDGE_MODEL"
    ask approval "Type FREEZE after validating the pilot" "cancel"
    [ "$approval" = "FREEZE" ] || exit 2
    run_module study.coding --study-id "$STUDY_ID" freeze \
      --model "$CODING_JUDGE_MODEL" --note "${CODING_FREEZE_NOTE:-pilot validated}"
    ;;
  finalize)
    ask approval "Type FINALIZE after completing human review" "cancel"
    [ "$approval" = "FINALIZE" ] || exit 2
    run_module study.coding --study-id "$STUDY_ID" import-human
    run_module study.coding --study-id "$STUDY_ID" finalize
    run_module study.coding --study-id "$STUDY_ID" agreement
    ;;
  export)
    export_dataset
    ;;
  analyze)
    export_dataset
    ask mode "Mode (--permute, --exploratory, or --confirmatory)" \
      "${ANALYSIS_MODE:---permute}"
    if [ "$mode" = "--confirmatory" ]; then
      ask approval "Type CONFIRM for the confirmatory run" "cancel"
      [ "$approval" = "CONFIRM" ] || exit 2
      export CONFIRMATORY=1
    fi
    case "$mode" in --permute|--exploratory|--confirmatory) ;; *) exit 2 ;; esac
    bash "$ANALYSIS_DIR/run.sh" "$EXPORT_DIR" "$mode"
    ;;
  status)
    run_module study.coding --study-id "$STUDY_ID" status
    ;;
  *) usage; exit 2 ;;
esac
