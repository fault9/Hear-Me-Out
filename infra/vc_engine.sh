#!/bin/bash
# Start/stop/status the selected VC engine (MeanVC or X-VC) on :5002, detached.
# Single source of truth for the VC-engine launch, shared by:
#   - run_all.sh in hmo mode (starts it at boot)
#   - the study prepare step (study-api starts it when a participant begins a run)
# Env (inherited from run_all, with standalone-safe defaults):
#   VC_ENGINE=meanvc|xvc   MEANVC_PORT   SSL_DIR   WORKSPACE
usage() { echo "usage: vc_engine.sh {start|stop|status}"; exit 2; }
ACTION="${1:-status}"

SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
HEARMEOUT_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)"
WORKSPACE="${WORKSPACE:-$(cd "$HEARMEOUT_DIR/.." 2>/dev/null && pwd || echo /workspace)}"
SERVICES="$HEARMEOUT_DIR/services"
VC_ENGINE="${VC_ENGINE:-meanvc}"
MEANVC_PORT="${MEANVC_PORT:-5002}"
LOG="${VC_ENGINE_LOG:-${TMPDIR:-/tmp}/hmo_vc_engine.log}"
PIDFILE="${VC_ENGINE_PIDFILE:-${TMPDIR:-/tmp}/hmo_vc_engine.pid}"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export MEANVC_PORT
export SSL_DIR="${SSL_DIR:-$WORKSPACE/ssl}"
export STUDY_APP_API_URL="${STUDY_APP_API_URL:-https://127.0.0.1:5001}"
export PERSONAPLEX_PROXY_HOST="${PERSONAPLEX_PROXY_HOST:-127.0.0.1}"
export PERSONAPLEX_PROXY_PORT="${PERSONAPLEX_PROXY_PORT:-8000}"

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$MEANVC_PORT") 2>/dev/null && { exec 3>&-; return 0; } || return 1; }

case "$ACTION" in
  start)
    if port_open; then echo "vc engine already listening on :$MEANVC_PORT"; exit 0; fi
    if [ "$VC_ENGINE" = "xvc" ]; then
      export XVC_DIR="${XVC_DIR:-$WORKSPACE/X-VC}"
      export XVC_CONFIG="${XVC_CONFIG:-$XVC_DIR/configs/xvc.yaml}"
      export XVC_CKPT="${XVC_CKPT:-$XVC_DIR/ckpts/xvc.pt}"
      [ -d "$XVC_DIR" ] || { echo "ERROR: X-VC not installed at $XVC_DIR"; exit 1; }
      nohup bash -c "cd '$XVC_DIR' && exec uv run --project '$SERVICES/xvc' python '$SERVICES/xvc/server.py'" >"$LOG" 2>&1 &
    else
      export MEANVC_CKPT_DIR="${MEANVC_CKPT_DIR:-$WORKSPACE/models/meanvc}"
      export MEANVC_SV_CKPT="${MEANVC_SV_CKPT:-$WORKSPACE/models/meanvc-sv/wavlm_large_finetune.pth}"
      export SPEAKER_VERIFICATION_ROOT="${SPEAKER_VERIFICATION_ROOT:-$WORKSPACE}"
      nohup bash -c "cd '$SERVICES/meanvc' && exec uv run python server.py" >"$LOG" 2>&1 &
    fi
    echo $! > "$PIDFILE"
    echo "vc engine ($VC_ENGINE) starting — pid $(cat "$PIDFILE"), log $LOG"
    ;;
  stop)
    [ -f "$PIDFILE" ] && { kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; }
    pkill -f "meanvc/server.py" 2>/dev/null || true
    pkill -f "xvc/server.py" 2>/dev/null || true
    echo "vc engine stopped"
    ;;
  status)
    if port_open; then echo "up"; exit 0; else echo "down"; exit 1; fi
    ;;
  *) usage;;
esac
