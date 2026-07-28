#!/bin/bash
# Hear-Me-Out: launch all services with SSL (uv per-service).
#   PersonaPlex|MiniCPM-o :8000 (GPU)   app-api :5001 (GPU)   MeanVC|X-VC :5002
# Each service runs in its own uv project venv (uv run --project / from its dir).
# Override the workspace with WORKSPACE=/dir.
# Pick the speech LM with SPEECH_LM_ENGINE=personaplex|minicpm_o; the VC engine with VC_ENGINE=meanvc|xvc.

SCRIPT_DIR="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
# Defaults to the repo's parent (this script lives at <workspace>/Hear-Me-Out/infra/run_all.sh).
WORKSPACE="${WORKSPACE:-$(cd "$SCRIPT_DIR/../.." 2>/dev/null && pwd || echo /workspace)}"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

echo
echo -e "${BOLD}╭──────────────────────────────────────────────╮${NC}"
echo -e "${BOLD}│        Hear-Me-Out — starting services       │${NC}"
echo -e "${BOLD}╰──────────────────────────────────────────────╯${NC}"
echo -e "  ${DIM}workspace${NC}  $WORKSPACE"

# uv must be present (provisions each service's venv).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo -e "${YELLOW}ERROR:${NC} uv not found — run infra/setup.sh first."; exit 1; }

# Locate the repo.
if [ -d "$WORKSPACE/Hear-Me-Out" ]; then HEARMEOUT_DIR="$WORKSPACE/Hear-Me-Out"
elif [ -d "$HOME/Hear-Me-Out" ]; then HEARMEOUT_DIR="$HOME/Hear-Me-Out"
else HEARMEOUT_DIR="$(cd "$SCRIPT_DIR/.." 2>/dev/null && pwd)"; fi
SERVICES="$HEARMEOUT_DIR/services"

# SSL certs (browser mic capture needs HTTPS).
SSL_DIR=""
for d in "$WORKSPACE/ssl" "$SCRIPT_DIR/ssl" "$HOME/ssl"; do
    if [ -f "$d/cert.pem" ] && [ -f "$d/key.pem" ]; then SSL_DIR="$d"; break; fi
done
if [ -z "$SSL_DIR" ]; then
    mkdir -p "$WORKSPACE/ssl"
    openssl req -x509 -newkey rsa:2048 -keyout "$WORKSPACE/ssl/key.pem" -out "$WORKSPACE/ssl/cert.pem" \
        -days 365 -nodes -subj "/CN=*" -addext "subjectAltName=IP:0.0.0.0" 2>/dev/null
    SSL_DIR="$WORKSPACE/ssl"
    echo -e "  ${DIM}ssl${NC}        generated in $SSL_DIR"
fi

# App mode: the HMO platform (default) or the participant Study platform. They
# are mutually exclusive — only one runs at a time (chosen here).
if [ -z "$APP_MODE" ]; then
  echo ""
  echo "  Which app on :5001?"
  echo "    1) HMO platform   (Chat / Convert / Metrics) [default]"
  echo "    2) Study platform (participant experiment + admin)"
  read -t 60 -p "  Choice [1/2]: " app_choice < /dev/tty 2>/dev/tty || app_choice="1"
  case "$app_choice" in 2) APP_MODE=study ;; *) APP_MODE=hmo ;; esac
fi
export APP_MODE

# Frontend is the Vite build for the selected app; build it if missing.
if [ "$APP_MODE" = "study" ]; then
    FRONTEND_PATH="$HEARMEOUT_DIR/study-frontend/dist"
else
    FRONTEND_PATH="$HEARMEOUT_DIR/frontend/dist"
fi
if [ ! -d "$FRONTEND_PATH" ]; then
    echo -e "  ${DIM}frontend${NC}   dist missing — building ($APP_MODE)..."
    APP_MODE="$APP_MODE" bash "$HEARMEOUT_DIR/infra/build-frontend.sh" || echo -e "  ${YELLOW}WARN:${NC} frontend build failed"
fi
# Fail clearly here rather than let app-api crash later on a missing static dir.
if [ ! -d "$FRONTEND_PATH" ]; then
    echo -e "  ${YELLOW}ERROR:${NC} $FRONTEND_PATH not found after build."
    echo -e "         Build it explicitly:  ${BOLD}APP_MODE=$APP_MODE bash infra/build-frontend.sh${NC}"
    echo -e "         (If it built the wrong app, this clone's infra/build-frontend.sh is stale — git pull.)"
    exit 1
fi
echo -e "  ${DIM}app${NC}        $APP_MODE  ($FRONTEND_PATH)"

# Pick the speech LM engine (only one runs on :8000).
if [ -z "$SPEECH_LM_ENGINE" ]; then
  echo ""
  echo "  Which speech LM on :8000?"
  echo "    1) PersonaPlex  (moshi fork, GPU) [default]"
  echo "    2) MiniCPM-o    (omni speech LM, GPU)"
  read -t 60 -p "  Choice [1/2]: " lm_choice < /dev/tty 2>/dev/tty || lm_choice="1"
  case "$lm_choice" in 2) SPEECH_LM_ENGINE=minicpm_o ;; *) SPEECH_LM_ENGINE=personaplex ;; esac
fi

# Pick the voice-conversion engine (only one runs on :5002).
# In study mode the engine is NOT started at boot (the participant prepare step
# starts it on demand), so don't prompt here — default meanvc, override with
# VC_ENGINE=xvc in the environment.
if [ -z "$VC_ENGINE" ]; then
  if [ "$APP_MODE" = "study" ]; then
    VC_ENGINE=meanvc
  else
    echo ""
    echo "  Which voice-conversion engine on :5002?"
    echo "    1) MeanVC  (CPU, streaming) [default]"
    echo "    2) X-VC    (GPU, streaming; needs the X-VC install from setup.sh)"
    read -t 60 -p "  Choice [1/2]: " vc_choice < /dev/tty 2>/dev/tty || vc_choice="1"
    case "$vc_choice" in 2) VC_ENGINE=xvc ;; *) VC_ENGINE=meanvc ;; esac
  fi
fi
if [ "$APP_MODE" = "study" ]; then
  echo -e "  ${DIM}vc engine${NC}  $VC_ENGINE  ${DIM}(starts on participant run; override with VC_ENGINE=xvc)${NC}"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# moshi uses torch.compile (inductor->triton) during warmup; skip it so PersonaPlex
# runs without a triton toolchain (eager mode). Set NO_CUDA_GRAPH=1 too if CUDA-graph
# capture errors. Remove these once triton is installed/working for the speedup.
export NO_TORCH_COMPILE=1

# Kill stale services.
pkill -f "personaplex/entrypoint" 2>/dev/null || true
pkill -f "minicpm_o/server.py" 2>/dev/null || true
pkill -f "llama-server" 2>/dev/null || true
pkill -f "app:create_app" 2>/dev/null || true
pkill -f "meanvc/server.py" 2>/dev/null || true
pkill -f "xvc/server.py" 2>/dev/null || true
sleep 2

# Shared env consumed by the service processes.
export FRONTEND_PATH SSL_DIR
export WHISPER_MODEL="${WHISPER_MODEL:-small}"

# Study-mode env (app-api mounts the study router; the VC engine is started later
# by the prepare step, so its launcher must inherit the engine env below).
if [ "$APP_MODE" = "study" ]; then
    if [ -z "$STUDY_ADMIN_TOKEN" ]; then
        STUDY_ADMIN_TOKEN="$(openssl rand -hex 12 2>/dev/null || echo changeme-study-admin)"
        echo -e "  ${YELLOW}study${NC}      generated STUDY_ADMIN_TOKEN=${BOLD}$STUDY_ADMIN_TOKEN${NC}"
    fi
    export STUDY_ADMIN_TOKEN
    # Storage for the study DB + audio. Put this on a MOUNTED VOLUME so it survives
    # container restarts (the repo dir is ephemeral). One root holds both; override
    # STUDY_DB_PATH / STUDY_DATA_DIR for finer control.
    if [ -z "$STUDY_DATA_ROOT" ]; then
        read -t 60 -p "  Study data dir (DB + audio) [/workspace/data]: " sdr < /dev/tty 2>/dev/tty || sdr=""
        STUDY_DATA_ROOT="${sdr:-/workspace/data}"
    fi
    # Expand a leading ~ (bash does NOT expand it inside a variable, so an entered
    # or exported "~/data" would otherwise create a literal "~" dir in the cwd).
    case "$STUDY_DATA_ROOT" in "~/"*) STUDY_DATA_ROOT="$HOME/${STUDY_DATA_ROOT#\~/}" ;; "~") STUDY_DATA_ROOT="$HOME" ;; esac
    export STUDY_DATA_ROOT
    export STUDY_DB_PATH="${STUDY_DB_PATH:-$STUDY_DATA_ROOT/study.db}"
    export STUDY_DATA_DIR="${STUDY_DATA_DIR:-$STUDY_DATA_ROOT/media}"
    # Create the dirs now and report if it fails (don't silently swallow, or the DB
    # ends up somewhere unexpected / nowhere).
    if ! mkdir -p "$STUDY_DATA_DIR" "$(dirname "$STUDY_DB_PATH")"; then
        echo -e "  ${YELLOW}ERROR:${NC} could not create study data dir under $STUDY_DATA_ROOT — check the path is writable (or set STUDY_DATA_ROOT to a writable dir)."
        exit 1
    fi
    echo -e "  ${DIM}storage${NC}    db=$STUDY_DB_PATH  media=$STUDY_DATA_DIR"
    # Observability (OpenTelemetry traces + logs). OFF by default. Two ways to opt in:
    #  • STUDY_OBSERVABILITY=1 — start the bundled OpenObserve backend (installed by
    #    setup.sh --observability) here, wire OTLP to it, and proxy its UI under /logs
    #    on this :5001 port (no extra container / exposed port).
    #  • STUDY_TRACING=1 — just export OTLP to an external OTEL_EXPORTER_OTLP_ENDPOINT
    #    (e.g. your own collector) without starting anything.
    # Either way these vars are inherited by app-api and (via engine.py) the on-demand
    # VC engine; each service self-names (study-app-api/xvc/meanvc/study-analysis), so
    # don't set OTEL_SERVICE_NAME.
    # Decide whether to run the bundled OpenObserve backend:
    #   installed on the volume  -> auto-on (no prompt, survives restarts)
    #   STUDY_OBSERVABILITY=1/0   -> force on / off
    #   otherwise (fresh box)     -> ASK, and install on first yes
    _o2_bin="${STUDY_DATA_ROOT}/observability/bin/openobserve"
    _o2_enable="no"
    if [ "$STUDY_OBSERVABILITY" = "0" ] || [ "$STUDY_OBSERVABILITY" = "false" ]; then
        _o2_enable="no"
    elif [ "$STUDY_OBSERVABILITY" = "1" ] || [ "$STUDY_OBSERVABILITY" = "true" ] || [ -x "$_o2_bin" ]; then
        _o2_enable="yes"
    else
        read -t 60 -p "  Enable observability UI (traces + logs at :5001/logs)? [y/N]: " _o2ans < /dev/tty 2>/dev/tty || _o2ans=""
        case "$_o2ans" in [Yy]*) _o2_enable="yes" ;; *) _o2_enable="no" ;; esac
    fi

    if [ "$_o2_enable" = "yes" ]; then
        if [ ! -x "$_o2_bin" ]; then   # one-time install onto the volume
            echo -e "  ${DIM}observ.${NC}    installing OpenObserve (one-time, persists on the volume)…"
            WORKSPACE="$WORKSPACE" STUDY_DATA_ROOT="$STUDY_DATA_ROOT" bash "$SCRIPT_DIR/observability.sh" install \
                || echo -e "  ${YELLOW}warn:${NC} observability install failed — set O2_VERSION and retry: bash infra/observability.sh install"
        fi
        if [ -x "$_o2_bin" ]; then
            O2_PORT="${O2_PORT:-5080}"
            WORKSPACE="$WORKSPACE" STUDY_DATA_ROOT="$STUDY_DATA_ROOT" O2_PORT="$O2_PORT" \
                bash "$SCRIPT_DIR/observability.sh" start || echo -e "  ${YELLOW}warn:${NC} observability backend failed to start (see log)"
            # OpenObserve serves under ZO_BASE_URI=/logs; OTLP ingest is org-scoped + Basic auth.
            _o2_email="${O2_ROOT_USER_EMAIL:-admin@example.com}"
            _o2_pass="${O2_ROOT_USER_PASSWORD:-ChangeMe123}"
            _o2_auth="$(printf '%s:%s' "$_o2_email" "$_o2_pass" | base64 | tr -d '\n')"
            export OTEL_TRACES_EXPORTER="otlp"
            export OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:$O2_PORT/logs/api/default"
            export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic $_o2_auth"
            export STUDY_OBSERVABILITY_URL="http://127.0.0.1:$O2_PORT"   # app-api reverse-proxies /logs -> here
            echo -e "  ${DIM}observ.${NC}    OpenObserve traces+logs  ${DIM}UI → http(s)://<host>:5001/logs  (login $_o2_email)${NC}"
        fi
    elif [ "$STUDY_TRACING" = "1" ] || [ "$STUDY_TRACING" = "true" ]; then
        export OTEL_TRACES_EXPORTER="${OTEL_TRACES_EXPORTER:-otlp}"
        export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://127.0.0.1:4318}"
        echo -e "  ${DIM}observ.${NC}    OpenTelemetry traces+logs → $OTEL_EXPORTER_OTLP_ENDPOINT  ${DIM}(external collector)${NC}"
    fi
    export STUDY_VC_HOST="${STUDY_VC_HOST:-127.0.0.1}"
    # Best-effort: locate the PersonaPlex voice prompts (.pt) in the HF cache so the
    # admin's assistant-voice dropdown (/api/study/voices) is populated.
    if [ -z "$PERSONAPLEX_VOICES_DIR" ]; then
        for _hub in "$HF_HOME/hub" "$HUGGINGFACE_HUB_CACHE" "$WORKSPACE/huggingface/hub" "$HOME/.cache/huggingface/hub"; do
            _vd=$(ls -d "$_hub"/models--nvidia--personaplex-7b-v1/snapshots/*/voices 2>/dev/null | head -1)
            [ -n "$_vd" ] && { export PERSONAPLEX_VOICES_DIR="$_vd"; break; }
        done
    fi
    [ -n "$PERSONAPLEX_VOICES_DIR" ] && echo -e "  ${DIM}voices${NC}     $PERSONAPLEX_VOICES_DIR"
fi
# MiniCPM-o (GGUF via llama.cpp-omni): point the bridge at the C++ engine + GGUF weights.
# Q4_K_M ≈ 9GB VRAM on a 24GB card, so there's plenty of headroom — app-api's Whisper
# stays on GPU (faster), and X-VC can co-load. (Override WHISPER_DEVICE=cpu if ever tight.)
if [ "$SPEECH_LM_ENGINE" = "minicpm_o" ]; then
    export LLAMA_OMNI_ROOT="${LLAMA_OMNI_ROOT:-$WORKSPACE/llama.cpp-omni}"
    export LLAMA_OMNI_BIN="${LLAMA_OMNI_BIN:-$LLAMA_OMNI_ROOT/build/bin/llama-server}"
    export MINICPM_O_GGUF_DIR="${MINICPM_O_GGUF_DIR:-$WORKSPACE/models/minicpm-o-gguf}"
    export MINICPM_O_LLM="${MINICPM_O_LLM:-MiniCPM-o-4_5-Q4_K_M.gguf}"
    export MINICPM_O_CPP_PORT="${MINICPM_O_CPP_PORT:-19080}"
    # Match the official cpp_backend ctx (32768). The duplex KV cache is never trimmed, so a
    # small ctx fills after ~2 turns of audio tokens and the model wedges in listen mode.
    # ~4.6GB KV at Q4 — still fits on the 24GB card alongside TTS + X-VC.
    export MINICPM_O_CTX="${MINICPM_O_CTX:-32768}"
    # Per-chunk speak-token budget. Engine duplex default 26 (~1s audio/chunk) clips long
    # replies (text outruns speech). 75 (~3s) lets Token2Wav finish phrases; lower it for
    # snappier barge-in, raise it if long sentences still get cut.
    export MINICPM_O_MAX_SPEAK_TOKENS="${MINICPM_O_MAX_SPEAK_TOKENS:-75}"
    export MINICPM_REF_AUDIO="${MINICPM_REF_AUDIO:-$HEARMEOUT_DIR/recordings/Target_2.wav}"
    export MINICPM_O_OUTPUT_DIR="${MINICPM_O_OUTPUT_DIR:-$SERVICES/minicpm_o/_omni_out}"
    # llama-server (CUDA build) needs its cudart at runtime — and it MUST match the toolkit
    # it was built with (the runfile toolkit at $WORKSPACE/cuda-*, which is <= the driver).
    # Build a lib path, applied ONLY to the MiniCPM-o launch (not exported globally), so it
    # never shadows app-api / the VC engine's torch-cu121 bundled CUDA. Toolkit libs go
    # FIRST; $CUDA_HOME (if set) wins over everything.
    MINICPM_O_LD=""
    _add_ld() { [ -n "$1" ] && [ -d "$1" ] && MINICPM_O_LD="${MINICPM_O_LD:+$MINICPM_O_LD:}$1"; }
    _add_ld "${CUDA_HOME:+$CUDA_HOME/lib64}"
    for _d in "$WORKSPACE"/cuda-*/lib64; do _add_ld "$_d"; done   # runfile toolkit(s)
    _add_ld /usr/local/cuda/lib64
    _add_ld /usr/local/cuda/targets/x86_64-linux/lib
    pkill -f "llama-server" 2>/dev/null || true   # clear a stale C++ engine
fi
export VC_CHECKPOINT_PATH="$WORKSPACE/models/seed-vc/DiT_uvit_tat_xlsr_ema.pth"
export VC_MODEL_CONFIG="${VC_MODEL_CONFIG:-configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml}"
export PERSONAPLEX_PROXY_HOST="${PERSONAPLEX_PROXY_HOST:-127.0.0.1}"
export PERSONAPLEX_PROXY_PORT="${PERSONAPLEX_PROXY_PORT:-8000}"

echo -e "${DIM}────────────────────────────────────────────────────${NC}"

# --- Speech LM :8000 ---
if [ "$SPEECH_LM_ENGINE" = "minicpm_o" ]; then
    echo -e "  ${CYAN}▶${NC} MiniCPM-o     :8000  ${DIM}(GPU)${NC}"
    # LD_LIBRARY_PATH scoped to THIS process only (not app-api / VC engine).
    ( cd "$SERVICES/minicpm_o" && LD_LIBRARY_PATH="${MINICPM_O_LD}${LD_LIBRARY_PATH:-}" \
        exec uv run python server.py \
        --host 0.0.0.0 --port 8000 --device cuda --ssl "$SSL_DIR" ) &
    PID1=$!; LM_LABEL="MiniCPM-o"
else
    echo -e "  ${CYAN}▶${NC} PersonaPlex   :8000  ${DIM}(GPU)${NC}"
    ( cd "$SERVICES/personaplex" && exec uv run python entrypoint.py \
        --host 0.0.0.0 --port 8000 --device cuda --ssl "$SSL_DIR" ) &
    PID1=$!; LM_LABEL="PersonaPlex"
fi

# --- app-api :5001 ---
echo -e "  ${CYAN}▶${NC} app-api       :5001  ${DIM}(GPU)${NC}"
( cd "$SERVICES/app_api" && exec uv run uvicorn app:create_app --factory \
    --host 0.0.0.0 --port 5001 \
    --ssl-keyfile "$SSL_DIR/key.pem" --ssl-certfile "$SSL_DIR/cert.pem" ) &
PID2=$!

# --- VC engine :5002 ---
# Set the engine env in all modes (so the study prepare step's launcher inherits
# it); the engine PROCESS is started here only in hmo mode. In study mode it is
# started on demand by infra/vc_engine.sh when a participant begins a run.
if [ "$VC_ENGINE" = "xvc" ]; then
    export XVC_DIR="$WORKSPACE/X-VC"
    export XVC_CONFIG="$XVC_DIR/configs/xvc.yaml"
    export XVC_CKPT="$XVC_DIR/ckpts/xvc.pt"
    export MEANVC_PORT=5002
    # X-VC streaming window (ms). MODEL-COUPLED — X-VC was designed for the stock
    # 2400/120/20/100; lowering CHUNK changes the forward's input length and yields
    # silent/garbled conversion (PersonaPlex then hears nothing at all). Keep these
    # values; only change them if you've validated X-VC output on the GPU box. Any
    # explicit env override wins.
    export XVC_CHUNK_MS="${XVC_CHUNK_MS:-2400}"
    export XVC_CURRENT_MS="${XVC_CURRENT_MS:-120}"
    export XVC_SMOOTH_MS="${XVC_SMOOTH_MS:-20}"
    export XVC_FUTURE_MS="${XVC_FUTURE_MS:-100}"
    echo -e "  ${DIM}xvc window${NC} chunk=${XVC_CHUNK_MS} current=${XVC_CURRENT_MS} smooth=${XVC_SMOOTH_MS} future=${XVC_FUTURE_MS} ms"
    VC_LABEL="X-VC"
    { [ "$APP_MODE" = "study" ] || [ -d "$XVC_DIR" ]; } || { echo -e "  ${YELLOW}ERROR:${NC} X-VC not installed — rerun setup.sh with --xvc."; exit 1; }
else
    export MEANVC_CKPT_DIR="$WORKSPACE/models/meanvc"
    export MEANVC_SV_CKPT="$WORKSPACE/models/meanvc-sv/wavlm_large_finetune.pth"
    export SPEAKER_VERIFICATION_ROOT="$WORKSPACE"
    export MEANVC_PORT=5002
    VC_LABEL="MeanVC"
fi

if [ "$APP_MODE" = "study" ]; then
    echo -e "  ${CYAN}⏸${NC} $VC_LABEL  :5002  ${DIM}(deferred — starts on participant run)${NC}"
    PID3=""
elif [ "$VC_ENGINE" = "xvc" ]; then
    echo -e "  ${CYAN}▶${NC} X-VC          :5002  ${DIM}(GPU, streaming)${NC}"
    ( cd "$XVC_DIR" && exec uv run --project "$SERVICES/xvc" python "$SERVICES/xvc/server.py" ) &
    PID3=$!
else
    echo -e "  ${CYAN}▶${NC} MeanVC        :5002  ${DIM}(CPU, streaming)${NC}"
    ( cd "$SERVICES/meanvc" && exec uv run python server.py ) &
    PID3=$!
fi

echo -e "${DIM}────────────────────────────────────────────────────${NC}"
if [ "$APP_MODE" = "study" ]; then
    echo -e "  ${GREEN}started${NC}  $LM_LABEL=$PID1  app-api=$PID2  ${DIM}($VC_LABEL deferred)${NC}"
else
    echo -e "  ${GREEN}started${NC}  $LM_LABEL=$PID1  app-api=$PID2  $VC_LABEL=$PID3"
fi
echo -e "  ${DIM}(models load on first connect; Ctrl-C to stop all)${NC}"
echo
wait
