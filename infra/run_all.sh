#!/bin/bash
# Hear-Me-Out: launch all services with SSL (uv per-service).
#   PersonaPlex|MiniCPM-o :8000 (GPU)   app-api :5001 (GPU)   MeanVC|X-VC :5002
# Each service runs in its own uv project venv (uv run --project / from its dir).
# Override the workspace with WORKSPACE=/dir.
# Pick the speech LM with SPEECH_LM_ENGINE=personaplex|minicpm_o; the VC engine with VC_ENGINE=meanvc|xvc.
# With VC_ENGINE=xvc, pick the checkpoint (original release vs an accent fine-tune)
# interactively or with XVC_CKPT=/path/to/ckpt.pt; fine-tunes live in $XVC_DIR/ckpts/
# (pull them there with X-VC's scripts/publish_checkpoint.py pull).

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

# Frontend is always the Vite build; build it if missing.
FRONTEND_PATH="$HEARMEOUT_DIR/frontend/dist"
if [ ! -d "$FRONTEND_PATH" ]; then
    echo -e "  ${DIM}frontend${NC}   dist missing — building..."
    bash "$HEARMEOUT_DIR/infra/build-frontend.sh" || echo -e "  ${YELLOW}WARN:${NC} frontend build failed"
fi
echo -e "  ${DIM}frontend${NC}   $FRONTEND_PATH"

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
if [ -z "$VC_ENGINE" ]; then
  echo ""
  echo "  Which voice-conversion engine on :5002?"
  echo "    1) MeanVC  (CPU, streaming) [default]"
  echo "    2) X-VC    (GPU, streaming; needs the X-VC install from setup.sh)"
  read -t 60 -p "  Choice [1/2]: " vc_choice < /dev/tty 2>/dev/tty || vc_choice="1"
  case "$vc_choice" in 2) VC_ENGINE=xvc ;; *) VC_ENGINE=meanvc ;; esac
fi

# X-VC only: pick WHICH checkpoint serves on :5002 (original release vs an accent
# fine-tune). Any *.pt in $XVC_DIR/ckpts/ is offered; XVC_CKPT env skips the menu.
if [ "$VC_ENGINE" = "xvc" ]; then
  export XVC_DIR="${XVC_DIR:-$WORKSPACE/X-VC}"
  export XVC_CONFIG="${XVC_CONFIG:-$XVC_DIR/configs/xvc.yaml}"
  [ -d "$XVC_DIR" ] || { echo -e "${YELLOW}ERROR:${NC} X-VC not installed — rerun setup.sh with --xvc."; exit 1; }
  if [ -z "$XVC_CKPT" ]; then
    xvc_ckpts=()
    [ -f "$XVC_DIR/ckpts/xvc.pt" ] && xvc_ckpts+=("$XVC_DIR/ckpts/xvc.pt")
    while IFS= read -r f; do
      [ "$(basename "$f")" = "xvc.pt" ] || xvc_ckpts+=("$f")
    done < <(ls "$XVC_DIR"/ckpts/*.pt 2>/dev/null | sort)
    if [ ${#xvc_ckpts[@]} -eq 0 ]; then
      echo -e "${YELLOW}ERROR:${NC} no checkpoints in $XVC_DIR/ckpts/ — download xvc.pt (setup.sh)"
      echo "       or pull a fine-tune: python scripts/publish_checkpoint.py pull ... --out ckpts/<name>.pt"
      exit 1
    elif [ ${#xvc_ckpts[@]} -eq 1 ]; then
      XVC_CKPT="${xvc_ckpts[0]}"
    else
      echo ""
      echo "  Which X-VC checkpoint on :5002?"
      i=1
      for f in "${xvc_ckpts[@]}"; do
        label="$(basename "$f")"
        [ "$label" = "xvc.pt" ] && label="xvc.pt  (original release)"
        [ "$i" -eq 1 ] && label="$label [default]"
        echo "    $i) $label"
        i=$((i+1))
      done
      read -t 60 -p "  Choice [1-${#xvc_ckpts[@]}]: " ck_choice < /dev/tty 2>/dev/tty || ck_choice="1"
      case "$ck_choice" in ''|*[!0-9]*) ck_choice=1 ;; esac
      { [ "$ck_choice" -ge 1 ] && [ "$ck_choice" -le ${#xvc_ckpts[@]} ]; } || ck_choice=1
      XVC_CKPT="${xvc_ckpts[$((ck_choice-1))]}"
    fi
  fi
  export XVC_CKPT
  # Fine-tuned checkpoints are saved WITHOUT EMA weights (ema_update: false in the
  # fine-tune configs); load_xvc falls back to raw generator weights on its own, but
  # set the flag explicitly so the served-weights intent is visible in the logs.
  if [ "$(basename "$XVC_CKPT")" != "xvc.pt" ]; then
    export XVC_EMA_LOAD="${XVC_EMA_LOAD:-0}"
  fi
  echo -e "  ${DIM}xvc ckpt${NC}   $XVC_CKPT ${DIM}(ema_load=${XVC_EMA_LOAD:-1})${NC}"
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
# MiniCPM-o (GGUF via llama.cpp-omni): point the bridge at the C++ engine + GGUF weights.
# Q4_K_M ≈ 9GB VRAM on a 24GB card, so there's plenty of headroom — app-api's Whisper
# stays on GPU (faster), and X-VC can co-load. (Override WHISPER_DEVICE=cpu if ever tight.)
if [ "$SPEECH_LM_ENGINE" = "minicpm_o" ]; then
    export LLAMA_OMNI_ROOT="${LLAMA_OMNI_ROOT:-$WORKSPACE/llama.cpp-omni}"
    export LLAMA_OMNI_BIN="${LLAMA_OMNI_BIN:-$LLAMA_OMNI_ROOT/build/bin/llama-server}"
    export MINICPM_O_GGUF_DIR="${MINICPM_O_GGUF_DIR:-$WORKSPACE/models/minicpm-o-gguf}"
    export MINICPM_O_LLM="${MINICPM_O_LLM:-MiniCPM-o-4_5-Q4_K_M.gguf}"
    export MINICPM_O_CPP_PORT="${MINICPM_O_CPP_PORT:-19080}"
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
if [ "$VC_ENGINE" = "xvc" ]; then
    echo -e "  ${CYAN}▶${NC} X-VC          :5002  ${DIM}(GPU, streaming) ckpt=$(basename "$XVC_CKPT")${NC}"
    export MEANVC_PORT=5002
    # XVC_DIR/XVC_CONFIG/XVC_CKPT resolved by the checkpoint picker above.
    # Run from the X-VC repo (relative pretrained/ paths) using the services/xvc venv.
    ( cd "$XVC_DIR" && exec uv run --project "$SERVICES/xvc" python "$SERVICES/xvc/server.py" ) &
    PID3=$!; VC_LABEL="X-VC"
else
    echo -e "  ${CYAN}▶${NC} MeanVC        :5002  ${DIM}(CPU, streaming)${NC}"
    export MEANVC_CKPT_DIR="$WORKSPACE/models/meanvc"
    export MEANVC_SV_CKPT="$WORKSPACE/models/meanvc-sv/wavlm_large_finetune.pth"
    export SPEAKER_VERIFICATION_ROOT="$WORKSPACE"
    export MEANVC_PORT=5002
    ( cd "$SERVICES/meanvc" && exec uv run python server.py ) &
    PID3=$!; VC_LABEL="MeanVC"
fi

echo -e "${DIM}────────────────────────────────────────────────────${NC}"
echo -e "  ${GREEN}started${NC}  $LM_LABEL=$PID1  app-api=$PID2  $VC_LABEL=$PID3"
echo -e "  ${DIM}(models load on first connect; Ctrl-C to stop all)${NC}"
echo
wait
