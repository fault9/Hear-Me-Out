#!/usr/bin/env bash
# OpenObserve — single-binary OTel backend (traces + logs + UI) for the study stack.
# No separate container: one static binary, installed by setup.sh (--observability)
# and started here by run_all.sh. Services export OTLP to it; app-api reverse-proxies
# its UI under /logs on :5001 (it serves itself under /logs via ZO_BASE_URI).
#
# Usage:  observability.sh <install|start|stop|status>
#
# Env:
#   WORKSPACE            install root (binary at $WORKSPACE/observability/openobserve)
#   O2_VERSION           OpenObserve release tag (default below); override if needed
#   O2_PORT              listen port (default 5080)
#   STUDY_DATA_ROOT      data dir base (persisted); data at $STUDY_DATA_ROOT/observability
#   O2_ROOT_USER_EMAIL / O2_ROOT_USER_PASSWORD   initial login (defaults below)
set -euo pipefail

ACTION="${1:-status}"
WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"
O2_VERSION="${O2_VERSION:-v0.14.4}"
O2_PORT="${O2_PORT:-5080}"
# Everything under the persistent volume (STUDY_DATA_ROOT, e.g. /workspace/data) so the
# binary AND data survive container restarts — install once, no reinstall each time.
O2_ROOT="${STUDY_DATA_ROOT:-/workspace/data}/observability"
O2_DIR="$O2_ROOT/bin"
O2_BIN="$O2_DIR/openobserve"
O2_DATA="$O2_ROOT/data"
PIDFILE="${TMPDIR:-/tmp}/hmo_observability.pid"
LOG="${OBSERVABILITY_LOG:-${TMPDIR:-/tmp}/hmo_observability.log}"

export ZO_ROOT_USER_EMAIL="${O2_ROOT_USER_EMAIL:-admin@example.com}"
export ZO_ROOT_USER_PASSWORD="${O2_ROOT_USER_PASSWORD:-ChangeMe123}"
export ZO_DATA_DIR="$O2_DATA"
export ZO_BASE_URI="/logs"                 # serve UI + API under /logs (proxied by app-api)
export ZO_HTTP_PORT="$O2_PORT"
export ZO_TELEMETRY="false"

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$O2_PORT") 2>/dev/null && { exec 3>&-; return 0; } || return 1; }

install_o2() {
  if [ -x "$O2_BIN" ]; then echo "OpenObserve already installed at $O2_BIN"; return 0; fi
  mkdir -p "$O2_DIR"
  local arch; arch="$(uname -m)"
  case "$arch" in
    x86_64|amd64) arch=amd64 ;;
    aarch64|arm64) arch=arm64 ;;
    *) echo "ERROR: unsupported arch '$arch' for OpenObserve auto-install"; return 1 ;;
  esac
  local tarball="openobserve-${O2_VERSION}-linux-${arch}.tar.gz"
  local url="https://github.com/openobserve/openobserve/releases/download/${O2_VERSION}/${tarball}"
  echo "Downloading OpenObserve $O2_VERSION ($arch) ..."
  if ! curl -fSL "$url" -o "$O2_DIR/$tarball"; then
    echo "ERROR: download failed: $url"
    echo "       Set O2_VERSION to a valid release from https://github.com/openobserve/openobserve/releases and retry."
    return 1
  fi
  tar -xzf "$O2_DIR/$tarball" -C "$O2_DIR"
  rm -f "$O2_DIR/$tarball"
  [ -x "$O2_BIN" ] || { chmod +x "$O2_BIN" 2>/dev/null || true; }
  [ -x "$O2_BIN" ] && echo "OpenObserve installed -> $O2_BIN" || { echo "ERROR: binary not found after extract"; return 1; }
}

start_o2() {
  [ -x "$O2_BIN" ] || { echo "OpenObserve not installed — run: bash infra/observability.sh install (or setup.sh --observability)"; exit 1; }
  if port_open; then echo "observability already listening on :$O2_PORT"; exit 0; fi
  mkdir -p "$O2_DATA"
  nohup "$O2_BIN" >"$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 30); do port_open && break; sleep 1; done
  if port_open; then
    echo "OpenObserve up on :$O2_PORT (UI via app-api at /logs), log $LOG"
  else
    echo "ERROR: OpenObserve did not open :$O2_PORT — see $LOG"; exit 1
  fi
}

stop_o2() {
  [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null || true
  pkill -f "$O2_BIN" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "observability stopped"
}

case "$ACTION" in
  install) install_o2 ;;
  start)   start_o2 ;;
  stop)    stop_o2 ;;
  status)  port_open && echo "observability: up (:$O2_PORT)" || echo "observability: down" ;;
  *) echo "Usage: observability.sh <install|start|stop|status>"; exit 1 ;;
esac
