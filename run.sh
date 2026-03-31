#!/usr/bin/env bash
# run.sh — video-chunker launcher
# Sets up venv on first run, prompts for API keys if .env is missing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
ENV_FILE="$SCRIPT_DIR/.env"
MARKER="$VENV_DIR/.install_ok"

# ── 1. First-run: prompt for API keys ────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo ""
  echo "┌─────────────────────────────────────────────┐"
  echo "│  video-chunker — first-run setup             │"
  echo "└─────────────────────────────────────────────┘"
  echo ""
  echo "Transcription uses local Whisper by default (free, no API key needed)."
  echo "DeepSeek is used for chunk analysis (recommended, cheap)."
  echo "OpenAI key is only needed if you want to use --whisper-mode openai."
  echo ""

  read -rsp "  DeepSeek API key (for chunk analysis, required): " DEEPSEEK_KEY
  echo ""
  read -rsp "  OpenAI API key   (optional, only for --whisper-mode openai): " OPENAI_KEY
  echo ""
  echo ""

  cat > "$ENV_FILE" <<EOF
# video-chunker environment variables
DEEPSEEK_API_KEY=${DEEPSEEK_KEY}
OPENAI_API_KEY=${OPENAI_KEY}
EOF

  echo "✓ .env saved to $ENV_FILE"
  echo ""
fi

# ── 2. Load .env into environment ─────────────────────────────────────────────
set -o allexport
# shellcheck disable=SC1090
source "$ENV_FILE"
set +o allexport

# ── 3. Create venv and install if not already done ───────────────────────────
if [[ ! -f "$MARKER" ]]; then
  echo "Setting up virtual environment at $VENV_DIR ..."
  # Remove stale venv if it exists
  [[ -d "$VENV_DIR" ]] && rm -rf "$VENV_DIR"
  # Prefer Homebrew Python 3.12+ over system Python (macOS ships 3.9)
  PYTHON3=""
  for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
      version=$("$candidate" -c "import sys; print(sys.version_info >= (3,10))")
      if [[ "$version" == "True" ]]; then
        PYTHON3="$candidate"
        break
      fi
    fi
  done
  if [[ -z "$PYTHON3" ]]; then
    echo "Error: Python 3.10+ is required. Install it via: brew install python@3.12"
    exit 1
  fi
  echo "Using $PYTHON3 ($(${PYTHON3} --version))"
  "$PYTHON3" -m venv "$VENV_DIR"
  echo "Installing video-chunker (this may take a minute on first run)..."
  "$VENV_DIR/bin/pip" install --quiet --upgrade pip
  "$VENV_DIR/bin/pip" install --quiet -e "$SCRIPT_DIR"
  touch "$MARKER"
  echo "✓ Ready"
  echo ""
fi

# ── 4. Run ────────────────────────────────────────────────────────────────────
exec "$VENV_DIR/bin/video-chunker" "$@"
