#!/usr/bin/env bash
# run.sh — video-chunker launcher
# Sets up venv on first run, prompts for API keys if .env is missing

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
ENV_FILE="$SCRIPT_DIR/.env"

# ── 1. First-run: prompt for API keys ────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo ""
  echo "┌─────────────────────────────────────────────┐"
  echo "│  video-chunker — first-run setup             │"
  echo "└─────────────────────────────────────────────┘"
  echo ""
  echo "Enter your API keys (input is hidden):"
  echo ""

  read -rsp "  OpenAI API key  (for Whisper transcription): " OPENAI_KEY
  echo ""
  read -rsp "  DeepSeek API key (for chunk analysis)       : " DEEPSEEK_KEY
  echo ""
  echo ""

  cat > "$ENV_FILE" <<EOF
# video-chunker environment variables
OPENAI_API_KEY=${OPENAI_KEY}
DEEPSEEK_API_KEY=${DEEPSEEK_KEY}
EOF

  echo "✓ .env saved to $ENV_FILE"
  echo ""
fi

# ── 2. Load .env into environment ─────────────────────────────────────────────
set -o allexport
# shellcheck disable=SC1090
source "$ENV_FILE"
set +o allexport

# ── 3. Create venv if it doesn't exist ───────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment at $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
  echo "Installing video-chunker ..."
  "$VENV_DIR/bin/pip" install --quiet --upgrade pip
  "$VENV_DIR/bin/pip" install --quiet -e "$SCRIPT_DIR"
  echo "✓ venv ready"
  echo ""
fi

# ── 4. Run ────────────────────────────────────────────────────────────────────
exec "$VENV_DIR/bin/video-chunker" "$@"
