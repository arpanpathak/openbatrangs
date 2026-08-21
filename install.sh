#!/usr/bin/env bash
set -euo pipefail

# openBatarangs installer — builds the CLI and installs it to ~/.local/bin.
# Also checks for Ollama and starts it if needed.

BIN_DIR="${HOME}/.local/bin"
BIN="${BIN_DIR}/openbatrangs"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🦇 Installing openBatarangs..."

if ! command -v cargo >/dev/null 2>&1; then
  echo "❌ Rust/Cargo not found. Install it first: https://rustup.rs"
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "⬇️  Ollama not found. Installing it..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "🔄 Starting Ollama..."
  nohup ollama serve >/tmp/openbatrangs-ollama.log 2>&1 &
  sleep 2
fi

echo "🔨 Building release binary..."
cargo build --release --manifest-path "${REPO_DIR}/Cargo.toml"

mkdir -p "${BIN_DIR}"
cp "${REPO_DIR}/target/release/openbatrangs" "${BIN}"
chmod +x "${BIN}"

echo ""
echo "✅ Installed: ${BIN}"
echo "   Run it with: ${BIN}"
echo "   One-time setup (pulls a coding model): ${BIN} setup"
echo "   Add to PATH: export PATH=\"${BIN_DIR}:\$PATH\""
