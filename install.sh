#!/bin/sh
set -eu

# openBatarangs installer/updater
#   Install:  curl -fsSL https://github.com/arpanpathak/openbatrangs/releases/latest/download/install.sh | sh
#   Update:   same command (overwrites the old binary)
#
# Downloads the prebuilt release binary and installs it to ~/.local/bin
# (or /usr/local/bin when run as root). Also ensures Ollama is present.

REPO="arpanpathak/openbatrangs"
VERSION="${OPENBATRANGS_VERSION:-latest}"
BASE_URL="https://github.com/${REPO}/releases/${VERSION}/download"

# --- Detect target ---------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"

case "${ARCH}" in
  aarch64|arm64) TARGET="aarch64-unknown-linux-gnu" ;;
  *)
    echo "❌ Unsupported architecture for the prebuilt binary: ${ARCH}"
    echo "   Prebuilt releases currently target aarch64 (Jetson/ARM64)."
    echo "   To build from source instead:"
    echo "     git clone https://github.com/arpanpathak/openbatrangs.git"
    echo "     cd openbatrangs && cargo build --release"
    exit 1
    ;;
esac

case "${OS}" in
  Linux) ;;
  Darwin)
    echo "❌ macOS is not supported by the prebuilt release yet."
    exit 1
    ;;
  *)
    echo "❌ Unsupported OS: ${OS}"
    exit 1
    ;;
esac

# --- Install location ------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
  BIN_DIR="/usr/local/bin"
else
  BIN_DIR="${HOME}/.local/bin"
fi

BIN="${BIN_DIR}/openbatrangs"

# --- Download --------------------------------------------------------------
echo "🦇 Installing/updating openBatarangs (${TARGET})..."
mkdir -p "${BIN_DIR}"
TMP_BIN="$(mktemp)"

echo "⬇️  Downloading ${BASE_URL}/openbatrangs-${TARGET} ..."
if ! curl -fsSL -o "${TMP_BIN}" "${BASE_URL}/openbatrangs-${TARGET}"; then
  echo "❌ Download failed. Check your network or GitHub availability."
  rm -f "${TMP_BIN}"
  exit 1
fi

chmod +x "${TMP_BIN}"
mv "${TMP_BIN}" "${BIN}"

echo "✅ Installed: ${BIN}"

# --- PATH ------------------------------------------------------------------
case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *)
    echo ""
    echo "⚠️  ${BIN_DIR} is not on your PATH yet."
    if [ "$(id -u)" -ne 0 ]; then
      RC_FILE="${HOME}/.bashrc"
      if [ -f "${HOME}/.zshrc" ]; then
        RC_FILE="${HOME}/.zshrc"
      fi
      if ! grep -q "${BIN_DIR}" "${RC_FILE}" 2>/dev/null; then
        echo "export PATH=\"${BIN_DIR}:\$PATH\"" >> "${RC_FILE}"
        echo "   Added to ${RC_FILE}. Restart your shell or run: export PATH=\"${BIN_DIR}:\$PATH\""
      fi
    else
      echo "   As root, it is already in a standard PATH location."
    fi
    ;;
esac

# --- Ollama ----------------------------------------------------------------
if ! command -v ollama >/dev/null 2>&1; then
  echo ""
  echo "⬇️  Ollama not found. Installing it with the official script..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "🔄 Starting Ollama..."
  nohup ollama serve >/tmp/openbatrangs-ollama.log 2>&1 &
  sleep 2
fi

echo ""
echo "🔄 Running first-time setup (pulls a coding model)..."
"${BIN}" setup || {
  echo "⚠️  First-time setup did not complete."
  echo "   You can retry it later with: ${BIN} setup"
}

echo ""
echo "🎉 Done!"
echo "   Run:    ${BIN}"
echo "   Update: curl -fsSL https://github.com/${REPO}/releases/latest/download/install.sh | sh"
