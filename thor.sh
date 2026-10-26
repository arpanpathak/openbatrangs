#!/bin/sh
#
# thor.sh — one-line setup for a Jetson Thor model endpoint.
#
#   curl -fsSL https://github.com/arpanpathak/openbatrangs/releases/latest/download/thor.sh | sh
#
# It installs the openBatarangs binary for this machine, asks for the API key
# once, writes ~/.config/thor-chat/api-key, installs a ready-to-run
# `openbatrangs-thor` wrapper pointed at the Thor, and verifies the connection.
# No other steps.
#
# Env knobs: THOR_URL, THOR_KEY (skip the prompt), THOR_MODEL.
#
set -eu

REPO="arpanpathak/openbatrangs"
VERSION="${OPENBATRANGS_VERSION:-latest}"
THOR_URL="${THOR_URL:-https://arpanpathak.taildb9a39.ts.net}"
THOR_MODEL="${THOR_MODEL:-NVIDIA-Nemotron-3-Nano-30B-A3B-Q8_0}"
KEY_FILE="${HOME}/.config/thor-chat/api-key"
BIN_DIR="${HOME}/.local/bin"
BIN="${BIN_DIR}/openbatrangs"

say() { printf '%s\n' "$*"; }
fail() { printf '\n%s\n' "✗ $*" >&2; exit 1; }

# --- Work out which prebuilt binary this machine needs ----------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
case "${OS}" in
  Darwin) SUFFIX="apple-darwin" ;;
  Linux)  SUFFIX="unknown-linux-gnu" ;;
  *) fail "Unsupported OS: ${OS}. Build from source instead: cargo install --git https://github.com/${REPO} --locked" ;;
esac
case "${ARCH}" in
  arm64|aarch64) CPU="aarch64" ;;
  x86_64|amd64)  CPU="x86_64" ;;
  *) fail "Unsupported CPU: ${ARCH}" ;;
esac
TARGET="${CPU}-${SUFFIX}"

say "🦇 openBatarangs → ${THOR_URL}"
say "   machine: ${OS} ${ARCH} (${TARGET})"

# --- Install the binary ----------------------------------------------------
mkdir -p "${BIN_DIR}"
URL="https://github.com/${REPO}/releases/${VERSION}/download/openbatrangs-${TARGET}"
if ! curl -fsSL -o "${BIN}.new" "${URL}"; then
  fail "Download failed: ${URL}"
fi
chmod +x "${BIN}.new"
mv "${BIN}.new" "${BIN}"
VERSION_OUT="$( "${BIN}" --version 2>/dev/null || echo unknown )"
say "   installed: ${BIN} (${VERSION_OUT})"

# --- Store the API key -----------------------------------------------------
mkdir -p "$(dirname "${KEY_FILE}")"
chmod 700 "$(dirname "${KEY_FILE}")"

if [ -z "${THOR_KEY:-}" ] && [ -s "${KEY_FILE}" ]; then
  say "   key: reusing ${KEY_FILE}"
fi

if [ -z "${THOR_KEY:-}" ] && [ ! -s "${KEY_FILE}" ]; then
  if [ -r /dev/tty ]; then
    printf '\n   Paste the API key, then press return: ' >&2
    # -s keeps it off screen; /dev/tty is used because stdin is the script
    # itself when this is piped from curl.
    read -r THOR_KEY < /dev/tty || fail "No key entered."
    printf '\n' >&2
  else
    fail "No terminal to read the key from. Set it and re-run:  THOR_KEY=<key> sh -c \"\$(curl -fsSL https://github.com/${REPO}/releases/latest/download/thor.sh)\""
  fi
fi

if [ -n "${THOR_KEY:-}" ]; then
  printf '%s' "${THOR_KEY}" > "${KEY_FILE}"
fi
chmod 600 "${KEY_FILE}"
[ -s "${KEY_FILE}" ] || fail "Key file is empty: ${KEY_FILE}"

# Trim a stray trailing newline a paste can carry.
KEY="$(tr -d '\r\n' < "${KEY_FILE}")"
printf '%s' "${KEY}" > "${KEY_FILE}"

# --- Install a wrapper so the URL and model are never typed again ----------
cat > "${BIN_DIR}/openbatrangs-thor" <<WRAPPER
#!/bin/sh
# openbatrangs-thor — installed by thor.sh. Routes to the Thor endpoint.
exec "${BIN}" \\
  --openai-url "${THOR_URL}/v1" \\
  --api-key-file "${KEY_FILE}" \\
  --model "\${THOR_MODEL:-${THOR_MODEL}}" \\
  "\$@"
WRAPPER
chmod +x "${BIN_DIR}/openbatrangs-thor"
say "   command:  openbatrangs-thor"

# --- Make sure ~/.local/bin is on PATH -------------------------------------
case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *)
    RC="${HOME}/.zshrc"
    [ -f "${HOME}/.bashrc" ] && [ ! -f "${RC}" ] && RC="${HOME}/.bashrc"
    if ! grep -q "${BIN_DIR}" "${RC}" 2>/dev/null; then
      printf 'export PATH="%s:$PATH"\n' "${BIN_DIR}" >> "${RC}"
      say "   added ${BIN_DIR} to ${RC} (restart your terminal)"
    fi
    ;;
esac

# --- Verify the endpoint actually answers ----------------------------------
say ""
say "🔌 checking ${THOR_URL} ..."
CODE="$(curl -s -o /tmp/thor-models.$$ -w '%{http_code}' -m 20 \
  -H "Authorization: Bearer ${KEY}" "${THOR_URL}/v1/models" || echo 000)"
case "${CODE}" in
  200)
    say "✓ connected. models available:"
    # No jq dependency: pull "id" values out of the JSON by hand.
    tr ',' '\n' < /tmp/thor-models.$$ | sed -n 's/.*"id": *"\([^"]*\)".*/    • \1/p' | sort -u
    ;;
  401|403) rm -f /tmp/thor-models.$$; fail "Key rejected (HTTP ${CODE}). Check ~/.config/thor-chat/api-key" ;;
  000) rm -f /tmp/thor-models.$$; fail "Could not reach ${THOR_URL}. The Thor may be off or rebooting." ;;
  *) rm -f /tmp/thor-models.$$; fail "Unexpected HTTP ${CODE} from ${THOR_URL}/v1/models" ;;
esac
rm -f /tmp/thor-models.$$

cat <<DONE

✅ Done. Try it:

   openbatrangs-thor                              # interactive
   openbatrangs-thor "add tests for src/main.rs"  # one-shot task

   openbatrangs-thor --model NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB   # bigger model

Your own harness works too — same endpoint, same key file:

   ANTHROPIC_BASE_URL=${THOR_URL}                 ANTHROPIC_AUTH_TOKEN="\$(cat ${KEY_FILE})"
   OPENAI_BASE_URL=${THOR_URL}/v1                 OPENAI_API_KEY="\$(cat ${KEY_FILE})"

DONE
