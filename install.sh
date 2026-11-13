#!/bin/sh
# Install or update openbatrangs from the latest GitHub release.
#
#   curl -fsSL https://github.com/arpanpathak/openbatrangs/releases/latest/download/install.sh | sh
#
# OPENBATRANGS_VERSION      release tag to install, such as v0.3.0 (default: latest)
# OPENBATRANGS_INSTALL_DIR  folder for the binary (default: ~/.local/bin, or
#                           /usr/local/bin for root)
set -eu

REPO="arpanpathak/openbatrangs"
VERSION="${OPENBATRANGS_VERSION:-latest}"

if [ "${VERSION}" = "latest" ]; then
  BASE_URL="https://github.com/${REPO}/releases/latest/download"
else
  BASE_URL="https://github.com/${REPO}/releases/download/${VERSION}"
fi

fail() {
  printf '\033[31merror:\033[0m %s\n' "$1" >&2
  exit 1
}

step() {
  printf '\033[36m==>\033[0m %s\n' "$1"
}

# The release target for this system, such as aarch64-unknown-linux-gnu.
target() {
  case "$(uname -s)" in
    Linux) suffix="unknown-linux-gnu" ;;
    Darwin) suffix="apple-darwin" ;;
    *) fail "no release binary for $(uname -s); build from source with cargo" ;;
  esac

  case "$(uname -m)" in
    aarch64 | arm64) echo "aarch64-${suffix}" ;;
    x86_64 | amd64) echo "x86_64-${suffix}" ;;
    *) fail "no release binary for $(uname -m); build from source with cargo" ;;
  esac
}

# The folder for the binary: the override, /usr/local/bin for root, or ~/.local/bin.
install_dir() {
  if [ -n "${OPENBATRANGS_INSTALL_DIR:-}" ]; then
    echo "${OPENBATRANGS_INSTALL_DIR}"
    return
  fi

  if [ "$(id -u)" -eq 0 ]; then
    echo "/usr/local/bin"
    return
  fi

  echo "${HOME}/.local/bin"
}

# The SHA-256 digest of a file, with whichever tool this system has.
digest() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
    return
  fi

  shasum -a 256 "$1" | cut -d' ' -f1
}

# Compare the download with its line in the release's SHA256SUMS.
verify() {
  expected="$(curl -fsSL "${BASE_URL}/SHA256SUMS" | grep "openbatrangs-$2\$" | cut -d' ' -f1)"
  [ -n "${expected}" ] || fail "SHA256SUMS has no entry for openbatrangs-$2"

  actual="$(digest "$1")"
  [ "${actual}" = "${expected}" ] || fail "checksum mismatch for openbatrangs-$2"
}

command -v curl >/dev/null 2>&1 || fail "curl is required"

TARGET="$(target)"
BIN_DIR="$(install_dir)"
TMP_BIN="$(mktemp)"
trap 'rm -f "${TMP_BIN}"' EXIT HUP INT TERM

step "Downloading openbatrangs for ${TARGET}"
curl -fsSL -o "${TMP_BIN}" "${BASE_URL}/openbatrangs-${TARGET}" ||
  fail "download failed from ${BASE_URL}"

step "Verifying the SHA-256 checksum"
verify "${TMP_BIN}" "${TARGET}"

step "Installing to ${BIN_DIR}/openbatrangs"
mkdir -p "${BIN_DIR}"
chmod 755 "${TMP_BIN}"
mv "${TMP_BIN}" "${BIN_DIR}/openbatrangs"
trap - EXIT HUP INT TERM

case ":${PATH}:" in
  *":${BIN_DIR}:"*) ;;
  *) printf '\n%s is not on your PATH. Add it with:\n  export PATH="%s:$PATH"\n' "${BIN_DIR}" "${BIN_DIR}" ;;
esac

printf '\n\033[32mopenbatrangs %s is installed.\033[0m Connect it to your model server:\n' \
  "$("${BIN_DIR}/openbatrangs" --version | cut -d' ' -f2)"
printf '  openbatrangs --openai-url http://127.0.0.1:8000/v1     # TensorRT-LLM, vLLM, SGLang, llama.cpp\n'
printf '  openbatrangs                                            # Ollama on localhost:11434\n'
