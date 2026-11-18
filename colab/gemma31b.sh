#!/usr/bin/env bash
# Runs openbatrangs with Gemma 4 31B on a Colab A100 80 GB, from Colab's terminal.
#
#   curl -fsSL https://raw.githubusercontent.com/arpanpathak/openbatrangs/main/colab/gemma31b.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/arpanpathak/openbatrangs/main/colab/gemma31b.sh | bash -s -- https://github.com/you/your-repo
#
# The first run installs openbatrangs and vLLM, downloads the model (about 62 GB of
# 16-bit weights and the draft model), starts the server in the background with
# the weights converted to FP8 at load, and opens the terminal UI in
# /content/project. A later run finds the server already running and opens the
# terminal UI at once.
set -euo pipefail

REPO="${1:-}"
PROJECT="/content/project"
MODEL="google/gemma-4-31B-it"
DRAFT_MODEL="google/gemma-4-31B-it-assistant"
SERVED_NAME="gemma-4-31b"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
SERVER="http://localhost:8000/v1"
LOG="/content/vllm.log"
PID_FILE="/content/vllm.pid"

# Whether the vLLM server answers.
server_up() {
    curl -fsS "$SERVER/models" > /dev/null 2>&1
}

# Whether the vLLM process started by an earlier run is still alive.
server_running() {
    [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2> /dev/null
}

# Installs openbatrangs and vLLM when they are missing.
install_tools() {
    if ! command -v openbatrangs > /dev/null; then
        curl -fsSL https://github.com/arpanpathak/openbatrangs/releases/latest/download/install.sh | sh
    fi

    if ! command -v vllm > /dev/null; then
        echo "Installing vLLM. This takes a few minutes."
        pip install -q vllm
    fi
}

# Starts vLLM in its own session, so it keeps running after this script ends.
start_server() {
    local draft="{\"method\": \"mtp\", \"model\": \"$DRAFT_MODEL\", \"num_speculative_tokens\": 3}"

    echo "Starting vLLM. The first start downloads the model, which takes 10 to 20 minutes."
    echo "Follow it with: tail -f $LOG"

    # FP8 weights take 31 GB instead of 62 GB, which leaves the rest of the GPU
    # for the KV cache and halves the bytes read for every generated token.
    setsid nohup vllm serve "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --quantization fp8 \
        --speculative-config "$draft" \
        --max-model-len "$MAX_MODEL_LEN" \
        --gpu-memory-utilization 0.95 \
        --enable-chunked-prefill --enable-prefix-caching \
        --reasoning-parser gemma4 \
        --enable-auto-tool-choice --tool-call-parser gemma4 \
        > "$LOG" 2>&1 < /dev/null &

    echo $! > "$PID_FILE"
}

# Waits until the server answers, or stops with the end of the log if vLLM exits.
wait_for_server() {
    until server_up; do
        if ! server_running; then
            tail -n 30 "$LOG"
            echo "vLLM stopped; the lines above are the end of $LOG."
            exit 1
        fi

        sleep 10
    done

    echo "Gemma 4 31B is ready."
}

# Clones REPO into the project folder, or creates an empty one when none is given.
prepare_project() {
    if [ -d "$PROJECT" ]; then
        return
    fi

    if [ -n "$REPO" ]; then
        git clone "$REPO" "$PROJECT"
        return
    fi

    mkdir -p "$PROJECT"
}

install_tools

# A server still downloading from an earlier run is waited for, not started twice.
if ! server_up; then
    server_running || start_server
    wait_for_server
fi

prepare_project
cd "$PROJECT"

# Under `curl | bash`, standard input is the script, so the terminal UI reads the terminal.
exec openbatrangs --no-sandbox --no-update-check --openai-url "$SERVER" --model "$SERVED_NAME" < /dev/tty
