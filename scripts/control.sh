#!/bin/bash
# Unified control script for the codex-proxy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Utility Functions ---
usage() {
    echo "Usage: $0 <command> [options]"
    echo
    echo "Commands:"
    echo "  start         - Start the proxy server."
    echo "  test          - Run the pytest test suite."
    echo "  lint          - Run ruff and mypy."
    echo "  format        - Format code with ruff."
    echo "  run           - Run a codex command through the proxy."
    echo
    echo "Options for 'run' command:"
    echo "  -p, --profile <name>  - Specify a codex profile to use."
    echo
    echo "Example:"
    echo "  $0 run -p glm -- 'hello'"
    exit 1
}

# --- Main Command Logic ---
COMMAND=${1:-}
if [[ -z "$COMMAND" ]]; then
    usage
fi
shift

cd "$PROJECT_ROOT"

case "$COMMAND" in
    start)
        echo "Starting codex-proxy..."
        uv run python -m codex_proxy
        ;;
    test)
        echo "Running tests..."
        uv run pytest tests/ -v
        ;;
    lint)
        echo "Running linters..."
        uv run ruff check src/ tests/
        uv run mypy src/ || true
        ;;
    format)
        echo "Formatting code..."
        uv run ruff format src/ tests/
        ;;
    run)
        PROFILE=""
        while [[ "$#" -gt 0 ]]; do
            case $1 in
                -p|--profile)
                    if [[ -n "${2:-}" ]]; then
                        PROFILE=$2
                        shift 2
                    else
                        echo "Error: --profile requires a value" >&2
                        exit 1
                    fi
                    ;;
                --)
                    shift
                    break
                    ;;
                *)
                    break
                    ;;
            esac
        done

        echo "Starting proxy in background..."
        uv run python -m codex_proxy &
        PROXY_PID=$!
        echo "Proxy PID: $PROXY_PID"
        echo "Waiting for proxy to be ready..."
        sleep 2

        echo "--------------------------------------------------------"
        echo "Running Codex..."
        if [[ -n "$PROFILE" ]]; then
            echo "Using profile: $PROFILE"
        fi
        echo "--------------------------------------------------------"

        if ! command -v codex &> /dev/null; then
            echo "Error: 'codex' command not found in PATH."
            kill "$PROXY_PID" 2>/dev/null || true
            exit 1
        fi

        CMD="codex exec"
        if [[ -n "$PROFILE" ]]; then
            CMD="$CMD -p $PROFILE"
        fi

        $CMD -- "$@" || true

        echo "--------------------------------------------------------"
        echo "Stopping proxy..."
        kill "$PROXY_PID" 2>/dev/null || true
        ;;
    *)
        usage
        ;;
esac
