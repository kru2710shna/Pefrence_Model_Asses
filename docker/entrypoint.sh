#!/usr/bin/env bash
# Entrypoint for the sandbox container. Just exec whatever command was passed.
# The sandbox layer (sandbox.py) decides what to run — the container itself
# is dumb on purpose, so the same image works for "run judge", "run agent
# command", "diagnostic shell", etc.
set -euo pipefail
exec "$@"