#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -f .deps_installed ]; then
  pip install -q -r requirements.txt
  touch .deps_installed
fi

python judge.py