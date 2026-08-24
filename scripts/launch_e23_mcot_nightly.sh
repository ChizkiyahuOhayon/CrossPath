#!/usr/bin/env bash
set -euo pipefail

E23_ROOT="/root/autodl-tmp/mcot_mvs"
E23_ENV="/root/autodl-tmp/envs/mcot_official_py310"
E23_INSTALL_PID="${1:-}"

if [[ -n "$E23_INSTALL_PID" ]]; then
  while kill -0 "$E23_INSTALL_PID" 2>/dev/null; do
    sleep 20
  done
fi

if [[ ! -x "$E23_ENV/bin/python" ]]; then
  /usr/bin/python3.10 -m venv "$E23_ENV"
fi

if ! "$E23_ENV/bin/python" -c \
  'import open_clip, pandas, supervision, torch; from PIL import Image' \
  >/dev/null 2>&1; then
  "$E23_ENV/bin/pip" install -r "$E23_ROOT/requirements.txt"
fi

"$E23_ENV/bin/pip" check
exec /root/miniconda3/bin/python \
  "$E23_ROOT/scripts/run_e23_mcot_official_env.py"
