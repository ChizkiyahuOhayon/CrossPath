#!/bin/bash
# 一键启动（macOS / Linux）：第一次跑会自动建虚拟环境、装依赖、建库、训练，
# 之后再跑就是直接起服务，几秒钟的事。双击 start.command 效果一样。
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python)
if [ -z "$PY" ]; then
  echo "没找到 python3，请先安装 Python 3.9 及以上版本：https://www.python.org/downloads/"
  exit 1
fi
exec "$PY" bootstrap.py "$@"
