#!/bin/bash
# 启动检索系统：必须用 system_venv 里的解释器（装了 flask/torch/open_clip），
# 直接用系统自带的 python3 会因为缺依赖启动失败。
cd "$(dirname "$0")"
exec ../system_venv/bin/python3 app.py
