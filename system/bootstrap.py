#!/usr/bin/env python3
"""One-command setup + launch for the retrieval demo system.

Works the same on macOS/Linux/Windows with nothing but a system Python 3.9+:
creates the virtualenv, installs dependencies, builds the (real or synthetic
placeholder) catalog and trains the two lightweight endpoint heads + gate the
first time it's run, then starts the Flask server and opens it in a browser.
Safe to re-run — every step is skipped if its output already exists, so the
second run just starts the server in a couple of seconds.

Usage:
    python3 bootstrap.py            # setup (first run) + start the server
    python3 bootstrap.py --rebuild  # force rebuilding the catalog/heads/gate
    python3 bootstrap.py --no-browser
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import venv
import webbrowser
from pathlib import Path

SYSTEM = Path(__file__).resolve().parent
REPO = SYSTEM.parent
VENV = REPO / "system_venv"
URL = "http://127.0.0.1:5057"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python3"


def run(cmd: list[str], **kw) -> None:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=SYSTEM, **kw)


def ensure_venv() -> None:
    if venv_python().exists():
        return
    print(f"[1/4] creating virtual environment at {VENV} ...")
    venv.EnvBuilder(with_pip=True).create(VENV)


def ensure_deps() -> None:
    marker = VENV / ".deps_installed"
    if marker.exists():
        return
    print("[2/4] installing dependencies (first run only, a couple of minutes) ...")
    run([str(venv_python()), "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
    run([str(venv_python()), "-m", "pip", "install", "--quiet", "-r",
         str(SYSTEM / "requirements.txt")])
    marker.write_text("ok")


def ensure_catalog(force: bool) -> None:
    db = SYSTEM / "data" / "crosspath.db"
    if db.exists() and not force:
        return
    print("[3/4] building catalog + training endpoint heads + gate (first run "
          "only) ...")
    py = str(venv_python())
    run([py, "db/build_db.py"])
    run([py, "tools/train_endpoints.py"])
    run([py, "tools/train_gate.py"])


def start_server(open_browser: bool) -> None:
    print(f"[4/4] starting the server at {URL} (Ctrl+C to stop) ...")
    if open_browser:
        try:
            webbrowser.open(URL)
        except Exception:
            pass
    subprocess.run([str(venv_python()), "app.py"], check=True, cwd=SYSTEM)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                     help="重新建库、重新训练端点和门控，即使已经有产物了")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args()

    ensure_venv()
    ensure_deps()
    ensure_catalog(force=args.rebuild)
    start_server(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
