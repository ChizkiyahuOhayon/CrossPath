#!/usr/bin/env python3
"""Run the E23 FashionIQ reproduction in the official MCoT-MVS environment."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/root/autodl-tmp/mcot_mvs")
OFFICIAL_PYTHON = Path("/root/autodl-tmp/envs/mcot_official_py310/bin/python")
RUN_ROOT = PROJECT_ROOT / "runs/E23_MCoT_MVS_FashionIQ_official_env_20260824_v1"
CATEGORIES = ("dress", "shirt", "toptee")
TARGETS = {
    "dress": (58.45, 78.92),
    "shirt": (63.24, 81.15),
    "toptee": (68.02, 85.97),
    "mean": (63.24, 82.01),
}


def parse_metrics(log_path):
    for line in reversed(log_path.read_text(encoding="utf-8").splitlines()):
        try:
            values = ast.literal_eval(line.strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(values, tuple) and len(values) == 4:
            return tuple(float(value) for value in values)
    raise RuntimeError(f"no four-value result tuple found in {log_path}")


def record_environment():
    code = """
import importlib.metadata
import json
import platform
import torch
from PIL import __version__ as pillow_version

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "open_clip_torch": importlib.metadata.version("open-clip-torch"),
    "pillow": pillow_version,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
}, indent=2))
"""
    completed = subprocess.run(
        [str(OFFICIAL_PYTHON), "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    (RUN_ROOT / "environment.json").write_text(completed.stdout, encoding="utf-8")


def run_category(category):
    log_path = RUN_ROOT / f"{category}.log"
    done_path = RUN_ROOT / f"{category}.done.json"
    if done_path.is_file():
        metrics = parse_metrics(log_path)
        print(f"[{category}] already complete: {metrics[:2]}", flush=True)
        return metrics

    command = [
        str(OFFICIAL_PYTHON),
        "fiq_validate.py",
        "--model_path",
        f"../checkpoints/mcot_mvs_{category}.pt",
        "--dress_type",
        category,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HOME": "/root/autodl-tmp/cache/hf",
            "PYTHONUNBUFFERED": "1",
        }
    )
    print(f"[{category}] starting", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT / "src",
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{category} failed with exit code {return_code}")

    metrics = parse_metrics(log_path)
    done_path.write_text(
        json.dumps(
            {
                "category": category,
                "exit_code": return_code,
                "val_r10": metrics[0],
                "val_r50": metrics[1],
                "original_r10": metrics[2],
                "original_r50": metrics[3],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def write_summary(results):
    categories = {}
    for category, metrics in results.items():
        target = TARGETS[category]
        deltas = (metrics[0] - target[0], metrics[1] - target[1])
        categories[category] = {
            "val_r10": metrics[0],
            "val_r50": metrics[1],
            "target_r10": target[0],
            "target_r50": target[1],
            "delta_r10": deltas[0],
            "delta_r50": deltas[1],
            "within_0.25": abs(deltas[0]) <= 0.25 and abs(deltas[1]) <= 0.25,
        }

    mean = tuple(
        sum(results[category][index] for category in CATEGORIES) / len(CATEGORIES)
        for index in (0, 1)
    )
    mean_delta = (mean[0] - TARGETS["mean"][0], mean[1] - TARGETS["mean"][1])
    summary = {
        "protocol": "FashionIQ val-split; official checkpoints and fiq_validate.py",
        "categories": categories,
        "mean": {
            "val_r10": mean[0],
            "val_r50": mean[1],
            "target_r10": TARGETS["mean"][0],
            "target_r50": TARGETS["mean"][1],
            "delta_r10": mean_delta[0],
            "delta_r50": mean_delta[1],
            "within_0.15": abs(mean_delta[0]) <= 0.15 and abs(mean_delta[1]) <= 0.15,
        },
    }
    summary["success"] = summary["mean"]["within_0.15"] and all(
        item["within_0.25"] for item in categories.values()
    )
    (RUN_ROOT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main():
    if not OFFICIAL_PYTHON.is_file():
        raise FileNotFoundError(f"official Python not found: {OFFICIAL_PYTHON}")
    for category in CATEGORIES:
        checkpoint = PROJECT_ROOT / f"checkpoints/mcot_mvs_{category}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    record_environment()
    results = {category: run_category(category) for category in CATEGORIES}
    write_summary(results)


if __name__ == "__main__":
    main()
