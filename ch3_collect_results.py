"""Collect the chapter-3 tables from the training logs.

Every number the thesis reports has to be traceable to an artifact, so this
reads ``train_log.jsonl`` and ``run_manifest.json`` from each run directory and
emits both the tables and, for each cell, the file and epoch it came from.

The reported epoch is the **last** one, not the best one: the validation split
is also the evaluation split, so picking the best epoch would select on the
numbers being reported.  Every epoch stays in the log for inspection.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

#: run directory name -> the row it fills
TABLE_3_2 = [("m1", "Baseline†"), ("m5", "本章模型†")]
TABLE_3_5 = [("m1", "1"), ("m2", "2"), ("m3", "3"), ("m4", "4"), ("m5", "5")]
TABLE_3_6 = [("L12", "L12"), ("L8", "L8"), ("L4_L8", "L4+L8"),
             ("L8_L12", "L8+L12"), ("m5", "L4+L8+L12")]
FIGURE_3_6 = [("m4", 0.0), ("lam0_1", 0.1), ("lam0_3", 0.3), ("m5", 0.5), ("lam1_0", 1.0)]


class Run:
    """One finished (or in-progress) training run on disk."""

    def __init__(self, root: str, name: str):
        self.name = name
        self.dir = os.path.join(root, name)
        self.manifest = self._load_json("run_manifest.json")
        self.epochs = self._load_epochs()

    def _load_json(self, name: str) -> Optional[dict]:
        path = os.path.join(self.dir, name)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _load_epochs(self) -> List[dict]:
        path = os.path.join(self.dir, "train_log.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    @property
    def last(self) -> Optional[dict]:
        scored = [e for e in self.epochs if "metrics" in e]
        return scored[-1] if scored else None

    def metrics(self, protocol: str) -> Optional[dict]:
        last = self.last
        return last["metrics"].get(protocol) if last else None

    def provenance(self, protocol: str) -> dict:
        return {"run": self.name,
                "log": os.path.join(self.dir, "train_log.jsonl"),
                "epoch": self.last["epoch"] if self.last else None,
                "protocol": protocol,
                "code_sha256": (self.manifest or {}).get("code_sha256"),
                "init": ((self.manifest or {}).get("init") or {}).get("checkpoint")}


def _row(run: Run, protocol: str, keys=("R@1", "R@5", "R@10")) -> Optional[List[float]]:
    m = run.metrics(protocol)
    if not m:
        return None
    return ([round(m["I2T"][k], 2) for k in keys]
            + [round(m["T2I"][k], 2) for k in keys]
            + [round(m["Mean R@1"], 2)])


def build(root: str) -> dict:
    runs: Dict[str, Run] = {}

    def run(name: str) -> Run:
        if name not in runs:
            runs[name] = Run(root, name)
        return runs[name]

    out: dict = {"tables": {}, "provenance": {}, "missing": []}

    def collect(table: str, rows, protocol: str, labels=("R@1", "R@5", "R@10")):
        body = []
        for name, label in rows:
            r = run(name)
            values = _row(r, protocol, labels)
            if values is None:
                out["missing"].append(f"{table}: {name}")
                continue
            body.append({"label": label, "values": values})
            out["provenance"].setdefault(table, []).append(r.provenance(protocol))
        out["tables"][table] = body

    collect("table_3_2_sample", TABLE_3_2, "sample")
    collect("table_3_3_full", [("m5", "本章模型")], "full")
    collect("table_3_5_ablation", TABLE_3_5, "sample", ("R@1",))
    collect("table_3_6_layers", TABLE_3_6, "sample", ("R@1",))
    collect("figure_3_6_lambda",
            [(name, str(w)) for name, w in FIGURE_3_6], "sample", ("R@1",))
    return out


def render(out: dict) -> str:
    lines = []
    for table, rows in out["tables"].items():
        lines.append(f"\n## {table}")
        if not rows:
            lines.append("  (no completed run yet)")
            continue
        width = len(rows[0]["values"])
        header = (["I2T R@1", "I2T R@5", "I2T R@10", "T2I R@1", "T2I R@5", "T2I R@10", "Mean R@1"]
                  if width == 7 else ["I2T R@1", "T2I R@1", "Mean R@1"])
        lines.append("  " + "  ".join(f"{h:>9s}" for h in ["row"] + header))
        for r in rows:
            cells = "  ".join(f"{v:9.2f}" for v in r["values"])
            lines.append(f"  {r['label']:>9s}  {cells}")
    if out["missing"]:
        lines.append("\n## not finished yet\n  " + "\n  ".join(out["missing"]))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/root/autodl-tmp/ch3/runs")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    out = build(args.runs)
    print(render(out))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
