#!/usr/bin/env python3
"""Dump CIRR train group members as gallery indices, for subset-aware training.

CIRR scores two things: global recall over the whole gallery, and subset recall
inside the six-image `img_set` group. Mining only global top-k confusions lifts
the former and slightly hurts the latter (observed: R@1 +0.45, subset R@1 -0.31).
The group members are the negatives the subset metric is actually defined over,
so they belong in the training pool alongside the global ones.

Writes train_group_members.npy, shape (Nq, 5) int32 -- group members with the
true target removed, padded by repetition when a group is short.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cirr-path", required=True, type=Path)
    parser.add_argument("--precomputed", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    captions = json.loads(
        (args.cirr_path / "cirr" / "captions" / "cap.rc2.train.json").read_text(
            encoding="utf-8"))
    gallery_ids = json.loads((args.precomputed / "train_gallery_ids.json").read_text())
    index = {name: i for i, name in enumerate(gallery_ids)}
    targets = json.loads((args.precomputed / "train_targets.json").read_text())

    width = 5
    out = np.zeros((len(targets), width), dtype=np.int32)
    short = 0
    for i in range(len(targets)):
        members = [index[m] for m in captions[i]["img_set"]["members"]
                   if m in index and index[m] != targets[i]]
        if not members:
            members = [targets[i]]
            short += 1
        while len(members) < width:
            members = members + members
        out[i] = np.asarray(members[:width], dtype=np.int32)

    np.save(args.precomputed / "train_group_members.npy", out)
    print(json.dumps({
        "queries": int(len(targets)),
        "members_per_query": width,
        "queries_with_no_usable_member": short,
    }, indent=2))


if __name__ == "__main__":
    main()
