#!/usr/bin/env python3
"""Train a zero-initialized candidate-wise four-path score residual on CIRR."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from weave_cirr_query_router import subset_target_ranks, target_ranks


class CandidateCompatibilityResidual(nn.Module):
    def __init__(self, hidden_width=32, residual_scale=0.1):
        super().__init__()
        self.residual_scale = residual_scale
        self.network = nn.Sequential(
            nn.Linear(4, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, path_scores):
        return path_scores[..., 3] + self.residual_scale * self.network(path_scores).squeeze(-1)


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_embeddings(path):
    path = Path(path)
    return {
        name: functional.normalize(torch.from_numpy(np.load(path / f"{name}.npy")).float(), dim=1)
        for name in ("base_gallery", "base_queries", "correction_gallery", "correction_queries")
    }


def metadata(path):
    path = Path(path)
    gallery_ids = json.loads((path / "gallery_ids.json").read_text())
    rows = load_jsonl(path / "queries.jsonl")
    index = {name: position for position, name in enumerate(gallery_ids)}
    sources = np.asarray([index[row["source_id"]] for row in rows], dtype=np.int64)
    targets = np.asarray([index[row["target_id"]] for row in rows], dtype=np.int64)
    groups = [np.asarray([index[name] for name in row["group_members"]], dtype=np.int64) for row in rows]
    return rows, sources, targets, groups


def hard_candidates(indices, targets, sources, hard, count):
    candidates = np.empty((len(indices), count + 1), dtype=np.int64)
    candidates[:, 0] = targets[indices]
    for row, query_index in enumerate(indices):
        negatives = [
            value for value in hard[query_index]
            if value != targets[query_index] and value != sources[query_index]
        ]
        if len(negatives) < count:
            raise ValueError("not enough hard negatives after source exclusion")
        candidates[row, 1:] = negatives[:count]
    return candidates


def group_candidates(indices, targets, sources, groups):
    width = max(len(groups[index]) - 1 for index in indices)
    candidates = np.zeros((len(indices), width), dtype=np.int64)
    mask = np.zeros((len(indices), width), dtype=bool)
    for row, query_index in enumerate(indices):
        target = targets[query_index]
        others = [
            value for value in groups[query_index]
            if value != sources[query_index] and value != target
        ]
        values = [target, *others]
        candidates[row, : len(values)] = values
        mask[row, : len(values)] = True
    return candidates, mask


def candidate_path_scores(arrays, indices, candidates, device):
    index_tensor = torch.as_tensor(indices, device=device)
    q0 = arrays["base_queries"][index_tensor]
    q1 = arrays["correction_queries"][index_tensor]
    candidate_tensor = torch.as_tensor(candidates, device=device)
    g0 = arrays["base_gallery"][candidate_tensor]
    g1 = arrays["correction_gallery"][candidate_tensor]
    return torch.stack(
        [
            torch.einsum("bd,bcd->bc", q0, g0),
            torch.einsum("bd,bcd->bc", q0, g1),
            torch.einsum("bd,bcd->bc", q1, g0),
            torch.einsum("bd,bcd->bc", q1, g1),
        ],
        dim=-1,
    )


def rank_metrics(global_ranks, subset_ranks):
    report = {
        f"R@{cutoff}": round(float(np.mean(global_ranks <= cutoff) * 100.0), 6)
        for cutoff in (1, 5, 10, 50)
    }
    report.update(
        {
            f"subset_R@{cutoff}": round(
                float(np.mean(subset_ranks <= cutoff) * 100.0), 6
            )
            for cutoff in (1, 2, 3)
        }
    )
    report["Avg"] = round(0.5 * (report["R@5"] + report["subset_R@1"]), 6)
    return report


@torch.no_grad()
def evaluate(model, arrays, sources, targets, groups, indices, device, batch_size):
    galleries = {
        name: arrays[name].to(device) for name in ("base_gallery", "correction_gallery")
    }
    all_base_global, all_base_subset = [], []
    all_model_global, all_model_subset = [], []
    max_group = max(len(groups[index]) for index in indices)
    model.eval()
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        selected_tensor = torch.as_tensor(selected, device=device)
        q0 = arrays["base_queries"][selected_tensor]
        q1 = arrays["correction_queries"][selected_tensor]
        paths = torch.stack(
            [
                q0 @ galleries["base_gallery"].T,
                q0 @ galleries["correction_gallery"].T,
                q1 @ galleries["base_gallery"].T,
                q1 @ galleries["correction_gallery"].T,
            ],
            dim=-1,
        )
        base_scores = paths[..., 3]
        model_scores = model(paths)
        batch_sources = torch.as_tensor(sources[selected], device=device)
        batch_targets = torch.as_tensor(targets[selected], device=device)
        rows = torch.arange(len(selected), device=device)
        base_scores[rows, batch_sources] = -torch.inf
        model_scores[rows, batch_sources] = -torch.inf
        batch_groups = np.full((len(selected), max_group), -1, dtype=np.int64)
        for row, query_index in enumerate(selected):
            batch_groups[row, : len(groups[query_index])] = groups[query_index]
        batch_groups = torch.as_tensor(batch_groups, device=device)
        for scores, global_store, subset_store in (
            (base_scores, all_base_global, all_base_subset),
            (model_scores, all_model_global, all_model_subset),
        ):
            global_store.append(target_ranks(scores, batch_targets).cpu().numpy())
            subset_store.append(
                subset_target_ranks(
                    scores, batch_groups, batch_targets, batch_sources
                ).cpu().numpy()
            )
    return {
        "base": rank_metrics(np.concatenate(all_base_global), np.concatenate(all_base_subset)),
        "model": rank_metrics(np.concatenate(all_model_global), np.concatenate(all_model_subset)),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-embedding-dir", required=True, type=Path)
    parser.add_argument("--partitions", required=True, type=Path)
    parser.add_argument("--hard-negatives", required=True, type=Path)
    parser.add_argument("--val-embedding-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hard-per-query", type=int, default=31)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--group-weight", type=float, default=1.0)
    parser.add_argument("--minimum-internal-gain", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    arrays = load_embeddings(args.train_embedding_dir)
    arrays = {name: values.to(device) for name, values in arrays.items()}
    _, sources, targets, groups = metadata(args.train_embedding_dir)
    partitions = np.load(args.partitions)
    hard = np.load(args.hard_negatives, mmap_mode="r")
    train_indices = np.flatnonzero(partitions == 0)
    development_indices = np.flatnonzero(partitions == 1)
    test_indices = np.flatnonzero(partitions == 2)
    model = CandidateCompatibilityResidual(
        args.hidden_width, args.residual_scale
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    best_state, best_avg, history = copy.deepcopy(model.state_dict()), -np.inf, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(train_indices)
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            global_candidates = hard_candidates(
                indices, targets, sources, hard, args.hard_per_query
            )
            group_values, group_mask = group_candidates(
                indices, targets, sources, groups
            )
            global_scores = model(
                candidate_path_scores(arrays, indices, global_candidates, device)
            )
            group_scores = model(
                candidate_path_scores(arrays, indices, group_values, device)
            )
            group_scores = group_scores.masked_fill(
                ~torch.as_tensor(group_mask, device=device), -torch.inf
            )
            labels = torch.zeros(len(indices), dtype=torch.long, device=device)
            loss = functional.cross_entropy(100.0 * global_scores, labels)
            loss = loss + args.group_weight * functional.cross_entropy(
                100.0 * group_scores, labels
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        development = evaluate(
            model,
            arrays,
            sources,
            targets,
            groups,
            development_indices,
            device,
            args.batch_size,
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "development": development,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if development["model"]["Avg"] > best_avg:
            best_avg = development["model"]["Avg"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    internal_test = evaluate(
        model, arrays, sources, targets, groups, test_indices, device, args.batch_size
    )
    gain = internal_test["model"]["Avg"] - internal_test["base"]["Avg"]
    passed = (
        gain >= args.minimum_internal_gain
        and internal_test["model"]["R@5"] >= internal_test["base"]["R@5"]
        and internal_test["model"]["subset_R@1"] >= internal_test["base"]["subset_R@1"]
    )
    report = {
        "history": history,
        "internal_test": internal_test,
        "internal_gain": round(gain, 6),
        "passed_internal_gate": passed,
        "minimum_internal_gain": args.minimum_internal_gain,
    }
    if passed and args.val_embedding_dir:
        val_arrays = load_embeddings(args.val_embedding_dir)
        val_arrays = {name: values.to(device) for name, values in val_arrays.items()}
        _, val_sources, val_targets, val_groups = metadata(args.val_embedding_dir)
        report["validation"] = evaluate(
            model,
            val_arrays,
            val_sources,
            val_targets,
            val_groups,
            np.arange(len(val_targets)),
            device,
            args.batch_size,
        )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden_width": args.hidden_width,
            "residual_scale": args.residual_scale,
        },
        args.output_dir / "best_model.pt",
    )
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
