#!/usr/bin/env python3
"""Build sharded CrossPath boundary records from aligned endpoint embeddings."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from weave_crosspath import RankPath, boundary_trace, build_rank_path


DEFAULT_KS = (1, 5, 10)
DEFAULT_ALPHAS = np.linspace(0.0, 1.0, 9)


def compose_correction_scores(path, score00, score01=None, score10=None, score11=None):
    if path in ("matched", "joint"):
        return score11
    if path == "cross-mean":
        return 0.5 * (score01 + score10)
    if path == "diagonal-mean":
        return 0.5 * (score00 + score11)
    if path == "all-mean":
        return 0.25 * (score00 + score01 + score10 + score11)
    raise ValueError(f"unknown correction path: {path}")


def compute_query_cache(
    candidate_ids,
    base_scores,
    correction_scores,
    target_index,
    ks=DEFAULT_KS,
    alphas=DEFAULT_ALPHAS,
    extra_scores=None,
):
    path = build_rank_path(
        candidate_ids, base_scores, correction_scores, alphas=alphas
    )
    if extra_scores is not None:
        extra_path = build_rank_path(
            candidate_ids, base_scores, extra_scores, alphas=alphas
        )
        path = RankPath(
            candidate_ids=path.candidate_ids,
            alphas=np.arange(path.alphas.size + extra_path.alphas.size - 1),
            endpoint_percentiles=np.vstack(
                [path.endpoint_percentiles, extra_path.endpoint_percentiles[1:]]
            ),
            orders=np.concatenate([path.orders, extra_path.orders[1:]], axis=0),
        )
    target_index = int(target_index)
    if not 0 <= target_index < path.candidate_ids.size:
        raise ValueError("target_index is outside the gallery")

    target_ranks = np.empty(path.alphas.size, dtype=np.int32)
    for action, order in enumerate(path.orders):
        target_ranks[action] = int(np.flatnonzero(order == target_index)[0]) + 1

    base = np.asarray(base_scores, dtype=np.float64)
    base_order = path.orders[0]
    margins = []
    boundaries = {}
    for k in ks:
        if not 1 <= k < path.candidate_ids.size:
            raise ValueError("each cutoff must be smaller than the gallery")
        margins.append(float(base[base_order[k - 1]] - base[base_order[k]]))
        trace = boundary_trace(path, k)
        candidates = trace.candidate_indices
        positions = np.flatnonzero(candidates == target_index)
        boundaries[int(k)] = {
            "candidate_indices": candidates.astype(np.int32, copy=False),
            "membership": trace.membership.astype(np.uint8, copy=False),
            "endpoint_percentiles": path.endpoint_percentiles[
                :, candidates
            ].T.astype(np.float32, copy=False),
            "target_position": int(positions[0]) if positions.size else len(candidates),
        }
    return {
        "target_ranks": target_ranks,
        "base_margins": np.asarray(margins, dtype=np.float32),
        "boundaries": boundaries,
    }


def pack_query_caches(query_indices, caches, ks):
    packed = {
        "query_indices": np.asarray(query_indices, dtype=np.int32),
        "target_ranks": np.stack(
            [cache["target_ranks"] for cache in caches], axis=0
        ),
        "base_margins": np.stack(
            [cache["base_margins"] for cache in caches], axis=0
        ),
    }
    for k in ks:
        records = [cache["boundaries"][k] for cache in caches]
        lengths = np.asarray(
            [len(record["candidate_indices"]) for record in records],
            dtype=np.int64,
        )
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        if offsets[-1]:
            candidates = np.concatenate(
                [record["candidate_indices"] for record in records], axis=0
            )
            membership = np.concatenate(
                [record["membership"] for record in records], axis=0
            )
            percentiles = np.concatenate(
                [record["endpoint_percentiles"] for record in records], axis=0
            )
        else:
            actions = caches[0]["target_ranks"].size
            candidates = np.empty(0, dtype=np.int32)
            membership = np.empty((0, actions), dtype=np.uint8)
            endpoint_count = records[0]["endpoint_percentiles"].shape[1]
            percentiles = np.empty((0, endpoint_count), dtype=np.float32)
        prefix = f"k{k}_"
        packed[prefix + "offsets"] = offsets
        packed[prefix + "candidate_indices"] = candidates
        packed[prefix + "membership"] = membership
        packed[prefix + "endpoint_percentiles"] = percentiles
        packed[prefix + "target_positions"] = np.asarray(
            [record["target_position"] for record in records], dtype=np.int32
        )
    return packed


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_queries(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            for field in ("query_index", "dataset", "source_id", "target_id"):
                if field not in row:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            if int(row["query_index"]) != len(rows):
                raise ValueError("query indices must be contiguous and ordered")
            rows.append(row)
    if not rows:
        raise ValueError("empty query metadata")
    return rows


def exclude_source_candidates(
    base_scores, correction_scores, rows, gallery_index, extra_scores=None
):
    """Move each FashionIQ source image below every retrievable candidate."""
    base = np.asarray(base_scores)
    correction = np.asarray(correction_scores)
    extra = None if extra_scores is None else np.asarray(extra_scores)
    if (
        base.shape != correction.shape
        or base.shape[0] != len(rows)
        or (extra is not None and extra.shape != base.shape)
    ):
        raise ValueError("score matrices and query rows must be aligned")
    for offset, row in enumerate(rows):
        source_id = str(row["source_id"])
        if source_id not in gallery_index:
            raise ValueError(f"source is absent from gallery: {source_id}")
        source_index = gallery_index[source_id]
        base[offset, source_index] = -1e12
        correction[offset, source_index] = -1e12
        if extra is not None:
            extra[offset, source_index] = -1e12


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query-batch-size", type=int, default=16)
    parser.add_argument("--cache-workers", type=int, default=1)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--cutoffs", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--exclude-source", action="store_true")
    parser.add_argument(
        "--correction-path",
        choices=("matched", "cross-mean", "diagonal-mean", "all-mean", "joint"),
        default="matched",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    if args.cache_workers <= 0:
        raise ValueError("--cache-workers must be positive")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("--max-queries must be positive")
    cutoffs = tuple(args.cutoffs)
    if not cutoffs or any(k <= 0 for k in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError("--cutoffs must contain unique positive integers")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as functional

    embedding_dir = args.embedding_dir
    gallery_ids = json.loads((embedding_dir / "gallery_ids.json").read_text())
    if gallery_ids != sorted(gallery_ids) or len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError("gallery IDs must be unique and sorted")
    candidate_ids = np.asarray(gallery_ids, dtype=str)
    queries = load_queries(embedding_dir / "queries.jsonl")
    if args.max_queries is not None:
        queries = queries[: args.max_queries]

    base_gallery_array = np.load(embedding_dir / "base_gallery.npy", mmap_mode="r")
    correction_gallery_array = np.load(
        embedding_dir / "correction_gallery.npy", mmap_mode="r"
    )
    base_query_array = np.load(embedding_dir / "base_queries.npy", mmap_mode="r")
    correction_query_array = np.load(
        embedding_dir / "correction_queries.npy", mmap_mode="r"
    )
    if base_gallery_array.shape != correction_gallery_array.shape:
        raise ValueError("endpoint gallery arrays differ in shape")
    if base_query_array.shape != correction_query_array.shape:
        raise ValueError("endpoint query arrays differ in shape")
    if base_gallery_array.shape[0] != len(gallery_ids):
        raise ValueError("gallery ID count does not match embeddings")
    if base_query_array.shape[0] < len(queries):
        raise ValueError("query metadata exceeds embedding rows")
    if base_gallery_array.shape[0] <= max(cutoffs):
        raise ValueError("gallery is too small for configured cutoffs")

    gallery_index = {product_id: index for index, product_id in enumerate(gallery_ids)}
    target_indices = []
    for row in queries:
        target_id = str(row["target_id"])
        if target_id not in gallery_index:
            raise ValueError(f"target is absent from gallery: {target_id}")
        target_indices.append(gallery_index[target_id])

    device = torch.device(args.device)
    base_gallery = functional.normalize(
        torch.tensor(base_gallery_array, dtype=torch.float32, device=device), dim=-1
    )
    correction_gallery = functional.normalize(
        torch.tensor(correction_gallery_array, dtype=torch.float32, device=device),
        dim=-1,
    )

    shard_records = []
    for shard_index, start in enumerate(
        range(0, len(queries), args.query_batch_size)
    ):
        stop = min(start + args.query_batch_size, len(queries))
        base_queries = functional.normalize(
            torch.tensor(
                base_query_array[start:stop], dtype=torch.float32, device=device
            ),
            dim=-1,
        )
        correction_queries = functional.normalize(
            torch.tensor(
                correction_query_array[start:stop],
                dtype=torch.float32,
                device=device,
            ),
            dim=-1,
        )
        with torch.no_grad():
            score00 = (base_queries @ base_gallery.T).float().cpu().numpy()
            score01 = score10 = score11 = None
            if args.correction_path in ("cross-mean", "all-mean", "joint"):
                score01 = (base_queries @ correction_gallery.T).float().cpu().numpy()
                score10 = (correction_queries @ base_gallery.T).float().cpu().numpy()
            if args.correction_path in ("matched", "diagonal-mean", "all-mean", "joint"):
                score11 = (
                    correction_queries @ correction_gallery.T
                ).float().cpu().numpy()
            base_scores = score00
            correction_scores = compose_correction_scores(
                args.correction_path, score00, score01, score10, score11
            )
            extra_scores = (
                0.5 * (score01 + score10)
                if args.correction_path == "joint"
                else None
            )
        if args.exclude_source:
            exclude_source_candidates(
                base_scores,
                correction_scores,
                queries[start:stop],
                gallery_index,
                extra_scores,
            )

        def build_cache(offset):
            return compute_query_cache(
                candidate_ids,
                base_scores[offset],
                correction_scores[offset],
                target_indices[start + offset],
                ks=cutoffs,
                extra_scores=(
                    extra_scores[offset] if extra_scores is not None else None
                ),
            )

        if args.cache_workers == 1:
            caches = [build_cache(offset) for offset in range(stop - start)]
        else:
            with ThreadPoolExecutor(max_workers=args.cache_workers) as executor:
                caches = list(executor.map(build_cache, range(stop - start)))
        packed = pack_query_caches(range(start, stop), caches, cutoffs)
        shard_path = args.output_dir / f"shard_{shard_index:05d}.npz"
        np.savez_compressed(shard_path, **packed)
        shard_records.append(
            {
                "path": shard_path.name,
                "queries": stop - start,
                "query_start": start,
                "query_stop": stop,
                "sha256": sha256_file(shard_path),
            }
        )
        print(f"wrote {shard_path.name}: queries {start}:{stop}", flush=True)

    manifest = {
        "embedding_dir": str(embedding_dir.resolve()),
        "embedding_manifest_sha256": sha256_file(embedding_dir / "manifest.json"),
        "queries": len(queries),
        "gallery": len(gallery_ids),
        "alphas": (
            list(range(2 * len(DEFAULT_ALPHAS) - 1))
            if args.correction_path == "joint"
            else DEFAULT_ALPHAS.tolist()
        ),
        "endpoint_count": 3 if args.correction_path == "joint" else 2,
        "cutoffs": list(cutoffs),
        "exclude_source": bool(args.exclude_source),
        "correction_path": args.correction_path,
        "path_units": (
            "joint_matched_and_cross_rank_percentiles"
            if args.correction_path == "joint"
            else "deterministic_endpoint_rank_percentiles"
        ),
        "boundary_definition": "top_k_membership_changes",
        "tie_break": "ascending_gallery_id",
        "query_batch_size": args.query_batch_size,
        "cache_workers": args.cache_workers,
        "device": str(device),
        "script_sha256": sha256_file(__file__),
        "shards": shard_records,
    }
    with (args.output_dir / "manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
