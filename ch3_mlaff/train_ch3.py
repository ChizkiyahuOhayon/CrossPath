"""Two-stage training driver for chapter 3 (schedule from section 3.3.1).

Stage 1 (``--stage pretrain``)
    FashionGen domain pre-training of the plain two-tower baseline.
    AdamW, batch 16, 30 epochs, 5 warmup epochs, lr 1e-5 -> 6e-5 -> 1e-5.

Stage 2 (``--stage retrieval``)
    Retrieval fine-tuning with the chapter-3 modules attached, initialised from
    the stage-1 epoch-25 weights.  AdamW, batch 16, 20 epochs, 1 warmup epoch,
    lr 1e-5 -> 1e-6.

Every run appends one JSON line per epoch to ``<out_dir>/train_log.jsonl`` and
writes ``<out_dir>/run_manifest.json`` describing the exact configuration, so
that any number reported in the thesis can be traced back to an artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import eval_ch3
from .albef_init import load_albef
from .configs_ch3 import ABLATION_CONFIGS, CONFIGS
from .data_fashiongen import (
    FashionGenEvalImages, FashionGenPretrain, FashionGenRetrievalTrain, SplitIndex,
    build_transform, full_protocol_labels, gallery_texts, sample_protocol_indices,
)
from .encoders_ch3 import MultiLayerViT, SplitBert
from .model_ch3 import Ch3Config, Ch3Model
from .pretrain_ch3 import Ch3PretrainModel


# --------------------------------------------------------------------------
# schedule (section 3.3.1)
# --------------------------------------------------------------------------


def lr_at(step: int, steps_per_epoch: int, *, warmup_epochs: float, total_epochs: int,
          lr_start: float, lr_peak: float, lr_min: float) -> float:
    """Linear warmup from ``lr_start`` to ``lr_peak``, then cosine decay to ``lr_min``."""
    warmup_steps = int(warmup_epochs * steps_per_epoch)
    total_steps = total_epochs * steps_per_epoch
    if warmup_steps > 0 and step < warmup_steps:
        return lr_start + (lr_peak - lr_start) * step / warmup_steps
    denom = max(total_steps - warmup_steps, 1)
    progress = min((step - warmup_steps) / denom, 1.0)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1.0 + math.cos(math.pi * progress))


def code_fingerprint() -> str:
    """SHA-256 over this package's sources, so a run names the code that made it."""
    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for name in sorted(f for f in os.listdir(here) if f.endswith(".py")):
        with open(os.path.join(here, name), "rb") as fh:
            digest.update(name.encode())
            digest.update(fh.read())
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def build_model(cfg: Ch3Config, device, pretrained: bool, momentum: bool = True) -> Ch3Model:
    vis = MultiLayerViT(cfg.vit_name, img_size=cfg.image_res, pretrained=pretrained)
    txt = SplitBert(cfg.bert_name, text_layers=cfg.text_layers, cross_width=cfg.width,
                    pretrained=pretrained)
    vis_m = txt_m = None
    if momentum:
        vis_m = MultiLayerViT(cfg.vit_name, img_size=cfg.image_res, pretrained=False,
                              drop_path_rate=0.0)
        txt_m = SplitBert(cfg.bert_name, text_layers=cfg.text_layers, cross_width=cfg.width,
                          pretrained=False)
    return Ch3Model(cfg, vis, txt, vis_m, txt_m).to(device)


def load_stage1_weights(model: Ch3Model, path: str) -> dict:
    """Load stage-1 encoder weights into a stage-2 model; report what was reused."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("encoder_state", blob.get("model", blob))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if model.use_momentum:
        model.copy_params()
    return {
        "checkpoint": os.path.abspath(path),
        "stage1_epoch": blob.get("epoch"),
        "loaded_tensors": len(state),
        "randomly_initialised": sorted(missing),
        "ignored_from_checkpoint": sorted(unexpected),
    }


# --------------------------------------------------------------------------
# evaluation during training
# --------------------------------------------------------------------------


@torch.no_grad()
def run_evaluation(model, val_index, val_h5, tokenizer, device, args, amp_dtype,
                   full_rerank: bool = False):
    """Sample recall every time; the Full rerank only when explicitly asked for.

    The Full protocol reranks ``topk_rerank`` candidates for each of the 32,528
    image and 7,519 text queries, which is minutes of GPU time, so it is run on
    the checkpoint that goes into table 3.3 rather than after every epoch.  The
    cheap CLS-only Full recall is always reported alongside as a training
    monitor.
    """
    transform = build_transform(model.cfg.image_res, is_train=False)
    dataset = FashionGenEvalImages(val_h5, val_index, tokenizer, transform, model.cfg.max_word_num)
    loader = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers,
                        pin_memory=True, shuffle=False)
    corpus = eval_ch3.encode_corpus(model, loader, gallery_texts(val_index), tokenizer, device,
                                    model.cfg.max_word_num, args.text_batch_size, amp_dtype)
    img2txt, txt2img = full_protocol_labels(val_index)
    idxs = sample_protocol_indices(val_index, seed=args.sample_seed, set_len=args.sample_set_len)

    sim_i2t, sim_t2i = eval_ch3.cls_similarity(corpus)
    metrics = {
        "sample": eval_ch3.evaluate_sample(model, dataset, corpus, idxs, device, amp_dtype,
                                           num_workers=args.num_workers),
        "full_cls_only": eval_ch3.full_recall(sim_i2t, sim_t2i, img2txt, txt2img),
    }
    if full_rerank:
        metrics["full"] = eval_ch3.evaluate_full(model, dataset, corpus, img2txt, txt2img,
                                                 device, amp_dtype, model.cfg.topk_rerank,
                                                 num_workers=args.num_workers)
    return metrics


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pretrain", "retrieval"])
    ap.add_argument("--config", default="m5", choices=sorted(CONFIGS))
    ap.add_argument("--data-root", default="/root/autodl-tmp/ch3/data/fashiongen_h5")
    ap.add_argument("--cache-dir", default="/root/autodl-tmp/ch3/data/index")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init-from", default="", help="stage-1 checkpoint for --stage retrieval")
    ap.add_argument("--albef", default="",
                    help="ALBEF.pth to start --stage pretrain from, as FashionSAP does")

    ap.add_argument("--epochs", type=int, default=0, help="0 = schedule default (30 / 20)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--text-batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--amp", default="bf16", choices=["off", "fp16", "bf16"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-seed", type=int, default=20260920)
    ap.add_argument("--sample-set-len", type=int, default=1000)

    ap.add_argument("--keep-epochs", default="25", help="comma-separated epochs to keep")
    ap.add_argument("--keep-last", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="debug only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--full-rerank", action="store_true",
                    help="also run the Full-protocol rerank after the last epoch")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    from transformers import BertTokenizer

    cfg = CONFIGS[args.config]
    is_pretrain = args.stage == "pretrain"
    if is_pretrain:
        cfg = ABLATION_CONFIGS["m1"]     # stage 1 trains the plain two-tower model
    epochs = args.epochs or (30 if is_pretrain else 20)
    schedule = dict(
        warmup_epochs=5 if is_pretrain else 1,
        total_epochs=epochs,
        lr_start=1e-5,
        lr_peak=6e-5 if is_pretrain else 1e-5,
        lr_min=1e-5 if is_pretrain else 1e-6,
    )

    tokenizer = BertTokenizer.from_pretrained(cfg.bert_name)
    train_h5 = os.path.join(args.data_root, "fashiongen_256_256_train.h5")
    val_h5 = os.path.join(args.data_root, "fashiongen_256_256_validation.h5")
    train_index = SplitIndex.load_or_build(train_h5, os.path.join(args.cache_dir, "train.json"))
    val_index = SplitIndex.load_or_build(val_h5, os.path.join(args.cache_dir, "validation.json"))

    transform = build_transform(cfg.image_res, is_train=True)
    if is_pretrain:
        train_ds = FashionGenPretrain(train_h5, train_index, tokenizer, transform, cfg.max_word_num)
    else:
        train_ds = FashionGenRetrievalTrain(train_h5, train_index, tokenizer, transform,
                                            cfg.max_word_num)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    base = build_model(cfg, device, pretrained=True)
    init_report = None
    if is_pretrain:
        model = Ch3PretrainModel(base, tokenizer.vocab_size).to(device)
        if args.albef:
            init_report = load_albef(model, args.albef, cfg.text_layers)
    else:
        if not args.init_from:
            raise SystemExit("--stage retrieval requires --init-from <stage1 checkpoint>")
        init_report = load_stage1_weights(base, args.init_from)
        model = base

    opt = torch.optim.AdamW(model.parameters(), lr=schedule["lr_start"],
                            weight_decay=args.weight_decay)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    steps_per_epoch = args.max_steps_per_epoch or len(loader)
    keep_epochs = {int(x) for x in args.keep_epochs.split(",") if x.strip()}
    manifest = {
        "stage": args.stage,
        "config": args.config if not is_pretrain else "m1(two-tower)",
        "config_fields": cfg.__dict__,
        "epochs": epochs,
        "schedule": schedule,
        "batch_size": args.batch_size,
        "amp": args.amp,
        "seed": args.seed,
        "sample_seed": args.sample_seed,
        "train_images": train_index.num_images,
        "train_products": train_index.num_products,
        "val_images": val_index.num_images,
        "val_products": val_index.num_products,
        "steps_per_epoch": steps_per_epoch,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "code_sha256": code_fingerprint(),
        "init": init_report,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(args.out_dir, "run_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    resume_path = os.path.join(args.out_dir, "resume.pth")
    start_epoch = 0
    if args.resume and os.path.exists(resume_path):
        blob = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optimizer"])
        start_epoch = blob["epoch"]
        print(f"resumed from epoch {start_epoch}", flush=True)

    global_step = start_epoch * steps_per_epoch
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        totals, seen = {}, 0
        alpha = cfg.alpha * min(1.0, (epoch * steps_per_epoch) / max(steps_per_epoch, 1))
        for step, batch in enumerate(loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
            for group in opt.param_groups:
                group["lr"] = lr_at(global_step, steps_per_epoch, **schedule)

            if is_pretrain:
                image, ids, mask, mlm_ids, mlm_labels, idx = batch
                fwd = (image.to(device, non_blocking=True), ids.to(device), mask.to(device),
                       mlm_ids.to(device), mlm_labels.to(device), idx.to(device))
            else:
                image, ids, mask, idx = batch
                fwd = (image.to(device, non_blocking=True), ids.to(device), mask.to(device),
                       idx.to(device))

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None and device.type == "cuda"):
                out = model(*fwd, alpha=alpha if epoch > 0 else 0.0)
            if args.amp == "fp16":
                scaler.scale(out["loss"]).backward()
                scaler.step(opt)
                scaler.update()
            else:
                out["loss"].backward()
                opt.step()

            for k, v in out.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach() if torch.is_tensor(v) else v)
            seen += 1
            global_step += 1
            if step % 200 == 0:
                print(f"ep{epoch + 1} step {step}/{steps_per_epoch} "
                      f"loss {float(out['loss']):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                      f"({(time.time() - t0) / max(seen, 1):.3f}s/step)", flush=True)

        record = {"epoch": epoch + 1, "seconds": round(time.time() - t0, 1),
                  **{k: round(v / max(seen, 1), 5) for k, v in totals.items()}}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == epochs:
            metrics = run_evaluation(model.base if is_pretrain else model, val_index, val_h5,
                                     tokenizer, device, args, amp_dtype,
                                     full_rerank=args.full_rerank and epoch + 1 == epochs)
            record["metrics"] = metrics
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

        weights = (model.encoder_state_dict() if is_pretrain
                   else {k: v for k, v in model.state_dict().items()
                         if not k.startswith(("visual_encoder_m.", "text_encoder_m.",
                                              "image_queue", "text_queue", "idx_queue",
                                              "queue_ptr"))})
        if (epoch + 1) in keep_epochs or epoch + 1 > epochs - args.keep_last:
            ckpt = os.path.join(args.out_dir, f"epoch{epoch + 1:02d}.pth")
            torch.save({"epoch": epoch + 1, "encoder_state": weights,
                        "config": cfg.__dict__, "metrics": record.get("metrics")}, ckpt)
        torch.save({"epoch": epoch + 1, "model": model.state_dict(),
                    "optimizer": opt.state_dict()}, resume_path + ".tmp")
        os.replace(resume_path + ".tmp", resume_path)

    print("training complete", flush=True)


if __name__ == "__main__":
    main()
