"""Throughput benchmark for the chapter 3 model.

Runs the real training step (ITC with momentum + queue, ITM with hard
negatives, region-enhanced loss) on synthetic tensors so that the cost of one
optimizer step can be measured without waiting for the dataset.

    python -m ch3_mlaff.bench_ch3 --config full --batch-size 16 --steps 30
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from .configs_ch3 import ABLATION_CONFIGS
from .encoders_ch3 import MultiLayerViT, SplitBert
from .model_ch3 import Ch3Model


def build(cfg, device, pretrained=False):
    vis = MultiLayerViT(cfg.vit_name, img_size=cfg.image_res, pretrained=pretrained)
    txt = SplitBert(cfg.bert_name, text_layers=cfg.text_layers, cross_width=cfg.width,
                    pretrained=pretrained)
    vis_m = MultiLayerViT(cfg.vit_name, img_size=cfg.image_res, pretrained=False, drop_path_rate=0.0)
    txt_m = SplitBert(cfg.bert_name, text_layers=cfg.text_layers, cross_width=cfg.width,
                      pretrained=False)
    return Ch3Model(cfg, vis, txt, vis_m, txt_m).to(device)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="full", choices=sorted(ABLATION_CONFIGS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--amp", default="bf16", choices=["off", "fp16", "bf16"])
    ap.add_argument("--train-images", type=int, default=260480)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ABLATION_CONFIGS[args.config]
    model = build(cfg, device, pretrained=False)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.02)

    b, m = args.batch_size, cfg.max_word_num
    image = torch.randn(b, 3, cfg.image_res, cfg.image_res, device=device)
    input_ids = torch.randint(1000, 20000, (b, m), device=device)
    attention_mask = torch.ones(b, m, dtype=torch.long, device=device)
    idx = torch.arange(b, device=device)

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    times = []
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None and device.type == "cuda"):
            out = model(image, input_ids, attention_mask, idx, alpha=cfg.alpha)
        if args.amp == "fp16":
            scaler.scale(out["loss"]).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out["loss"].backward()
            opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    per_step = elapsed / args.steps
    steps_per_epoch = (args.train_images + b - 1) // b
    report = {
        "config": args.config,
        "batch_size": b,
        "amp": args.amp,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "sec_per_step": round(per_step, 4),
        "images_per_sec": round(b / per_step, 2),
        "steps_per_epoch": steps_per_epoch,
        "hours_per_epoch": round(steps_per_epoch * per_step / 3600, 3),
        "peak_mem_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else 0.0,
        "loss": {k: float(v) for k, v in out.items() if k != "loss"},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
