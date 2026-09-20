# Chapter 3 — Multi-Level Adaptive Feature Fusion (MLAFF)

Reference implementation of the fashion image–text retrieval model specified in
section 3.2 of the thesis, together with the FashionGen protocol of section
3.3.2 and the training schedule of section 3.3.1.

The backbone is the ALBEF / FashionSAP two-tower + fusion layout that the
chapter's baseline reproduces. On top of it the chapter adds three modules:

| Module | Section | Equations | Code |
|---|---|---|---|
| Multi-level ViT patch features (blocks 4 / 8 / 12) | 3.2.1 | — | `encoders_ch3.MultiLayerViT.forward_layers` |
| Hierarchical cross-modal cross attention | 3.2.2 | (3.1)–(3.7) | `model_ch3.CrossModalBlock` |
| Text-guided gated fusion | 3.2.3 | (3.8)–(3.10) | `model_ch3.TextGuidedGate`, `Ch3Model.region_representation` |
| Region-enhanced alignment loss | 3.2.4 | (3.11)–(3.13) | `model_ch3.Ch3Model._rea_loss` |

## Layout

```
model_ch3.py      Ch3Config, CrossModalBlock, TextGuidedGate, Ch3Model (losses + pairwise scoring)
encoders_ch3.py   MultiLayerViT (timm), SplitBert (6-layer text + 6-layer fusion encoder)
pretrain_ch3.py   stage-1 wrapper: the two-tower baseline plus an MLM head
data_fashiongen.py  h5 datasets, the Sample/Full protocols, FashionSAP text normalisation
eval_ch3.py       recall, CLS shortlist, pairwise rerank, the two protocols
train_ch3.py      two-stage driver, schedule, run manifest with a code fingerprint
configs_ch3.py    the 13 configurations behind tables 3.5 / 3.6 and figure 3.6
bench_ch3.py      step-time and memory benchmark on synthetic tensors
```

## Configurations

`configs_ch3.CONFIGS` holds every configuration the chapter reports.

| Name | Visual layers | Cross attn | Fusion | λ | Fills |
|---|---|---|---|---|---|
| `m1` | 12 | — | — | — | table 3.2 `Baseline†`, table 3.5 row 1 |
| `m2` | 12 | ✓ | — | 0 | table 3.5 row 2 |
| `m3` | 4/8/12 | ✓ | direct | 0 | table 3.5 row 3 |
| `m4` | 4/8/12 | ✓ | gated | 0 | table 3.5 row 4, figure 3.6 at λ=0 |
| `m5` | 4/8/12 | ✓ | gated | 0.5 | tables 3.2/3.3/3.5/3.6, figure 3.6 at λ=0.5 |
| `L12` `L8` `L4_L8` `L8_L12` | as named | ✓ | gated | 0.5 | table 3.6 |
| `lam0_1` `lam0_3` `lam1_0` | 4/8/12 | ✓ | gated | as named | figure 3.6 |

## Reproducing

```bash
# stage 1 — FashionGen domain pre-training, 30 epochs, shared by every configuration
python -m ch3_mlaff.train_ch3 --stage pretrain \
  --data-root <dir with fashiongen_256_256_{train,validation}.h5> \
  --cache-dir <index cache> --out-dir runs/stage1_pretrain \
  --epochs 30 --batch-size 16 --eval-every 5 --keep-epochs 25 --resume

# stage 2 — retrieval fine-tuning from the epoch-25 weights, one run per configuration
python -m ch3_mlaff.train_ch3 --stage retrieval --config m5 \
  --init-from runs/stage1_pretrain/epoch25.pth \
  --out-dir runs/m5 --epochs 20 --batch-size 16 --full-rerank --resume
```

`../ch3_run_stage2.sh` runs all twelve stage-2 configurations in order, and
`../ch3_collect_results.py` turns the logs into the tables with a provenance
record for every cell.

The FashionGen h5 files are the `hieupth/fashiongen` mirror, verified by
SHA-256: train `bb55646511c2656bf59cb09aec3f188697f9bf2f635c3feef61bc626fbc8f81f`,
validation `f96d7630301328ef8b8a8e84c6fbdbb17e7c4fea8c2cd1186cde1133e4eff689`.

## Measured cost

RTX 4090 (24 GB), bf16, batch 16, 24 dataloader workers:

| | s/step | steps/epoch | h/epoch |
|---|---|---|---|
| stage 1 (ITC+ITM+MLM) | 0.229 | 16,280 | 1.04 |
| stage 2 (m5) | 0.168 | 3,759 | 0.18 |

Peak memory 10.8 GiB. Encoding the validation corpus takes 38 s; the Sample
protocol adds about 30 s and the Full rerank about 15 min, so the Full rerank
runs once, on the checkpoint that fills table 3.3, rather than every epoch.

## Decisions the thesis text does not fix

Recorded here because each one changes what the numbers mean.

1. **Where `F_final` enters the objective.** Section 3.2 makes `F_final` the
   retrieval representation, but `F_final` is a function of the image *and* the
   text, so it cannot also be the gallery embedding that table 3.1's single
   `sim_t2i` matrix implies. Reading it as a plain CLS cosine leaves `F_final`
   reachable only through `L_rea`, and then rows 1–4 of table 3.5 train
   identically — the cross-attention blocks receive exactly zero gradient.
   `F_final` is therefore contrasted inside `L_itc` alongside the CLS
   embedding, which keeps a text-independent embedding for the Full shortlist
   and gives the chapter-3 modules a training signal in every row that has
   them. `test_cross_attention_and_gate_receive_gradient` guards this.
2. **`L_rea` versus the contrastive term on `F_final`.** `L_rea` uses neither
   the momentum queue nor soft distillation targets and has its own temperature
   `tau_r`, so m5 differs from m4 by a sharper in-batch constraint rather than
   by a duplicate loss.
3. **The baseline.** Model 1 is the plain two-tower model that section 3.3.5
   describes (ITC + ITM, layer-12 features only), not a full FashionSAP
   reproduction — FashionSAP's symbol prompts and attribute prediction are not
   part of the chapter's description.
4. **Stage-1 objective.** Section 3.3.1 fixes the schedule but not the loss;
   ITC + ITM + MLM is used, the ALBEF/FashionSAP domain pre-training objective
   the baseline inherits.
5. **Which epoch is reported.** The last one. FashionGen's validation split is
   also the evaluation split, so selecting the best epoch would select on the
   numbers being reported. Every epoch stays in `train_log.jsonl`.
6. **Sample-protocol sampling.** The 1,000 queries and their 100 negatives are
   drawn with a fixed seed (`--sample-seed`), recorded in the run manifest, and
   identical across configurations.
