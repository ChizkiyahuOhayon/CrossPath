# CrossPath 主表：FashionMV / FashionGen

官方协议：FashionGen validation，5,292 个 gallery 商品，9,031 条 queries。公开方法数字来自 FashionMV 原论文 Table 2；R@1 未报告处记为 `—`。

| Method | Params | R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|
| CLIP4CIR (MaxSim) | 0.25B | — | 17.1 | 25.0 |
| SPRC (MaxSim) | 1.2B | — | 42.7 | 53.0 |
| Qwen3-VL-2B (Joint) | 2B | — | 63.0 | 74.1 |
| Qwen3-VL-8B (Joint) | 8B | — | 74.7 | 83.5 |
| Doubao-E-V (MaxSim) | — | — | 67.2 | 77.1 |
| ProCIR (paper) | 0.8B | — | 75.0 | 85.3 |
| A1 (strong reproduced base) | 0.8B | 42.73 | 79.26 | 87.75 |
| CrossPath matched-only (Ours) | 2×0.8B + 0.526M | 43.65 | 80.27 | **88.33** |
| **CrossPath joint matrix (Ours)** | **2×0.8B + 0.526M** | **44.09** | **80.36** | 88.28 |

CrossPath joint matrix 相对同协议强 A1 提升 `+1.36 R@1 / +1.10 R@5 / +0.53 R@10`；matched-only 保持最高 R@10。joint 相对 matched-only 为 `+0.44/+0.09/-0.04`，SumR 提高 0.49。

结果来源：`results/CrossPath_A1seedpair_20260816_official_manifest.json` 与 `results/fashiongen_joint_matrix_official_manifest.json`。公开基线来源：[FashionMV](https://arxiv.org/abs/2604.10297)，Table 2。

# FashionIQ val-split

官方 DQU-CIR val-split 协议；三类目等权平均，报告 R@10/R@50。

| Method | R@10 | R@50 |
|---|---:|---:|
| DQU-CIR (paper / reproduced Base) | 62.00 / 61.98 | 81.58 / 81.57 |
| Endpoint 2 (GradCache-b128) | 62.19 | 81.76 |
| CrossPath matched gate (Ours) | 62.72 | 81.61 |
| **CrossPath cross matrix (Ours)** | 62.85 | **82.03** |
| MCoT-MVS (2026) | **63.24** | 82.01 |

零参数 cross matrix 相对同协议 reproduced Base 提升 `+0.87 R@10 / +0.46 R@50`，并全面超过两个 frozen endpoints 与 learned gate；R@50 略高于 MCoT-MVS 0.02，R@10 仍低 0.39。结果来源：`results/fashioniq_valsplit_cross_matrix_summary.json`。公开来源：[DQU-CIR](https://arxiv.org/abs/2404.15875)、[MCoT-MVS](https://arxiv.org/abs/2603.17360)。
