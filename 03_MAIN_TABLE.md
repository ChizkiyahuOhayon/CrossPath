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

三类目等权平均。下表对所有本地 endpoint 统一移除 source image，报告 R@10/R@50；MCoT-MVS 公开数字单列为文献参照，不用于本地增益计算。

| Method | R@10 | R@50 |
|---|---:|---:|
| DQU-CIR (paper / reproduced Base) | 62.00 / 61.98 | 81.58 / 81.57 |
| DQU GradCache-b128 | 62.19 | 81.76 |
| DQU × DQU cross mean (Ours) | 62.85 | 82.03 |
| MCoT-MVS (paper) | 63.24 | 82.01 |
| MCoT-MVS (E24 same-pipeline) | 63.56 | 82.33 |
| DQU Base × MCoT cross mean (Ours) | **64.61** | 82.57 |
| **DQU GradCache × MCoT cross mean (Ours)** | 64.58 | **82.82** |

总体主候选相对同管线 MCoT 提升 `+1.02 R@10 / +0.48 R@50`，并相对 DQU GradCache 提升 `+2.39/+1.06`。DQU Base × MCoT 保持最高 R@10，DQU GradCache × MCoT 保持最高 R@50 与 SumR。

为对齐 MCoT 作者代码保留 source image 的执行方式，E24 也冻结重算了 include-source 版本：MCoT 62.95/82.14，DQU GradCache × MCoT cross mean **64.16/82.68**，提升 `+1.21/+0.55`；该 MCoT 数字与 E23 官方脚本复现的 62.96/82.09 相差仅 `−0.01/+0.05`。结果来源：`results/e24_heterogeneous/`。公开来源：[DQU-CIR](https://arxiv.org/abs/2404.15875)、[MCoT-MVS](https://arxiv.org/abs/2603.17360)。

## Mechanism controls

E26 对 endpoint 1 的 query/gallery 同时施加固定正交 signed permutation。该操作保持 endpoint 1 的全部内部点积（最大数值误差 FashionGen `7.75e-7`、FashionIQ `3.58e-7`），也保持 diagonal mean 不变，但破坏 off-diagonal 坐标对应：FashionIQ cross mean 从 64.58/82.82 降至 0.85/3.08 R@10/R@50；FashionGen cross mean 从 44.10/79.74/88.16 降至 0.14/0.59/0.95 R@1/R@5/R@10。

E27 的 3×3 原始路径矩阵进一步显示方向性：MCoT query 搭配 DQU Base/GradCache gallery 分别达到 63.99/82.50 和 63.94/82.45，均高于 MCoT 自身的 63.56/82.33；DQU query 搭配 MCoT gallery 则较弱。该非对称结构解释了双向 cross mean 的增益，并反对“任意两个 endpoint 平均都会提升”的说法。完整结果：`results/e26_controls/` 与 `results/e27_fashioniq_matrix3/`。
