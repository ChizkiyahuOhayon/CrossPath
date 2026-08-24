# CrossPath 实验记录（论文闭环）

本文件只记录进入 CrossPath 论文主线的正式实验。指标来自保存的 manifest/evaluation 原始产物；失败实验也保留，以便直接据此撰写对比与消融部分。

## E0 — FashionIQ 工程 smoke（2026-08-20，完成）

- dress train 前 32 条查询、64 张 source/target gallery；只验证协议与代码，不作为论文数字。
- 双 endpoint embedding 维度 1024、metadata 对齐、R@1/R@10/R@50 cache 构建成功。
- cache manifest 明确记录 `exclude_source=true`；gate 可读取动态 cutoffs 并完成一次训练/评测。
- 远端相关测试 21/21 通过；正式实验随后使用完整 train/val 数据。

## E1 — FashionGen 主结果（2026-08-16，完成）

- Base endpoint：A1 seed20260718，R@1/R@5/R@10 = 42.73/79.26/87.75。
- Endpoint 2：A1 seed20260722，42.40/79.35/88.14。
- CrossPath：43.65/80.27/88.33。
- 相对 Base：+0.92/+1.01/+0.58；三个指标均提高。
- 原始产物：`results/CrossPath_A1seedpair_20260816_official_manifest.json` 与 `results/CrossPath_A1seedpair_20260816_official_evaluation.npz`。

## E2 — FashionIQ 第二 benchmark（2026-08-20，完成）

目标：以 DQU-CIR seed42 为 base、GradCache-b128 为第二 endpoint，在 FashionIQ train labels 上训练三类目 CrossPath gate，并在 DQU-CIR val-split 协议上报告三类目平均 R@10/R@50。

固定设置：

- cutoffs：R@1/R@10/R@50；论文主指标为 R@10/R@50。
- path：9 个 endpoint rank-percentile actions。
- gate：width 128/256，3 epochs，train/calibration/test = 70/15/15；每类目独立训练，符合 DQU-CIR 类别训练协议。
- 评测：source image 从候选中排除；gallery 为 val triplets 中 source/target 并集。
- Backbone 不重训；直接使用已保存的六个 endpoint checkpoint。

当前 official 结果：

| 类目 | Base R@10/R@50 | Endpoint 2 | CrossPath | Δ vs Base |
|---|---:|---:|---:|---:|
| dress | 57.21 / 78.73 | 57.02 / 79.03 | **58.21 / 78.93** | **+0.99 / +0.20** |
| shirt | 62.90 / 80.37 | 62.95 / 80.62 | **63.40 / 80.37** | **+0.49 / +0.00** |
| toptee | 65.83 / 85.62 | 66.60 / 85.62 | **66.55 / 85.52** | **+0.71 / -0.10** |
| **三类平均** | **61.98 / 81.57** | **62.19 / 81.76** | **62.72 / 81.61** | **+0.73 / +0.03** |

dress 补充：R@1 为 Base 25.29、Endpoint 2 24.34、CrossPath 24.94；objective oracle R@10/R@50 = 60.93/80.96；intervention rate = 53.00%。论文主指标只采用 R@10/R@50。

shirt 补充：R@1 为 Base 31.01、Endpoint 2 30.96、CrossPath 31.35；objective oracle R@10/R@50 = 65.36/81.99；intervention rate = 21.54%。CrossPath 的 R@10 高于两个 endpoint，R@50 与 Base 持平。

toptee 补充：R@1 为 Base 32.64、Endpoint 2 33.10、CrossPath 32.94；objective oracle R@10/R@50 = 68.79/86.89；intervention rate = 63.90%。

最终判断：CrossPath 在第二 benchmark 上复现了相对强 Base 的正向迁移，三类平均 R@1/R@10/R@50 提升 +0.10/+0.73/+0.03；R@10 同时高于 Base 与 Endpoint 2，R@50 高于 Base 但低于 Endpoint 2 0.15。最佳固定 action（action 3）为 29.69/62.64/81.87，CrossPath 的 R@10 更高 0.08，但 R@50 更低 0.27。原始汇总：`results/FashionIQ_CrossPath_20260820_summary.json`。

## E3 — 常规消融（2026-08-20，完成）

- Query-only：只保留 query embedding 与 cutoff，其余输入块置零。
- Margin-only：只保留两个 endpoint rank-percentile（即相对排名 margin）与 cutoff，其余输入块置零。
- Full CrossPath：完整 query/candidate interaction、percentiles 与 membership trace。
- 三种设置沿用同一 split、seed、epoch、width 与 threshold 流程，不单独调参。

当前 FashionGen official 结果：

| Variant | R@1 | R@5 | R@10 | Intervention |
|---|---:|---:|---:|---:|
| Base | 42.73 | 79.26 | 87.75 | 0.00% |
| Query-only | 42.73 | 79.26 | 87.75 | 0.00% |
| Margin-only | 42.73 | 79.26 | 87.75 | 0.00% |
| Full CrossPath | **43.65** | **80.27** | **88.33** | 45.68% |

Query-only 的 calibration 流程选择不干预（threshold 0.0，所有 predicted utility 未越过阈值），因此精确回退到 Base；这说明 query identity 本身不足以支持安全的 candidate-level path routing。原始产物：`results/FashionGen_query_only_20260820_{gate,official}_manifest.json`。

Margin-only 同样由 calibration 选择不干预（threshold 0.0，intervention 0%），精确回退 Base。两个 endpoint 的相对 rank margin 单独也不足以支持安全路由；只有 Full 的 query/candidate interaction + rank percentiles + membership trace 组合产生正向结果。原始产物：`results/FashionGen_margin_only_20260820_{gate,official}_manifest.json`。

## E4 — 效率实测（2026-08-20，完成）

RTX 4090，post-embedding gate+policy latency 使用 500 queries 实测；该数字包含 boundary feature gathering、gate forward 和 utility aggregation，不包含两个 endpoint 编码与离线 cache 构建。

| Dataset | Gate params | Gate ckpt | Two gallery embeddings | Avg. boundary candidates | Gate+policy ms/query |
|---|---:|---:|---:|---:|---:|
| FashionGen | 0.526M | 2.01 MiB | 41.34 MiB | 0.37 / 2.02 / 3.85 @ R1/R5/R10 | 4.48 |
| FashionIQ（3 类平均） | 0.526–1.052M | 2.01–4.02 MiB | 22.45 MiB | 0.54 / 4.11 / 15.22 @ R1/R10/R50 | 4.35 |

FashionIQ shirt/toptee 的 query encoder 实测均值：Base 12.29 ms/query，Endpoint 2 12.03 ms/query；双 endpoint + gate/policy 合计约 28.68 ms/query。gallery encoding 为离线成本。原始结果：`results/efficiency/*.json`。

## E5 — 定性结果（暂缓）

按当前优先级先完成实验闭环，暂不继续投入定性图。DQU-CIR 的 CLIP 双塔没有适合作为主证据的原生 cross-attention map，因此后续若补图，优先采用 retrieval case/path transition，而不制作名实不符的 attention heatmap。

- 路径案例图已完成：`figures/fig_rank_paths.pdf`（矢量）及 `figures/fig_rank_paths.png`（300 dpi）。
- 图中三个真实 FashionGen query 分别将 target rank 从 4→1、6→5、43→10；数据直接读取 E1 的 official evaluation，不进行人工改写。
- 可复现脚本：`figures/gen_fig_rank_paths.py`。

## E6 — FashionIQ 官方 cutoff policy（2026-08-20，完成；不采用）

动机：FashionIQ 官方主指标只包含 R@10/R@50；E2 的 gate utility 额外等权包含 R@1。此实验预先将 policy cutoffs 固定为 `{10,50}`，其余 endpoint、embedding、split、seed、width、epoch、threshold 流程全部不变，不重训 backbone、不扫参。

- 当前对照：CrossPath 62.72/81.61 R@10/R@50。
- 成功标准：三类平均优于当前 CrossPath；SOTA 目标为 63.24/82.01。
- 可复现脚本：`scripts/run_fashioniq_official_cutoffs.sh`。

结果：

| 类目 | E2 Full（R@10/R@50） | Official-cutoff only | 差值 |
|---|---:|---:|---:|
| dress | 58.21 / 78.93 | 58.06 / 79.03 | -0.15 / +0.10 |
| shirt | 63.40 / 80.37 | 62.90 / 80.37 | -0.49 / +0.00 |
| toptee | 66.55 / 85.52 | 65.94 / 85.62 | -0.61 / +0.10 |
| **三类平均** | **62.72 / 81.61** | **62.30 / 81.67** | **-0.42 / +0.07** |

裁决：不替换 E2。移除 R@1 后，R@50 仅提高 0.07，但 R@10 降低 0.42，整体不满足“优于当前 CrossPath”的预设标准；主表继续使用 E2 的三 cutoff policy。原始汇总：`results/FashionIQ_CrossPath_k10k50_20260820_summary.json`。

## E7 — FashionIQ original-split 完整 gallery（2026-08-20，完成）

动机：E2 使用 DQU-CIR val-split 的 source/target 并集 gallery；近期公开方法通常在 FashionIQ original-split 完整验证 gallery 上报告结果。本实验保持 E2 的两个 endpoint、已有 gate checkpoint、cutoffs 与 source exclusion 不变，只将每类验证 gallery 扩展为官方 `split.<category>.val.json` 中的全部图像。训练 gate 不重跑。

| 类目 | Base R@10/R@50 | Endpoint 2 | CrossPath | Δ vs Base |
|---|---:|---:|---:|---:|
| dress | 52.35 / 75.06 | 52.31 / 74.81 | **53.10 / 75.36** | **+0.74 / +0.30** |
| shirt | 54.07 / 73.99 | 55.10 / 73.41 | **54.86 / 73.99** | **+0.79 / +0.00** |
| toptee | 56.86 / 78.89 | 58.18 / 79.70 | **58.59 / 79.30** | **+1.73 / +0.41** |
| **三类平均** | **54.43 / 75.98** | **55.20 / 75.97** | **55.52 / 76.22** | **+1.09 / +0.24** |

补充结果：三类平均 R@1 为 Base 24.50、Endpoint 2 24.53、CrossPath 24.62；objective oracle 为 27.64/58.17/78.08 R@1/R@10/R@50。最佳固定 action 5 为 24.34/55.51/76.50，CrossPath 的 R@10 略高 0.004，但 R@50 低 0.29。

裁决：作为 FashionIQ 后续主线的正式协议。CrossPath 在完整 gallery 上仍稳定提升 Base，且 R@10 增益由 E2 的 +0.73 扩大到 +1.09；但与 objective oracle 仍有 +2.65 R@10/+1.86 R@50 的差距，下一步进入边界候选匹配，不继续调 cutoff。原始汇总：`results/fashioniq_original/summary.json`；远端 run：`runs/CrossPath_FashionIQ_DQU_original_20260820_v1`。

## E8 — 零参数 2×2 cross-compatibility 路径（2026-08-20，完成）

动机：两个共享 CLIP 空间的 endpoint 天然定义四条兼容路径 `q0·g0`、`q0·g1`、`q1·g0`、`q1·g1`。本实验不训练新模型，直接在 E7 的完整 gallery embeddings 上评估四条路径及其简单均值，验证交叉路径是否含有现有 matched endpoints 未利用的信息。

三类平均结果：

| Path | R@1 | R@10 | R@50 |
|---|---:|---:|---:|
| q0·g0 | 24.50 | 54.43 | 75.98 |
| q0·g1 | 24.42 | 54.93 | 76.10 |
| q1·g0 | 24.07 | 55.17 | 75.82 |
| q1·g1 | 24.53 | 55.20 | 75.97 |
| diagonal mean | **24.73** | 55.61 | 76.38 |
| cross mean | 24.58 | **55.61** | 76.40 |
| all mean | 24.68 | 55.60 | **76.43** |

对照 E7 CrossPath 24.62/55.52/76.22，零参数 cross mean 的差值为 -0.05/+0.10/+0.18。裁决：交叉兼容路径有效但不能替换原路径；下一实验先验证替换式 gate，随后按结果决定是否构造联合路径池。原始结果：`results/fashioniq_original/cross_compatibility.json`；可复现脚本：`scripts/eval_cross_compatibility.py`。所有排名使用与正式 cache 相同的 ascending gallery-ID tie-break。

## E9 — Cross-mean 替换式 responsibility gate（2026-08-20，完成；不采用）

设置：保持 E7 的 Base 和 gate 结构不变，将原 `q0·g0 → q1·g1` rank path 替换为 `q0·g0 → (q0·g1+q1·g0)/2`，重新构建 train/official cache 并训练三类 gate。

| 类目 | Base R@10/R@50 | Cross-mean endpoint | Gated cross path | Δ gate vs Base |
|---|---:|---:|---:|---:|
| dress | 52.35 / 75.06 | 52.65 / 75.41 | 52.55 / 75.51 | +0.20 / +0.45 |
| shirt | 54.07 / 73.99 | 55.64 / 74.29 | 54.56 / 74.14 | +0.49 / +0.15 |
| toptee | 56.86 / 78.89 | 58.54 / 79.50 | 58.18 / 79.30 | +1.33 / +0.41 |
| **三类平均** | **54.43 / 75.98** | **55.61 / 76.40** | **55.10 / 76.32** | **+0.67 / +0.33** |

裁决：不替换 E7。替换式 gate 的 R@10 比 E7 CrossPath 低 0.42；直接 cross-mean endpoint 反而更强。类别结果显示明显互补：dress 的 matched CrossPath 更强，shirt 的 cross-mean 更强，toptee 两者分别偏向 R@10/R@50。因此后续只考虑保留 matched path 并增加 cross path 的联合路径，不再做替换。原始汇总：`results/fashioniq_original/cross_matrix_summary.json`；远端 run：`runs/CrossPath_FashionIQ_CrossMatrix_original_20260820_v1`。

## E10 — FashionIQ matched+cross 联合动作池（2026-08-20，完成；不采用）

设置：将 E7 的 9 个 matched-path actions 与 E9 的 8 个非重复 cross-path actions 合并为 17 个动作，使用同一个 responsibility gate 联合选择；新增 cross endpoint rank-percentile，其余训练、校准与 official original-split 评测设置不变。一次早期 cache 因 cross scores 未排除 source 被立即作废，修复并加入回归测试后从头重跑；下表只记录修复后的正式结果。

| 方法 | R@1 | R@10 | R@50 |
|---|---:|---:|---:|
| Base | 24.50 | 54.43 | 75.98 |
| E7 matched CrossPath | **24.62** | **55.52** | **76.22** |
| 最佳固定 joint action（cross mean） | 24.58 | 55.61 | **76.40** |
| 17-action joint gate | 24.56 | 55.35 | 76.15 |

裁决：joint gate 相对 Base 提升 +0.05/+0.92/+0.17，但 R@10/R@50 均低于最佳固定 action，且未全面超过 E7；扩展动作空间没有转化为更好的泛化，故不作为主方法。停止继续扩展 gate/action pool，转向验证零参数兼容矩阵本身的跨数据集收益。原始汇总：`results/fashioniq_original/joint_matrix_summary.json`；远端 run：`runs/CrossPath_FashionIQ_JointMatrix_original_20260820_v1`。

## E11 — FashionGen 零参数 2×2 compatibility matrix（2026-08-20，完成）

设置：复用 E1 两个 endpoint 的 official embeddings，评估四条 query-gallery 路径及固定均值；沿用 FashionGen 正式协议，不排除 source，cutoffs 为 R@1/R@5/R@10。一次错误排除 source 的诊断输出因协议不符被作废，未计入任何结果。

| Path | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| q0·g0（Base） | 42.73 | 79.26 | 87.75 |
| q1·g1 | 42.40 | 79.35 | 88.14 |
| diagonal mean | 42.85 | 79.74 | **88.24** |
| all mean | 43.54 | 79.73 | 88.17 |
| cross mean | **44.10** | **79.74** | 88.16 |
| E1 matched CrossPath | 43.65 | **80.27** | **88.33** |

结论：cross mean 相对 Base 为 +1.37/+0.48/+0.41，并将 E1 CrossPath 的 R@1 再提高 0.45；但 E1 CrossPath 的 R@5/R@10 仍高 0.53/0.17。FashionGen 与 FashionIQ 均证明 cross paths 含有 matched endpoints 未利用的互补信号；下一实验使用已实现的联合动作池验证能否兼得 R@1 与 R@5/R@10。原始结果：`results/fashiongen_cross_matrix_summary.json`；远端 run：`runs/CrossMatrix_FashionGen_20260820_v1`。

## E12 — FashionIQ val-split 零参数 2×2 compatibility matrix（2026-08-20，完成）

动机：MCoT-MVS 代码同时输出 val-split 与 original-split，其论文主表 63.24/82.01 与 DQU-CIR 61.97/81.56 对应 val-split；因此 E2 才是直接可比协议。本实验将 E8 的固定兼容矩阵无训练地应用到 E2 embeddings，source exclusion 与 cutoffs 完全沿用 E2。

| 方法 | R@1 | R@10 | R@50 |
|---|---:|---:|---:|
| Base | 29.64 | 61.98 | 81.57 |
| Endpoint 2 | 29.47 | 62.19 | 81.76 |
| diagonal mean | 29.93 | **62.87** | 81.95 |
| all mean | 29.93 | 62.79 | 82.00 |
| cross mean | **29.95** | 62.85 | **82.03** |
| E2 CrossPath gate | 29.74 | 62.72 | 81.61 |
| MCoT-MVS（公开主表） | — | **63.24** | 82.01 |

裁决：cross mean 是当前 FashionIQ val-split 最佳总体候选，相对 Base 为 +0.30/+0.87/+0.46，并全面超过 E2 gate；R@50 与 MCoT-MVS 基本持平且高 0.02，R@10 低 0.39。该结果与 FashionGen E11 共同支持“非对角兼容路径具有互补信息”的简洁主线。原始结果：`results/fashioniq_valsplit_cross_matrix_summary.json`；远端 run：`runs/CrossMatrix_FashionIQ_valsplit_20260820_v1`。

## E13 — 4 参数 learned CrossMatrix（2026-08-20，完成；不采用）

设置：每类别只学习四个全局 softmax 权重，score 为四条兼容路径的凸组合；两个 endpoint 完全冻结。权重只用 train embeddings 上的全 gallery retrieval cross-entropy 学习 30 epochs，随后直接用于 E2 official val。首次实现用 `-inf` 屏蔽 source 导致权重梯度 NaN，其 100% 诊断输出作废；改为与余弦范围充分分离的有限屏蔽值后以 v2 从头运行。

结果：三类别的训练权重均将 97.5%–97.9% 质量压到 `q1·g1`，official 平均 R@1/R@10/R@50 = 29.43/62.22/81.75，低于 E12 cross mean 的 29.95/62.85/82.03。

裁决：不采用。全排序交叉熵偏向单 endpoint，未保留 cutoff 附近的跨路径互补性；不继续增加正则或基于 official 指标扫权重。原始结果：`results/fashioniq_valsplit_learned_cross_matrix_summary.json`；可复现脚本：`scripts/fit_cross_matrix.py`；远端 run：`runs/CrossMatrix_FashionIQ_Learned_20260820_v2`。

## E14 — Compatibility max reducers（2026-08-20，完成；不采用）

设置：只增加与 mean 同等自然的零参数 max reducers（diagonal/cross/all）。先在 internal 数据上选择归约，再冻结选择用于 official，避免根据 official 挑函数。internal 上 `diagonal_max` 最优，R@1/R@10/R@50 = 38.84/74.32/90.65；因此预选它进入 official。

official `diagonal_max` 为 29.53/62.42/81.75，低于 E12 cross mean 的 29.95/62.85/82.03，也低于 internal 选择前已存在的 diagonal mean。裁决：不采用并停止扩展 reducer；max 放大了 internal 上的高分路径，但未泛化。原始结果：`results/fashioniq_internal_matrix_reducers_summary.json` 与 `results/fashioniq_valsplit_matrix_reducers_summary.json`。

## E15 — Frozen joint gate on FashionIQ val-split（2026-08-20，完成；不采用）

设置：直接复用 E10 在 FashionIQ train split 上训练的 17-action joint gates，不重训、不改阈值；只对 E2 val-split embeddings 建 joint cache 并评估，从而与 MCoT 主表同协议对齐。

结果：Base 29.64/61.98/81.57，最佳固定 action（cross mean）29.95/62.85/82.03，joint gate 29.68/62.60/81.67（R@1/R@10/R@50）。joint gate 相对 Base 为 +0.03/+0.62/+0.10，但低于最佳固定 action -0.27/-0.25/-0.36。

裁决：不采用。联合 gate 已在 val-split 与 original-split 两个 gallery 协议上均未超过固定 cross mean，故停止扩展 action/gate 路线。原始结果：`results/fashioniq_valsplit_joint_matrix_summary.json`；远端 run：`runs/CrossPath_FashionIQ_JointMatrix_valsplit_20260820_v1`。

## E16 — Rank-percentile Borda reducers（2026-08-20，internal 止损）

设置：将四条兼容路径分别转换为完整候选 rank，再对 diagonal/cross/all ranks 做等权 Borda 融合；这是对 score-scale 不一致的零参数控制。按预设先在 internal 比较，只有优于已有 mean 才进入 official。

结果：`cross_borda` 是 Borda 中 R@10 最好者，R@1/R@10/R@50 = 36.06/73.20/90.39；低于 diagonal mean 的 37.10/73.55/90.50。裁决：internal 未通过，不运行 official；停止所有固定 reducer 扩展。原始结果：`results/fashioniq_internal_borda_summary.json`。

## E17 — FashionGen matched+cross joint routing（2026-08-20，完成；总体主候选）

设置：在 E1 的 9 个 matched-path actions 基础上，加入 8 个非重复 cross-path actions，形成 17-action pool；同一个 responsibility gate 使用新增的 cross endpoint rank-percentile，仍输出单一 ranking。endpoint、train/official embeddings、cutoffs、widths、epochs、regression cost 与 threshold calibration 流程均沿用 E1。cache builder 新增默认关闭的 `cache_workers`，正式 run 使用 8 threads；17 项回归测试通过，manifest 记录 workers=8。两个早期未完成 cache（qbatch16 单线程、qbatch128 单线程）因速度过慢被停止并保留为 `aborted_*`，未训练 gate、未产生指标；正式 run 从空目录完整重建。

| 方法 | R@1 | R@5 | R@10 | Δ vs Base |
|---|---:|---:|---:|---:|
| Base | 42.73 | 79.26 | 87.75 | — |
| 固定 cross mean | **44.10** | 79.74 | 88.16 | +1.37/+0.48/+0.41 |
| E1 matched-only CrossPath | 43.65 | 80.27 | **88.33** | +0.92/+1.01/+0.58 |
| **E17 joint CrossPath** | 44.09 | **80.36** | 88.28 | **+1.36/+1.10/+0.53** |

internal gate 通过 go/no-go：Base 30.89/63.00/74.35，joint 32.34/64.42/75.06 R@1/R@5/R@10；selected width=128，intervention rate=41.96%。official selected width=128，intervention rate=38.53%，objective oracle=47.71/82.75/90.36。

裁决：E17 是 SumR/总体表现最强主候选，相对 E1 matched-only 为 +0.44 R@1/+0.09 R@5/−0.04 R@10，SumR +0.49；它不逐项支配 E1，因此主表同时保留 matched-only（R@10 最优）与 joint（R@1/R@5/总体最优），不声称 joint 在每个 cutoff 都更强。原始产物：`results/fashiongen_joint_matrix_official_manifest.json`、`results/fashiongen_joint_matrix_gate_manifest.json`、`results/fashiongen_joint_matrix_evaluation.npz`；远端 run：`runs/CrossPath_FashionGen_JointMatrix_20260820_v1`。

## E18 — Frozen-embedding CrossPath residual adapter pilot（2026-08-24，完成；不晋级）

动机：将四条 compatibility paths 从测试时固定均值推进为可训练模块，同时避免同时加载两个 4GB DQU-CIR endpoint。两个 endpoint 完全冻结；每个 endpoint 共享一个用于 query/gallery 的 rank-64 residual adapter，共 0.262M 参数。训练目标为四路径联合对比损失

\[
\mathcal L_{\mathrm{CP}}=\frac{1}{4}\sum_{a,b\in\{0,1\}}
\operatorname{CE}\!\left(q'_a g_b'^{\top}/\tau\right),
\]

其中 \(q'_a=\operatorname{norm}(q_a+0.1A_a(q_a))\)，gallery 同理；\(A_a\) 是共享的低秩 residual adapter。只在 FashionIQ toptee 上进行预设 pilot：10 epochs、temperature 0.07、AdamW、lr 0.001、weight decay 0.0001、batch 128、每 batch 2048 个 gallery negatives。train queries 以 SHA-256 固定划分 85/15，checkpoint 仅按内部验证集 all-mean 的 R@10+R@50 选择；official val 训练结束后评测一次。

预设晋级线：official toptee 超过已有固定 all-mean 67.06/85.57（R@10/R@50，Sum=152.63）。

| Variant | R@1 | R@10 | R@50 | Sum R@10+R@50 |
|---|---:|---:|---:|---:|
| 固定 all-mean（E12） | **33.45** | 67.06 | **85.57** | **152.63** |
| adapted all-mean | 33.35 | 67.06 | 85.36 | 152.42 |
| 固定 cross-mean（E12） | 33.25 | 67.01 | **85.57** | 152.58 |
| adapted cross-mean | **33.50** | **67.11** | 85.47 | 152.58 |

裁决：不晋级，不扩展到 dress/shirt，也不搜索 rank、scale 或 learning rate。最佳内部 checkpoint 出现在 epoch 1；其 official adapted all-mean 相对固定 all-mean 为 −0.00/−0.20 R@10/R@50。adapted cross-mean 只是将约 0.10 从 R@50 转移到 R@10，Sum 与固定 cross-mean 基本相同。冻结 embedding 后的低秩适配没有产生净增益，下一步应在单模型 encoder/composer 内部定义 CrossPath，而不是继续后处理 adapter/gate 搜索。

完整本地产物：`results/e18_crosspath_adapter_toptee/{manifest.json,history.json,result.json,train.log,best_adapter.pt}`；训练与评测代码：`weave_train_crosspath_adapter.py`；远端 run：`runs/CrossPathAdapter_FashionIQ_toptee_E18_20260824_v1`。

## E19 — Single-endpoint relational CrossPath pilot（2026-08-24，完成；不晋级）

动机：移除双 endpoint 依赖，在单个 DQU-CIR endpoint 内显式构造保真路径与关系路径。给定 composed query \(q\) 和 source image embedding \(s\)，模型计算

\[
d=\operatorname{norm}(q-s),\qquad
q_{\mathrm{rel}}=\operatorname{norm}(q+\alpha(q,s)d),\qquad
q_{\mathrm{out}}=\operatorname{norm}(q+q_{\mathrm{rel}}).
\]

其中 \(\alpha(q,s)\in[-1,1]\) 由两层 scalar head 预测，输入为 \([q,s,q-s,q\odot s]\)。base endpoint 与 gallery 完全冻结，新增 0.525M 参数。toptee pilot 使用 10 epochs、temperature 0.07、relation loss weight 0.5、AdamW、lr 0.001、weight decay 0.0001、batch 128、2048 gallery negatives；沿用 E18 的固定 85/15 train/validation split，checkpoint 只按内部 fused R@10+R@50 选择，official val 只评测一次。

| Variant | R@1 | R@10 | R@50 | Sum R@10+R@50 |
|---|---:|---:|---:|---:|
| 单模型 Base | 32.64 | 65.83 | 85.62 | 151.45 |
| relation path | 32.53 | 65.73 | **85.93** | 151.66 |
| **fused relational CrossPath** | **32.94** | **65.94** | 85.67 | **151.61** |
| 固定双模型 all-mean（E12） | 33.45 | 67.06 | 85.57 | 152.63 |

最佳 checkpoint 为 epoch 9，official step 为 mean 0.057、std 0.183、范围 [−0.499, 0.719]。fused 相对单模型 Base 为 +0.31/+0.10/+0.05 R@1/R@10/R@50，证明 source→query 关系方向含有小幅正信号；但没有超过 E12 固定双模型融合的主指标，故不扩展到 dress/shirt，也不搜索 hidden size、step bound 或 loss weight。

裁决：不作为当前主方法。单 endpoint 关系路径满足简洁性与效率，但增益不足以支撑 CVPR 主表。完整本地产物：`results/e19_relational_crosspath_toptee/{manifest.json,history.json,result.json,train.log,best_composer.pt}`；代码：`weave_train_relational_crosspath.py`；远端 run：`runs/RelationalCrossPath_FashionIQ_toptee_E19_20260824_v1`。

## E20 — Full-gallery composition CrossPath toptee pilot（2026-08-24，完成；通过单类晋级线）

动机：DQU-CIR 的 composed query 为 \(q=\operatorname{norm}(\lambda t+(1-\lambda)v)\)。既有 official-val 诊断显示，固定 encoder 下的 per-query oracle \(\lambda\) 在 toptee 可达到 74.30/89.75 R@10/R@50，而原 head 只有约 65.83/85.62，说明 composition path 选择是明确瓶颈。本实验冻结单个 DQU-CIR endpoint，导出 text path \(t\)、visual path \(v\)、原预测 \(\lambda_0\) 与完整 gallery，新增一个 0.525M-parameter correction head：

\[
\lambda=\operatorname{clip}\!\left(\lambda_0+0.5\tanh h(t,v,t-v,t\odot v,\lambda_0),0,1\right).
\]

head 使用完整训练 gallery 中每 batch 2048 个 negatives 训练；10 epochs、hidden 128、temperature 0.07、AdamW、lr 0.001、weight decay 0.0001、batch 128。checkpoint 只按固定 85/15 internal split 的 R@10+R@50 选择；official-val 训练结束后评一次。随后将改进后的 \(q_0\) 无调参代入已有 2×2 compatibility matrix。

| toptee variant | R@1 | R@10 | R@50 |
|---|---:|---:|---:|
| 单模型 Base | 32.64 | 65.83 | 85.62 |
| Composition head only | 32.99 | 66.04 | 85.77 |
| E12 固定 cross-mean | 33.25 | 67.01 | 85.57 |
| **Composition + cross-mean** | **33.61** | **67.21** | **85.82** |

结果：composition head only 相对单模型 Base 为 +0.36/+0.20/+0.15；与 compatibility matrix 组合后，相对 E12 toptee fixed cross-mean 为 +0.36/+0.20/+0.25，三项同时提升并通过单类晋级线。最佳内部 checkpoint 为 epoch 5，official composition \(\lambda\) mean/std = 0.590/0.102。裁决：固定全部设置，扩展 dress/shirt（E21）。

完整本地产物：`results/e20_composition_toptee/`；代码：`weave_extract_dqu_branches.py`、`weave_train_composition_crosspath.py`、`weave_eval_composition_cross_matrix.py`；远端 run：`runs/CompositionCrossPath_FashionIQ_toptee_E20_20260824_v1`。大于 15MB 的 branch/gallery `.npy` 缓存未复制到公开仓库，可由记录的 endpoint checkpoint 与 extractor 重建。

## E21 — Full-gallery composition CrossPath 三类扩展（2026-08-24，完成；总体不晋级）

设置：E20 的 hidden size、训练目标、negative gallery size、optimizer、epoch、split、seed、checkpoint selection 与 reducer 全部冻结；只将类别扩展到 dress/shirt。每类独立使用对应 DQU-CIR seed42 endpoint 与既有 GradCache endpoint，符合 FashionIQ 类别训练协议。

统一使用 cross-mean 的 official-val 结果：

| 类目 | E12 fixed cross-mean R@10/R@50 | E21 composition + cross-mean | 差值 |
|---|---:|---:|---:|
| dress | 58.11 / 79.47 | **58.50** / 78.98 | +0.40 / −0.50 |
| shirt | **63.44 / 81.06** | 63.00 / 80.91 | −0.44 / −0.15 |
| toptee | 67.01 / 85.57 | **67.21 / 85.82** | +0.20 / +0.25 |
| **三类平均** | 62.85 / **82.03** | **62.91** / 81.90 | +0.05 / −0.13 |

补充统一 reducer 平均：diagonal mean 29.77/62.80/81.86，cross mean 29.87/62.91/81.90，all mean 29.83/62.85/81.91（R@1/R@10/R@50）。MCoT-MVS 公开主表为 63.24/82.01；E21 未超过其 R@10，也未保持 E12 的 R@50。

裁决：总体不晋级，不按 official 为每类选择不同 reducer，也不搜索 head 超参数。E20 证明 full-gallery composition calibration 可与 compatibility paths 叠加，但 shirt 的 internal selection 未迁移，导致三类平均只交换 cutoff 指标。当前 FashionIQ 主结果继续使用 E12 fixed cross-mean 29.95/62.85/82.03。

完整汇总：`results/e21_composition_summary.json`；逐类曲线、manifest、结果与 checkpoint：`results/e21_composition_{dress,shirt}/` 与 `results/e20_composition_toptee/`；远端 run：`runs/CompositionCrossPath_FashionIQ_{dress,shirt}_E21_20260824_v1`。

## E22 — Composition head 100% train fixed-epoch refit（2026-08-24，完成；不采用）

动机：E20/E21 为选择 checkpoint 固定保留 15% train queries；常规最终训练应在 epoch 冻结后使用 100% train data 重新拟合。各类 epoch 直接取 E20/E21 internal 最优：dress=10、shirt=1、toptee=5；其余 hidden size、optimizer、learning rate、negative gallery size、seed 与 reducer 全部不变。refit 期间不再查看 internal 或 official 选择 checkpoint，训练到固定 epoch 后 official 评测一次。

统一 cross-mean 结果：

| 类目 | E21 R@10/R@50 | 100% train refit | 差值 |
|---|---:|---:|---:|
| dress | **58.50 / 78.98** | 57.66 / 78.78 | −0.84 / −0.20 |
| shirt | 63.00 / **80.91** | **63.10** / 80.81 | +0.10 / −0.10 |
| toptee | **67.21** / 85.82 | 67.01 / **85.98** | −0.20 / +0.15 |
| **三类平均** | **62.91 / 81.90** | 62.59 / 81.86 | −0.32 / −0.05 |

裁决：不采用。100% train refit 没有解决类别迁移，反而进一步降低三类平均；因此 E20–E22 composition calibration 路线正式止损，不再重训、调 epoch 或修改 loss。FashionIQ 主结果保持 E12 fixed cross-mean 29.95/62.85/82.03。

完整汇总：`results/e22_composition_refit_summary.json`；逐类 manifest/history/result/checkpoint/joint matrix：`results/e22_composition_refit_{dress,shirt,toptee}/`；远端 run：`runs/CompositionCrossPath_FashionIQ_{dress,shirt,toptee}_E22_refit_20260824_v1`。

## E23 — MCoT-MVS FashionIQ 官方基线复现（2026-08-24，协议已锁定；进行中）

目的：在构造 DQU × MCoT-MVS 异构 CrossPath 前，先原样复现公开强基线，排除 checkpoint、数据协议或实现差异造成的虚假增益。本实验只运行 MCoT-MVS 作者发布的 FashionIQ checkpoint 与官方 `fiq_validate.py`；不修改模型结构、不重训、不调参，也不使用 CrossPath reducer。

公开复现目标来自 MCoT-MVS 论文 Table 2：

| 类目 | R@10 | R@50 |
|---|---:|---:|
| dress | 58.45 | 78.92 |
| shirt | 63.24 | 81.15 |
| toptee | 68.02 | 85.97 |
| **三类平均** | **63.24** | **82.01** |

固定协议：FashionIQ val-split；每类使用作者对应 checkpoint、LLM annotations 与 segmentation features；gallery、query caption 合并、图像预处理和指标计算均沿用作者代码。运行硬件为单张 RTX 4090；checkpoint、分割特征、元数据及代码均记录 SHA-256。正式结果只在资产完整性检查、checkpoint 严格加载和数据样本计数检查通过后生成。

成功标准：三类平均 R@10/R@50 与 63.24/82.01 的绝对差均不超过 0.15，且每类两个指标的绝对差均不超过 0.25。若不满足，先定位协议或环境差异；在复现闭环前不进入异构 CrossPath，也不修改 baseline 以追分。

资产准备状态（2026-08-24）：作者发布的 FashionIQ segmentation features 已下载并通过 ZIP CRC 校验；原始压缩包大小为 675,930,204 bytes，SHA-256 为 `49935ee900783f792abd5e910d596777c8e7352aa74ea1eadce957ce28b46fda`。按官方代码路径解压至 `data/fiq/segment/seg_features_vit-h_patch/`，共 60,033 个 `seg_feature.pt`。val query 的 reference-feature 覆盖为 dress 2017/2017、shirt 2038/2038、toptee 1961/1961，缺失均为 0。

作者发布的 dress checkpoint 大小为 4,947,539,267 bytes，SHA-256 为 `f222cbcd17f1fcc83ac27df4c96b0510c96a0e0032b8b1af051165233100cae8`；shirt checkpoint 大小同为 4,947,539,267 bytes，SHA-256 为 `ed2a5fb7a4d8c98f7e789cd9e6a630fe92a3810299f9a9da842eb67270a61424`。两者均完成本地与服务器端 SHA-256 对照及 PyTorch ZIP 全量 CRC 校验。无卡容器的 cgroup 配额为 0.5 CPU、2 GiB RAM，无法对约 4.95 GB checkpoint 执行 `torch.load`；严格 `load_state_dict` 与指标计算延后至 GPU 模式。

toptee checkpoint 的官方 OneDrive 链接在本地第二跳返回 HTTP 403，在服务器第二跳连接重置；`aria2c` 复核停留于 0 B。用户随后从官方 README 的第四项 checkpoint 下载该文件，并提供 Google Drive mirror（file id `1btYT1q-WMB8nzGMYnO-uUuxIsuXXVB6Y`）。toptee checkpoint 大小为 4,947,539,267 bytes，SHA-256 为 `b23a68d1b660354de834d78e958b40d2101ee1b9779f4d54e656ea62fdc7f84c`；本地与服务器端 SHA-256、PyTorch ZIP 全量 CRC 均通过。未使用 DQU checkpoint 或自行训练权重替代作者权重。至此 E23 三类官方 checkpoint 与 segmentation features 已全部就绪；严格 `load_state_dict` 和指标计算等待 GPU 模式执行。
