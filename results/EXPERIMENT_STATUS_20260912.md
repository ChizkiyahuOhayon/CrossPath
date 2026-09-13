# CrossPath 实验总览与下一步裁决（2026-09-12）

## 结论

CrossPath 已形成完整的三数据集证据链。FashionGen 和 FashionIQ 是论文主结果；CIRR 已从 E25 的完整融合退化，推进到 E30/E29 组合相对强 MCoT 提升 `+0.741 Avg`，可作为正向跨数据集泛化结果，但尚未超过 ReCALL 的 `82.81 Avg`。

当前不应继续搜索 query router、融合权重、几何映射或候选级后处理。E30/E31/E32 已覆盖这四类高价值方向；E32 虽提高全局 R@5，却持续损害 subset R@1，未通过内部评测门槛。冻结的 E29+E30 最佳配置已生成 test1 官方提交文件；下一步停止 GPU 实验，上传 CIRR evaluation server 取得最终测试数值，并使用现有结果完成论文定稿与可视化整合。

## 主结果

| 数据集 | 强基线 | CrossPath 当前最佳 | 提升 | 论文定位 |
|---|---|---|---|---|
| FashionGen val | A1 `42.73/79.26/87.75` R@1/5/10 | joint matrix `44.09/80.36/88.28` | `+1.36/+1.10/+0.53` | 主表结果 |
| FashionIQ val | MCoT `63.56/82.33` R@10/50 | DQU GradCache × MCoT `64.58/82.82` | `+1.02/+0.48` | 主表结果 |
| CIRR val | MCoT `81.643 Avg` | E29 re-ranker + 0.25 weak CrossPath `82.385 Avg` | `+0.741 Avg` | 正向泛化，非 SOTA |

CIRR 最佳组合的完整指标为：R@1 `55.78`、R@5 `85.77`、R@10 `92.42`、R@50 `98.40`、subset R@1/2/3 `79.00/92.15/96.99`、Avg `82.385`。

## CIRR 实验链

| 实验 | 方法 | Avg | 相对 MCoT | 裁决 |
|---|---|---:|---:|---|
| E25 | 完整 cross mean | 79.909 | -1.734 | 融合过强，不采用 |
| E29 | query-conditioned segment re-ranker | 81.823 | +0.179 | 单独收益不足 |
| E30 | 0.125/0.25 单向弱融合 | 82.193 | +0.550 | 保留 |
| E30 | oracle-action 分类路由 | 81.643 | +0.000 | 停止 |
| E30 | delta regression 路由 | 81.727 | +0.084 | 停止 |
| E29+E30 | re-ranker + 0.25 弱融合 | **82.385** | **+0.741** | CIRR 当前最佳 |
| E31 | orthogonal map + 0.25 弱融合 | 82.157 | +0.514 | 映射拟合成功但指标下降，不采用 |
| E32 v1 | candidate-wise residual，group weight 1（internal test） | 84.797 | +0.082 | R@5 +0.895、subset R@1 -0.730，停止 |
| E32 v2 | candidate-wise residual，group weight 4（internal test） | 84.845 | +0.130 | R@5 +0.542、subset R@1 -0.283，停止 |
| 上限 | 13-action per-query oracle | 87.144 | +5.501 | 仅说明互补空间，不作模型结果 |

E30 的 train/development/internal-test 为按 source-target 图像对哈希划分的 `19,795/4,184/4,246` 条查询。delta router 在 development 为 `+0.215 Avg`，在未参与阈值选择的 internal test 仅为 `+0.047 Avg`，说明路由信号可以学习但很弱；外部 val 的 `+0.084` 与此一致。E31 将训练图库成对表示余弦从 `0.6080` 提高到 `0.7032`，但检索 Avg 由未映射的 `82.193` 降至 `82.157`，因此不再扩大映射模型。

## 已完成的常规顶会实验组件

- 强基线与公开方法对比：FashionGen、FashionIQ、CIRR 均已具备。
- 主模块与固定路径消融：两个 endpoint、四条原始路径、diagonal/cross/all、learned routing、candidate re-ranking 和 oracle ceiling 均有记录。
- 机制证据：E26 正交坐标扰动保持 diagonal 不变并使 off-diagonal 接近随机；E27 3×3 endpoint matrix 复现方向性。
- 多数据集泛化：FashionGen、FashionIQ、CIRR 三个标准 benchmark。
- 可视化：FashionGen/FashionIQ 原图检索案例、兼容矩阵热力图和 CIRR 检索案例已保存；论文阶段再统一排版，不需要为当前实验重新生成图片。
- 效率与复现：checkpoint、embedding manifest、运行脚本、延迟和存储结果已记录。

## E32 candidate-wise compatibility residual：执行结果

目标不是再调固定 alpha，而是把四条兼容路径直接变成候选级学习信号：

\[
s(q, x)=s_{11}(q,x)+\Delta_\theta\!\left(s_{00},s_{01},s_{10},s_{11}\right),
\]

其中 `s11` 是强 MCoT 分数，`Delta` 的末层零初始化，因此初始排序严格等于 MCoT。训练只使用 CIRR train 的 target、MCoT top-64 hard negatives 和 img_set negatives；DQU/MCoT 全部冻结。与 E30 的区别是它不为整条 query 选择一个全局 action，而是允许不同候选使用不同的四路径证据，直接优化 top-k 内部顺序。

预先设定的通过条件为：

1. CIRR internal test 同时不降低 R@5 和 subset R@1，Avg 至少提高 `0.30`；
2. 冻结后 CIRR val 超过当前 `82.385`；
3. 在 FashionIQ 一个预定类目上方向为正，再扩展三类目。

v1 在 internal test 将 R@5 从 `84.809` 提高到 `85.704`，但 subset R@1 从 `84.621` 降至 `83.891`，Avg 仅 `+0.082`。v2 将 img_set loss 权重从 1 提高到 4，其余配置不变；R@5 达到 `85.351`，subset R@1 为 `84.338`，Avg `+0.130`。两次均未达到 `+0.30 Avg` 且 subset 不降的门槛，因此没有运行 CIRR val，也不扩展到 FashionIQ。

最终方向：不再新增 CIRR 后处理模块，以现有三数据集实验进入论文定稿。方法主线使用 FashionGen learned routing 与 FashionIQ heterogeneous cross paths；CIRR 使用 E29+E30 的 82.385 正向泛化结果，并把 E31/E32 留在实验记录而非正文主表。

## 本地产物与远端缓存

- 汇总表：`results/ALL_RESULTS.md`
- E25：`results/e25_cirr/`
- E29：`results/e29_sate/`，完整 8 epoch 重排记录位于 `reranker_v2_full/`
- E30：`results/e30_cirr_query_router/`
- E31：`results/e31_cirr_procrustes/`
- E32：`results/e32_candidate_compatibility/`
- CIRR 原始高清案例：`results/e25_cirr/qualitative_export/`（54 张官方原图及案例 JSON），压缩包 SHA-256 为 `2eadb9cc3ad3eb7e37916b46f4a8e50ec3eadfc4b9365eb0f26465dbd744ec24`
- E25 固定路径 test1 提交：`results/e25_cirr/test1_submission/`（general/subset 共 6 份 JSON）
- 最佳 E29+E30 test1 提交：`results/e30_cirr_query_router/test1_submission/`；4,148 条 query 的 top-50/top-3 与 source exclusion 已校验
- 最佳 test1 general SHA-256：`a25d01e66452be7d0ddf7a57370b33eafeb5497f4c9dcdadc10881bff27c79c3`
- 最佳 test1 subset SHA-256：`cfc1dc6f712d26d53a6ec0388a03e61701ad47a0bf727511ae09c52cbe43e0da`
- 权重 SHA-256：
  - E29 re-ranker: `9400615c91638f08821316b45c3277374eabdbb8a7dcad623466d1adac65f73d`
  - E30 classification router: `6a284f9007f91a0b1cc6a55dbc9a842bc4d124e636d7fa57b788def83e41309b`
  - E30 delta router: `4eb596e59b46640d44c9ebefe951cdc555e96bada8dcf5e1db7914342cdc748d`
  - E31 orthogonal map: `ef9838f4b44a2d98cd6c236499eb0eb159ba5cce2f842abfee210fb61df109be`
  - E32 group-weight-1 residual: `3ef4eaad132788916b6935189bb9ed0dec666bd590711c39aa263d8839af53c5`
  - E32 group-weight-4 residual: `c4485f9b19ed0401ec8dee3448a810834647550b4398c93f02c915578e0c9dd9`

本机磁盘仅余约 3.1 GiB，因此未复制约 0.7 GiB 的训练/验证 feature cache，也未复制 3.7 GiB 的 DQU checkpoint；这些可再生大文件保留在服务器 `/root/autodl-tmp/weave/runs/E30_CIRR_QueryRouter_20260912/`。本地已保存全部指标 JSON、训练历史、manifest、选中 action、轻量模型权重与生成代码。
