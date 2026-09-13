# CrossPath 效率表（RTX 4090）

| Dataset | Gate params | Two gallery embeddings | Boundary candidates/query | Post-embedding latency |
|---|---:|---:|---:|---:|
| FashionGen | 0.526M | 41.34 MiB | 0.37 / 2.02 / 3.85 @ R1/R5/R10 | 4.48 ms/query |
| FashionIQ (3-category mean) | 0.526–1.052M | 22.45 MiB | 0.54 / 4.11 / 15.22 @ R1/R10/R50 | 4.35 ms/query |

FashionIQ shirt/toptee 平均 query encoder latency：Base 12.29 ms，Endpoint 2 12.03 ms；双 endpoint 与 CrossPath gate/policy 合计约 28.68 ms/query。gallery encoding 离线执行。所有数字来自 `results/efficiency/*.json`；每个 latency 使用 500 queries 测量。

E24 的异构 cross mean 不训练 gate 或融合权重，新增融合参数为 0。MCoT 三类 query encoder 的宏平均为 43.41 ms/query（dress 46.35、shirt 43.29、toptee 40.59），与 DQU GradCache 的 12.03 ms/query 合计约 55.44 ms/query，尚未计入 gallery dot products。MCoT 三类完整 gallery 离线编码分别用时 41.31/50.69/55.57 s；每类两份 1024 维 FP32 gallery embeddings 的平均存储仍为 22.45 MiB。原始计时来自 `results/e24_heterogeneous/mcot/*/manifest.json`。
