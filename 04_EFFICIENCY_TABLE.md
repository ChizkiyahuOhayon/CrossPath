# CrossPath 效率表（RTX 4090）

| Dataset | Gate params | Two gallery embeddings | Boundary candidates/query | Post-embedding latency |
|---|---:|---:|---:|---:|
| FashionGen | 0.526M | 41.34 MiB | 0.37 / 2.02 / 3.85 @ R1/R5/R10 | 4.48 ms/query |
| FashionIQ (3-category mean) | 0.526–1.052M | 22.45 MiB | 0.54 / 4.11 / 15.22 @ R1/R10/R50 | 4.35 ms/query |

FashionIQ shirt/toptee 平均 query encoder latency：Base 12.29 ms，Endpoint 2 12.03 ms；双 endpoint 与 CrossPath gate/policy 合计约 28.68 ms/query。gallery encoding 离线执行。所有数字来自 `results/efficiency/*.json`；每个 latency 使用 500 queries 测量。
