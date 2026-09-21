# 交接文档：毕业论文第三章实验复现

交接时间：2026-09-21 13:20 (UTC+8)。接手方：Codex。
本文档写给一个**没有任何前文上下文**的接手者，请通读后再动手。

---

## 0. 三条不可违背的底线

这是委托方的原话，任何情况下不得违背：

1. **所有数字必须来自真实跑出来的实验结果，不允许编造或凑数。**
2. **重跑结果和旧稿数字有 0.1–0.2 个百分点的出入是正常的**（不同随机种子、硬件非确定性），
   如实报告测出来的数字就行，**不需要也不允许为了接近旧数字去调整**。
3. **每个数字要能追溯到具体的实验产物（脚本 + 输出文件）**，不能是"看起来合理"就写上去。

> ⚠️ 关于第 2 条的一个**关键修正**：调查已证实旧稿的表 3.2/3.3/3.5/3.6 和图 3.5/3.6
> **从未真实跑过**（证据见 §6）。因此"0.1–0.2 pp 容差"这个前提不成立——重跑结果与旧稿
> 差几个百分点属于正常，**不是复现失败**。如果差距大到论文论述撑不住，要改的是正文论述，
> 不是数字。**这一点务必在汇报时对委托方讲清楚，不要默默把数字往旧稿上靠。**

另有代码质量要求：代码将开源，需经得起同行挑刺，遵循「最小改动、不做投机性抽象、
不加未被要求的可配置性、每一行改动都能追溯到需求」。

---

## 1. 任务是什么

复现论文《多层自适应特征融合的服饰图文检索模型》第三章 3.3 节的全部实验。

论文当前版本：`/Users/zhao/Desktop/NLP/Thesis/v9/普娟毕业论文-v9-CIRR替换.docx`

四项任务：

| # | 任务 | 状态 |
|---|---|---|
| 1 | 重跑表 3.3 对比实验 + 消融实验，拿到真实实测数字 | 🟡 训练进行中 |
| 2 | 删除表 3.4 和图 3.5，并处理正文中对它们的引用 | ✅ 已完成（v8） |
| 3 | 核查表 3.2 是否真跑过；若是编造的则真跑一遍并写入真实结果 | 🟡 已证实未跑过，实验进行中 |
| 4 | 表 3.2/3.3 数字变化后，同步摘要、结论、其他章节中所有引用处 | ⬜ 待办（等数字出来） |

附加任务（委托方后续追加，均已完成）：
- 第二章 2.5.1 节 Flickr30K → 替换为 CIRR 数据集 ✅（v9）
- 删除孤立的 PFAN 参考文献条目 ✅（v9）

---

## 2. 现在正在发生什么（最重要的一节）

### 两台 AutoDL 4090，双机并行

| | 1 号机 | 2 号机 |
|---|---|---|
| SSH | `ssh -p 16569 root@connect.cqa1.seetacloud.com` | `ssh -p 38822 root@connect.cqa1.seetacloud.com` |
| 密码 | `128YdeOGLBL3` | `N084ZLMIYIL0` |
| 本机别名 | `ssh ch3gpu`（已装公钥，免密） | `ssh ch3gpu2`（已装公钥，免密） |
| 状态 | **运行中**（阶段一预训练） | **已关机**（待 09-22 05:30 前开机） |
| GPU | RTX 4090 24 GB | RTX 4090 24 GB |

两台机器的环境**逐项相同**（已核对，不是"差不多"）：
`Python 3.12.3 / torch 2.8.0+cu128 / timm 1.0.28 / transformers 5.3.0 / h5py 3.14.0 / numpy 2.3.2`

1 号机训练用 `/root/autodl-tmp/envs/procir-eval/bin/python`；
2 号机用 `/root/miniconda3/bin/python`（两者版本相同）。

### 阶段一：领域预训练（1 号机，进行中）

```
启动     2026-09-21 03:54:57 (UTC+8)
进度     ep10 / 30，0.224 s/step，16280 step/epoch，约 1.02 h/epoch
loss     ep7 2.703 → ep8 2.490 → ep9 2.308（正常下降）
screen   26091.stage1
日志     /root/autodl-tmp/ch3/runs/stage1.log
逐轮记录 /root/autodl-tmp/ch3/runs/stage1_pretrain/train_log.jsonl
```

查看进度：
```bash
ssh ch3gpu 'tail -3 /root/autodl-tmp/ch3/runs/stage1.log; tail -3 /root/autodl-tmp/ch3/runs/stage1_pretrain/train_log.jsonl'
```

**关键时间点：epoch25.pth 预计 2026-09-22 05:30 (UTC+8) 落盘**，这是阶段二唯一的起跑门槛。

### 阶段二：检索微调（12 个配置，尚未开始）

12 个 run 全部从同一个 `epoch25.pth` 出发、互不依赖，所以拆两台机器并行：

**1 号机（5 个 run，≈19 h）** — 编排脚本 `ch3_run_stage2_m1.sh`，已在 `screen 42753.stage2` 中等待：
```
m5 (--full-rerank)  m1  m2  m3  m4      → 表3.3 + 表3.5（消融梯队）
```

**2 号机（7 个 run，≈25 h）** — 编排脚本 `ch3_run_stage2_m2.sh`，**开机后需手动启动**：
```
L12  L8  L4_L8  L8_L12                  → 表3.6（层数消融）
lam0_1  lam0_3  lam1_0                  → 图3.6（λ 敏感性）
```

> 注意命名：`ch3_collect_results.py` 里 **`table_3_6_layers` 是层数表、`figure_3_6` 是 λ 图**。
> 别弄反。

**checkpoint 传输是自动的**：1 号机脚本在 `epoch25.pth` 落盘后会起一个后台重试循环，
每 5 分钟尝试 `rsync` 到 2 号机，直到成功。2 号机哪怕关机 20 小时，开机后自己就收到了。
1 号机 → 2 号机的免密 SSH（别名 `m2`）已打通。

### 预计完成时间

```
09-22 05:30   epoch25.pth 落盘 → 双机同时开跑阶段二
09-23 06:30   全部 12 个 run 结束
```
（单机串行的话是 09-24 03:30，双机省约 21 小时。）

---

## 3. 开机后要做的事（2 号机）

委托方在 09-22 05:30 前于 AutoDL 控制台开机后，执行：

```bash
ssh ch3gpu2 'screen -dmS stage2 bash -c "exec >> /root/autodl-tmp/ch3/runs/stage2.log 2>&1; bash /root/autodl-tmp/ch3/repo/ch3_run_stage2_m2.sh"'
```

2 号机上留了同样内容的备忘：`/root/autodl-tmp/ch3/RESUME_AFTER_BOOT.txt`。

脚本会自己等 `/root/autodl-tmp/ch3/runs/epoch25.pth` 出现再开跑，不用等人盯。

---

## 4. ⚠️ 磁盘：1 号机会写满，必须处理

**这是交接时最紧迫的技术风险。**

1 号机 `/root/autodl-tmp` 共 260 G，**当前仅剩 19 G**。其余空间被委托方的其他项目占着
（`weave` 64G、`mcot_mvs` 37G、`dqucir_2026-08-05` 28G、`cache` 31G、`envs` 29G）——
**这些都不是本任务的产物，不要删。**

算账：
- 阶段一按 `--keep-epochs 25 --keep-last 2` 会存 epoch25 / epoch29 / epoch30 三个 ×3.7 G = **11.1 G**
- 加上 `resume.pth` 3.77 G → 阶段一结束时占 **≈15 G**
- 阶段二原计划每个 run 留 3.7 G ×5 = 18.5 G → **必然写满**

**已做的缓解**：`ch3_run_stage2_m1.sh` 中 m1–m4 的 `--keep-last` 由 1 改为 0，
只保留 m5 的最终权重。表格数字读的是 `train_log.jsonl` 而非权重，所以不影响任何结果。
省约 14.8 G。（脚本已改，编排进程已重启加载新版本。）

**仍需接手方执行的清理**（按需，从安全到激进）：

| 可删对象 | 大小 | 安全性 |
|---|---|---|
| `/root/autodl-tmp/ch3/smoke` | 15 M | 随时可删，冒烟测试残留 |
| `/root/autodl-tmp/ch3/ckpt/ALBEF.pth` | 3.3 G | **epoch 25 过后**可删（仅阶段一初始化用；从 `resume.pth` 续跑不需要它） |
| `stage1_pretrain/resume.pth` | 3.77 G | 阶段一**完全结束后**可删 |
| `stage1_pretrain/epoch29.pth` `epoch30.pth` | 7.4 G | 阶段二只从 epoch25 出发，二者无人使用——但属判断题，删前请示委托方 |

全删可回收 ≈14.5 G → 约 33 G 可用，足够。

**每次汇报请顺带报一次 `df -h /root/autodl-tmp`。**

---

## 5. 代码在哪、怎么组织

### 本地仓库
`/Users/zhao/Desktop/NLP/CrossPath_CVPR2027`，分支 **`ch3-mlaff`**

```
1685c7c  Split stage 2 across two GPUs
4c461cf  Fill the rerank batches across images
10e80fb  Initialise pre-training from ALBEF, and swap Flickr30K for CIRR
ee98b03  Add the chapter-3 multi-level adaptive feature fusion model
```

> ⚠️ **尚未 push。** 本地 remote 是 `git@github.com:ChizkiyahuOhayon/CrossPath.git`，
> 但委托方口头指定的是 `https://github.com/Weirdbuf/CrossPath`。两个仓库都存在、都不是 fork、
> HEAD 的 commit message 相同但 SHA 不同。**push 前必须先跟委托方确认推哪个。**

### 模型代码 `ch3_mlaff/`

| 文件 | 内容 |
|---|---|
| `model_ch3.py` | `Ch3Config` / `CrossModalBlock`（式 3.1–3.7）/ `TextGuidedGate`（式 3.9）/ `Ch3Model`，含 ITC/ITM/REA 损失与 `score_pairs` |
| `encoders_ch3.py` | `MultiLayerViT`（timm ViT-B/16，取 block 4/8/12）/ `SplitBert`（6 层文本 + 6 层融合） |
| `albef_init.py` | 把官方 `ALBEF.pth` 映射到本模型 |
| `data_fashiongen.py` | FashionGen h5 读取、Sample/Full 协议索引构建 |
| `eval_ch3.py` | `evaluate_sample` / `evaluate_full` / 成对重排 |
| `train_ch3.py` | 两阶段训练入口，写 `run_manifest.json` 与 `train_log.jsonl` |
| `pretrain_ch3.py` | 阶段一的 MLM 头与包装模型 |
| `configs_ch3.py` | 17 个名字、**12 个不同的 run**（有别名，见下） |
| `README.md` | 公式→代码对照表 + **6 条设计决策**（§7 必读） |

配置别名（容易重复跑，注意）：
`baseline`≡`m1`；`full`≡`L4_L8_L12`≡`lam0_5`≡`m5`；`lam0_0`≡`m4`。

### 论文编辑工具（纯 Python 操作 .docx XML，不依赖 Word）

| 文件 | 用途 |
|---|---|
| `ch3_thesis_docx.py` | `Document` 类：`find` / `delete` / `replace` / `truncate_after` / `set_table` / `set_paragraph` / `renumber_plan` |
| `ch3_apply_thesis_edits.py` | 任务 2：删表 3.4 / 图 3.5 并重编号 → v8 |
| `ch3_replace_flickr_with_cirr.py` | Flickr30K → CIRR 替换 + 删 PFAN → v9 |
| `ch3_collect_results.py` | 从各 run 的日志汇总成表格 + 每个数字的来源记录 |

### 测试

```bash
cd /Users/zhao/Desktop/NLP/CrossPath_CVPR2027
python3 -m pytest test_ch3_mlaff.py test_ch3_thesis_docx.py test_ch3_albef_init.py -q
# 当前：69 passed, 5 skipped   （skip 的 5 个需要 timm，本地没装）
```

几个**不要删**的关键回归测试：
- `test_cross_attention_and_gate_receive_gradient` —— 守护 §7 决策 1 的致命缺陷
- `test_evaluate_sample_reads_the_pairwise_score_not_the_cls_matrix`
- `test_set_paragraph_does_not_trap_the_text_inside_a_leading_hyperlink`
- `test_truncate_after_removes_whole_fields_in_the_tail`

### 服务器目录

```
/root/autodl-tmp/ch3/
├── frozen_<fingerprint>/     ← 冻结的代码快照，训练**只从这里跑**
├── repo/                     ← 可改的脚本副本（编排脚本放这）
├── data/fashiongen_h5/       ← 两个 h5
├── data/index/               ← 预构建的划分索引 JSON
├── ckpt/ALBEF.pth            ← 阶段一初始化权重
└── runs/                     ← 所有输出
```

**可复现性约定：训练永远从 `frozen_<fingerprint>/` 跑，且该目录在任务运行期间绝对不能改。**
`run_manifest.json` 里记了 `code_sha256`；当前值 `a9e5c558e2a9…`，两台机器一致（已核对）。

---

## 6. 已经查明的事实（不要重新调查）

### 表 3.2/3.3/3.5/3.6、图 3.5/3.6 从未真实跑过

证据：论文 v1–v6（截至 2026-09-14）的第三章是**完全不同的内容**（CrossPath 组合检索）；
"多层自适应"在 v1–v6 中出现 **0 次**、在 v7 中出现 13 次；数字 71.70 / 75.10 / 73.45
在 v7 之前的任何版本中都**不存在**。结论：这些表格没有任何实验支撑。

### 论文 3.3.1 的超参数是逐字抄自 FashionSAP 的配置文件

`configs/fashion_pretrain.yaml`：lr 6e-5 / 30 epochs / min_lr 1e-5 / warmup 5 /
warmup_lr 1e-5 / batch 16 / queue 65536 / momentum 0.995 / alpha 0.4 / temp 0.07
—— 与 3.3.1 完全一致。这本身又是"只抄配置、没跑实验"的佐证。
我方已据此把 `queue_size` 从 16384 改为 65536 对齐。

### 数据集

FashionGen 从 `hieupth/fashiongen`（经 hf-mirror.com）下载，两台机器的 SHA-256 均已核对：
```
train       bb55646511c2656bf59cb09aec3f188697f9bf2f635c3feef61bc626fbc8f81f
validation  f96d7630301328ef8b8a8e84c6fbdbb17e7c4fea8c2cd1186cde1133e4eff689
```
与 `cinder/configs/fashiongen_e0.json` 中历史记录的值一致。

### ALBEF 初始化（一个重要修正）

FashionSAP 的 `fashion_pretrain.py` 默认 `--pre_point ALBEF.pth`——它的领域预训练是
**从 ALBEF 的 400 万图文对预训练权重开始的**。最初我方从 ImageNet ViT + bert-base 起步，
等于拿没见过图文预训练的模型去对标见过的数字。现已改为从官方 ALBEF.pth 初始化：
**420 个张量全部加载，随机初始化 0 项，checkpoint 中 0 个张量被浪费。**
m5 检索模型中只有 54 个第三章模块张量保持随机（本来就该随机）。

### CIRR 统计（写进 v9 的数字）

不是引用论文，是从官方 `split.rc2.*.json` / `cap.rc2.*.json` 数出来的：
```
train 16,939 images / 28,225 triplets ; val 2,297 / 4,181 ; test1 2,315 / 4,148
合计 21,551 images / 36,554 triplets ; subset 大小恰为 6
```
（原论文称 21,552，去重后的并集是 21,551，按数出来的写。）脚本：`cirr_stats.py`。

---

## 7. 已锁定的设计决策（不要重新讨论）

完整版见 `ch3_mlaff/README.md` 的「Decisions the thesis text does not fix」。摘要：

1. **`F_final` 接进 `L_itc`。** 论文 3.2 把 `F_final` 当作检索表示，但它同时依赖图像和文本，
   不可能充当表 3.1 那个单一 `sim_t2i` 矩阵所隐含的图库嵌入。若只按 CLS 余弦解读，
   `F_final` 就只能通过 `L_rea` 获得梯度——**那样表 3.5 的第 1–4 行会训练出完全相同的模型，
   cross-attention 模块收到的梯度精确为 0**（实测：m3 的 `grad|cross_blocks| = 0.000000`，
   m5 为 2.068865）。此缺陷在烧掉 40+ GPU 小时**之前**被发现并上报，委托方选择了
   「接进 ITC 损失」方案。`test_cross_attention_and_gate_receive_gradient` 守护此项。
2. `L_rea` 不用动量队列、不用软标签，有独立温度 `tau_r`；m5 与 m4 的差别是更锐的批内约束，
   而非重复的损失项。
3. **基线是朴素双塔**（ITC+ITM，仅第 12 层特征），不是完整的 FashionSAP 复现。
4. 阶段一目标函数 = ITC + ITM + MLM。
5. **报告最后一个 epoch，不是最好的那个**（FashionGen 的验证集即评测集，选最好等于在被报告的
   数字上做选择）。每个 epoch 都留在 `train_log.jsonl` 里。
6. Sample 协议的 1000 个 query 及其 100 个负例用固定种子抽取（`--sample-seed 20260920`），
   记录在 manifest 中，所有配置完全一致。

---

## 8. 已知的坑（都是真踩过的，别再踩）

| 坑 | 后果 | 正确做法 |
|---|---|---|
| `pkill -f "<模式>"` | **会匹配到自己的 ssh 命令行，直接杀掉会话（exit 255）。已发生两次。** | `pgrep -f "ch3_mlaff[.]train_ch3"` 拿 PID 再 `kill`，或用方括号技巧 |
| `grep -c "[m]ch3_mlaff"` | 永远匹配不到，会误判训练已死 | 用 `pgrep -af` |
| 同时用 `setsid nohup` 和 `screen -dmS` 启动 | 双重启动竞争，一个 OOM、日志被覆写 | 只用 `screen`，并在启动脚本内 `exec >> log 2>&1` |
| 通过 ssh 传嵌套 heredoc | 被 shell 展开/弄乱 | `rsync` 脚本文件过去再执行 |
| 往**正在运行**的冻结代码目录里 rsync 新代码 | 破坏代码指纹的可复现性保证 | 改代码 → 新建冻结快照 → 重启任务 |
| `rsync` 覆盖**正在运行**的 bash 脚本 | 旧 inode 仍被持有，改动**不生效**（易误判已生效） | 改完必须重启编排进程 |
| 训练期间跑 pytest / benchmark | GPU 争用，实测 s/step 从 0.229 虚高到 0.382 | 别在训练机上跑基准 |
| 重排逐张读图 | 单线程 ~121 img/s，全部 Sample 评测要 ~55 小时纯 I/O | 已修：DataLoader 按图序流式 + 跨图像凑批，1,289 pairs/s |

---

## 9. 待办清单

### 训练期间
- [ ] 定期汇报进度（委托方明确要求**中途汇报，不要几十小时后才出现**）。
      建议节奏：阶段一每 5 个 epoch 一次、阶段二每个 run 结束一次。
- [ ] 每次汇报带上 `df -h /root/autodl-tmp`（见 §4）
- [ ] 09-22 05:30 前提醒委托方开 2 号机，开机后执行 §3 的命令
- [ ] 按 §4 清理磁盘

### 跑完之后
- [ ] **先把 2 号机的 run 目录合并到 1 号机**，再汇总。`ch3_collect_results.py` 只接受单个
      `--runs` 目录（双机拆分是后加的，脚本没跟着改）：
      ```bash
      # 在 1 号机上执行
      ssh ch3gpu 'for c in L12 L8 L4_L8 L8_L12 lam0_1 lam0_3 lam1_0; do \
          rsync -a m2:/root/autodl-tmp/ch3/runs/$c /root/autodl-tmp/ch3/runs/; done'
      ssh ch3gpu '/root/autodl-tmp/envs/procir-eval/bin/python \
          /root/autodl-tmp/ch3/repo/ch3_collect_results.py \
          --runs /root/autodl-tmp/ch3/runs --out /root/autodl-tmp/ch3/runs/ch3_results.json'
      ```
      合并后请确认 12 个 run 目录齐全（每个都应有 `DONE`、`run_manifest.json`、`train_log.jsonl`）。
- [ ] 把真实数字写入表 3.2 / 3.3 / 3.5 / 3.6 / 图 3.6
- [ ] **任务 4**：同步摘要（中/英）、结论、以及其他章节中所有引用这些数字的地方
- [ ] 每个数字附上来源（run 名、日志路径、epoch、`code_sha256`、初始化 checkpoint）

### 需要跟委托方确认的开放问题
- [ ] **push 到哪个 GitHub 仓库**（`Weirdbuf` vs 本地 remote `ChizkiyahuOhayon`）——未经确认不要 push
- [ ] **第四章是否引用 CIRR 结果**。目前 CIRR 在论文正文中出现 **0 次**，而仓库里有真实的
      CIRR 实验结果（`results/e25_cirr/val_summary.json`，官方 val 4181 query / 2297 gallery）。
      若第四章不引用，新写的 2.5.1 CIRR 节就和刚被删掉的 Flickr30K 犯同样的毛病（孤立数据集）。
- [ ] **表 3.3 缺来源注**（表 3.2 有，注明引自 [42]；表 3.3 的 EI-CLIP / ALBEF / FashionViL
      数字没有任何出处说明）
- [ ] **第一章对第四章的描述与第四章正文不符**：第一章说「类别—属性语义引导」，
      第四章正文是「跨模型编码器重组（CrossPath）」

---

## 10. 对结果的预判（供汇报时参考）

我方环境现已与 FashionSAP 高度对齐（ALBEF 初始化、queue 65536、batch 16、30+20 epoch、
学习率表、256×256、`max_word_num=180`、文本前缀、评测协议代码均一致）。

仍存在两处对不齐，也是预判偏保守的原因：

1. **FashionSAP 用 4 卡 DDP，有效 batch 64；我方单卡 batch 16。**
   对比学习对有效批量敏感，这是最大的剩余差距。论文明写 batch 16，**不应为了刷高而改**。
2. **FashionSAP 的预训练含符号提示与属性预测**（那正是它的论文贡献），
   而论文 3.3.5 把模型 1 定义为"基础双流模型"，所以我方 Baseline† 实质是
   "ALBEF + FashionGen 领域适配"，比真正的 FashionSAP 弱。

由此判断：
- **Baseline† 大概率低于旧稿的 71.70**；
- **m5 相对 m1 能否涨到旧稿宣称的 +3.40 (I2T) / +1.20 (T2I)，是真正的未知数**；
- **表 3.5 那种 m1<m2<m3<m4<m5 的完美单调基本不会出现**，真实消融出现一两处倒挂是常态。

再次强调 §0 的底线：**测出多少报多少，不调参去靠近旧数字。**

---

## 11. 给 Codex 的启动 prompt

> 你接手一个毕业论文第三章的实验复现任务。请先完整阅读仓库根目录的 `HANDOFF_CH3.md`，
> 然后阅读 `ch3_mlaff/README.md` 中的「Decisions the thesis text does not fix」一节。
>
> 当前状态：两台 AutoDL 4090（别名 `ch3gpu` / `ch3gpu2`，已配置免密 SSH）。1 号机正在跑
> 阶段一领域预训练（约 09-22 05:30 完成 epoch 25），2 号机已关机待命。阶段二的 12 个
> 微调任务已编排好并拆分到两台机器，预计 09-23 06:30 全部结束。
>
> 你的职责：
> 1. 监控训练，按 `HANDOFF_CH3.md` §9 的节奏向委托方汇报进度（不要沉默几十小时）；
> 2. 处理 §4 的磁盘风险（1 号机仅剩 19 G，会在阶段二写满）；
> 3. 跑完后用 `ch3_collect_results.py` 汇总真实数字，写入论文表 3.2/3.3/3.5/3.6 和图 3.6，
>    并同步摘要、结论、其他章节中的所有引用处；
> 4. 严格遵守 §0 的三条底线——特别是：**旧稿数字从未真实跑过，因此不存在"对不上就是错了"
>    这回事，绝不允许为了接近旧数字而调参。**
>
> 动手前若有任何理解偏差或不确定处，先提出来问，不要自行假设。
