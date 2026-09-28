# 服饰图文跨模态检索系统

学位论文第五章描述的系统实现。前端是页面，后端是 Flask，数据落在关系数据库里，
检索链路跑的是第四章的 CrossPath：两个冻结基础模型交叉重组查询编码器与图库编码器，
再按多个召回截断做边界感知自适应重排序。

## 一键启动（推荐，任何系统都一样）

只需要装好 Python 3.9+，不用手动建虚拟环境、不用手动跑训练脚本：

* **macOS**：在 Finder 里双击 `system/start.command`（或在终端里 `./system/start.sh`）
* **Windows**：双击 `system\start.bat`
* **Linux / 任何终端**：`python3 system/bootstrap.py`

第一次跑会自动建虚拟环境、装依赖、建库、训练两个端点和边界门控（笔记本上几分钟，
不需要 GPU），跑完直接打开浏览器到 `http://127.0.0.1:5057`；之后再跑同一个命令，
因为产物都在了，几秒钟直接起服务。想强制重新建库/重新训练：加 `--rebuild`。

`system_venv/`、`system/data/` 下的图库图片/数据库/模型权重都没有随仓库提交
（见仓库根目录 `.gitignore` 的说明——FashionGen 图片有授权限制，不能重分发），
所以这些是一键脚本第一次跑时**现场生成**的，不是打包进仓库的。

`db/build_db.py` 会按顺序找真实数据源，找不到时自动退化，**不会报错中断**：

1. 内部审核清单 `WEAVE_HANDOFF_2026-07-31/server_code_and_results/runs/*/gold_manifest.csv`
   （仓库外部的私有数据，只有能访问这份内部交接资料的人才有，放在 `CrossPath_CVPR2027`
   同级目录即可自动被发现）——有的话建出完整 721 件商品的图库；
2. 论文定性图源图 `paper_assets/source_images/`（同样因为授权原因没进仓库）；
3. 上面两个都没有时，退回到仓库自带、完全原创手绘、随便重新分发都没问题的
   `system/data/demo_placeholder/`（15 件占位商品），保证**只 clone 这个仓库、什么内部数据
   都没有的人，装完依赖也能看到一个真正能搜索出结果的 demo**，只是商品是简笔画不是真实服饰照片。
   `db/build_db.py` 跑完会在终端提示是否用了占位数据。

想重新生成占位图库：`python system/data/demo_placeholder/generate.py`。

### 手动分步搭建（等价于一键脚本内部做的事，方便定制）

```bash
cd CrossPath_CVPR2027                  # 仓库根目录
python3 -m venv system_venv            # 虚拟环境和 system/ 同级，run.sh/bootstrap.py 按这个位置找
system_venv/bin/pip install -r system/requirements.txt

cd system
../system_venv/bin/python db/build_db.py           # 建表、灌数据
../system_venv/bin/python tools/train_endpoints.py # 抽特征、挖三元组、训练两个端点
../system_venv/bin/python tools/train_gate.py       # 训练并标定边界门控

./run.sh                                            # 等价于 python app.py，http://127.0.0.1:5057
python tools/run_tests.py           # 表 5.5 的六条功能用例
python tools/screenshot.py          # 重新抓图 5.3–5.6（另需 playwright）
```

依赖版本见 `requirements.txt`：`flask`、`pillow`、`numpy`、`torch`、`torchvision`、
`open_clip_torch`，截图另需 `playwright`。

## 对应论文的哪几张表

| 论文 | 实现 |
| --- | --- |
| 表 5.2 技术栈 | Python + Flask、MySQL、HTML/JS/CSS |
| 表 5.3 数据库核心表 | `db/schema.sql` 的六张表，字段逐个对齐 |
| 表 5.4 核心接口 | `/api/search/text`、`/api/search/image`、`/api/search/fusion`、`/api/item/detail`、`/api/db/update`、`/api/log/write` |
| 5.3 节七个页面接口 | `/upload`、`/chose`、`/reference`、`/relative`、`/custom`、`/results`、`/get_img` |
| 图 5.3–5.6 | `tools/screenshot.py` 抓出来的四张界面图 |
| 表 5.5 功能测试 | `tools/run_tests.py`，六条全部通过 |

## 分层

```
表现层   templates/ + static/        检索模式选择、上传、属性筛选、结果展示
业务层   app.py                      请求解析、参数校验、任务分发、日志写入
模型层   crosspath/                  端点编码、2×2 兼容矩阵、17 个排序动作、边界门控
数据层   db/                         六张表 + 图库向量索引
```

模型层没有重写论文的算法。`crosspath/pipeline.py` 直接 import 仓库根目录的
`weave_crosspath.py` 和 `weave_crosspath_gate.py`——也就是论文实验用的同一份
秩路径、边界追踪、效用估计和精确回退代码。系统只补了三件工程上的事：把两个端点
的表示凑成兼容矩阵、把三条打分拼成 17 个动作、在 K=1/5/10 上选出唯一排序。

## 数据库

SQLite 开箱即用（`data/crosspath.db`）。要走论文写的 MySQL，设一个环境变量即可，
DDL 和 SQL 完全一样：

```bash
export CROSSPATH_MYSQL_URL="mysql://root:password@127.0.0.1:3306/crosspath"
python db/build_db.py
```

图库里的 721 件商品、类目、描述文本和属性标签全部来自已有的真实标注文件
（`gold_manifest.csv` 与论文定性图的素材），没有生成任何商品。FashionGen 原始
数据不含价格和上架季节，这两列保持为空，界面显示「—」。图片按 256×256 原尺寸
直接拷贝，服务端不缩放、不重编码。

## 两个基础模型

系统要跑起来需要两个**表示空间彼此兼容**的冻结端点——这正是第四章交叉路径成立
的前提。仓库里给的是一对可以在笔记本上重建的演示端点：

* 主干：CLIP ViT-B/32，全程冻结；
* 每个端点带一组轻量头（图库投影 V_i + 查询组合器 C_i），从恒等映射初始化，
  用种子 20260718 / 20260722 各自训练，对应论文 FashionGen 设置里「结构相同、
  独立训练的两个基础模型」。

论文正式实验用的是 ProCIR / DQU-CIR / MCoT-MVS 的检查点，体量在 GB 级、依赖上游
仓库的环境，且要完整数据集才能导出表示，所以没有打包进来。换成那套端点只需要把
`data/gallery_e0.npy`、`data/gallery_e1.npy` 和 `data/endpoint_*.pt` 替换掉，
`crosspath/` 下的代码一行都不用改。**演示端点只用来验证系统功能，它跑出来的检索
指标不能和论文表格里的数字混为一谈。**

## 接口示例

```bash
curl "http://127.0.0.1:5057/api/search/text?query_text=a%20black%20wool%20coat&topk=10"
curl "http://127.0.0.1:5057/api/search/fusion?query_image=2303857&query_text=make%20it%20white&topk=10"
curl -X POST http://127.0.0.1:5057/api/db/update -H 'Content-Type: application/json' \
     -d '{"item_info":{"item_id":"X1","title":"新商品","category":"TOPS","image_path":"gallery/x1.jpg"}}'
```

`/api/search/fusion` 的返回里带着 `action`、`branch`、`alpha`，也就是这次查询
落在哪个排序动作上、属于对角分支还是交叉分支、秩插值系数是多少。结果页把这三个量
和完整的 2×2 兼容矩阵一起画了出来。
