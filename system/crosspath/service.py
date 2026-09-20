"""模型层与检索服务层：把数据库、索引和第四章模型接在一起。

对外只暴露三个检索模式（对应表 5.4 的 /search/text、/search/image、
/search/fusion），三者共用同一条 CrossPath 推理路径，区别只在于送进查询组合器
C_i 的两路特征里哪一路被置零：

    以文搜图   图像特征置零，只给文本
    以图搜图   文本特征置零，只给图像
    图文联合   两路都给，即第四章的组合式查询

属性筛选在候选进入重排序之前生效，走数据库的 t_attr_label 表。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SYSTEM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYSTEM / "db"))

from crosspath.backbone import EMBED_DIM, SEEDS, Backbone, Endpoint
from crosspath.pipeline import CrossPathRanker
from database import Database
from weave_crosspath_gate import CrossPathGate

DATA = SYSTEM / "data"
INDEX_VERSION = "crosspath-clip-b32-v1"


class RetrievalService:
    def __init__(self) -> None:
        self.db = Database()
        self.order: list[str] = json.loads((DATA / "gallery_order.json").read_text())
        self.position = {item: i for i, item in enumerate(self.order)}
        self.image_features = np.load(DATA / "clip_image.npy")
        self.endpoints = []
        for i in range(2):
            endpoint = Endpoint(EMBED_DIM)
            endpoint.load_state_dict(torch.load(DATA / f"endpoint_{i}.pt")["state_dict"])
            self.endpoints.append(endpoint.eval().requires_grad_(False))
        galleries = [np.load(DATA / f"gallery_e{i}.npy") for i in range(2)]

        gate, threshold = None, 0.0
        gate_path = DATA / "gate.pt"
        if gate_path.exists():
            blob = torch.load(gate_path)
            gate = CrossPathGate(blob["embedding_dim"], num_actions=blob["num_actions"],
                                 hidden_width=128)
            gate.load_state_dict(blob["state_dict"])
            gate.eval()
            threshold = blob["threshold"]
        self.ranker = CrossPathRanker(galleries[0], galleries[1], self.order,
                                      gate=gate, threshold=threshold)
        self.backbone = Backbone()
        self._catalog: dict[str, dict] | None = None

    # -------------------------------------------------------------- 元数据
    @property
    def catalog(self) -> dict[str, dict]:
        if self._catalog is None:
            rows = self.db.query(
                "SELECT c.item_id, c.title, c.category, c.image_path,"
                " a.color, a.neckline, a.sleeve_length, a.material, a.pattern"
                " FROM t_clothing_item c LEFT JOIN t_attr_label a"
                " ON a.item_id = c.item_id")
            self._catalog = {r["item_id"]: r for r in rows}
        return self._catalog

    def item(self, item_id: str) -> dict | None:
        return self.catalog.get(item_id)

    def describe(self, item_id: str, limit: int = 4) -> list[str]:
        rows = self.db.query(
            "SELECT clean_text FROM t_text_desc WHERE item_id = ? AND source = 'fashiongen'"
            " ORDER BY text_id LIMIT ?", (item_id, limit))
        return [r["clean_text"] for r in rows]

    # ---------------------------------------------------------------- 编码
    def text_feature(self, text: str) -> np.ndarray:
        return self.backbone.encode_texts([text])[0]

    def image_feature_for(self, item_id: str) -> np.ndarray:
        return self.image_features[self.position[item_id]]

    def image_feature_from_file(self, path: Path) -> np.ndarray:
        from PIL import Image
        return self.backbone.encode_images([Image.open(path).convert("RGB")])[0]

    def compose(self, image_feature, text_feature) -> tuple[np.ndarray, np.ndarray]:
        """按查询里到底有哪几路模态，走对应的编码分支。

        只有图像时走图库编码器 V_i（以图搜图就是图库空间里的最近邻），其余情况
        走查询组合器 C_i，缺失的那一路特征置零。
        """
        zeros = np.zeros(EMBED_DIM, dtype=np.float32)
        if text_feature is None and image_feature is not None:
            image = torch.from_numpy(image_feature[None])
            with torch.no_grad():
                return (self.endpoints[0].encode_gallery(image).numpy()[0],
                        self.endpoints[1].encode_gallery(image).numpy()[0])
        image = torch.from_numpy((image_feature if image_feature is not None else zeros)[None])
        text = torch.from_numpy((text_feature if text_feature is not None else zeros)[None])
        with torch.no_grad():
            q0 = self.endpoints[0].encode_query(image, text).numpy()[0]
            q1 = self.endpoints[1].encode_query(image, text).numpy()[0]
        return q0, q1

    # ---------------------------------------------------------------- 检索
    def search(self, *, image_feature=None, text_feature=None, top_k: int = 10,
               attr_filter: dict | None = None, exclude: str | None = None) -> dict:
        start = time.perf_counter()
        q0, q1 = self.compose(image_feature, text_feature)
        result = self.ranker.rank(q0, q1, top_k=max(top_k, 10) + 2)
        if exclude:
            # 与论文的评测协议一致：参考图像本身不参与候选排序。
            result["path_recall"] = {
                name: [i for i in ids if i != exclude]
                for name, ids in result["path_recall"].items()}
            for key in ("final_ids", "base_ids"):
                keep = [i for i, item in enumerate(result[key]) if item != exclude]
                score_key = key.replace("_ids", "_scores")
                result[score_key] = [result[score_key][i] for i in keep]
                result[key] = [result[key][i] for i in keep]
        if attr_filter:
            keep = self.filter_ids(attr_filter)
            result["final_ids"] = [i for i in result["final_ids"] if i in keep][:top_k]
            result["base_ids"] = [i for i in result["base_ids"] if i in keep][:top_k]
        else:
            result["final_ids"] = result["final_ids"][:top_k]
            result["base_ids"] = result["base_ids"][:top_k]
        result["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        result["index_version"] = INDEX_VERSION
        return result

    def filter_ids(self, attr_filter: dict) -> set[str]:
        clauses, params = [], []
        for column in ("color", "neckline", "sleeve_length", "material", "pattern"):
            value = (attr_filter.get(column) or "").strip()
            if value:
                clauses.append(f"LOWER({column}) LIKE ?")
                params.append(f"%{value.lower()}%")
        category = (attr_filter.get("category") or "").strip()
        if not clauses and not category:
            return set(self.order)
        sql = "SELECT a.item_id FROM t_attr_label a JOIN t_clothing_item c" \
              " ON c.item_id = a.item_id"
        where = list(clauses)
        if category:
            where.append("LOWER(c.category) LIKE ?")
            params.append(f"%{category.lower()}%")
        sql += " WHERE " + " AND ".join(where)
        return {r["item_id"] for r in self.db.query(sql, params)}

    # ------------------------------------------------------------------ 日志
    def log(self, *, query_type: str, query_text: str | None, query_image: str | None,
            topk_ids: list[str], latency_ms: float, user_id: str = "demo") -> None:
        self.db.execute(
            "INSERT INTO t_retrieval_log (user_id,query_type,query_text,query_image,"
            "topk_ids,model_version,latency_ms,query_time) VALUES (?,?,?,?,?,?,?,?)",
            (user_id, query_type, query_text, query_image, ",".join(topk_ids),
             INDEX_VERSION, latency_ms,
             time.strftime("%Y-%m-%d %H:%M:%S")))

    def register_models(self) -> None:
        """把两个冻结端点和门控写进 t_model_info，界面上要显示版本。"""
        if self.db.query("SELECT model_id FROM t_model_info LIMIT 1"):
            return
        rows = [
            ("CLIP ViT-B/32 (frozen)", "open_clip:ViT-B-32/openai", "WIT-400M",
             "backbone", "主干"),
            ("Endpoint E0", f"data/endpoint_0.pt (seed {SEEDS[0]})", "FashionGen-demo",
             INDEX_VERSION, "基础模型"),
            ("Endpoint E1", f"data/endpoint_1.pt (seed {SEEDS[1]})", "FashionGen-demo",
             INDEX_VERSION, "基础模型"),
            ("CrossPath 边界门控", "data/gate.pt", "FashionGen-demo-internal",
             INDEX_VERSION, "重排序"),
        ]
        self.db.executemany(
            "INSERT INTO t_model_info (model_name,checkpoint_path,train_set,version,role)"
            " VALUES (?,?,?,?,?)", rows)

    def register_feature_index(self) -> None:
        if self.db.query("SELECT feat_id FROM t_feature_index LIMIT 1"):
            return
        rows = [(item, "data/gallery_e0.npy", "data/gallery_e1.npy", INDEX_VERSION, i)
                for i, item in enumerate(self.order)]
        self.db.executemany(
            "INSERT INTO t_feature_index (item_id,image_feat_path,text_feat_path,"
            "index_version,row_offset) VALUES (?,?,?,?,?)", rows)
