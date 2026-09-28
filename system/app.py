"""服饰图文跨模态检索系统 · Flask 入口。

页面（对应论文图 5.3–5.6）：

    /                首页，检索模式总入口
    /search/text     文字检索图像
    /search/image    图像检索文本
    /results         检索结果页

表 5.4 的 6 个核心接口和第 5.3 节列的 7 个页面接口都在下面，路径与论文一致。
"""

from __future__ import annotations

import json
import random
import sys
import time
import uuid
from pathlib import Path

from flask import (Flask, abort, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "db"))

from crosspath.service import RetrievalService  # noqa: E402

DATA = HERE / "data"
UPLOADS = DATA / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

service = RetrievalService()
service.register_models()
service.register_feature_index()
DEMO_CASES = json.loads((DATA / "demo_cases.json").read_text(encoding="utf-8")) \
    if (DATA / "demo_cases.json").exists() else []


# ---------------------------------------------------------------- 工具函数
def card(item_id: str, score: float | None = None, rank: int | None = None) -> dict:
    meta = service.item(item_id) or {}
    return {
        "item_id": item_id,
        "title": meta.get("title", item_id),
        "category": meta.get("category", "—"),
        "image": url_for("gallery_image", filename=Path(meta.get("image_path", "")).name),
        "color": meta.get("color") or "—",
        "material": meta.get("material") or "—",
        "neckline": meta.get("neckline") or "—",
        "sleeve_length": meta.get("sleeve_length") or "—",
        "pattern": meta.get("pattern") or "—",
        "score": None if score is None else round(score, 4),
        "rank": rank,
    }


def cards(ids, scores=None):
    scores = scores or [None] * len(ids)
    return [card(i, s, r + 1) for r, (i, s) in enumerate(zip(ids, scores))]


@app.context_processor
def inject_globals():
    # 键名避开 dict.items：Jinja 会先命中方法再命中键。
    return {"stats": {
        "item_count": len(service.order),
        "endpoint_count": 2,
        "gate_loaded": service.ranker.gate is not None,
    }}


# -------------------------------------------------------------------- 页面
@app.route("/")
def index():
    models = service.db.query("SELECT model_name, role, version FROM t_model_info")
    recent = service.db.query(
        "SELECT query_type, query_text, query_time FROM t_retrieval_log"
        " ORDER BY log_id DESC LIMIT 5")
    return render_template("index.html", models=models, recent=recent,
                           total=len(service.order))


@app.route("/search/text", methods=["GET", "POST"])
def page_text():
    if request.method == "GET" and not request.args.get("q"):
        history = service.db.query(
            "SELECT DISTINCT query_text FROM t_retrieval_log WHERE query_type='text'"
            " AND query_text IS NOT NULL ORDER BY log_id DESC LIMIT 6")
        return render_template("text_search.html",
                               history=[h["query_text"] for h in history],
                               examples=EXAMPLES)
    query = (request.values.get("q") or "").strip()
    if not query:
        return redirect(url_for("page_text"))
    return redirect(url_for("results", mode="text", q=query,
                            topk=request.values.get("topk", 10)))


@app.route("/search/image", methods=["GET", "POST"])
def page_image():
    if request.method == "GET" and not request.args.get("item"):
        sample_ids = reference_sample(12)
        # 「最近解析结果」展示的是真实的一次以图搜图：分值就是该路径的余弦相似度。
        probe = service.search(image_feature=service.image_feature_for(sample_ids[0]),
                               top_k=5, exclude=sample_ids[0])
        parsed = cards(probe["final_ids"], probe["final_scores"])
        return render_template("image_search.html", samples=cards(sample_ids),
                               parsed=parsed, probe=probe,
                               probe_item=card(sample_ids[0]))
    item_id = request.values.get("item")
    if request.method == "POST" and "file" in request.files and request.files["file"].filename:
        stored = save_upload(request.files["file"])
        return redirect(url_for("results", mode="image", upload=stored,
                                topk=request.values.get("topk", 10)))
    return redirect(url_for("results", mode="image", item=item_id,
                            topk=request.values.get("topk", 10)))


@app.route("/results")
def results():
    mode = request.args.get("mode", "fusion")
    top_k = max(1, min(int(request.args.get("topk", 10)), 24))
    text = (request.args.get("q") or "").strip() or None
    item_id = request.args.get("item") or None
    upload = request.args.get("upload") or None
    order_by = request.args.get("order", "crosspath")
    attr_filter = {k: request.args.get(k) for k in
                   ("color", "material", "neckline", "sleeve_length", "pattern", "category")}

    image_feature = None
    reference = None
    if upload:
        image_feature = service.image_feature_from_file(UPLOADS / upload)
        reference = {"image": url_for("get_img", filename=upload), "title": "用户上传图片",
                     "item_id": "—", "category": "—"}
    elif item_id:
        image_feature = service.image_feature_for(item_id)
        reference = card(item_id)
    text_feature = service.text_feature(text) if text else None
    if image_feature is None and text_feature is None:
        return redirect(url_for("index"))

    outcome = service.search(image_feature=image_feature, text_feature=text_feature,
                             top_k=top_k, attr_filter=attr_filter, exclude=item_id)
    service.log(query_type=mode, query_text=text, query_image=item_id or upload,
                topk_ids=outcome["final_ids"], latency_ms=outcome["latency_ms"])

    shown = outcome["final_ids"] if order_by == "crosspath" else outcome["base_ids"]
    scores = outcome["final_scores"] if order_by == "crosspath" else outcome["base_scores"]
    # 重排序后展示「基线名次 → 当前名次」比展示 S00 分值有意义：分值来自基线路径，
    # 在重排序后的序列里本来就不是单调的。
    base_rank = {item: i + 1 for i, item in enumerate(outcome["base_ids"])}
    rows = cards(shown, scores[:len(shown)])
    for row in rows:
        row["base_rank"] = base_rank.get(row["item_id"])
    return render_template(
        "results.html",
        mode=mode, text=text, reference=reference, order_by=order_by,
        results=rows,
        baseline=cards(outcome["base_ids"][:top_k]),
        outcome=outcome, top_k=top_k,
        attr_filter={k: v for k, v in attr_filter.items() if v},
    )


@app.route("/item/detail")
def item_detail_page():
    item_id = request.args.get("item_id")
    meta = service.item(item_id or "")
    if not meta:
        abort(404)
    return render_template("detail.html", item=card(item_id),
                           descriptions=service.describe(item_id, limit=8))


# --------------------------------------------------- 表 5.4 · 六个核心接口
@app.get("/api/search/text")
def api_search_text():
    query = (request.args.get("query_text") or "").strip()
    if not query:
        return jsonify({"code": 400, "message": "query_text 不能为空"}), 400
    top_k = int(request.args.get("topk", 10))
    outcome = service.search(text_feature=service.text_feature(query), top_k=top_k)
    service.log(query_type="text", query_text=query, query_image=None,
                topk_ids=outcome["final_ids"], latency_ms=outcome["latency_ms"])
    return jsonify({"code": 0, "items": cards(outcome["final_ids"]),
                    "latency_ms": outcome["latency_ms"]})


@app.get("/api/search/image")
def api_search_image():
    item_id = request.args.get("query_image")
    if item_id not in service.position:
        return jsonify({"code": 404, "message": "query_image 不在图库中"}), 404
    top_k = int(request.args.get("topk", 10))
    outcome = service.search(image_feature=service.image_feature_for(item_id), top_k=top_k,
                             exclude=item_id)
    service.log(query_type="image", query_text=None, query_image=item_id,
                topk_ids=outcome["final_ids"], latency_ms=outcome["latency_ms"])
    return jsonify({"code": 0, "items": cards(outcome["final_ids"]),
                    "latency_ms": outcome["latency_ms"]})


@app.get("/api/search/fusion")
def api_search_fusion():
    item_id = request.args.get("query_image")
    query = (request.args.get("query_text") or "").strip()
    if item_id not in service.position or not query:
        return jsonify({"code": 400, "message": "需要同时给出 query_image 与 query_text"}), 400
    attr_filter = json.loads(request.args.get("attr_filter", "{}") or "{}")
    outcome = service.search(image_feature=service.image_feature_for(item_id),
                             text_feature=service.text_feature(query),
                             top_k=int(request.args.get("topk", 10)),
                             attr_filter=attr_filter, exclude=item_id)
    service.log(query_type="fusion", query_text=query, query_image=item_id,
                topk_ids=outcome["final_ids"], latency_ms=outcome["latency_ms"])
    return jsonify({"code": 0, "items": cards(outcome["final_ids"]),
                    "action": outcome["action"], "branch": outcome["branch"],
                    "alpha": outcome["alpha"], "latency_ms": outcome["latency_ms"]})


@app.get("/api/item/detail")
def api_item_detail():
    item_id = request.args.get("item_id", "")
    meta = service.item(item_id)
    if not meta:
        return jsonify({"code": 404, "message": "item_id 不存在"}), 404
    return jsonify({"code": 0, "item": card(item_id),
                    "descriptions": service.describe(item_id, limit=8)})


@app.post("/api/db/update")
def api_db_update():
    payload = request.get_json(silent=True) or {}
    item = payload.get("item_info") or {}
    if not item.get("item_id"):
        return jsonify({"code": 400, "message": "item_info.item_id 必填"}), 400
    service.db.execute(
        "INSERT OR REPLACE INTO t_clothing_item"
        " (item_id,title,category,season,gender,price,image_path) VALUES (?,?,?,?,?,?,?)"
        if service.db.backend == "sqlite" else
        "REPLACE INTO t_clothing_item"
        " (item_id,title,category,season,gender,price,image_path) VALUES (?,?,?,?,?,?,?)",
        (item["item_id"], item.get("title", ""), item.get("category", ""),
         item.get("season"), item.get("gender"), item.get("price"),
         item.get("image_path", "")))
    attrs = payload.get("attr_info") or {}
    if attrs:
        service.db.execute(
            "INSERT INTO t_attr_label (item_id,color,neckline,sleeve_length,material,pattern)"
            " VALUES (?,?,?,?,?,?)",
            (item["item_id"], attrs.get("color"), attrs.get("neckline"),
             attrs.get("sleeve_length"), attrs.get("material"), attrs.get("pattern")))
    feature = payload.get("feature_info") or {}
    if feature:
        service.db.execute(
            "INSERT INTO t_feature_index"
            " (item_id,image_feat_path,text_feat_path,index_version,row_offset)"
            " VALUES (?,?,?,?,?)",
            (item["item_id"], feature.get("image_feat_path", ""),
             feature.get("text_feat_path"), feature.get("index_version", "manual"),
             int(feature.get("row_offset", -1))))
    service._catalog = None
    return jsonify({"code": 0})


@app.post("/api/log/write")
def api_log_write():
    payload = request.get_json(silent=True) or {}
    info = payload.get("query_info") or {}
    service.log(query_type=info.get("query_type", "manual"),
                query_text=info.get("query_text"),
                query_image=info.get("query_image"),
                topk_ids=[str(i) for i in payload.get("result_ids", [])],
                latency_ms=float(info.get("latency_ms", 0.0)),
                user_id=info.get("user_id", "api"))
    return jsonify({"code": 0})


# ------------------------------------------ 第 5.3 节 · 七个页面级接口
@app.post("/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"code": 400, "message": "没有收到文件"}), 400
    stored = save_upload(file)
    return jsonify({"code": 0, "filename": stored, "database": "FashionGen",
                    "url": url_for("get_img", filename=stored)})


@app.get("/chose")
def chose():
    return redirect(url_for("page_image", db=request.args.get("db", "FashionGen")))


@app.get("/reference")
def reference():
    picked = reference_sample(int(request.args.get("n", 8)))
    return jsonify({"code": 0, "database": request.args.get("db", "FashionGen"),
                    "items": cards(picked)})


@app.get("/relative")
def relative():
    """按参考图片回查数据库里已有的相关描述，兼顾真实修改文本和商品描述。"""
    item_id = request.args.get("item")
    if item_id not in service.position:
        return jsonify({"code": 404, "message": "参考图片不在图库中"}), 404
    texts = [c["modification"] for c in DEMO_CASES if c["source_id"] == item_id]
    texts += service.describe(item_id, limit=4)
    return jsonify({"code": 0, "database": "FashionGen", "reference": item_id,
                    "count": len(texts), "captions": texts})


@app.post("/custom")
def custom():
    text = (request.form.get("text") or (request.get_json(silent=True) or {}).get("text") or "").strip()
    if not text:
        return jsonify({"code": 400, "message": "自定义描述为空"}), 400
    return jsonify({"code": 0, "text": text})


@app.get("/get_img/<path:filename>")
def get_img(filename):
    return send_from_directory(UPLOADS, filename)


@app.get("/gallery/<path:filename>")
def gallery_image(filename):
    # 原图直出，不做任何缩放或重编码，前端只用 CSS 控制显示尺寸。
    return send_from_directory(DATA / "gallery", filename)


# -------------------------------------------------------------------- 辅助
EXAMPLES = [
    "a black wool coat with notched lapels",
    "slim fit blue denim jeans with whiskering",
    "long sleeve striped cotton shirt",
    "leather ankle boots with a block heel",
]


def reference_sample(n: int) -> list[str]:
    """从有完整类目和描述的商品里随机取样。

    图库里另有 20 件是论文定性图的素材，只带图片没有类目标注，它们照样可以被
    检索到，但不适合摆进「选一张参考图片」的入口。
    """
    pool = [i for i in service.order
            if (service.item(i) or {}).get("category") != "FASHIONGEN"]
    rng = random.Random(int(time.time() // 30))
    return rng.sample(pool, min(n, len(pool)))


def save_upload(file) -> str:
    suffix = Path(file.filename).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        abort(400)
    stored = f"{uuid.uuid4().hex}{suffix}"
    file.save(UPLOADS / stored)
    return stored


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5057, debug=False)
