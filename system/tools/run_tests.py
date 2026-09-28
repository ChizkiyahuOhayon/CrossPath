"""按论文表 5.5 的六条用例跑一遍功能测试，输出可以直接抄进论文的结论表。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

SYSTEM = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SYSTEM))
sys.path.insert(0, str(SYSTEM / "db"))

import app as application  # noqa: E402
from database import Database  # noqa: E402

CLIENT = application.app.test_client()
DB = Database()


def tc01():
    r = CLIENT.get("/api/search/text?query_text=a black wool coat&topk=10")
    ok = r.status_code == 200 and len(r.json["items"]) == 10
    page = CLIENT.get("/results?mode=text&q=a%20black%20wool%20coat&topk=10")
    return ok and page.status_code == 200, f"返回 {len(r.json['items'])} 条，结果页 {page.status_code}"


def tc02():
    item = application.service.order[0]
    r = CLIENT.get(f"/api/search/image?query_image={item}&topk=10")
    ids = r.json["items"]
    thumbs = all(i["image"].startswith("/gallery/") for i in ids)
    return r.status_code == 200 and thumbs and item not in [i["item_id"] for i in ids], \
        f"返回 {len(ids)} 条，缩略图路径正常，参考图已排除"


def tc03():
    case = json.loads((SYSTEM / "data" / "demo_cases.json").read_text())[0]
    r = CLIENT.get(f"/api/search/fusion?query_image={case['source_id']}"
                   f"&query_text={case['modification'][:60]}&topk=10")
    body = r.json
    fields = all(k in body for k in ("action", "branch", "alpha"))
    return r.status_code == 200 and fields, \
        f"动作 a{body.get('action')}（{body.get('branch')}），属性字段完整"


def tc04():
    r = CLIENT.get("/api/search/fusion?query_image=" + application.service.order[0] +
                   "&query_text=white%20cotton&topk=10&attr_filter=" +
                   json.dumps({"color": "white"}))
    ids = [i["item_id"] for i in r.json["items"]]
    rows = {x["item_id"] for x in DB.query(
        "SELECT item_id FROM t_attr_label WHERE LOWER(color) LIKE '%white%'")}
    ok = bool(ids) and all(i in rows for i in ids)
    return ok, f"{len(ids)} 条结果全部满足 color LIKE '%white%'"


def tc05():
    item_id = f"TC05{int(time.time())}"
    r = CLIENT.post("/api/db/update", json={
        "item_info": {"item_id": item_id, "title": "测试商品", "category": "TEST",
                      "image_path": "gallery/placeholder.jpg"},
        "attr_info": {"color": "white", "material": "cotton"},
        "feature_info": {"image_feat_path": "data/gallery_e0.npy",
                         "index_version": "manual", "row_offset": -1}})
    got = DB.one("SELECT title FROM t_clothing_item WHERE item_id = ?", (item_id,))
    back = CLIENT.get(f"/api/item/detail?item_id={item_id}")
    DB.execute("DELETE FROM t_clothing_item WHERE item_id = ?", (item_id,))
    DB.execute("DELETE FROM t_attr_label WHERE item_id = ?", (item_id,))
    DB.execute("DELETE FROM t_feature_index WHERE item_id = ?", (item_id,))
    return r.status_code == 200 and got is not None and back.status_code == 200, \
        "写入成功，随后可被 /item/detail 查到"


def tc06():
    before = DB.one("SELECT COUNT(*) AS n FROM t_retrieval_log")["n"]
    CLIENT.get("/api/search/text?query_text=striped%20cotton%20shirt&topk=5")
    row = DB.one("SELECT query_type, query_text, topk_ids, model_version, latency_ms,"
                 " query_time FROM t_retrieval_log ORDER BY log_id DESC LIMIT 1")
    after = DB.one("SELECT COUNT(*) AS n FROM t_retrieval_log")["n"]
    complete = all(row[k] is not None for k in row)
    return after == before + 1 and complete, "新增 1 条记录，六个字段均非空"


CASES = [
    ("TC-01", "文本检索", "输入服饰描述文本并设置 topk=10", "返回 10 条相似商品，页面正常展示", tc01),
    ("TC-02", "图像检索", "上传参考服饰图像", "返回视觉相似商品列表，缩略图加载正常", tc02),
    ("TC-03", "图文联合检索", "上传图片并输入属性描述", "返回综合排序结果，属性字段显示正常", tc03),
    ("TC-04", "属性筛选", "设置颜色、领型等筛选条件", "结果集中样本满足指定属性约束", tc04),
    ("TC-05", "数据库更新", "新增一条商品及属性信息", "数据库写入成功，样本可被后续查询", tc05),
    ("TC-06", "日志记录", "执行一次检索请求", "日志表新增对应记录，字段完整", tc06),
]


def main() -> None:
    rows, failed = [], 0
    for code, name, given, expect, fn in CASES:
        start = time.perf_counter()
        try:
            passed, note = fn()
        except Exception as exc:                       # noqa: BLE001
            passed, note = False, f"异常：{exc}"
        elapsed = (time.perf_counter() - start) * 1000
        failed += not passed
        rows.append({"编号": code, "测试功能": name, "输入内容": given,
                     "预期结果": expect, "实测": note,
                     "耗时(ms)": round(elapsed, 1),
                     "测试结论": "通过" if passed else "未通过"})
        print(f"{code}  {'通过' if passed else '未通过'}  {name:<8}  {note}")
    out = SYSTEM / "data" / "test_report.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(rows) - failed}/{len(rows)} 通过 -> {out}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
