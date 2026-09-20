"""起一个真实的 Flask 进程，用 Chromium 抓第五章需要的四张界面截图。

截图按 3 倍设备像素比渲染（1440 逻辑宽 → 4320 物理像素），论文里按整栏
宽度 14 cm 排版时仍在 700 dpi 以上，不需要任何放大。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

SYSTEM = Path(__file__).resolve().parent.parent
OUT = SYSTEM / "data" / "screenshots"
PORT = 5057
BASE = f"http://127.0.0.1:{PORT}"

# 图 5.6 用一条真实的重排序样例（DRESSES 类目）：目标商品在基础模型排序里
# 排第 2，CrossPath 选中动作 a6 之后升到第 1。
SHOWCASE = {
    "item": "2303857",
    "q": "change the colour to white, make the sleeves short, use cotton jersey instead.",
}

SHOTS = [
    ("fig5_3_homepage", "/", "图5.3 系统首页页面"),
    ("fig5_4_text_search", "/search/text", "图5.4 文字检索图像界面"),
    ("fig5_5_image_search", "/search/image", "图5.5 图搜文界面"),
    ("fig5_6_results",
     f"/results?mode=fusion&item={SHOWCASE['item']}&q={SHOWCASE['q'].replace(' ', '%20')}&topk=10",
     "图5.6 检索结果界面"),
]


def wait_for_server(timeout: float = 90.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urlopen(BASE + "/", timeout=2).read(64)
            return
        except Exception:
            time.sleep(1.0)
    raise RuntimeError("Flask 没有在超时时间内起来")


def warm_up() -> None:
    """清空日志表再发几次检索，让首页的「最近检索记录」只剩本次的真实记录。"""
    sys.path.insert(0, str(SYSTEM / "db"))
    from database import Database
    Database().execute("DELETE FROM t_retrieval_log")
    for query in ("a black wool coat with notched lapels",
                  "slim fit blue denim jeans with whiskering",
                  "long sleeve striped cotton shirt"):
        urlopen(BASE + "/api/search/text?query_text=" +
                query.replace(" ", "%20") + "&topk=10", timeout=60).read()


def main() -> None:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [str(SYSTEM.parent / "system_venv" / "bin" / "python"), str(SYSTEM / "app.py")],
        cwd=SYSTEM, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
        warm_up()
        manifest = []
        with sync_playwright() as play:
            browser = play.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 980},
                                    device_scale_factor=3)
            for name, path, caption in SHOTS:
                page.goto(BASE + path, wait_until="networkidle")
                page.wait_for_timeout(700)
                target = OUT / f"{name}.png"
                page.screenshot(path=str(target), full_page=True)
                manifest.append({"name": name, "path": path, "caption": caption,
                                 "file": target.name})
                print(f"{caption:<24} -> {target.name}")
            browser.close()
        (OUT / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
