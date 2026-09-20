"""数据层：统一的连接、建表和查询封装。

论文第五章表 5.2 把数据存储定为 MySQL。开发机上不一定跑得起 MySQL，所以这里
保留同一套 DDL 和同一套 SQL，只在连接层分流：设了 ``CROSSPATH_MYSQL_URL`` 就
连 MySQL，没设就落到同目录下的 SQLite 文件。两条路径共用 schema.sql。
"""

from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
SCHEMA = Path(__file__).with_name("schema.sql")
SQLITE_PATH = ROOT / "data" / "crosspath.db"


def _mysql_url() -> str | None:
    return os.environ.get("CROSSPATH_MYSQL_URL") or None


class Database:
    """极薄的 DB-API 包装，只提供本系统用得到的四个动作。"""

    def __init__(self, url: str | None = None) -> None:
        self.url = url if url is not None else _mysql_url()
        self.backend = "mysql" if self.url else "sqlite"
        self._conn = None

    # ---------------------------------------------------------------- 连接
    def connect(self):
        if self._conn is not None:
            return self._conn
        if self.backend == "mysql":
            import pymysql  # 只在真的要连 MySQL 时才 import

            from urllib.parse import urlparse

            u = urlparse(self.url)
            self._conn = pymysql.connect(
                host=u.hostname or "127.0.0.1",
                port=u.port or 3306,
                user=u.username or "root",
                password=u.password or "",
                database=(u.path or "/crosspath").lstrip("/"),
                charset="utf8mb4",
                autocommit=True,
            )
        else:
            SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------ 占位符
    @property
    def ph(self) -> str:
        return "%s" if self.backend == "mysql" else "?"

    def _bind(self, sql: str) -> str:
        return sql.replace("?", self.ph) if self.backend == "mysql" else sql

    # -------------------------------------------------------------- 查询
    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict]:
        cur = self.connect().cursor()
        cur.execute(self._bind(sql), tuple(params))
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.close()
        return rows

    def one(self, sql: str, params: Sequence[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.execute(self._bind(sql), tuple(params))
        cur.close()
        if self.backend == "sqlite":
            conn.commit()

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        conn = self.connect()
        cur = conn.cursor()
        cur.executemany(self._bind(sql), [tuple(r) for r in rows])
        cur.close()
        if self.backend == "sqlite":
            conn.commit()

    # -------------------------------------------------------------- 建表
    def create_schema(self) -> None:
        ddl = SCHEMA.read_text(encoding="utf-8")
        if self.backend == "mysql":
            # MySQL 用 AUTO_INCREMENT，且 AUTO_INCREMENT 列必须是 INT 主键。
            ddl = ddl.replace("INTEGER      NOT NULL PRIMARY KEY AUTOINCREMENT",
                              "INT NOT NULL AUTO_INCREMENT PRIMARY KEY")
            ddl = ddl.replace("INTEGER     NOT NULL PRIMARY KEY AUTOINCREMENT",
                              "INT NOT NULL AUTO_INCREMENT PRIMARY KEY")
            ddl = re.sub(r"CREATE INDEX IF NOT EXISTS[^;]+;", "", ddl)
            ddl = re.sub(r"(\)\s*);", r"\1 ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;", ddl)
        conn = self.connect()
        cur = conn.cursor()
        for stmt in [s.strip() for s in ddl.split(";") if s.strip()]:
            cur.execute(stmt)
        cur.close()
        if self.backend == "sqlite":
            conn.commit()

    def drop_all(self) -> None:
        conn = self.connect()
        cur = conn.cursor()
        for t in ("t_retrieval_log", "t_model_info", "t_feature_index",
                  "t_attr_label", "t_text_desc", "t_clothing_item"):
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.close()
        if self.backend == "sqlite":
            conn.commit()


def get_db() -> Database:
    return Database()
