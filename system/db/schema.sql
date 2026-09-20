-- 服饰图文跨模态检索系统 · 数据库核心表
-- 对应学位论文第五章 表5.3「数据库核心表设计」。
-- 六张表在 SQLite 与 MySQL 上共用同一份 DDL；两者的差异（AUTO_INCREMENT /
-- AUTOINCREMENT、ENGINE 子句）由 db/database.py 在执行前做文本替换。

CREATE TABLE IF NOT EXISTS t_clothing_item (
    item_id     VARCHAR(32)  NOT NULL PRIMARY KEY,
    title       VARCHAR(255) NOT NULL,
    category    VARCHAR(64)  NOT NULL,
    season      VARCHAR(32),
    gender      VARCHAR(16),
    price       DECIMAL(10,2),
    image_path  VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS t_text_desc (
    text_id     INTEGER      NOT NULL PRIMARY KEY AUTOINCREMENT,
    item_id     VARCHAR(32)  NOT NULL,
    raw_text    TEXT         NOT NULL,
    clean_text  TEXT         NOT NULL,
    source      VARCHAR(32)  NOT NULL
);

CREATE TABLE IF NOT EXISTS t_attr_label (
    attr_id        INTEGER     NOT NULL PRIMARY KEY AUTOINCREMENT,
    item_id        VARCHAR(32) NOT NULL,
    color          VARCHAR(64),
    neckline       VARCHAR(64),
    sleeve_length  VARCHAR(64),
    material       VARCHAR(64),
    pattern        VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS t_feature_index (
    feat_id          INTEGER     NOT NULL PRIMARY KEY AUTOINCREMENT,
    item_id          VARCHAR(32) NOT NULL,
    image_feat_path  VARCHAR(255) NOT NULL,
    text_feat_path   VARCHAR(255),
    index_version    VARCHAR(32) NOT NULL,
    row_offset       INTEGER     NOT NULL
);

CREATE TABLE IF NOT EXISTS t_model_info (
    model_id        INTEGER      NOT NULL PRIMARY KEY AUTOINCREMENT,
    model_name      VARCHAR(64)  NOT NULL,
    checkpoint_path VARCHAR(255) NOT NULL,
    train_set       VARCHAR(64)  NOT NULL,
    version         VARCHAR(32)  NOT NULL,
    role            VARCHAR(32)  NOT NULL
);

CREATE TABLE IF NOT EXISTS t_retrieval_log (
    log_id        INTEGER      NOT NULL PRIMARY KEY AUTOINCREMENT,
    user_id       VARCHAR(32),
    query_type    VARCHAR(16)  NOT NULL,
    query_text    TEXT,
    query_image   VARCHAR(255),
    topk_ids      TEXT,
    model_version VARCHAR(32),
    latency_ms    REAL,
    query_time    VARCHAR(32)  NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_text_item  ON t_text_desc  (item_id);
CREATE INDEX IF NOT EXISTS ix_attr_item  ON t_attr_label (item_id);
CREATE INDEX IF NOT EXISTS ix_feat_item  ON t_feature_index (item_id);
CREATE INDEX IF NOT EXISTS ix_log_time   ON t_retrieval_log (query_time);
