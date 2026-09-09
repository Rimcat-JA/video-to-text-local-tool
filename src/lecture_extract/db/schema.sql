-- 設計 9.1: SQLite を解析結果・進捗・参照関係の正本とする。
-- 原画像とモデルの原応答はローカルファイルに置き、ここから参照する。

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS media (
    id              TEXT PRIMARY KEY,
    source_path     TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    size_bytes      INTEGER NOT NULL,
    duration_us     INTEGER NOT NULL,
    container       TEXT,
    stream_info     TEXT NOT NULL,          -- JSON: ffprobe の要約
    time_origin_us  INTEGER NOT NULL,       -- t0
    video_start_us  INTEGER NOT NULL,
    audio_start_us  INTEGER NOT NULL,
    width           INTEGER,
    height          INTEGER,
    nominal_fps     REAL,
    is_vfr          INTEGER NOT NULL DEFAULT 0,
    timeline_flags  TEXT NOT NULL DEFAULT '[]',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS frame_observation (
    id            TEXT PRIMARY KEY,
    media_id      TEXT NOT NULL REFERENCES media(id),
    pts           INTEGER,
    time_base_num INTEGER,
    time_base_den INTEGER,
    t_us          INTEGER NOT NULL,
    image_hash    TEXT,                     -- 保存画像の sha256
    crop          TEXT,                     -- JSON: null なら全画面
    evidence_ref  TEXT,                     -- 保存画像への相対パス
    kind          TEXT NOT NULL DEFAULT 'representative',
    width         INTEGER,
    height        INTEGER
);
CREATE INDEX IF NOT EXISTS idx_frame_t ON frame_observation(media_id, t_us);

CREATE TABLE IF NOT EXISTS extraction_attempt (
    id               TEXT PRIMARY KEY,
    frame_id         TEXT NOT NULL REFERENCES frame_observation(id),
    cache_key        TEXT NOT NULL,
    model_revision   TEXT NOT NULL,
    prompt_version   TEXT NOT NULL,
    runtime_version  TEXT NOT NULL,
    params_hash      TEXT NOT NULL,
    request_kind     TEXT NOT NULL,          -- 'full' | 'crop' | 'tile'
    raw_response_ref TEXT,                   -- 原応答ファイルへの相対パス
    status           TEXT NOT NULL,          -- 'ok' | 'truncated' | 'schema_invalid' | 'error'
    finish_reason    TEXT,
    error            TEXT,
    latency_ms       INTEGER,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempt_cache ON extraction_attempt(cache_key, status);

CREATE TABLE IF NOT EXISTS screen_content (
    id                 TEXT PRIMARY KEY,
    text_hash          TEXT NOT NULL,
    body_text          TEXT NOT NULL,
    regions            TEXT NOT NULL,        -- JSON
    reading_order      TEXT NOT NULL,        -- JSON
    screen_kind        TEXT NOT NULL,
    context            TEXT NOT NULL,        -- JSON
    structure_notes    TEXT NOT NULL,        -- JSON
    quality_flags      TEXT NOT NULL,        -- JSON
    source_attempt_ids TEXT NOT NULL         -- JSON
);
CREATE INDEX IF NOT EXISTS idx_content_hash ON screen_content(text_hash);

CREATE TABLE IF NOT EXISTS screen_occurrence (
    id                    TEXT PRIMARY KEY,
    media_id              TEXT NOT NULL REFERENCES media(id),
    content_id            TEXT REFERENCES screen_content(id),
    start_us              INTEGER NOT NULL,
    end_us                INTEGER NOT NULL,
    boundary_start_lo_us  INTEGER NOT NULL,
    boundary_start_hi_us  INTEGER NOT NULL,
    boundary_end_lo_us    INTEGER NOT NULL,
    boundary_end_hi_us    INTEGER NOT NULL,
    state_kind            TEXT NOT NULL,
    evidence_refs         TEXT NOT NULL DEFAULT '[]',
    parent_block_id       TEXT,
    change_summary        TEXT NOT NULL DEFAULT '',
    quality_flags         TEXT NOT NULL DEFAULT '[]',
    CHECK (end_us >= start_us)
);
CREATE INDEX IF NOT EXISTS idx_occ_time ON screen_occurrence(media_id, start_us);

CREATE TABLE IF NOT EXISTS utterance (
    id            TEXT PRIMARY KEY,
    media_id      TEXT NOT NULL REFERENCES media(id),
    start_us      INTEGER NOT NULL,
    end_us        INTEGER NOT NULL,
    text_raw      TEXT NOT NULL,
    language      TEXT NOT NULL DEFAULT '',
    tokens        TEXT NOT NULL DEFAULT '[]',
    quality_flags TEXT NOT NULL DEFAULT '[]',
    source        TEXT NOT NULL DEFAULT 'asr',
    chunk_id      TEXT NOT NULL DEFAULT '',
    CHECK (end_us >= start_us)
);
CREATE INDEX IF NOT EXISTS idx_utt_time ON utterance(media_id, start_us);

CREATE TABLE IF NOT EXISTS alignment (
    occurrence_id TEXT NOT NULL REFERENCES screen_occurrence(id),
    utterance_id  TEXT NOT NULL REFERENCES utterance(id),
    overlap_us    INTEGER NOT NULL,
    is_primary    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (occurrence_id, utterance_id)
);
CREATE INDEX IF NOT EXISTS idx_align_utt ON alignment(utterance_id);

CREATE TABLE IF NOT EXISTS reading_block (
    id             TEXT PRIMARY KEY,
    media_id       TEXT NOT NULL REFERENCES media(id),
    block_index    INTEGER NOT NULL,
    start_us       INTEGER NOT NULL,
    end_us         INTEGER NOT NULL,
    occurrence_ids TEXT NOT NULL,           -- JSON
    kind           TEXT NOT NULL DEFAULT 'single',
    title_hint     TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_block_time ON reading_block(media_id, start_us);

CREATE TABLE IF NOT EXISTS visual_event (
    id         TEXT PRIMARY KEY,
    media_id   TEXT NOT NULL REFERENCES media(id),
    start_us   INTEGER NOT NULL,
    end_us     INTEGER NOT NULL,
    region_ref TEXT NOT NULL DEFAULT '',
    event_kind TEXT NOT NULL,
    annotation TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_vevent_time ON visual_event(media_id, start_us);

CREATE TABLE IF NOT EXISTS review_item (
    id            TEXT PRIMARY KEY,
    media_id      TEXT NOT NULL REFERENCES media(id),
    target_kind   TEXT NOT NULL,
    target_ref    TEXT NOT NULL,
    reason        TEXT NOT NULL,
    detail        TEXT NOT NULL DEFAULT '',
    resolution    TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT 'open',
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_status ON review_item(media_id, review_status);

CREATE TABLE IF NOT EXISTS coverage_span (
    id       TEXT PRIMARY KEY,
    media_id TEXT NOT NULL REFERENCES media(id),
    track    TEXT NOT NULL,
    start_us INTEGER NOT NULL,
    end_us   INTEGER NOT NULL,
    state    TEXT NOT NULL,
    detail   TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_cov ON coverage_span(media_id, track, start_us);

CREATE TABLE IF NOT EXISTS job (
    id          TEXT PRIMARY KEY,
    media_id    TEXT,
    stage       TEXT NOT NULL,
    input_hash  TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status      TEXT NOT NULL,               -- 'running' | 'done' | 'failed'
    checkpoint  TEXT NOT NULL DEFAULT '{}',  -- JSON
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    error       TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_job_stage ON job(media_id, stage);

-- 設計 9.1: 手修正は元の抽出結果を残したまま訂正版を作る。
CREATE TABLE IF NOT EXISTS correction (
    id              TEXT PRIMARY KEY,
    target_kind     TEXT NOT NULL,
    target_ref      TEXT NOT NULL,
    field           TEXT NOT NULL,
    original_value  TEXT NOT NULL,
    corrected_value TEXT NOT NULL,
    author          TEXT NOT NULL DEFAULT 'user',
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corr_target ON correction(target_kind, target_ref);

-- 設計 12.1: 抽出結果のキャッシュ。
CREATE TABLE IF NOT EXISTS cache_entry (
    cache_key  TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,               -- JSON: 解析済み結果
    attempt_id TEXT,
    created_at TEXT NOT NULL
);

-- 性能計測 (設計 14.2)
CREATE TABLE IF NOT EXISTS metric (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id   TEXT,
    stage      TEXT NOT NULL,
    name       TEXT NOT NULL,
    value      REAL NOT NULL,
    unit       TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
