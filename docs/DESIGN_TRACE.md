# 設計案と実装の対応表

`local_lecture_extraction_design.md`（版 0.1）の各節が、どこで実装されているかの一覧です。
未実装の項目も、隠さずここに書きます。

## 1. 目的と採用方針

| 設計 | 実装 |
|---|---|
| 1.1 入力（元動画を変更しない） | `pipeline/ingest.py` — 読み取りのみ。書き出しは作業フォルダと出力フォルダに限定 |
| 1.1 実行場所（解析中は通信なし） | `adapters/vision/llama_server.py:_ensure_loopback`、モデル自動取得は未実装（意図的） |
| 1.1 画面抽出（全文記録） | `pipeline/vision_extract.py` — 全画面パス + 領域別クロップ再認識 |
| 1.1 発話抽出（原言語・時刻付き） | `pipeline/asr_run.py`、`adapters/asr/whisper_cpp.py` |
| 1.1 グループ化（文字内容の変化） | `pipeline/frame_scan.py`（第 1 段階）、`pipeline/screen_tracker.py`（第 2 段階） |
| 1.1 コード保持 | `adapters/vision/prompts.py`、`util/textnorm.py:normalize_for_compare(is_code=True)` |
| 1.1 出力 | `pipeline/exporter.py` |
| 1.1 不明箇所を明示 | `models.py` の品質フラグ、`ReviewItem`、`pipeline/quality.py` |
| 1.1 中断・再開・部分再解析 | `db/store.py` の `job`/`checkpoint`、`pipeline/orchestrator.py` |
| 1.2 抽出と解釈の分離 | `adapters/vision/prompts.py:EXTRACTION_CONTRACT`、構造注記を原文と別に保持 |

## 2. 全体構造

| 設計 | 実装 |
|---|---|
| 処理の担当分け（表 2） | `pipeline/` の各モジュール。モデルは `adapters/` に隔離 |
| GPU 1 枚では重い推論を交互に | `pipeline/orchestrator.py` — ASR → 解放 → VLM の順、`util/gpulock.py` |
| 動画全体を VLM へ渡さない | 表示状態ごとに画像 1 枚を単位に読み取り、時間軸は `pipeline/` 側が管理 |

## 3. 推奨する初期構成

| 設計 | 実装 |
|---|---|
| 3.1 技術構成 | `pyproject.toml`（PyAV / OpenCV / NumPy）、`adapters/vision/llama_server.py`、`adapters/asr/whisper_cpp.py` |
| 3.2 AMD GPU / Vulkan | `docs/SETUP.md`。バックエンドは llama.cpp / whisper.cpp のビルド側で選ぶ |
| 3.3 初期リソース計画 | `config.py` の既定値（`n_ctx=8192`、`concurrency=1`、画像キャッシュ上限 20GB ほか） |

## 4. 時間軸の設計

| 設計 | 実装 |
|---|---|
| 4.1 共通の時刻・t0・整数マイクロ秒 | `util/timeutil.py`、`pipeline/probe.py` |
| 4.1 PTS × time_base で求める | `util/timeutil.py:pts_to_us`、`pipeline/frame_scan.py` |
| 4.1 映像・音声の開始オフセット保存 | `MediaProbe.video_offset_us` / `audio_offset_us`、`media` テーブル |
| 4.1 ffprobe による連続性検査 | `pipeline/probe.py:check_frame_timestamps`、異常は `ReviewItem` に記録 |
| 4.1 切り出しても元動画の時刻へ戻す | `pipeline/ingest.py:analysis_range`、`asr_run.py` のチャンク時刻変換 |
| 4.2 境界が存在しうる区間を保存 | `ScreenOccurrence.boundary_start_lo/hi_us`、`boundary_end_lo/hi_us` |
| 4.2 半開区間 | `util/timeutil.py:contains_half_open`、`aligner.py` |

## 5. ブロック分割

| 設計 | 実装 |
|---|---|
| 5.1 画面の領域分け（本文 / 文脈 / その他） | `models.py` の `ROLE_*`、`prompts.py` の役割指示 |
| 5.1 UI 文字を黙って除外しない | `exporter.py:_ui_appendix_lines`（時刻付き付録） |
| 5.1 焼き込み字幕は出典を分ける | `ROLE_BURNED_CAPTION`、`exporter.py` で別掲 |
| 5.2 第 1 段階（画像差分） | `frame_scan.py` — 全体差分 + タイル局所差分 + 変化画素の実数 |
| 5.2 第 2 段階（文字内容の確認） | `screen_tracker.py` — 本文ハッシュが同じなら継続、判定できなければ統合を保留 |
| 5.2 比較用正規化と原文を分ける | `util/textnorm.py`、`Region.text`（原文）と署名の分離 |
| 5.3 保全モード / 高速モード | `ScanConfig.mode`、`orchestrator._verification_scope` が検証範囲を明記 |
| 5.3 定期的な全文再認識 | `ScanConfig.periodic_full_recheck_us` |
| 5.4 表示状態 / 読書ブロック / 変化履歴 | `ScreenOccurrence` / `ReadingBlock` / `exporter._sub_state_lines` |
| 5.4 短時間表示も残す | `frame_scan.py` の `STATE_SHORT`、`vision_extract.py` で「全文未確定」として期間保持 |
| 5.5 カーソルと強調表示 | `frame_scan.py` の保留判定、`VisualEvent(event_kind='cursor')` |
| 5.6 重複表示 | `vision_extract._make_content`（本文は再利用）、出現は別レコード |

## 6. VLM による画面全文の抽出

| 設計 | 実装 |
|---|---|
| 6.1 処理手順 1〜7 | `vision_extract.VisionExtractor.extract_occurrence` |
| 6.1 ぼけ・カーソルの被りが少ない画像を選ぶ | `frame_scan.FrameScanner.save_representative`（同一状態の期間内から選択） |
| 6.2 認識は元解像度 | 領域クロップは常に原寸。全画面パスの縮小は `low_resolution` として記録 |
| 6.2 生成型の超解像を使わない | `_maybe_upscale` は `INTER_CUBIC` のみ |
| 6.2 タイル重複 15〜20% | `VisionConfig.tile_overlap_ratio = 0.18` |
| 6.2 切り詰めを成功扱いにしない | `llama_server.parse_completion`（`finish_reason=length` と `output_complete=false`） |
| 6.2 文脈長より先に領域分割 | `_split_tiles`、`_band_fallback` |
| 6.3 コードの保持 | `prompts.py` の指示、`line_numbers` を本文と分離、自動結合は行わない |
| 6.4 スライドの構造 | `Region.kind`、`structure_notes`（原文と分離、`relation_unknown`） |
| 6.5 抽出規約 | `prompts.py:EXTRACTION_CONTRACT`（設計の文言をそのまま実装） |
| 6.5 低温度・固定 seed | `VisionConfig.temperature=0.0`、`seed=42`。自信の数値は表示しない |

## 7. 発話の抽出

| 設計 | 実装 |
|---|---|
| 7.1 16kHz モノラル PCM と時刻対応 | `audio_prepare.extract_master_wav`、`verify_audio_alignment` |
| 7.1 チャンネル選択 | `AsrConfig.channel`（mix / left / right / 番号） |
| 7.1 チャンク 10 分・重複 2 秒 | `AsrConfig.chunk_us` / `chunk_overlap_us` |
| 7.1 チャンク内時刻を元時刻へ戻す | `asr_run.run_asr`（`chunk.start_us + seg.start_us`） |
| 7.1 重複範囲の同一発話だけ整理 | 中心時刻による担当決め + 正規化一致による重複除去 |
| 7.1 無音を切り詰めない | `detect_silence` は検出のみ。時間軸は変えない |
| 7.2 逐語性 | 原文をそのまま保存。清書は行わない |
| 7.3 発話セグメント単位で成立 | `Utterance`。単語時刻は `tokens`（`experimental: true`）として任意保存 |

## 8. 表示期間と発話の同期

| 設計 | 実装 |
|---|---|
| 8.1 overlap 計算と走査 | `util/timeutil.overlap_us`、`aligner.align_occurrences`（時刻順の走査） |
| 8.1 意味上の対応と区別 | 時刻を変更しない。`Alignment` は重なりの事実だけを持つ |
| 8.2 原文は 1 度だけ正本保存 | `utterance` テーブル、`alignment` で多対多参照 |
| 8.2 主ブロックは重複最大・同率なら先 | `aligner._mark_primary` |
| 8.2 その他は継続・関連発話として参照 | `exporter._utterance_lines` |
| 8.2 発話なし画面 / 画面なし発話 | `exporter._orphan_utterance_lines`、`aligner.spans_without_screen` |

## 9. データモデル

| 設計 | 実装 |
|---|---|
| 9.1 各エンティティ | `db/schema.sql`、`models.py` |
| 9.1 座標系と逆変換情報 | `Region.coord_space`、`evidence["crop"]` |
| 9.1 手修正は元を残す | `correction` テーブル、`cli.py:cmd_correct` |
| 9.2 不変条件 | `pipeline/quality.py:collect_quality` が検査し、品質報告に出す |
| 9.2 未抽出区間も状態を記録 | `coverage_span` テーブル、`_update_screen_coverage` / `_update_audio_coverage` |
| 9.2 失敗を空欄の成功にしない | `vision_extract` のクロップ再認識失敗時に原文を残す処理、`FLAG_NOT_EXTRACTED` |

## 10. 出力形式

| 設計 | 実装 |
|---|---|
| 出力ファイル一式 | `exporter.Exporter.export_all`、`quality.write_quality_report` / `write_manifest` |
| 10.1 ブロックの出力順序 1〜7 | `exporter.write_markdown` |
| 10.1 画像を埋め込まない | Markdown には根拠画像を埋め込まず、参照だけを持つ |
| 10.1 見出しがなければブロック番号 | `_title_for` が見出し・題名・ファイル名のみを使う |
| 10.1 再生時刻を必ず併記 | `video_time_link` とあわせて時刻を常に表示 |
| 10.1 エスケープとフェンス長 | `escape_markdown_text`、`fence_for` |

## 11. モジュール境界

`pipeline/` が設計 11 の表に対応します。

| 設計のモジュール | 実装 |
|---|---|
| ingest | `pipeline/ingest.py` |
| timeline | `pipeline/probe.py`、`util/timeutil.py` |
| frame_scan | `pipeline/frame_scan.py` |
| vision_adapter | `adapters/vision/` |
| screen_tracker | `pipeline/screen_tracker.py`、`pipeline/vision_extract.py` |
| audio_prepare | `pipeline/audio_prepare.py` |
| asr_adapter | `adapters/asr/` |
| aligner | `pipeline/aligner.py` |
| block_builder | `pipeline/block_builder.py` |
| reviewer | `cli.py:cmd_review` / `cmd_correct`（Web UI は未実装） |
| exporter | `pipeline/exporter.py` |

## 12. 中断・再開・性能管理

| 設計 | 実装 |
|---|---|
| 12.1 キャッシュ鍵の構成要素 | `vision_extract.VisionExtractor.cache_key`（設計の 7 要素をそのまま使用） |
| 12.1 類似画像を同一本文と断定しない | 類似ハッシュによる吸収は実装していない |
| 12.2 段階と処理単位ごとの完了状態 | `job` テーブル、`frame_scan` / `asr` / `vision` の checkpoint |
| 12.2 一時書き込みと確定を分ける | `util/atomic.py`、画像・音声チャンクの `.partial` |
| 12.2 ハッシュ比較で必要部分だけ再解析 | `RunConfig.stage_config_hash`、`Store.stage_is_done`、`STAGE_DOWNSTREAM` |
| 12.2 再試行上限 2 回 | `VisionConfig.retry_limit` / `AsrConfig.retry_limit` |
| 12.2 疑義への再認識と障害の再試行を区別 | `vision_extract._call` の分岐 |
| 12.2 GPU 要求のロック | `util/gpulock.py` |
| 12.2 保存領域不足で停止 | `pipeline/cache.py:check_free_space`（走査の開始時と 25 状態ごと） |
| 12.3 処理量の見積もり | `metric` テーブルに各ステージの件数と所要時間を記録し、品質報告へ出力 |
| 12.3 画像キャッシュの上限と優先度 | `pipeline/cache.py:prune_image_cache`（要確認・未抽出・短時間表示を優先して残す） |
| 12.3 削除画像は再生成できることを確認 | `pipeline/cache.py:regenerate_frame`（元動画と時刻から再取得、`cache --restore`） |

## 13. ローカル完結の運用

| 設計 | 実装 |
|---|---|
| 13.1 通常処理で外部通信なし | `_ensure_loopback`、モデル自動取得なし、`doctor` が不足を事前に一覧表示 |
| 13.1 抽出したコードを実行しない | 文字データとしてのみ扱う。HTML としての埋め込み・実行も行わない |
| 13.2 再現可能な環境 | `quality.write_manifest`（モデル SHA-256、ランタイム版、設定、依存の版） |
| 13.2 GGUF と whisper.cpp のモデルを区別 | `docs/SETUP.md`、別々の設定項目 |

## 14. 品質評価と完了条件

| 設計 | 実装 |
|---|---|
| 14.1 評価用データ | `tests/fixtures.py`（合成講義動画：スライド・コード・1 文字変更・短時間表示・境界をまたぐ発話・無音） |
| 14.2 評価指標 | `quality_report.md` に観測値を出力。CER / WER / 回収率は正解データが必要なため、達成値としては表示しない |
| 14.2 時間軸の網羅 | `_unknown_coverage`（処理状態が不明な区間の件数） |
| 14.3 優先する検証 | `tests/test_pipeline.py`、`tests/test_units.py` |
| 14.4 完了状態 | `quality.completion_status`（processing / completed_with_review / reviewed / failed） |

## 15. 実装の進め方

| 段階 | 状態 |
|---|---|
| 0：環境とモデル確認 | `doctor`、`scripts/try_vision.py`、`docs/SETUP.md` |
| 1：最小の一連処理 | 実装済み |
| 2：保全と再開 | 実装済み |
| 3：読みやすさ | 実装済み（親ブロック化、コード版の追跡、継続発話の参照） |
| 4：確認画面 | **未実装**。`review` / `correct` コマンドで代替 |
| 5：速度改善 | **未実装**。4B との比較や補助 OCR は入れていない |

## 16. 初期設定

`config.py` の既定値が設計 16 の表に対応します。見直す条件が生じた場合は、
設定を変更して該当ステージだけを再解析できます（`--force vision` など）。
