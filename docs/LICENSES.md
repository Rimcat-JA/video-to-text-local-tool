# ライセンス情報

設計 13.2 に従い、採用したソフトウェアとモデルのライセンス情報をまとめます。
再配布する場合は、それぞれの配布元の条件を確認してください。

**注意**: 以下は配布ページの表記に基づく整理です。実際に取得したファイルの版で
表記が変わることがあるため、導入時に配布ページを確認し、`manifest.json` に記録された
ハッシュとあわせて保管してください。

## このプロジェクト

`lecture-extract` 本体のライセンスは未設定です。用途が決まった時点で決めてください。

## 実行時に必要な外部ソフトウェア

| ソフトウェア | 役割 | 配布元 |
|---|---|---|
| FFmpeg / ffprobe | メディア情報の取得、音声の書き出し | <https://ffmpeg.org/> |
| llama.cpp（llama-server） | 画像モデルのローカル推論 | <https://github.com/ggml-org/llama.cpp> |
| whisper.cpp（whisper-cli） | 音声認識のローカル推論 | <https://github.com/ggml-org/whisper.cpp> |

FFmpeg はビルド構成によって LGPL / GPL のどちらかになります。この PC では gyan.dev の
full build（GPL 構成を含む）を使用しています。再配布時は特に確認が必要です。

## Python の依存

| パッケージ | 役割 |
|---|---|
| PyAV | PTS 付きのフレーム読み出し |
| OpenCV (opencv-python-headless) | 画像差分、クロップ、エンコード |
| NumPy | 数値計算 |
| Pillow | 画像の補助処理 |
| requests | ループバック接続の HTTP 呼び出し |

各パッケージの版は `manifest.json` の `runtime.packages` に記録されます。

## モデル

| モデル | 用途 | 配布ページのライセンス表記 |
|---|---|---|
| Qwen3-VL-8B-Instruct GGUF | 画面の文字抽出 | Apache-2.0（<https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF>） |
| Qwen3-VL-4B-Instruct GGUF | 軽量構成の比較用 | Apache-2.0（<https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF>） |
| Whisper large-v3-turbo | 音声認識 | MIT（<https://huggingface.co/openai/whisper-large-v3-turbo>） |

whisper.cpp 用の `ggml-*.bin` は変換済み配布です。取得元と版を `manifest.json` に記録してください。

## 費用

ソフトウェア利用料・モデル推論料は発生しません。既存 PC の電気代と保存領域は必要です。
