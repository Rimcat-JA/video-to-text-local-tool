# 導入手順（段階 0：環境とモデル確認）

設計 15 の「段階 0」に対応する準備手順です。ここを終えてから `lecture-extract run` を実行します。

外部通信が発生するのは**この初期準備だけ**です。通常の解析中にモデルの自動ダウンロードは行いません。
必要ファイルが欠けている場合は、解析を始める前に `lecture-extract doctor` が不足を一覧表示して停止します。

---

## 0. この PC の確認済み状態（2026-09-09 時点）

| 項目 | 状態 |
|---|---|
| Python | 3.12.0 / 3.13.13（本プロジェクトは 3.12 の venv を使用） |
| FFmpeg / ffprobe | 8.1.2（winget の gyan.dev ビルド）導入済み |
| llama.cpp | winget `ggml.llamacpp` 導入済み。version 9935 (f2d1c2f39)、`ggml-vulkan.dll` を同梱 |
| GPU | AMD Radeon RX 9070 XT（VRAM 16GB） |
| メインメモリ | 32GB |
| whisper.cpp | **未導入**（下記 2 章で導入） |
| 画像モデル / mmproj | **未取得**（下記 1 章で取得） |

`ggml-vulkan.dll` が同梱されていることは「Vulkan バックエンドのビルドである」ことを示すだけで、
この GPU・ドライバ・モデルの組み合わせで実際に動くことを保証しません（設計 3.2）。
1 章の最後にある動作確認を必ず実施してください。

---

## 1. 画像モデル（VLM）の取得

第一候補は Qwen3-VL-8B-Instruct の GGUF（Q4_K_M）です。モデル本体に加えて、
マルチモーダル用の **mmproj ファイル**が必要です（設計 3.1）。

### 1.1 取得

`huggingface_hub` の CLI を使うのが確実です。ファイル名は配布側で変わることがあるため、
名前を直接書かずにパターンで指定します。

```bash
.venv/Scripts/python.exe -m pip install "huggingface_hub[cli]"
```

```bash
.venv/Scripts/hf.exe download Qwen/Qwen3-VL-8B-Instruct-GGUF --include "*Q4_K_M*" "*mmproj*" --local-dir models/qwen3-vl-8b
```

うまくいかない場合は、ブラウザで次のページを開き、Q4_K_M の本体と mmproj を `models/qwen3-vl-8b/` へ手動で置いてください。

- <https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF>
- 軽量構成の比較用: <https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF>

取得後、ファイル名を確認します。

```bash
ls -la models/qwen3-vl-8b/
```

### 1.2 動作確認（対象画像を読めるか）

まず llama-server を単体で起動します。`<MODEL>` と `<MMPROJ>` は 1.1 で確認した実際のファイル名に置き換えてください。

```bash
llama-server -m models/qwen3-vl-8b/<MODEL>.gguf --mmproj models/qwen3-vl-8b/<MMPROJ>.gguf -c 8192 -ngl 99 --host 127.0.0.1 --port 8080
```

別のシェルで、講義のコード画面・スライド画面のスクリーンショットを 1 枚ずつ読ませて、
文字が正しく転記されるかを目視で確認します。

```bash
.venv/Scripts/python.exe scripts/try_vision.py path/to/code_screenshot.png
```

確認する点（設計 15 段階 0 の通過条件）:

- 記号・字下げ・大小文字が保たれているか
- 画面外の行を勝手に補完していないか
- 誤ったコードを「修正」していないか
- 判読できない箇所を推測で埋めていないか

VRAM が足りない場合の順序は設計 3.2 の通りです。同時要求数 → 画像タイルの大きさ → 4B モデル →
部分 CPU 実行、の順に下げます。**文字を潰すような解像度低下は最初の対処にしません。**

---

## 2. 音声認識（whisper.cpp）の導入

### 2.1 実行ファイル

いずれかを選びます。

**(a) 公式のビルド済みバイナリを使う**

<https://github.com/ggml-org/whisper.cpp/releases> を開き、Windows x64 向けの
アーカイブ（Vulkan 版があればそれ）を展開して、`whisper-cli.exe` の場所を控えます。
リリースごとに資産名が異なるため、ページで実際の名前を確認してください。

**(b) 自分でビルドする（Vulkan）**

```bash
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build -j --config Release
```

`build/bin/Release/whisper-cli.exe` が生成されます。

### 2.2 モデル

初期候補は large-v3-turbo です。whisper.cpp 用の `ggml-*.bin` 形式が必要で、
VLM 用の GGUF とは別形式です（設計 13.2）。

whisper.cpp のリポジトリを取得している場合:

```bash
cd whisper.cpp && ./models/download-ggml-model.sh large-v3-turbo
```

取得していない場合は Hugging Face から直接取得します。

```bash
.venv/Scripts/hf.exe download ggerganov/whisper.cpp ggml-large-v3-turbo.bin --local-dir models/whisper
```

モデル管理の公式手順: <https://github.com/ggml-org/whisper.cpp/blob/master/models/README.md>

### 2.3 動作確認（短い音声が時刻付きで出るか）

```bash
<whisper-cli のパス> -m models/whisper/ggml-large-v3-turbo.bin -f test.wav -l ja --output-json -of tmp/test_asr
```

`tmp/test_asr.json` の `transcription[].offsets` に開始・終了のミリ秒が入っていれば通過です。

---

## 3. 不足の確認

```bash
.venv/Scripts/lecture-extract.exe doctor --vision-model models/qwen3-vl-8b/<MODEL>.gguf --mmproj models/qwen3-vl-8b/<MMPROJ>.gguf --asr-model models/whisper/ggml-large-v3-turbo.bin --asr-binary <whisper-cli のパス>
```

「不足はありません。」と表示されたら段階 0 は完了です。

---

## 4. 実行

```bash
.venv/Scripts/lecture-extract.exe run "D:/lectures/week01.mp4" -o out/week01 --vision-model models/qwen3-vl-8b/<MODEL>.gguf --mmproj models/qwen3-vl-8b/<MMPROJ>.gguf --manage-vision-server --asr-model models/whisper/ggml-large-v3-turbo.bin --asr-binary <whisper-cli のパス> --language ja --save-config out/week01/run_config.json
```

2 回目以降は設定ファイルを使えます。

```bash
.venv/Scripts/lecture-extract.exe run "D:/lectures/week01.mp4" --config out/week01/run_config.json
```

### まず短い区間で試す

1 時間の動画をいきなり流さず、代表的な数分から始めて処理時間を測ってください（設計 3.3 / 12.3）。

```bash
.venv/Scripts/lecture-extract.exe run "D:/lectures/week01.mp4" -o out/probe --start 0:10:00 --end 0:13:00 --config out/week01/run_config.json
```

出力の時刻は、切り出し区間ではなく**元動画に対する時刻**で書かれます。

### 中断と再開

`Ctrl+C` で止めた場合、同じコマンドをもう一度実行すれば、完了済みのステージと
処理単位を飛ばして続きから再開します。入力・設定・モデルのハッシュが変わった部分だけが再解析されます。

```bash
.venv/Scripts/lecture-extract.exe status -w out/week01/work
```

---

## 5. ネット切断状態での確認

設計 14.2 の「ローカル完結」を確認するには、ネットワークを切った状態で
取り込みから出力まで通してください。モデルの自動取得や外部サービスへの切り替えは実装していないため、
不足があれば `doctor` と同じ形式で停止します。
