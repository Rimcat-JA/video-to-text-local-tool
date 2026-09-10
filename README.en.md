[日本語](README.md) | **English**

# lecture-extract — Local lecture-video text extraction

Extracts **the on-screen text of slides, code, and program output** together with
**what the lecturer says** from a downloaded lecture video, and produces a Markdown
document you can read chronologically plus structured data for further processing.

This implements the design in `local_lecture_extraction_design.md` (v0.1), covering
stages 0 through 3.

> The repository is named `video-to-text-local-tool`, but the Python package and the
> CLI command are both `lecture-extract`. The commands below work as written.

Once the software and models are obtained the first time, analysis and output run
**entirely offline**. No paid services or cloud APIs are used.

---

## What it does

- Cuts the video into **screen states** — periods during which the displayed text stays the same
- Transcribes the full screen text of each state with a local VLM (llama.cpp + Qwen3-VL)
- Transcribes speech with a local ASR (whisper.cpp + Whisper large-v3-turbo)
- Links screen states to utterances by **interval overlap**
- Writes Markdown, JSONL and SRT without altering the original text

## What it does not do

Section 1.2 of the design ("separate extraction from interpretation") is implemented literally.

- Never summarizes or translates slide text
- Never "fixes" code or fills in lines that were off-screen
- Never rewrites speech into more fluent prose
- Never guesses at illegible text — it is marked unknown, and any guesses go in a
  separate field from the transcription
- Never reports "processing finished" as "text verified"

---

## Installation

**[docs/SETUP.md](docs/SETUP.md) has step-by-step instructions (in Japanese).**

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/lecture-extract.exe doctor   # lists whatever is missing
```

Requirements:

| Kind | Details |
|---|---|
| Software | Python 3.12+, FFmpeg / ffprobe, llama.cpp (llama-server), whisper.cpp (whisper-cli) |
| Models | Qwen3-VL-8B-Instruct GGUF (Q4_K_M) + mmproj, Whisper large-v3-turbo (ggml format) |

## Running

```bash
lecture-extract run "D:/lectures/week01.mp4" -o out/week01 \
  --vision-model models/qwen3-vl-8b/<MODEL>.gguf \
  --mmproj models/qwen3-vl-8b/<MMPROJ>.gguf \
  --manage-vision-server \
  --asr-model models/whisper/ggml-large-v3-turbo.bin \
  --language en
```

### Commands

| Command | Purpose |
|---|---|
| `run` | Analyze and export. Re-run the same command to resume after an interruption |
| `probe` | Inspect the time base, stream information and timestamp anomalies |
| `doctor` | List missing software and models |
| `status` | Stage progress; counts of screen states, utterances and open review items |
| `export` | Regenerate the outputs from the SQLite source of truth |
| `review list` / `review resolve` | Inspect and resolve uncertain extractions |
| `cache` | Check and prune the image cache; regenerate evidence images |
| `correct` | Hand-correct a region's text (the original extraction is kept) |

### Key options

| Option | Meaning |
|---|---|
| `--mode preserve` (default) | Decode every frame and collect change candidates. Prioritizes catching momentary changes |
| `--mode fast` | Starts around 2 fps, for trials and estimates. Because it can miss changes, its output is never labelled "verified" |
| `--start` / `--end` | Limit the range. Output timestamps still refer to the original video |
| `--only` / `--force` | Limit which stages run / re-run stages already marked done |
| `--extract-min-state` | Minimum display duration sent to the VLM. Shorter states still keep their time span |
| `--channel left` etc. | Pick an audio channel when a plain stereo downmix cancels the voice |

---

## Output

| File | Contents |
|---|---|
| `lecture.md` | Full screen text and speech per display period. Code in code blocks, each utterance with its own timestamps |
| `blocks.jsonl` | Reading blocks with references to screen states and utterances |
| `screen_contents.jsonl` | Screen text, regions, structure, extraction status |
| `screen_occurrences.jsonl` | Display start/end, content id, evidence, change status |
| `utterances.jsonl` | Verbatim speech with timestamps (stored once, linked by reference) |
| `transcript.srt` | Subtitles of speech only. Screen OCR text is never mixed in |
| `quality_report.md` | Unextracted spans, illegible text, timing anomalies, review items, measurements |
| `manifest.json` | Input hashes, models, runtime environment, configuration, output version |
| `work/state.sqlite` | Internal source of truth for re-analysis and resuming |

Images are never embedded in the Markdown. Evidence images stay in `work/frames/` and
raw model responses in `work/responses/`, both referenced from SQLite.

---

## Design points

- **Time**: integer microseconds relative to the start of the original video (t0).
  Timestamps come from PTS × time_base — never from "frame number ÷ nominal FPS".
  Variable frame rate and audio start offsets are handled.
- **Boundaries are estimates**: for every transition, both the last time the old state
  was observed and the first time the new one was observed are stored. A boundary is
  never treated as a single exact value.
- **Small changes are not dropped**: detection combines whole-frame difference,
  per-tile local difference and the absolute count of changed pixels. A one-character
  code edit is detected; a blinking cursor is separated out as a visual event.
- **Brief displays are kept**: a state shown for under 0.5 s still gets a recorded time
  span, marked "brief display, text unconfirmed" if it could not be read. Intermediate
  states are never silently discarded in favour of the final one.
- **Repeated screens**: when the same slide reappears, the text is stored once but each
  occurrence (display period) is kept separately.
- **Interrupt and resume**: completion is checkpointed per stage and per work unit.
  Input, configuration and model hashes are compared so only what changed is redone.
- **Local only**: the inference server is restricted to 127.0.0.1. There is no automatic
  fallback to an external service. Extracted code is treated as text and never executed.

A section-by-section map from the design document to the implementation is in
[docs/DESIGN_TRACE.md](docs/DESIGN_TRACE.md).
Measurements from a real lecture video, the defects found, and what remains unverified
are recorded in [docs/RUN_NOTES.md](docs/RUN_NOTES.md). (Both are in Japanese.)

---

## Tests

The whole pipeline can be verified without any model. The tests generate a synthetic
lecture video and substitute an adapter that reads a marker embedded in each frame to
look up scripted extraction results.

```bash
.venv/Scripts/python.exe -m pytest
```

The suite covers the checks listed in section 14.3 of the design:

- A blinking cursor alone is not treated as a text change
- A single-character code edit is not lost
- A repeated slide is kept as a separate display period
- Code before and after an edit is never merged into one version
- An utterance spanning a screen boundary is reachable from both sides
- No utterance is lost or duplicated at an audio chunk boundary
- Resuming after an interruption loses nothing and duplicates nothing
- No span of the timeline is left in an unknown processing state

## Not implemented (design stage 4 onward)

- A review UI (local web app) for viewing the video, evidence image and extracted text
  side by side with partial re-analysis. The `review` and `correct` commands stand in
  for it today.
- Speed work such as auxiliary OCR and adaptive retries.

## Known limitations

Verified against a 7-hour 38-minute lecture video. See
[docs/RUN_NOTES.md](docs/RUN_NOTES.md) for the full record.

- A managed `llama-server` is not restarted if it dies. After about nine hours of
  continuous use it crashed, and subsequent requests failed with connection errors.
- With the 4B model, 46% of full-screen passes hit the output token limit on that video.
  Truncated responses are never treated as success and fall back to region splitting,
  but the result is still worse than a clean read.

## License

MIT ([LICENSE](LICENSE)). For the licenses of the software and models this depends on,
see [docs/LICENSES.md](docs/LICENSES.md).
