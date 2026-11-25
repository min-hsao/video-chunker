# video-chunker

A Python CLI tool that splits long video recording sessions (20–45 min) into labeled chunks using silence detection, speech transcription, and AI analysis.

Designed for creators who record multiple takes in a single session — product demos, tutorials, talking-head videos — and need each take separated, labeled, and assessed for completeness.

## Features

- **Silence-based splitting** — uses ffmpeg `silencedetect` to find natural breaks
- **Transcript-aware** — validates split points against a Whisper transcript so cuts never land mid-sentence
- **Fast Mac transcription** — whisper.cpp on the Apple GPU (`large-v3-turbo` by default) with accurate word timings (DTW); no PyTorch needed
- **Cleaner audio for transcription** — a copy of the audio is high-passed, gently denoised and normalised to −20 LUFS before Whisper hears it; clips are still cut losslessly from the original
- **Timing check** — word times are checked against the silences found in the original audio; if the cleaned transcript drifts, the original audio is transcribed and the better one kept
- **Vocabulary** — your names, terms and spelling fixes (`Sahm` → `Sam`) steer Whisper, correct the transcript and keep clip names spelled right 
- **Grounded AI cuts** — every AI retake, filler or content cut must quote words that are really in the transcript, or it is dropped
- **Verbal cue detection** — configurable keywords (e.g. "cut", "next") that force a split
- **AI analysis** — each chunk is analyzed by an LLM for completeness, content description, and optional script comparison
- **Lossless output** — splits use `-c copy` (stream copy) cutting on the nearest keyframe — no re-encoding
- **Smart naming** — output files are named `001_brief_description_complete.mp4`
- **Rich CLI** — progress bars, color output, and a detailed JSON manifest option
- **EXIF/metadata aware** — extracts camera model, GPS coordinates, creation time, and lens info from video files using ffprobe and optional exiftool
- **Reverse-geocode** — optionally convert GPS coordinates to place names (offline via bundled dataset, or online via OpenStreetMap/Nominatim)
- **Context-aware labeling** — feeds camera/location/date context to the LLM for smarter chunk descriptions (e.g. `shoreditch_street_intro_complete.mp4`)
- **Sidecar JSONs** — per-chunk metadata files with transcript, analysis, GPS, camera info, and QC results
- **GPS redaction** — `--redact-gps` excludes raw coordinates from outputs, using place names only

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/) and ffprobe installed and on your PATH
- For the default engine: [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`brew install whisper-cpp`). The model (`large-v3-turbo`, about 1.6 GB) downloads to `~/.cache/video-chunker/models/` on first use
- For `--whisper-mode local` (Python Whisper on the CPU, slow): `pip install 'video-chunker[local-whisper]'` (installs PyTorch)
- For OpenAI Whisper mode: An [OpenAI API key](https://platform.openai.com/api-keys) set as `OPENAI_API_KEY`
- For LLM analysis (optional): `DEEPSEEK_API_KEY` for DeepSeek models, or `OPENAI_API_KEY` for GPT-4o
- For exiftool enrichment (optional): [exiftool](https://exiftool.org/) installed and on your PATH — auto-detected if available
- For offline reverse-geocoding (optional): `pip install 'video-chunker[geo]'` (installs `reverse_geocoder`)

## Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USER/video-chunker.git
cd video-chunker

# Install in a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -e .          # add '.[dev]' for pytest
brew install whisper-cpp  # the default transcription engine
```

Run the tests with `pytest`.

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Usage

### Basic usage

```bash
video-chunker recording.mp4
```

Splits `recording.mp4` into chunks in `./chunks/`.

### Specify output directory and video type

```bash
video-chunker recording.mov -o ./output --type tutorial
```

### Compare against a draft script

```bash
video-chunker session.mp4 --script draft.txt --detailed
```

The `--detailed` flag outputs a full JSON manifest with transcripts, analysis, and script-match info.

### Custom verbal cues and silence settings

```bash
video-chunker recording.mp4 \
  --cues "cut,next,stop,redo" \
  --silence-duration 3.0 \
  --silence-threshold -30
```

### Dry run (preview without splitting)

```bash
video-chunker recording.mp4 --dry-run
```

Shows detected chunks and analysis in a table without writing any files.

### Choose models

**whisper.cpp (default when `whisper-cli` is installed, free, Mac GPU):**

```bash
video-chunker recording.mp4                                    # large-v3-turbo
video-chunker recording.mp4 --whisper-model large-v3-turbo-q5_0  # smaller download (~550 MB)
video-chunker recording.mp4 --whisper-model ~/models/ggml-small.en.bin
```

Any [whisper.cpp model name](https://huggingface.co/ggerganov/whisper.cpp) works, or a path to a `.bin`.
Word timings use whisper.cpp's DTW alignment for the standard models. `--vad` (experimental)
skips non-speech with Silero VAD; it uses sentence-level timings because whisper.cpp leaves
word times in VAD-compressed time.

**Python Whisper on the CPU (`--whisper-mode local`):**

```bash
pip install 'video-chunker[local-whisper]'
video-chunker recording.mp4 --whisper-mode local --whisper-model small
```

Available models: `tiny`, `base`, `small` (default), `medium`, `large-v3`. Slow on long recordings (Apple's GPU path returns empty transcripts, so it runs on the CPU).

**Audio cleanup:** on by default (`--no-clean-audio` to turn it off). It only changes what Whisper hears.

### Vocabulary: names, terms and spelling fixes

Create `~/.config/video-chunker/vocab.json` (used automatically) or pass `--vocab FILE`:

```json
{
  "my_name": "Sam",
  "aliases": ["Samantha"],
  "people": ["Dana"],
  "words": ["Samtrix", "rocketdoc"],
  "corrections": [{"from": "Sahm", "to": "Sam"}, {"from": "sam tricks", "to": "Samtrix"}]
}
```

The names and terms go into Whisper's prompt and the AI prompts; corrections are applied to the
transcript before anything else sees it. Corrections are whole-word, keep the original
capitalisation (`SAHM` → `SAM`, `sahm` → `sam`), longest first, one pass (test cases in
`tests/data/vocabulary-cases.json`).

**OpenAI Whisper API (paid, faster for long videos):**

```bash
export OPENAI_API_KEY=your_key_here
video-chunker recording.mp4 --whisper-mode openai --whisper-model whisper-1
```

**LLM analysis:**

```bash
video-chunker recording.mp4 --llm-model deepseek-chat
```

Set `DEEPSEEK_API_KEY` for DeepSeek models, or `OPENAI_API_KEY` for GPT-4o.

## CLI Reference

```
Usage: video-chunker [OPTIONS] INPUT_VIDEO

  Split a long video recording into labeled chunks.

Options:
  -o, --output PATH           Output directory (default: ./chunks/)
  --type TEXT                  Video type: product-demo, tutorial, talking-head, or custom
  --script PATH               Path to draft script file for comparison
  --cues TEXT                  Comma-separated verbal cue keywords (default: "cut,next,take")
  --silence-duration FLOAT    Minimum silence duration in seconds (default: 2.0)
  --silence-threshold FLOAT   Silence threshold in dB (default: -35)
  --detailed                  Output full JSON manifest
  --whisper-mode [auto|cpp|local|openai]  Engine (default: auto = cpp if whisper-cli is installed, else local)
  --whisper-model TEXT        cpp: large-v3-turbo (default) or a .bin path; local: small; openai: whisper-1
  --clean-audio/--no-clean-audio  Clean the transcription copy of the audio (default: on)
  --vad/--no-vad              Experimental, cpp only: Silero VAD (default: off)
  --vocab FILE                Vocabulary JSON (default: ~/.config/video-chunker/vocab.json if present)
  --llm-model TEXT            LLM model for analysis (default: deepseek-chat)
  --dry-run                   Show detected chunks without splitting
  --metadata/--no-metadata     Extract embedded camera/GPS metadata (default: on)
  --geocode [off|offline|online]  Reverse-geocode GPS to place names (default: off)
  --redact-gps                Exclude raw GPS coordinates from outputs
  --sidecar                   Write per-chunk JSON sidecar files
  -v, --verbose               Enable debug logging
  --version                   Show this message and exit
  --help                      Show this message and exit
```

## EXIF & GPS Metadata

video-chunker extracts embedded metadata from video files in two tiers:

1. **ffprobe (always available)** — reads GPS coordinates (ISO 6709), creation time, camera make/model, and software from QuickTime/MP4 container tags.
2. **exiftool (auto-detected)** — fills gaps with lens info, Fuji/Canon/Nikon maker notes, GoPro/DJI GPS variants, and timezone offset data.

### Reverse-geocoding

When GPS data is present, you can convert coordinates to place names:

```bash
# Offline (bundled dataset, no network, city-level)
pip install 'video-chunker[geo]'
video-chunker recording.mp4 --geocode offline

# Online (OpenStreetMap Nominatim, neighborhood-level)
video-chunker recording.mp4 --geocode online
```

Geocode results are cached at `~/.cache/video-chunker/geocode.json` to avoid repeat lookups.

### Privacy

- Raw GPS coordinates are never sent to the LLM — only derived place names, dates, and camera info.
- Use `--redact-gps` to exclude coordinates from manifest/sidecar files entirely.
- Location data is only sent to DeepSeek/OpenAI if you use their API. Use `--llm-model local/...` for fully local processing.
- Online geocoding uses OpenStreetMap's Nominatim API with a proper User-Agent and rate limiting.

## Output

### File naming

Files are named with a three-digit index, a brief AI-generated description, and a completeness status:

```
chunks/
├── 001_product_overview_intro_complete.mp4
├── 002_feature_demo_incomplete.mp4
├── 003_pricing_walkthrough_complete.mp4
└── 004_closing_remarks_complete.mp4
```

### JSON manifest (--detailed)

When `--detailed` is passed, a full manifest is printed with per-chunk data:

```json
{
  "input": "recording.mp4",
  "video_info": {
    "codec": "h264",
    "resolution": "3840x2160",
    "duration": 1847.3,
    "fps": 29.97
  },
  "chunks": [
    {
      "index": 1,
      "start": 0.0,
      "end": 312.5,
      "duration": 312.5,
      "transcript": "Welcome to the product demo...",
      "cue_triggered": false,
      "output_path": "chunks/001_product_overview_intro_complete.mp4",
      "analysis": {
        "description": "product overview intro",
        "is_complete": true,
        "confidence": 0.95,
        "notes": "Clean intro with full greeting",
        "script_match": ""
      }
    }
  ]
}
```

## How It Works

1. **Probe** — `ffprobe` reads codec, resolution, duration, and fps
2. **Silence detection** — `ffmpeg silencedetect` finds gaps exceeding the threshold
3. **Transcription** — a cleaned copy of the audio is transcribed with whisper.cpp (default), Python Whisper or the OpenAI API, biased toward your vocabulary; word times are checked against the detected silences and corrections applied
4. **Split point refinement** — silence midpoints are adjusted to nearest sentence boundaries; verbal cue keywords force additional splits
5. **Keyframe snapping** — split points are snapped to the nearest video keyframe for clean cuts
6. **LLM analysis** — each chunk's transcript is sent to DeepSeek or OpenAI to determine completeness and generate a brief description
7. **Lossless split** — `ffmpeg -c copy` extracts each chunk without re-encoding
8. **Naming** — files are named `{index}_{description}_{status}.{ext}`

## Supported Formats

- **Containers**: MP4, MOV, MKV
- **Video codecs**: H.264, H.265/HEVC
- **Tested with**: iPhone 4K, Fuji X-T3, Insta360

## License

MIT — see [LICENSE](LICENSE).
