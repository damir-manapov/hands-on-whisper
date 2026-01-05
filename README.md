# hands-on-whisper

Exploring speech recognition with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [OpenAI Whisper](https://github.com/openai/whisper), and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [gitleaks](https://github.com/gitleaks/gitleaks) for secret scanning
- [FFmpeg](https://ffmpeg.org/) (required by whisper)

### Install FFmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

## Setup

```bash
uv sync
```

## Download whisper.cpp models

For whisper.cpp you need to download ggml models:

```bash
# Download from huggingface
curl -L -o models/ggml-base.en.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin
```

## Usage

```bash
# Basic transcription (uses faster-whisper by default)
uv run python src/transcribe.py transcribe audio.wav

# Specify backend
uv run python src/transcribe.py transcribe audio.wav --backend openai
uv run python src/transcribe.py transcribe audio.wav --backend whispercpp --model-path models/ggml-base.bin

# Specify model size
uv run python src/transcribe.py transcribe audio.wav --model large-v3

# Specify language (auto-detect if not set)
uv run python src/transcribe.py transcribe audio.wav --language ru

# Use GPU
uv run python src/transcribe.py transcribe audio.wav --device cuda

# Compare multiple backends/models (runs all combinations)
uv run python src/transcribe.py transcribe audio.wav --backend faster-whisper openai --model base large-v3

# Short alias: 't' instead of 'transcribe'
uv run python src/transcribe.py t audio.wav

# Show all options
uv run python src/transcribe.py transcribe --help
```

## Report

A markdown report is automatically generated after each transcription run (e.g., `audio.md`).

To regenerate or customize the report:

```bash
# Regenerate report from JSON
uv run python src/transcribe.py report audio.json

# Save to custom path
uv run python src/transcribe.py report audio.json -o custom-report.md

# Short alias: 'r' instead of 'report'
uv run python src/transcribe.py r audio.json
```

The report includes:
- Performance summary table (sorted by duration)
- Detailed transcription results for each run

## Output

Results are automatically saved to a JSON file named after the audio file (e.g., `audio.json`):

```json
{
  "audio": "audio.wav",
  "runs": [
    {
      "id": "a1b2c3d4e5f6",
      "timestamp": "2026-01-05T16:00:00+00:00",
      "duration_seconds": 12.34,
      "backend": "faster-whisper",
      "model": "base",
      "language": null,
      "device": "cpu",
      "text": "transcribed text..."
    }
  ]
}
```

Each run has a unique ID based on settings (backend, model, language, device). Re-running with the same settings updates the existing entry.

## Audio preprocessing

Whisper expects 16kHz mono audio. Use ffmpeg to convert:

```bash
# Check audio format
ffprobe input.mp3

# Convert to 16kHz mono WAV (recommended for whisper.cpp)
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav

# For stereo call recordings - mix both channels with volume compensation
ffmpeg -i input.mp3 -af "pan=mono|c0=0.5*c0+0.5*c1,volume=2" -ar 16000 output.wav
```

## Development

```bash
# Run all checks (format, lint, tests)
./check.sh

# Run health checks (gitleaks, outdated deps, vulnerabilities)
./health.sh

# Run both
./all-checks.sh
```

## Scripts

- `check.sh` - Format, lint, type check, and run tests
- `health.sh` - Check for secrets, outdated dependencies, and vulnerabilities
- `all-checks.sh` - Run both check.sh and health.sh
