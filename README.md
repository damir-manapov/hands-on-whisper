# hands-on-whisper

Exploring speech recognition with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [OpenAI Whisper](https://github.com/openai/whisper), and [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [gitleaks](https://github.com/gitleaks/gitleaks) for secret scanning
- FFmpeg (required by whisper)

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
# Transcribe with faster-whisper
uv run python src/transcribe_faster.py audio.wav

# Transcribe with OpenAI whisper
uv run python src/transcribe_openai.py audio.wav

# Transcribe with whisper.cpp
uv run python src/transcribe_whispercpp.py models/ggml-base.en.bin audio.wav
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
