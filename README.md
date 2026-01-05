# hands-on-whisper

Exploring speech recognition with [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and [OpenAI Whisper](https://github.com/openai/whisper).

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- [gitleaks](https://github.com/gitleaks/gitleaks) for secret scanning
- FFmpeg (required by whisper)

## Setup

```bash
uv sync
```

## Usage

```bash
# Transcribe with faster-whisper
uv run python src/transcribe_faster.py audio.wav

# Transcribe with OpenAI whisper
uv run python src/transcribe_openai.py audio.wav
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
