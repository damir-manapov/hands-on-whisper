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

## Download models

Pre-download models for offline use:

```bash
# Download all models for all backends
uv run python src/download_models.py --backend all

# Download for specific backend
uv run python src/download_models.py --backend faster-whisper
uv run python src/download_models.py --backend openai
uv run python src/download_models.py --backend whispercpp
```

Model sizes: tiny, base, small, medium, large-v3, large-v3-turbo, distil-large-v3

### Model availability

| Model | faster-whisper | openai-whisper | whisper.cpp | Notes |
|-------|---------------|----------------|-------------|-------|
| tiny, base, small, medium | ✅ | ✅ | ✅ f16 + q8_0 | |
| large-v3 | ✅ | ✅ | ✅ f16 + q5_0 | Best multilingual quality |
| large-v3-turbo | ✅ | ✅ | ✅ f16 + q8_0 | Fast, good quality |
| distil-large-v3 | ✅ | ✅ (via HF) | ✅ f16 only | ⚠️ English only |

Quantization notes:
- **q8_0**: 8-bit quantization, ~2x smaller, similar quality (used with `--compute-type int8`)
- **q5_0**: 5-bit quantization, ~3x smaller, slight quality loss
- **f16**: float16 (default), best quality

## Usage

```bash
# Basic transcription (uses faster-whisper by default)
uv run python src/transcribe.py transcribe audio.wav

# Specify backend
uv run python src/transcribe.py transcribe audio.wav --backend openai
uv run python src/transcribe.py transcribe audio.wav --backend whispercpp -m base

# Specify model size
uv run python src/transcribe.py transcribe audio.wav --model large-v3

# Specify language (auto-detect if not set)
uv run python src/transcribe.py transcribe audio.wav --language ru

# Use GPU
uv run python src/transcribe.py transcribe audio.wav --device cuda

# Compare multiple backends/models (runs all combinations)
uv run python src/transcribe.py transcribe audio.wav --backend faster-whisper openai --model base large-v3

# Adjust decoding parameters
uv run python src/transcribe.py transcribe audio.wav --beam-size 10 --temperature 0.2

# Reduce repetitive hallucinations
uv run python src/transcribe.py transcribe audio.wav --no-condition-on-prev

# Set compute precision
uv run python src/transcribe.py transcribe audio.wav --compute-type float16

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
- Performance summary table (sorted by WER when reference exists) with all parameters
- WER/CER metrics (if `audio.txt` reference transcription exists)
- Detailed transcription results for each run

### Sample Reports

**sherbakov_call** - Sample audio from [Sherbakov Call Prank](https://www.youtube.com/watch?v=vZQNgvIoPr0) (Russian):

- [sherbakov_call.json](calls/sherbakov_call.json) - Raw benchmark data
- [sherbakov_call.md](calls/sherbakov_call.md) - Formatted report with WER/CER metrics

**finance** - Sample audio from [Russian Speech Recognition Dataset](https://huggingface.co/datasets/AxonData/russian-speech-recognition-dataset) (Russian):

- [finance.json](calls/finance.json) - Raw benchmark data
- [finance.md](calls/finance.md) - Formatted report with WER/CER metrics

### Metrics

To calculate WER (Word Error Rate) and CER (Character Error Rate), place a reference transcription file alongside your audio:

```
calls/
  sherbakov_call.wav    # Audio file
  sherbakov_call.txt    # Manual reference transcription
  sherbakov_call.json   # Generated results
  sherbakov_call.md     # Generated report with WER/CER
```

The report command auto-detects the `.txt` file and includes metrics in the table.

**Normalization**: Text is normalized before comparison (lowercase, remove punctuation, collapse whitespace). This is standard practice and works with any language including Russian.

## Optimization

Find optimal parameters using Optuna (Bayesian optimization):

```bash
# Full search across all backends, models, compute types (default)
uv run python src/transcribe.py optimize audio.wav -l ru --n-trials 50

# Optimize for CER instead of WER
uv run python src/transcribe.py o audio.wav -l ru --metric cer

# Limit search to specific backends
uv run python src/transcribe.py o audio.wav --backends faster-whisper -l ru

# Limit search to specific models
uv run python src/transcribe.py o audio.wav --models large-v3 large-v3-turbo -l ru

# Quick test with fewer trials
uv run python src/transcribe.py o audio.wav -l ru --n-trials 10
```

### Search space

By default, Optuna searches across **all** backends, models, and compute types:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `backend` | all 3 | faster-whisper, openai, whispercpp |
| `model` | all 6 | tiny, base, small, medium, large-v3, large-v3-turbo |
| `compute_type` | int8, float16, float32 | Precision options |
| `beam_size` | 1-10 | Beam search width |
| `temperature` | 0.0-0.5 | Sampling temperature |
| `condition_on_prev` | True/False | Condition on previous text |
| `--metric` | wer | Optimize for `wer` or `cer` |

Note: `condition_on_prev` is always `False` for whispercpp (not supported).

### Features

- **Learns from history**: Optuna is initialized with all previous runs from JSON
- **Saves results**: Each trial is saved to JSON and MD report is regenerated
- **Caching**: If a run with same settings exists, cached result is reused
- **Resumable**: Stop and restart - previous runs inform better suggestions

Requires `audio.txt` reference transcription for WER calculation.

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
      "memory_delta_mb": 657.8,
      "memory_peak_mb": 681.5,
      "backend": "faster-whisper",
      "model": "base",
      "language": null,
      "device": "cpu",
      "text": "transcribed text..."
    }
  ]
}
```

- `memory_delta_mb`: Memory increase during transcription (model loading + inference)
- `memory_peak_mb`: Peak memory usage during run

Each run has a unique ID based on settings (backend, model, language, device). Re-running with the same settings skips existing runs (use this to resume interrupted comparison runs).

## Benchmarks

Tested on Russian audio (sherbakov_call.wav), CPU, tiny model:

| Backend | Duration | Notes |
|---------|----------|-------|
| whispercpp | 16.75s ⚡ | 2.3x faster than faster-whisper |
| faster-whisper | 39.00s | Good Russian quality |
| openai | 51.79s | Multi-language output with tiny model |

Note: Results vary by model size, language, and hardware. Larger models (base, small, medium, large-v3) generally produce better quality.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--beam-size` | 5 | Beam search width. Higher = better quality, slower |
| `--temperature` | 0.0 | Sampling temperature. 0 = greedy, >0 = more varied output |
| `--compute-type` | auto | Precision: auto, float32, float16, int8. For whispercpp: int8 uses q8_0 models |
| `--no-condition-on-prev` | false | Don't condition on previous text. Helps break repetition loops |

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

## GPU Benchmarking

Run benchmarks on cloud GPU instances using Terraform:

```bash
cd terraform/selectel
cp terraform.tfvars.example terraform.tfvars
# Set credentials (see terraform/selectel/README.md)

terraform init
terraform apply

# SSH and run optimization
ssh root@<vm-ip>
cd /root/hands-on-whisper
uv run python src/transcribe.py optimize calls/sherbakov_call.wav -l ru --n-trials 50

# Don't forget to destroy when done!
terraform destroy
```

See [terraform/selectel/README.md](terraform/selectel/README.md) for full instructions.

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
