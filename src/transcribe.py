"""Unified transcription script supporting multiple backends."""

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def generate_run_id(backend: str, model: str, language: str | None, device: str) -> str:
  """Generate a unique ID based on settings."""
  settings = f"{backend}:{model}:{language}:{device}"
  return hashlib.sha256(settings.encode()).hexdigest()[:12]


def transcribe_faster_whisper(
  audio_path: str,
  model_size: str,
  language: str | None,
  device: str,
) -> str:
  """Transcribe using faster-whisper."""
  from faster_whisper import WhisperModel

  compute_type = "float16" if device == "cuda" else "int8"
  model = WhisperModel(model_size, device=device, compute_type=compute_type)
  segments, _info = model.transcribe(audio_path, beam_size=5, language=language)
  return " ".join(segment.text.strip() for segment in segments)


def transcribe_openai_whisper(
  audio_path: str,
  model_size: str,
  language: str | None,
  device: str,
) -> str:
  """Transcribe using OpenAI whisper."""
  import whisper

  model = whisper.load_model(model_size, device=device)
  result = model.transcribe(audio_path, language=language)
  return result["text"].strip()


def transcribe_whispercpp(
  audio_path: str,
  model_path: str,
) -> str:
  """Transcribe using whisper.cpp."""
  from pywhispercpp.model import Model

  model = Model(model_path)
  segments = model.transcribe(audio_path)
  return " ".join(segment.text.strip() for segment in segments)


def generate_report(data: dict[str, Any]) -> str:
  """Generate a markdown report from transcription data."""
  lines = ["# Transcription Report", ""]
  lines.append(f"**Audio file:** `{data.get('audio', 'unknown')}`")
  lines.append("")

  runs = data.get("runs", [])
  if not runs:
    lines.append("No transcription runs found.")
    return "\n".join(lines)

  # Sort by duration for comparison
  sorted_runs = sorted(runs, key=lambda r: r.get("duration_seconds", 0))

  lines.append(f"**Total runs:** {len(runs)}")
  lines.append("")

  # Summary table
  lines.append("## Performance Summary")
  lines.append("")
  lines.append("| # | Backend | Model | Language | Device | Duration (s) |")
  lines.append("|---|---------|-------|----------|--------|--------------|")

  for i, run in enumerate(sorted_runs, 1):
    backend = run.get("backend", "?")
    model = run.get("model", "?")
    lang = run.get("language") or "auto"
    device = run.get("device", "?")
    duration = run.get("duration_seconds", 0)
    lines.append(f"| {i} | {backend} | {model} | {lang} | {device} | {duration:.2f} |")

  lines.append("")

  # Detailed results
  lines.append("## Transcription Results")
  lines.append("")

  for i, run in enumerate(sorted_runs, 1):
    run_id = run.get("id", "?")
    backend = run.get("backend", "?")
    model = run.get("model", "?")
    lang = run.get("language") or "auto"
    device = run.get("device", "?")
    duration = run.get("duration_seconds", 0)
    timestamp = run.get("timestamp", "?")
    text = run.get("text", "")

    lines.append(f"### {i}. {backend} / {model}")
    lines.append("")
    lines.append(f"- **ID:** `{run_id}`")
    lines.append(f"- **Language:** {lang}")
    lines.append(f"- **Device:** {device}")
    lines.append(f"- **Duration:** {duration:.2f}s")
    lines.append(f"- **Timestamp:** {timestamp}")
    lines.append("")
    lines.append("**Text:**")
    lines.append("")
    lines.append(f"> {text}")
    lines.append("")

  return "\n".join(lines)


def cmd_transcribe(args: argparse.Namespace) -> None:
  """Handle transcribe command."""
  if not Path(args.audio).exists():
    print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
    sys.exit(1)

  if "whispercpp" in args.backend and not args.model_path:
    print("Error: --model-path is required for whispercpp backend", file=sys.stderr)
    sys.exit(1)

  # Build all combinations
  combinations = list(itertools.product(args.backend, args.model, args.language, args.device))
  print(f"Running {len(combinations)} transcription(s)...")

  audio_path = Path(args.audio)
  json_path = audio_path.with_suffix(".json")

  if json_path.exists():
    data = json.loads(json_path.read_text(encoding="utf-8"))
  else:
    data = {"audio": str(audio_path), "runs": []}

  for backend, model, language, device in combinations:
    # For whispercpp, use model_path instead of model
    if backend == "whispercpp":
      model_paths = args.model_path or []
      for model_path in model_paths:
        run_single(args.audio, backend, model_path, language, device, data)
    else:
      run_single(args.audio, backend, model, language, device, data)

  json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"\nAll results saved to {json_path}")


def cmd_report(args: argparse.Namespace) -> None:
  """Handle report command."""
  json_path = Path(args.json_file)
  if not json_path.exists():
    print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
    sys.exit(1)

  data = json.loads(json_path.read_text(encoding="utf-8"))
  report = generate_report(data)

  # Default output path: same name as JSON but with .md extension
  output_path = Path(args.output) if args.output else json_path.with_suffix(".md")
  output_path.write_text(report, encoding="utf-8")
  print(f"Report saved to {output_path}")


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Transcribe audio using various Whisper backends",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  subparsers = parser.add_subparsers(dest="command", help="Available commands")

  # Transcribe command
  trans_parser = subparsers.add_parser(
    "transcribe",
    aliases=["t"],
    help="Transcribe audio file",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s audio.wav --backend openai --model large-v3
  %(prog)s audio.wav --backend whispercpp --model-path models/ggml-base.bin
  %(prog)s audio.wav --language ru --device cuda

  # Compare multiple backends/models (runs all combinations)
  %(prog)s audio.wav --backend faster-whisper openai --model base large-v3
    """,
  )
  trans_parser.add_argument("audio", help="Path to audio file")
  trans_parser.add_argument(
    "--backend",
    "-b",
    nargs="+",
    choices=["faster-whisper", "openai", "whispercpp"],
    default=["faster-whisper"],
    help="Transcription backend(s) (default: faster-whisper)",
  )
  trans_parser.add_argument(
    "--model",
    "-m",
    nargs="+",
    default=["base"],
    help="Model size(s): tiny, base, small, medium, large-v3 (default: base)",
  )
  trans_parser.add_argument(
    "--model-path",
    nargs="+",
    help="Path(s) to ggml model file (required for whispercpp backend)",
  )
  trans_parser.add_argument(
    "--language",
    "-l",
    nargs="+",
    default=[None],
    help="Language code(s) (e.g., en, ru). Auto-detect if not specified",
  )
  trans_parser.add_argument(
    "--device",
    "-d",
    nargs="+",
    choices=["cpu", "cuda"],
    default=["cpu"],
    help="Device(s) to use (default: cpu)",
  )
  trans_parser.set_defaults(func=cmd_transcribe)

  # Report command
  report_parser = subparsers.add_parser(
    "report",
    aliases=["r"],
    help="Generate report from transcription results",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s audio.json                    # Save to audio.md
  %(prog)s audio.json -o custom.md       # Save to custom.md
    """,
  )
  report_parser.add_argument("json_file", help="Path to JSON results file")
  report_parser.add_argument(
    "--output",
    "-o",
    help="Output markdown file path (default: <json_file>.md)",
  )
  report_parser.set_defaults(func=cmd_report)

  args = parser.parse_args()

  if not args.command:
    parser.print_help()
    sys.exit(1)

  args.func(args)


def run_single(  # noqa: PLR0913
  audio: str,
  backend: str,
  model: str,
  language: str | None,
  device: str,
  data: dict,
) -> None:
  """Run a single transcription and update data."""
  run_id = generate_run_id(backend, model, language, device)
  print(f"\n[{run_id}] {backend} / {model} / lang={language} / {device}")

  start_time = time.perf_counter()

  if backend == "faster-whisper":
    result = transcribe_faster_whisper(audio, model, language, device)
  elif backend == "openai":
    result = transcribe_openai_whisper(audio, model, language, device)
  elif backend == "whispercpp":
    result = transcribe_whispercpp(audio, model)

  duration = time.perf_counter() - start_time

  run_record = {
    "id": run_id,
    "timestamp": datetime.now(UTC).isoformat(),
    "duration_seconds": round(duration, 2),
    "backend": backend,
    "model": model,
    "language": language,
    "device": device,
    "text": result,
  }

  # Update existing run with same ID or append new one
  existing_idx = next((i for i, r in enumerate(data["runs"]) if r.get("id") == run_id), None)
  if existing_idx is not None:
    data["runs"][existing_idx] = run_record
    print(f"  Updated existing run ({duration:.2f}s)")
  else:
    data["runs"].append(run_record)
    print(f"  Added new run ({duration:.2f}s)")

  print(f"  Text: {result[:100]}..." if len(result) > 100 else f"  Text: {result}")


if __name__ == "__main__":
  main()
