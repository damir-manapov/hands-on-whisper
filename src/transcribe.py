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


def generate_run_id(  # noqa: PLR0913
  backend: str,
  model: str,
  language: str | None,
  device: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
) -> str:
  """Generate a unique ID based on settings."""
  settings = (
    f"{backend}:{model}:{language}:{device}:beam{beam_size}:temp{temperature}:{compute_type}"
  )
  return hashlib.sha256(settings.encode()).hexdigest()[:12]


def resolve_whispercpp_model_path(model: str, compute_type: str) -> str:
  """Resolve whisper.cpp model path from model name and compute type.

  Different models have different quantizations available:
  - tiny, base, small, medium, large-v3-turbo: f16 and q8_0
  - large-v3: f16 and q5_0 (no q8_0)
  - distil-large-v3: f16 only (from distil-whisper/distil-large-v3-ggml repo)
  """
  if compute_type == "int8":
    if model == "large-v3":
      suffix = "-q5_0"  # No q8_0, only q5_0
    elif model == "distil-large-v3":
      suffix = ""  # No quantized version available
    else:
      suffix = "-q8_0"
  else:
    suffix = ""
  return f"models/ggml-{model}{suffix}.bin"


def transcribe_faster_whisper(  # noqa: PLR0913
  audio_path: str,
  model_size: str,
  language: str | None,
  device: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
) -> str:
  """Transcribe using faster-whisper."""
  from faster_whisper import WhisperModel

  # Map unified compute_type to faster-whisper compute_type
  if compute_type == "auto":
    ct = "float16" if device == "cuda" else "int8"
  elif compute_type == "float32":
    ct = "float32"
  elif compute_type == "float16":
    ct = "float16"
  elif compute_type == "int8":
    ct = "int8"
  else:
    ct = compute_type  # Allow raw values like int8_float16

  model = WhisperModel(model_size, device=device, compute_type=ct)
  segments, _info = model.transcribe(
    audio_path, beam_size=beam_size, language=language, temperature=temperature
  )
  return " ".join(segment.text.strip() for segment in segments)


def transcribe_openai_whisper(  # noqa: PLR0913
  audio_path: str,
  model_size: str,
  language: str | None,
  device: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
) -> str:
  """Transcribe using OpenAI whisper."""
  import whisper

  # Map compute_type to fp16 flag
  fp16 = compute_type != "float32" and device == "cuda"

  # distil-large-v3 requires loading from local path
  if model_size == "distil-large-v3":
    model_path = "models/distil-large-v3-openai/model.bin"
    model = whisper.load_model(model_path, device=device)
  else:
    model = whisper.load_model(model_size, device=device)
  result = model.transcribe(
    audio_path, language=language, beam_size=beam_size, temperature=temperature, fp16=fp16
  )
  return result["text"].strip()


def transcribe_whispercpp(  # noqa: PLR0913
  audio_path: str,
  model_path: str,
  language: str | None,
  beam_size: int,
  temperature: float,
  compute_type: str,  # noqa: ARG001 - used for run_id, model selected via path
) -> str:
  """Transcribe using whisper.cpp."""
  from pywhispercpp.model import Model

  model = Model(model_path)
  segments = model.transcribe(
    audio_path, language=language or "", beam_size=beam_size, temperature=temperature
  )
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
    # For whispercpp, auto-resolve model path or use explicit --model-path
    if backend == "whispercpp":
      if args.model_path:
        # Use explicit model paths
        for model_path in args.model_path:
          run_single(
            args.audio,
            backend,
            model_path,
            language,
            device,
            args.beam_size,
            args.temperature,
            args.compute_type,
            data,
          )
      else:
        # Auto-resolve from model name and compute_type
        model_path = resolve_whispercpp_model_path(model, args.compute_type)
        if not Path(model_path).exists():
          print(f"Error: Model file not found: {model_path}", file=sys.stderr)
          print("Run: uv run python src/download_models.py --backend whispercpp", file=sys.stderr)
          sys.exit(1)
        run_single(
          args.audio,
          backend,
          model_path,
          language,
          device,
          args.beam_size,
          args.temperature,
          args.compute_type,
          data,
        )
    else:
      run_single(
        args.audio,
        backend,
        model,
        language,
        device,
        args.beam_size,
        args.temperature,
        args.compute_type,
        data,
      )

  json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
  print(f"\nResults saved to {json_path}")

  # Generate report
  report = generate_report(data)
  md_path = json_path.with_suffix(".md")
  md_path.write_text(report, encoding="utf-8")
  print(f"Report saved to {md_path}")


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
    help="Model size(s): tiny, base, small, medium, large-v3, large-v3-turbo, "
    "distil-large-v3 (default: base)",
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
  trans_parser.add_argument(
    "--beam-size",
    type=int,
    default=5,
    help="Beam size for decoding (default: 5, higher = better quality but slower)",
  )
  trans_parser.add_argument(
    "--temperature",
    type=float,
    default=0.0,
    help="Sampling temperature (default: 0.0 = greedy, >0 = more varied)",
  )
  trans_parser.add_argument(
    "--compute-type",
    "-c",
    default="auto",
    help="Compute type: auto, float32, float16, int8 (default: auto)",
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
  beam_size: int,
  temperature: float,
  compute_type: str,
  data: dict,
) -> None:
  """Run a single transcription and update data."""
  run_id = generate_run_id(backend, model, language, device, beam_size, temperature, compute_type)

  # Skip if we already have this run
  existing_idx = next((i for i, r in enumerate(data["runs"]) if r.get("id") == run_id), None)
  if existing_idx is not None:
    print(f"\n[{run_id}] {backend} / {model} / lang={language} / {device} - skipped (exists)")
    return

  print(f"\n[{run_id}] {backend} / {model} / lang={language} / {device}")
  print(f"  beam_size={beam_size}, temperature={temperature}, compute_type={compute_type}")

  start_time = time.perf_counter()

  if backend == "faster-whisper":
    result = transcribe_faster_whisper(
      audio, model, language, device, beam_size, temperature, compute_type
    )
  elif backend == "openai":
    result = transcribe_openai_whisper(
      audio, model, language, device, beam_size, temperature, compute_type
    )
  elif backend == "whispercpp":
    result = transcribe_whispercpp(audio, model, language, beam_size, temperature, compute_type)

  duration = time.perf_counter() - start_time

  run_record = {
    "id": run_id,
    "timestamp": datetime.now(UTC).isoformat(),
    "duration_seconds": round(duration, 2),
    "backend": backend,
    "model": model,
    "language": language,
    "device": device,
    "beam_size": beam_size,
    "temperature": temperature,
    "compute_type": compute_type,
    "text": result,
  }

  data["runs"].append(run_record)
  print(f"  Done ({duration:.2f}s)")
  print(f"  Text: {result[:100]}..." if len(result) > 100 else f"  Text: {result}")


if __name__ == "__main__":
  main()
