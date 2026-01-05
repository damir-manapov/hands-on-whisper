"""Unified transcription script supporting multiple backends."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


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


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Transcribe audio using various Whisper backends",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s audio.wav
  %(prog)s audio.wav --backend openai --model large-v3
  %(prog)s audio.wav --backend whispercpp --model-path models/ggml-base.bin
  %(prog)s audio.wav --language ru --device cuda
    """,
  )

  parser.add_argument("audio", help="Path to audio file")
  parser.add_argument(
    "--backend",
    "-b",
    choices=["faster-whisper", "openai", "whispercpp"],
    default="faster-whisper",
    help="Transcription backend (default: faster-whisper)",
  )
  parser.add_argument(
    "--model",
    "-m",
    default="base",
    help="Model size: tiny, base, small, medium, large-v3 (default: base)",
  )
  parser.add_argument(
    "--model-path",
    help="Path to ggml model file (required for whispercpp backend)",
  )
  parser.add_argument(
    "--language",
    "-l",
    default=None,
    help="Language code (e.g., en, ru). Auto-detect if not specified",
  )
  parser.add_argument(
    "--device",
    "-d",
    choices=["cpu", "cuda"],
    default="cpu",
    help="Device to use (default: cpu)",
  )
  parser.add_argument(
    "--output",
    "-o",
    help="Output file path (prints to stdout if not specified)",
  )

  args = parser.parse_args()

  if not Path(args.audio).exists():
    print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
    sys.exit(1)

  if args.backend == "whispercpp" and not args.model_path:
    print("Error: --model-path is required for whispercpp backend", file=sys.stderr)
    sys.exit(1)

  if args.backend == "faster-whisper":
    result = transcribe_faster_whisper(args.audio, args.model, args.language, args.device)
  elif args.backend == "openai":
    result = transcribe_openai_whisper(args.audio, args.model, args.language, args.device)
  elif args.backend == "whispercpp":
    result = transcribe_whispercpp(args.audio, args.model_path)

  # Build run record with unique ID based on settings
  model_used = args.model_path if args.backend == "whispercpp" else args.model
  run_id = generate_run_id(args.backend, model_used, args.language, args.device)

  run_record = {
    "id": run_id,
    "timestamp": datetime.now(UTC).isoformat(),
    "backend": args.backend,
    "model": model_used,
    "language": args.language,
    "device": args.device,
    "text": result,
  }

  # Save to JSON file named after audio file
  audio_path = Path(args.audio)
  json_path = audio_path.with_suffix(".json")

  if json_path.exists():
    data = json.loads(json_path.read_text(encoding="utf-8"))
  else:
    data = {"audio": str(audio_path), "runs": []}

  # Update existing run with same ID or append new one
  existing_idx = next((i for i, r in enumerate(data["runs"]) if r.get("id") == run_id), None)
  if existing_idx is not None:
    data["runs"][existing_idx] = run_record
    print(f"Updated existing run {run_id} in {json_path}")
  else:
    data["runs"].append(run_record)
    print(f"Added new run {run_id} to {json_path}")

  json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

  if args.output:
    Path(args.output).write_text(result, encoding="utf-8")
    print(f"Transcription also saved to {args.output}")
  else:
    print(result)


if __name__ == "__main__":
  main()
