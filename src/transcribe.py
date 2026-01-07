"""Unified transcription CLI supporting multiple Whisper backends."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

from backends import CLOUD_BACKENDS, resolve_whispercpp_model_path
from optimize import cmd_optimize
from report import generate_report
from runner import (
  get_result_suffix,
  load_reference,
  run_single,
  save_results,
)


def cmd_transcribe(args: argparse.Namespace) -> None:
  """Handle transcribe command."""
  if not Path(args.audio).exists():
    print(f"Error: Audio file not found: {args.audio}", file=sys.stderr)
    sys.exit(1)

  combinations = list(itertools.product(args.backend, args.model, args.language, args.runtime))
  print(f"Running {len(combinations)} transcription(s)...")

  audio_path = Path(args.audio)
  condition_on_prev = not args.no_condition_on_prev

  saved_files: set[Path] = set()
  data_cache: dict[Path, dict[str, Any]] = {}

  def get_data_for_path(json_path: Path) -> dict[str, Any]:
    if json_path not in data_cache:
      if json_path.exists():
        data_cache[json_path] = json.loads(json_path.read_text(encoding="utf-8"))
      else:
        data_cache[json_path] = {"audio": str(audio_path), "runs": []}
    return data_cache[json_path]

  for backend, model, language, runtime_arg in combinations:
    runtime = "cloud" if backend in CLOUD_BACKENDS else runtime_arg

    suffix = get_result_suffix(backend, runtime)
    json_path = audio_path.parent / f"{audio_path.stem}{suffix}.json"
    data = get_data_for_path(json_path)

    if backend == "whispercpp" and args.model_path:
      for mp in args.model_path:
        if not Path(mp).exists():
          print(f"Error: Model file not found: {mp}", file=sys.stderr)
          sys.exit(1)
        model_name = Path(mp).stem.replace("ggml-", "").replace("-q8_0", "").replace("-q5_0", "")
        run_single(
          args.audio,
          backend,
          model_name,
          language,
          runtime,
          args.beam_size,
          args.temperature,
          args.compute_type,
          condition_on_prev,
          args.batch_size if backend == "faster-whisper" else 0,
          data,
          user=args.user,
        )
        save_results(data, json_path)
        saved_files.add(json_path)
    else:
      if backend == "whispercpp":
        model_path = resolve_whispercpp_model_path(model, args.compute_type)
        if not Path(model_path).exists():
          print(f"Error: Model file not found: {model_path}", file=sys.stderr)
          print("Run: uv run python src/download_models.py --backend whispercpp", file=sys.stderr)
          sys.exit(1)

      run_single(
        args.audio,
        backend,
        model,
        language,
        runtime,
        args.beam_size,
        args.temperature,
        args.compute_type,
        condition_on_prev,
        args.batch_size if backend == "faster-whisper" else 0,
        data,
        user=args.user,
      )
      save_results(data, json_path)
      saved_files.add(json_path)

  print("\nResults saved:")
  for json_path in sorted(saved_files):
    print(f"  - {json_path}")
    print(f"  - {json_path.with_suffix('.md')}")


def cmd_report(args: argparse.Namespace) -> None:
  """Handle report command."""
  json_path = Path(args.json_file)
  if not json_path.exists():
    print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
    sys.exit(1)

  data = json.loads(json_path.read_text(encoding="utf-8"))

  reference = load_reference(data)
  if reference:
    print(f"Reference: {len(reference)} chars, {len(reference.split())} words")

  report = generate_report(data, reference)

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
  %(prog)s audio.wav --language ru --runtime cuda
  %(prog)s audio.wav --backend openai-api --model gpt-4o --runtime cloud

  # Compare multiple backends/models (runs all combinations)
  %(prog)s audio.wav --backend faster-whisper openai --model base large-v3
    """,
  )
  trans_parser.add_argument("audio", help="Path to audio file")
  trans_parser.add_argument(
    "--backend",
    "-b",
    nargs="+",
    choices=["faster-whisper", "openai", "whispercpp", "yandex", "openai-api"],
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
    "--runtime",
    "-d",
    nargs="+",
    choices=["cpu", "cuda", "cloud"],
    default=["cpu"],
    help="Runtime(s) to use: cpu, cuda for local, cloud for cloud backends (default: cpu)",
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
  trans_parser.add_argument(
    "--no-condition-on-prev",
    action="store_true",
    help="Don't condition on previous text (helps reduce repetitive hallucinations)",
  )
  trans_parser.add_argument(
    "--batch-size",
    type=int,
    default=0,
    help="Batch size for faster-whisper (0=sequential, 1-32=parallel segments, default: 0)",
  )
  trans_parser.add_argument(
    "--user",
    type=str,
    default=None,
    help="Username to record in results",
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

  # Optimize command
  optim_parser = subparsers.add_parser(
    "optimize",
    aliases=["o"],
    help="Find optimal parameters using Optuna (requires reference.txt)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s audio.wav                        # Full search across all backends/models
  %(prog)s audio.wav -l ru                  # Full search with Russian language
  %(prog)s audio.wav --backends faster-whisper  # Limit to one backend
  %(prog)s audio.wav --models large-v3      # Limit to one model
  %(prog)s audio.wav --n-trials 50          # Run 50 optimization trials
    """,
  )
  optim_parser.add_argument("audio", help="Audio file to transcribe")
  optim_parser.add_argument(
    "--backends",
    "-b",
    nargs="+",
    default=None,
    choices=["faster-whisper", "openai", "whispercpp", "yandex", "openai-api"],
    help="Backends to search (default: all local backends)",
  )
  optim_parser.add_argument(
    "--models",
    "-m",
    nargs="+",
    default=None,
    help="Models to search (default: tiny,base,small,medium,large-v3,large-v3-turbo)",
  )
  optim_parser.add_argument(
    "--compute-types",
    "-c",
    nargs="+",
    default=None,
    choices=["auto", "int8", "float16", "float32"],
    help="Compute types to search (default: int8,float16,float32)",
  )
  optim_parser.add_argument(
    "--batch-sizes",
    nargs="+",
    type=int,
    default=None,
    help="Batch sizes to search for faster-whisper (default: 0-32 range)",
  )
  optim_parser.add_argument(
    "--beam-sizes",
    nargs="+",
    type=int,
    default=None,
    help="Beam sizes to search (default: 1-10 range)",
  )
  optim_parser.add_argument(
    "--language", "-l", default=None, help="Language code (auto-detect if not set)"
  )
  optim_parser.add_argument(
    "--runtime",
    "-d",
    default="cpu",
    choices=["cpu", "cuda", "cloud"],
    help="Runtime (default: cpu)",
  )
  optim_parser.add_argument(
    "--n-trials", type=int, default=10, help="Number of optimization trials (default: 10)"
  )
  optim_parser.add_argument(
    "--metric",
    default="wer",
    choices=["wer", "cer"],
    help="Metric to optimize: wer (Word Error Rate) or cer (Character Error Rate) (default: wer)",
  )
  optim_parser.add_argument(
    "--user",
    type=str,
    default=None,
    help="Username to record in results",
  )
  optim_parser.set_defaults(func=cmd_optimize)

  args = parser.parse_args()

  if not args.command:
    parser.print_help()
    sys.exit(1)

  args.func(args)


if __name__ == "__main__":
  main()
