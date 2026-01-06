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


def normalize_text(text: str) -> str:
  """Normalize text for WER comparison: lowercase, remove punctuation, collapse whitespace."""
  import re

  text = text.lower()
  text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation (Unicode-aware)
  text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
  return text


def calculate_metrics(reference: str, hypothesis: str) -> tuple[float, float]:
  """Calculate WER and CER between reference and hypothesis (normalizes both)."""
  from jiwer import cer, wer

  ref_norm = normalize_text(reference)
  hyp_norm = normalize_text(hypothesis)
  wer_score = wer(ref_norm, hyp_norm) * 100
  cer_score = cer(ref_norm, hyp_norm) * 100
  return wer_score, cer_score


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
  condition_on_prev: bool = True,
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
    audio_path,
    beam_size=beam_size,
    language=language,
    temperature=temperature,
    condition_on_previous_text=condition_on_prev,
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
  condition_on_prev: bool = True,
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
    audio_path,
    language=language,
    beam_size=beam_size,
    temperature=temperature,
    fp16=fp16,
    condition_on_previous_text=condition_on_prev,
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
    audio_path,
    language=language or "",
    beam_search={"beam_size": beam_size, "patience": -1.0},
    temperature=temperature,
  )
  return " ".join(segment.text.strip() for segment in segments)


def generate_report(data: dict[str, Any], reference: str | None = None) -> str:
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
  if reference:
    lines.append(f"**Reference:** {len(reference)} chars, {len(reference.split())} words")
  lines.append("")

  # Summary table
  lines.append("## Performance Summary")
  lines.append("")
  base_header = "| # | Backend | Model | Language | Device | Duration (s) | Mem Δ | Mem Peak |"
  base_sep = "|---|---------|-------|----------|--------|--------------|-------|----------|"
  if reference:
    lines.append(f"{base_header} WER % | CER % |")
    lines.append(f"{base_sep}-------|-------|")
  else:
    lines.append(base_header)
    lines.append(base_sep)

  for i, run in enumerate(sorted_runs, 1):
    backend = run.get("backend", "?")
    model = run.get("model", "?")
    lang = run.get("language") or "auto"
    device = run.get("device", "?")
    duration = run.get("duration_seconds", 0)
    mem_delta = run.get("memory_delta_mb", 0)
    mem_peak = run.get("memory_peak_mb", 0)
    base_row = (
      f"| {i} | {backend} | {model} | {lang} | {device} "
      f"| {duration:.2f} | {mem_delta} | {mem_peak} |"
    )
    if reference:
      wer_score, cer_score = calculate_metrics(reference, run.get("text", ""))
      lines.append(f"{base_row} {wer_score:.1f} | {cer_score:.1f} |")
    else:
      lines.append(base_row)

  lines.append("")

  # Detailed results
  _append_detailed_results(lines, sorted_runs)

  return "\n".join(lines)


def _append_detailed_results(lines: list[str], sorted_runs: list[dict]) -> None:
  """Append detailed transcription results to report lines."""
  lines.append("## Transcription Results")
  lines.append("")

  for i, run in enumerate(sorted_runs, 1):
    run_id = run.get("id", "?")
    backend = run.get("backend", "?")
    model = run.get("model", "?")
    lang = run.get("language") or "auto"
    device = run.get("device", "?")
    duration = run.get("duration_seconds", 0)
    mem_delta = run.get("memory_delta_mb", 0)
    mem_peak = run.get("memory_peak_mb", 0)
    beam_size = run.get("beam_size", 5)
    temperature = run.get("temperature", 0.0)
    compute_type = run.get("compute_type", "auto")
    condition_on_prev = run.get("condition_on_prev", True)
    timestamp = run.get("timestamp", "?")
    text = run.get("text", "")

    lines.append(f"### {i}. {backend} / {model}")
    lines.append("")
    lines.append(f"- **ID:** `{run_id}`")
    lines.append(f"- **Language:** {lang}")
    lines.append(f"- **Device:** {device}")
    lines.append(f"- **Duration:** {duration:.2f}s")
    lines.append(f"- **Memory:** Δ {mem_delta} MB, peak {mem_peak} MB")
    lines.append(f"- **Beam size:** {beam_size}")
    lines.append(f"- **Temperature:** {temperature}")
    lines.append(f"- **Compute type:** {compute_type}")
    lines.append(f"- **Condition on prev:** {condition_on_prev}")
    lines.append(f"- **Timestamp:** {timestamp}")
    lines.append("")
    lines.append("**Text:**")
    lines.append("")
    lines.append(f"> {text}")
    lines.append("")


def save_results(data: dict[str, Any], json_path: Path) -> None:
  """Save JSON data and regenerate MD report."""
  json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

  # Auto-detect reference file
  ref_path = json_path.with_suffix(".txt")
  reference = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None

  report = generate_report(data, reference)
  md_path = json_path.with_suffix(".md")
  md_path.write_text(report, encoding="utf-8")


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

  condition_on_prev = not args.no_condition_on_prev

  for backend, model, language, device in combinations:
    # Resolve model path for whispercpp
    if backend == "whispercpp":
      if args.model_path:
        model_paths = args.model_path
      else:
        model_path = resolve_whispercpp_model_path(model, args.compute_type)
        if not Path(model_path).exists():
          print(f"Error: Model file not found: {model_path}", file=sys.stderr)
          print("Run: uv run python src/download_models.py --backend whispercpp", file=sys.stderr)
          sys.exit(1)
        model_paths = [model_path]

      for mp in model_paths:
        run_single(
          args.audio,
          backend,
          mp,
          language,
          device,
          args.beam_size,
          args.temperature,
          args.compute_type,
          condition_on_prev,
          data,
        )
        save_results(data, json_path)
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
        condition_on_prev,
        data,
      )
      save_results(data, json_path)

  print(f"\nResults saved to {json_path}")
  print(f"Report saved to {json_path.with_suffix('.md')}")


def cmd_report(args: argparse.Namespace) -> None:
  """Handle report command."""
  json_path = Path(args.json_file)
  if not json_path.exists():
    print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
    sys.exit(1)

  data = json.loads(json_path.read_text(encoding="utf-8"))

  # Auto-detect reference file
  ref_path = json_path.with_suffix(".txt")
  reference = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None
  if reference:
    print(f"Reference: {len(reference)} chars, {len(reference.split())} words")

  report = generate_report(data, reference)

  # Default output path: same name as JSON but with .md extension
  output_path = Path(args.output) if args.output else json_path.with_suffix(".md")
  output_path.write_text(report, encoding="utf-8")
  print(f"Report saved to {output_path}")


def _run_transcription(  # noqa: PLR0913
  audio: str,
  backend: str,
  model: str,
  language: str | None,
  device: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
  condition_on_prev: bool,
) -> dict:
  """Run transcription and return run record with timing/memory stats."""
  import psutil

  run_id = generate_run_id(backend, model, language, device, beam_size, temperature, compute_type)

  process = psutil.Process()
  mem_before = process.memory_info().rss
  start_time = time.perf_counter()

  if backend == "faster-whisper":
    result = transcribe_faster_whisper(
      audio, model, language, device, beam_size, temperature, compute_type, condition_on_prev
    )
  elif backend == "openai":
    result = transcribe_openai_whisper(
      audio, model, language, device, beam_size, temperature, compute_type, condition_on_prev
    )
  elif backend == "whispercpp":
    model_path = resolve_whispercpp_model_path(model, compute_type)
    result = transcribe_whispercpp(
      audio, model_path, language, beam_size, temperature, compute_type
    )
  else:
    msg = f"Unknown backend: {backend}"
    raise ValueError(msg)

  duration = time.perf_counter() - start_time
  mem_after = process.memory_info().rss
  mem_used_mb = round((mem_after - mem_before) / 1024 / 1024, 1)
  mem_peak_mb = round(mem_after / 1024 / 1024, 1)

  return {
    "id": run_id,
    "timestamp": datetime.now(UTC).isoformat(),
    "duration_seconds": round(duration, 2),
    "memory_delta_mb": mem_used_mb,
    "memory_peak_mb": mem_peak_mb,
    "backend": backend,
    "model": model,
    "language": language,
    "device": device,
    "beam_size": beam_size,
    "temperature": temperature,
    "compute_type": compute_type,
    "condition_on_prev": condition_on_prev,
    "text": result,
  }


def _run_optimization_trial(  # noqa: PLR0913
  trial_number: int,
  backend: str,
  model: str,
  compute_type: str,
  beam_size: int,
  temperature: float,
  condition_on_prev: bool,
  audio: str,
  language: str | None,
  device: str,
  reference: str,
  data: dict,
  json_path: Path,
) -> float:
  """Run a single optimization trial and return WER score."""
  # Check if already exists (use cached result)
  run_id = generate_run_id(backend, model, language, device, beam_size, temperature, compute_type)
  existing = next((r for r in data["runs"] if r.get("id") == run_id), None)
  if existing:
    print(f"\n[Trial {trial_number}] {run_id} - using cached result")
    wer_score, _ = calculate_metrics(reference, existing.get("text", ""))
    print(f"  WER: {wer_score:.1f}%")
    return wer_score / 100

  print(
    f"\n[Trial {trial_number}] {backend}/{model} compute={compute_type} "
    f"beam={beam_size} temp={temperature:.2f} cond={condition_on_prev}"
  )

  run_record = _run_transcription(
    audio, backend, model, language, device, beam_size, temperature, compute_type, condition_on_prev
  )
  data["runs"].append(run_record)
  save_results(data, json_path)

  wer_score, _ = calculate_metrics(reference, run_record["text"])
  mem_used_mb = run_record["memory_delta_mb"]
  duration = run_record["duration_seconds"]
  print(f"  Done ({duration:.2f}s, mem: +{mem_used_mb}MB) WER: {wer_score:.1f}%")
  return wer_score / 100


def cmd_optimize(args: argparse.Namespace) -> None:
  """Find optimal parameters using Optuna."""
  import optuna

  audio_path = Path(args.audio)
  if not audio_path.exists():
    print(f"Error: Audio file not found: {audio_path}", file=sys.stderr)
    sys.exit(1)

  ref_path = audio_path.with_suffix(".txt")
  if not ref_path.exists():
    print(f"Error: Reference file required: {ref_path}", file=sys.stderr)
    sys.exit(1)

  reference = ref_path.read_text(encoding="utf-8").strip()
  print(f"Reference: {len(reference.split())} words")

  # Load or create JSON data
  json_path = audio_path.with_suffix(".json")
  if json_path.exists():
    data = json.loads(json_path.read_text(encoding="utf-8"))
  else:
    data = {"audio": str(audio_path), "runs": []}

  # Parse search space from args (defaults to all options)
  all_backends = ["faster-whisper", "openai", "whispercpp"]
  all_models = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
  all_compute_types = ["int8", "float16", "float32"]

  backends = args.backends if args.backends else all_backends
  models = args.models if args.models else all_models
  compute_types = args.compute_types if args.compute_types else all_compute_types

  print(f"Search space: backends={backends}, models={models}, compute_types={compute_types}")

  def objective(trial: optuna.Trial) -> float:
    backend = trial.suggest_categorical("backend", backends)
    model = trial.suggest_categorical("model", models)
    compute_type = trial.suggest_categorical("compute_type", compute_types)
    beam_size = trial.suggest_int("beam_size", 1, 10)
    temperature = trial.suggest_float("temperature", 0.0, 0.5)

    # whispercpp doesn't support condition_on_prev
    if backend == "whispercpp":
      condition_on_prev = False
    else:
      condition_on_prev = trial.suggest_categorical("condition_on_prev", [True, False])

    return _run_optimization_trial(
      trial.number,
      backend,
      model,
      compute_type,
      beam_size,
      temperature,
      condition_on_prev,
      args.audio,
      args.language,
      args.device,
      reference,
      data,
      json_path,
    )

  study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
  study.optimize(objective, n_trials=args.n_trials)

  print("\n" + "=" * 50)
  print("OPTIMIZATION RESULTS")
  print("=" * 50)
  print(f"Best WER: {study.best_value * 100:.1f}%")
  print("Best parameters:")
  for key, value in study.best_params.items():
    print(f"  {key}: {value}")


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
  trans_parser.add_argument(
    "--no-condition-on-prev",
    action="store_true",
    help="Don't condition on previous text (helps reduce repetitive hallucinations)",
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
    choices=["faster-whisper", "openai", "whispercpp"],
    help="Backends to search (default: all)",
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
    "--language", "-l", default=None, help="Language code (auto-detect if not set)"
  )
  optim_parser.add_argument(
    "--device", "-d", default="cpu", choices=["cpu", "cuda"], help="Device (default: cpu)"
  )
  optim_parser.add_argument(
    "--n-trials", type=int, default=10, help="Number of optimization trials (default: 10)"
  )
  optim_parser.set_defaults(func=cmd_optimize)

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
  condition_on_prev: bool,
  data: dict,
) -> None:
  """Run a single transcription and update data."""
  run_id = generate_run_id(backend, model, language, device, beam_size, temperature, compute_type)

  # Skip if we already have this run
  existing = next((r for r in data["runs"] if r.get("id") == run_id), None)
  if existing:
    print(f"\n[{run_id}] {backend} / {model} / lang={language} / {device} - skipped (exists)")
    return

  print(f"\n[{run_id}] {backend} / {model} / lang={language} / {device}")
  cond_str = "" if condition_on_prev else ", no_cond_prev"
  print(f"  beam={beam_size}, temp={temperature}, compute={compute_type}{cond_str}")

  run_record = _run_transcription(
    audio, backend, model, language, device, beam_size, temperature, compute_type, condition_on_prev
  )
  data["runs"].append(run_record)

  duration = run_record["duration_seconds"]
  mem_used_mb = run_record["memory_delta_mb"]
  mem_peak_mb = run_record["memory_peak_mb"]
  result = run_record["text"]

  print(f"  Done ({duration:.2f}s, mem: +{mem_used_mb}MB, peak: {mem_peak_mb}MB)")
  print(f"  Text: {result[:100]}..." if len(result) > 100 else f"  Text: {result}")


if __name__ == "__main__":
  main()
