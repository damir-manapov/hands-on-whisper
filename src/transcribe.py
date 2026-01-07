"""Unified transcription script supporting multiple backends."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  import optuna


# Default search space for optimization
ALL_BACKENDS = ["faster-whisper", "openai", "whispercpp"]
ALL_MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
ALL_COMPUTE_TYPES = ["int8", "float16", "float32"]


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


def load_reference(data: dict[str, Any]) -> str | None:
  """Load reference text file based on audio path in JSON data."""
  audio_path = Path(data.get("audio", ""))
  ref_path = audio_path.with_suffix(".txt")
  if ref_path.exists():
    return ref_path.read_text(encoding="utf-8").strip()
  return None


def find_existing_run(data: dict[str, Any], run_id: str) -> dict[str, Any] | None:
  """Find an existing run by ID in the data."""
  return next((r for r in data.get("runs", []) if r.get("id") == run_id), None)


def generate_run_id(  # noqa: PLR0913
  backend: str,
  model: str,
  language: str | None,
  device: str,
  beam_size: int,
  temperature: float,
  compute_type: str,
  condition_on_prev: bool = True,
) -> str:
  """Generate a unique ID based on settings."""
  cond = "cond" if condition_on_prev else "nocond"
  settings = (
    f"{backend}:{model}:{language}:{device}:beam{beam_size}:temp{temperature}:{compute_type}:{cond}"
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

  # Calculate WER for sorting if reference provided
  def get_wer(run: dict) -> float:
    if reference:
      wer_score, _ = calculate_metrics(reference, run.get("text", ""))
      return wer_score
    return run.get("duration_seconds", 0)

  # Sort by WER (best first) if reference provided, otherwise by duration
  sorted_runs = sorted(runs, key=get_wer)

  lines.append(f"**Total runs:** {len(runs)}")
  if reference:
    lines.append(f"**Reference:** {len(reference)} chars, {len(reference.split())} words")
  lines.append("")

  # Summary table
  lines.append("## Performance Summary")
  lines.append("")
  hdr = "| # | Backend | Model | Compute | Beam | Temp | Cond | Lang | Dur(s) | MemΔ | Peak |"
  sep = "|---|---------|-------|---------|------|------|------|------|--------|------|------|"
  if reference:
    lines.append(f"{hdr} WER% | CER% |")
    lines.append(f"{sep}------|------|")
  else:
    lines.append(hdr)
    lines.append(sep)

  for i, run in enumerate(sorted_runs, 1):
    backend = run.get("backend", "?")
    model = run.get("model", "?")
    compute = run.get("compute_type") or "-"
    beam = run.get("beam_size", 5)
    temp = run.get("temperature", 0.0)
    cond_prev = "Y" if run.get("condition_on_prev", True) else "N"
    lang = run.get("language") or "auto"
    duration = run.get("duration_seconds", 0)
    mem_delta = run.get("memory_delta_mb", 0)
    mem_peak = run.get("memory_peak_mb", 0)
    row = (
      f"| {i} | {backend} | {model} | {compute} | {beam} | {temp:.2f} "
      f"| {cond_prev} | {lang} | {duration:.1f} | {mem_delta} | {mem_peak} |"
    )
    if reference:
      wer_score, cer_score = calculate_metrics(reference, run.get("text", ""))
      lines.append(f"{row} {wer_score:.1f} | {cer_score:.1f} |")
    else:
      lines.append(row)

  lines.append("")

  # Detailed results
  _append_detailed_results(lines, sorted_runs, reference)

  return "\n".join(lines)


def _append_detailed_results(
  lines: list[str], sorted_runs: list[dict], reference: str | None
) -> None:
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
    lines.append(f"- **Temperature:** {temperature:.2f}")
    lines.append(f"- **Compute type:** {compute_type}")
    lines.append(f"- **Condition on prev:** {condition_on_prev}")
    if reference:
      wer_score, cer_score = calculate_metrics(reference, text)
      lines.append(f"- **WER:** {wer_score:.1f}%")
      lines.append(f"- **CER:** {cer_score:.1f}%")
    lines.append(f"- **Timestamp:** {timestamp}")
    lines.append("")
    lines.append("**Text:**")
    lines.append("")
    lines.append(f"> {text}")
    lines.append("")


def save_results(data: dict[str, Any], json_path: Path) -> None:
  """Save JSON data and regenerate MD report."""
  json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

  reference = load_reference(data)
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
  # Auto-detect suffix from device: cuda->_gpu, cpu->_cpu
  suffix = "_gpu" if "cuda" in args.device else "_cpu"
  json_path = audio_path.parent / f"{audio_path.stem}{suffix}.json"

  if json_path.exists():
    data = json.loads(json_path.read_text(encoding="utf-8"))
  else:
    data = {"audio": str(audio_path), "runs": []}

  condition_on_prev = not args.no_condition_on_prev

  for backend, model, language, device in combinations:
    # For whispercpp with custom model paths
    if backend == "whispercpp" and args.model_path:
      for mp in args.model_path:
        if not Path(mp).exists():
          print(f"Error: Model file not found: {mp}", file=sys.stderr)
          sys.exit(1)
        # Extract model name from path: models/ggml-large-v3-turbo.bin -> large-v3-turbo
        model_name = Path(mp).stem.replace("ggml-", "").replace("-q8_0", "").replace("-q5_0", "")
        run_single(
          args.audio,
          backend,
          model_name,
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
      # Check existence for whispercpp
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

  reference = load_reference(data)
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

  run_id = generate_run_id(
    backend, model, language, device, beam_size, temperature, compute_type, condition_on_prev
  )

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
    # Model might already be a path (from cmd_transcribe) or a name (from optimization)
    if model.endswith(".bin"):
      model_path = model
    else:
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
  total_trials: int,
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
  metric: str = "wer",
) -> float:
  """Run a single optimization trial and return WER or CER score."""
  # Format trial header
  trial_header = (
    f"[Trial {trial_number + 1}/{total_trials}] "
    f"{backend}/{model} | {compute_type} | beam={beam_size} temp={temperature:.2f} "
    f"cond={condition_on_prev}"
  )

  # Check if already exists (use cached result)
  run_id = generate_run_id(
    backend, model, language, device, beam_size, temperature, compute_type, condition_on_prev
  )
  existing = find_existing_run(data, run_id)
  if existing:
    print(f"\n{trial_header} [cached]")
    wer_score, cer_score = calculate_metrics(reference, existing.get("text", ""))
    score = wer_score if metric == "wer" else cer_score
    print(f"  → {metric.upper()}: {score:.1f}%")
    return score / 100

  print(f"\n{trial_header}")

  run_record = _run_transcription(
    audio, backend, model, language, device, beam_size, temperature, compute_type, condition_on_prev
  )
  data["runs"].append(run_record)
  save_results(data, json_path)

  wer_score, cer_score = calculate_metrics(reference, run_record["text"])
  score = wer_score if metric == "wer" else cer_score
  mem_used_mb = run_record["memory_delta_mb"]
  duration = run_record["duration_seconds"]
  print(f"  → {metric.upper()}: {score:.1f}% ({duration:.1f}s, +{mem_used_mb:.0f}MB)")
  return score / 100


def _init_study_with_history(  # noqa: PLR0913
  study: optuna.Study,
  data: dict,
  reference: str,
  backends: list[str],
  models: list[str],
  compute_types: list[str],
  metric: str = "wer",
) -> tuple[int, int]:
  """Initialize Optuna study with previous runs so it can learn from them.

  Returns:
    Tuple of (loaded_count, total_count) for reporting.
  """
  import optuna as opt

  total_runs = len(data.get("runs", []))
  loaded_count = 0

  for run in data.get("runs", []):
    # Only add runs that are within current search space
    run_backend = run.get("backend")
    run_model = run.get("model")
    run_compute = run.get("compute_type")
    if run_backend not in backends or run_model not in models:
      continue
    if run_compute and run_compute not in compute_types:
      continue

    wer_score, cer_score = calculate_metrics(reference, run.get("text", ""))
    score = wer_score if metric == "wer" else cer_score
    params = {
      "backend": run_backend,
      "model": run_model,
      "compute_type": run_compute or "int8",
      "beam_size": run.get("beam_size", 5),
      "temperature": run.get("temperature", 0.0),
    }
    if run_backend != "whispercpp":
      params["condition_on_prev"] = run.get("condition_on_prev", True)

    study.add_trial(
      opt.trial.create_trial(
        params=params,
        distributions={
          "backend": opt.distributions.CategoricalDistribution(backends),
          "model": opt.distributions.CategoricalDistribution(models),
          "compute_type": opt.distributions.CategoricalDistribution(compute_types),
          "beam_size": opt.distributions.IntDistribution(1, 10),
          "temperature": opt.distributions.FloatDistribution(0.0, 0.5),
          **(
            {"condition_on_prev": opt.distributions.CategoricalDistribution([True, False])}
            if run_backend != "whispercpp"
            else {}
          ),
        },
        values=[score / 100],
      )
    )
    loaded_count += 1

  return loaded_count, total_runs


def _print_optimization_results(study: optuna.Study, metric: str) -> None:
  """Print optimization results summary."""
  print("\n" + "=" * 50)
  print("OPTIMIZATION RESULTS")
  print("=" * 50)
  print(f"Best {metric.upper()}: {study.best_value * 100:.1f}%")
  print("Best parameters:")
  for key, value in study.best_params.items():
    print(f"  {key}: {value}")


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

  # Load or create JSON data - auto-detect suffix from device
  suffix = "_gpu" if args.device == "cuda" else "_cpu"
  json_path = audio_path.parent / f"{audio_path.stem}{suffix}.json"
  if json_path.exists():
    data = json.loads(json_path.read_text(encoding="utf-8"))
  else:
    data = {"audio": str(audio_path), "runs": []}

  # Parse search space from args (defaults to all options)
  backends = args.backends if args.backends else ALL_BACKENDS
  # whispercpp doesn't support GPU (pywhispercpp is CPU-only)
  if args.device == "cuda" and "whispercpp" in backends:
    backends = [b for b in backends if b != "whispercpp"]
    print("Note: whispercpp excluded (no GPU support in pywhispercpp)")
  models = args.models if args.models else ALL_MODELS
  compute_types = args.compute_types if args.compute_types else ALL_COMPUTE_TYPES

  metric = args.metric
  print(f"Search space: backends={backends}, models={models}, compute_types={compute_types}")
  print(f"Optimizing: {metric.upper()}")

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

    # Skip invalid combinations (faster-whisper doesn't support float16 on CPU)
    if backend == "faster-whisper" and compute_type == "float16" and args.device == "cpu":
      raise optuna.TrialPruned("faster-whisper float16 not supported on CPU")

    return _run_optimization_trial(
      trial.number,
      args.n_trials,
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
      metric,
    )

  study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())
  loaded, total = _init_study_with_history(
    study, data, reference, backends, models, compute_types, metric
  )

  if loaded > 0:
    filtered = total - loaded
    if filtered > 0:
      print(f"Loaded {loaded} previous trials ({filtered} filtered by search space)")
    else:
      print(f"Loaded {loaded} previous trials")

  study.optimize(objective, n_trials=args.n_trials)
  _print_optimization_results(study, metric)


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
  optim_parser.add_argument(
    "--metric",
    default="wer",
    choices=["wer", "cer"],
    help="Metric to optimize: wer (Word Error Rate) or cer (Character Error Rate) (default: wer)",
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
  run_id = generate_run_id(
    backend, model, language, device, beam_size, temperature, compute_type, condition_on_prev
  )

  # Skip if we already have this run
  if find_existing_run(data, run_id):
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
