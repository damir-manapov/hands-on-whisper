"""Optimization functions using Optuna."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from backends import (
  ALL_COMPUTE_TYPES,
  ALL_MODELS,
  CLOUD_BACKENDS,
  DEEPGRAM_MODELS,
  LOCAL_BACKENDS,
  OPENAI_API_MODELS,
)
from report import calculate_metrics
from runner import find_existing_run, generate_run_id, run_transcription, save_results

if TYPE_CHECKING:
  import optuna


def _load_results_json(audio_path: Path, suffix: str) -> tuple[Path, dict]:
  """Load or create results JSON file."""
  json_path = audio_path.parent / f"{audio_path.stem}{suffix}.json"
  if json_path.exists():
    data = json.loads(json_path.read_text(encoding="utf-8"))
  else:
    data = {"audio": str(audio_path), "runs": []}
  return json_path, data


def _prepare_search_space(
  args: argparse.Namespace,
) -> tuple[list[str], list[str], list[str]]:
  """Prepare search space for optimization."""
  backends = args.backends if args.backends else LOCAL_BACKENDS
  if args.runtime == "cuda" and "whispercpp" in backends:
    backends = [b for b in backends if b != "whispercpp"]
    print("Note: whispercpp excluded (no GPU support in pywhispercpp)")
  models = args.models if args.models else ALL_MODELS
  compute_types = args.compute_types if args.compute_types else ALL_COMPUTE_TYPES
  return backends, models, compute_types


def _format_trial_header(  # noqa: PLR0913
  trial_number: int,
  total_trials: int,
  backend: str,
  model: str,
  compute_type: str,
  beam_size: int,
  temperature: float,
  condition_on_prev: bool,
  batch_size: int,
) -> str:
  """Format trial header based on backend type."""
  if backend == "yandex":
    return f"[Trial {trial_number + 1}/{total_trials}] {backend}/{model}"
  elif backend == "openai-api":
    return f"[Trial {trial_number + 1}/{total_trials}] {backend}/{model} | temp={temperature:.2f}"
  else:
    batch_str = f" batch={batch_size}" if batch_size > 0 else ""
    return (
      f"[Trial {trial_number + 1}/{total_trials}] "
      f"{backend}/{model} | {compute_type} | beam={beam_size} temp={temperature:.2f} "
      f"cond={condition_on_prev}{batch_str}"
    )


def run_optimization_trial(  # noqa: PLR0913
  trial_number: int,
  total_trials: int,
  backend: str,
  model: str,
  compute_type: str,
  beam_size: int,
  temperature: float,
  condition_on_prev: bool,
  batch_size: int,
  audio: str,
  language: str | None,
  runtime: str,
  reference: str,
  data: dict,
  json_path: Path,
  metric: str = "wer",
  smart_format: bool = True,
  diarize: bool = False,
  user: str | None = None,
) -> float:
  """Run a single optimization trial and return WER or CER score."""
  trial_header = _format_trial_header(
    trial_number,
    total_trials,
    backend,
    model,
    compute_type,
    beam_size,
    temperature,
    condition_on_prev,
    batch_size,
  )

  run_id = generate_run_id(
    backend,
    model,
    language,
    runtime,
    beam_size,
    temperature,
    compute_type,
    condition_on_prev,
    batch_size,
    smart_format,
    diarize,
  )
  existing = find_existing_run(data, run_id)
  if existing:
    print(f"\n{trial_header} [cached]")
    wer_score, cer_score = calculate_metrics(reference, existing.get("text", ""))
    score = wer_score if metric == "wer" else cer_score
    print(f"  → {metric.upper()}: {score:.2f}%")
    return score / 100

  print(f"\n{trial_header}")

  run_record = run_transcription(
    audio,
    backend,
    model,
    language,
    runtime,
    beam_size,
    temperature,
    compute_type,
    condition_on_prev,
    batch_size,
    smart_format,
    diarize,
    user=user,
  )
  data["runs"].append(run_record)
  save_results(data, json_path)

  wer_score, cer_score = calculate_metrics(reference, run_record["text"])
  score = wer_score if metric == "wer" else cer_score
  mem_used_mb = run_record["memory_delta_mb"]
  duration = run_record["duration_seconds"]
  print(f"  → {metric.upper()}: {score:.2f}% ({duration:.1f}s, +{mem_used_mb:.0f}MB)")
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
  """Initialize Optuna study with previous runs."""
  import optuna as opt

  total_runs = len(data.get("runs", []))
  loaded_count = 0

  for run in data.get("runs", []):
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
    if run_backend == "faster-whisper":
      params["batch_size"] = run.get("batch_size", 0)

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
          **(
            {"batch_size": opt.distributions.IntDistribution(0, 32)}
            if run_backend == "faster-whisper"
            else {}
          ),
        },
        values=[score / 100],
      )
    )
    loaded_count += 1

  return loaded_count, total_runs


def _print_results(study: optuna.Study, metric: str) -> None:
  """Print optimization results summary."""
  print("\n" + "=" * 50)
  print("OPTIMIZATION RESULTS")
  print("=" * 50)
  print(f"Best {metric.upper()}: {study.best_value * 100:.1f}%")
  print("Best parameters:")
  for key, value in study.best_params.items():
    print(f"  {key}: {value}")


def optimize_cloud(
  args: argparse.Namespace,
  audio_path: Path,
  reference: str,
  metric: str,
) -> None:
  """Run cloud backend optimization."""
  import optuna

  backends = args.backends if args.backends else ["openai-api"]

  for backend in backends:
    # Determine models based on backend
    if backend == "openai-api":
      models = args.models if args.models else OPENAI_API_MODELS
    elif backend == "deepgram":
      models = args.models if args.models else DEEPGRAM_MODELS
    elif backend == "yandex":
      models = args.models if args.models else ["general"]
    else:
      models = args.models if args.models else []

    json_path, data = _load_results_json(audio_path, f"_{backend}")

    print(f"\nCloud optimization: backend={backend}, models={models}")
    print(f"Optimizing: {metric.upper()}")

    def _make_objective(
      backend_val: str,
      models_val: list[str],
      data_val: dict,
      json_path_val: Path,
    ) -> callable:
      """Create objective function with bound variables."""

      def obj_func(trial: optuna.Trial) -> float:
        model = trial.suggest_categorical("model", models_val) if models_val else None
        temperature = trial.suggest_float("temperature", 0.0, 0.5)

        # Deepgram-specific parameters
        if backend_val == "deepgram":
          smart_format = trial.suggest_categorical("smart_format", [True, False])
          diarize = trial.suggest_categorical("diarize", [True, False])
        else:
          smart_format = True
          diarize = False

        return run_optimization_trial(
          trial.number,
          args.n_trials,
          backend_val,
          model if model else "general",
          "auto",
          5,
          temperature,
          True,
          0,
          args.audio,
          args.language,
          "cloud",
          reference,
          data_val,
          json_path_val,
          metric,
          smart_format=smart_format,
          diarize=diarize,
          user=args.user,
        )

      return obj_func

    objective = _make_objective(backend, models, data, json_path)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

    for run in data.get("runs", []):
      if run.get("backend") != backend or run.get("model") not in models:
        continue
      text = run.get("text", "")
      wer_score, cer_score = calculate_metrics(reference, text)
      score = wer_score if metric == "wer" else cer_score
      study.add_trial(
        optuna.trial.create_trial(
          params={"model": run.get("model"), "temperature": run.get("temperature", 0.0)},
          distributions={
            "model": optuna.distributions.CategoricalDistribution(models),
            "temperature": optuna.distributions.FloatDistribution(0.0, 0.5),
          },
          values=[score],
        )
      )

    if len(study.trials) > 0:
      print(f"Loaded {len(study.trials)} previous trials")

    study.optimize(objective, n_trials=args.n_trials)
    _print_results(study, metric)


def optimize_local(
  args: argparse.Namespace,
  audio_path: Path,
  reference: str,
  metric: str,
) -> None:
  """Run local backend optimization."""
  import optuna

  suffix = "_gpu" if args.runtime == "cuda" else "_cpu"
  json_path, data = _load_results_json(audio_path, suffix)

  backends, models, compute_types = _prepare_search_space(args)
  print(f"Search space: backends={backends}, models={models}, compute_types={compute_types}")
  print(f"Optimizing: {metric.upper()}")

  def objective(trial: optuna.Trial) -> float:
    backend = trial.suggest_categorical("backend", backends)
    model = trial.suggest_categorical("model", models)
    compute_type = trial.suggest_categorical("compute_type", compute_types)
    if args.beam_sizes:
      beam_size = trial.suggest_categorical("beam_size", args.beam_sizes)
    else:
      beam_size = trial.suggest_int("beam_size", 1, 10)
    temperature = trial.suggest_float("temperature", 0.0, 0.5)

    if backend == "whispercpp":
      condition_on_prev = False
    else:
      condition_on_prev = trial.suggest_categorical("condition_on_prev", [True, False])

    if backend == "faster-whisper":
      if args.batch_sizes:
        batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)
      else:
        batch_size = trial.suggest_int("batch_size", 0, 32)
    else:
      batch_size = 0

    if backend == "faster-whisper" and compute_type == "float16" and args.runtime == "cpu":
      raise optuna.TrialPruned("faster-whisper float16 not supported on CPU")

    return run_optimization_trial(
      trial.number,
      args.n_trials,
      backend,
      model,
      compute_type,
      beam_size,
      temperature,
      condition_on_prev,
      batch_size,
      args.audio,
      args.language,
      args.runtime,
      reference,
      data,
      json_path,
      metric,
      user=args.user,
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
  _print_results(study, metric)


def cmd_optimize(args: argparse.Namespace) -> None:
  """Find optimal parameters using Optuna."""
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

  # Validate backend selection
  backends = args.backends if args.backends else LOCAL_BACKENDS
  cloud_backends = [b for b in backends if b in CLOUD_BACKENDS]
  local_backends = [b for b in backends if b not in CLOUD_BACKENDS]

  if cloud_backends and local_backends:
    msg = f"Cannot mix cloud ({cloud_backends}) and local ({local_backends}) backends"
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)

  if cloud_backends:
    optimize_cloud(args, audio_path, reference, args.metric)
  else:
    optimize_local(args, audio_path, reference, args.metric)
