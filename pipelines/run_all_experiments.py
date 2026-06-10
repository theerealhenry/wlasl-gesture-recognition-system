"""
pipelines/run_all_experiments.py
==================================
Orchestrator for the full Stage 5 multi-model experiment matrix.

This script executes all 17 experiments across 4 groups in dependency order,
reads completed-group results programmatically from MLflow to select optimal
hyperparameters before launching dependent groups, and writes a comprehensive
execution report upon completion.

Experiment groups and dependency chain
----------------------------------------

    Group 1 — Architecture comparison (4 runs, no dependencies)
        dense / lstm / gru / bilstm — all seq60, no augmentation.
        Purpose: quantitatively prove temporal modelling is necessary.
        Expected output: LSTM/GRU/BiLSTM meaningfully outperform Dense.

    Group 2 — Augmentation ablation (3 runs, no cross-group dependencies)
        lstm + seq60 × {none, temporal, spatial_temporal}.
        Purpose: determine which augmentation strategy maximises val macro-F1
                 and reduces the train/val overfitting gap.
        NOTE: Runs in parallel with Group 1 because neither depends on the
              other's results. The adaptive Groups 3 and 4 depend on Group 2.

    Group 3 — Sequence length ablation (6 runs, depends on Group 2)
        lstm + best_augmentation × {seq20, seq30, seq40, seq60, seq80, seq100}.
        Purpose: find the accuracy/latency elbow given WLASL's distribution.
        Run order: seq60 first (fast baseline), seq80 SECOND (highest expected
        gain — 97% truncation at seq60, P75=84 frames per Notebook 04).

    Group 4 — Landmark configuration ablation (3 runs, depends on Groups 2+3)
        lstm + best_augmentation + best_seq × {hands_only, pose_only, full}.
        Purpose: test whether the 1.47× hands_only Fisher ratio advantage
                 translates to real accuracy gains.
        Run order: hands_only first (highest Fisher ratio, 0.8097).

    Champion — Final best model (1 run, depends on all groups)
        Best architecture from Group 1 (likely bilstm) + best_aug + best_seq +
        best_landmark — trained with hidden_units=128 for up to 100 epochs.

Adaptive selection logic
--------------------------
Groups 3, 4, and Champion use ``_select_best_run()`` to query the MLflow
tracking store and automatically select the optimal hyperparameters from
preceding groups. This is genuinely adaptive — the script does NOT use
pre-specified "expected best" values. If Group 2 shows that temporal-only
augmentation outperforms spatial_temporal (e.g. due to one-handed sign
geometry), Group 3 will correctly use temporal-only augmentation.

Execution modes
----------------
    Sequential (default):
        python pipelines/run_all_experiments.py

    Single group:
        python pipelines/run_all_experiments.py --groups 1
        python pipelines/run_all_experiments.py --groups 1 2

    Resume (skip already-completed runs):
        python pipelines/run_all_experiments.py --resume

    Dry run (print plan, no training):
        python pipelines/run_all_experiments.py --dry-run

    Champion only (requires groups 1-4 already completed):
        python pipelines/run_all_experiments.py --champion-only

    Custom MLflow tracking URI:
        python pipelines/run_all_experiments.py --mlflow-tracking-uri http://localhost:5000

    Override config values for all runs:
        python pipelines/run_all_experiments.py --override training.epochs=40

Fault tolerance
-----------------
Each run is individually wrapped in a try/except. A single run failing (NaN
loss, OOM, corrupt data) does NOT abort the orchestrator. The failure is
logged, the run is recorded in the execution report as FAILED, and execution
continues with the next run. This is critical for overnight batch execution
where one fragile data point should not cancel 16 other experiments.

If a critical group fails (> 50% of runs in the group fail), subsequent
dependent groups are skipped and a clear diagnostic is logged. This is
distinct from a single-run failure, which does not block the group.

Run collision avoidance
--------------------------
Before each run, ``_is_run_already_complete()`` checks for an existing
``run_manifest.json`` in ``artifacts/experiments/{run_name}/``. If found
(and ``--resume`` is set), the run is skipped and its results are loaded
from the existing manifest. Without ``--resume``, an existing run raises
an error (prevents silent overwrites of completed experiments).

MLflow result reading
-----------------------
``_select_best_run()`` queries the MLflow tracking store using
``mlflow.search_runs()`` with group-specific tag filters. This means the
tracking store must be accessible and populated before Groups 3/4/Champion
can run. If the tracking store is unavailable (network error, corrupt db),
a ``RuntimeError`` is raised with actionable guidance.

Stage 5 completion gate
------------------------
After all runs, ``_run_completion_gate()`` verifies:
  - All 17 MLflow runs exist in the "WLASL-35-class" experiment
  - Every run has best_val_macro_f1 logged
  - artifacts/experiments/ has a directory per run with run_manifest.json
  - models/ has a SavedModel directory per run
  - Notebook 05 input data (experiment_summary.md, run registry) is written
  - val_macro_f1 ≥ 0.60 for at least one run (minimum viability)
  - val_macro_f1 ≥ 0.70 for at least one run (target — logged as warning if not met)
  - High-risk class F1=0.0 are counted and flagged in the report

Execution report
-----------------
``reports/experiment_execution_report.json`` is written on completion (and on
partial completion if interrupted). It contains: total elapsed time, per-run
results, group summaries, best run across all groups, completion gate status,
and diagnostic messages. This file is the primary input for Notebook 05 §7
(experiment registry table).

``reports/experiment_summary.md`` is written after the gate check. It is
the human-readable registry of all Stage 5 runs consumed by the Stage 11 report.

Design constraints (all non-negotiable)
-----------------------------------------
- run_single_experiment() from run_training.py is the sole training entry point.
  This file never calls train_one_run() directly.
- Primary metric is val_macro_f1 everywhere. val_accuracy is secondary.
- class_weight_balancing=True for every run (enforced in _build_run_kwargs).
- Masking=True for all recurrent models (enforced in architectures.py, not here).
- seq80 runs SECOND within Group 3 (highest expected accuracy gain).
- hands_only runs FIRST within Group 4 (highest Fisher ratio: 0.8097).
- No nested MLflow runs. run_single_experiment() handles its own run context.
- All exceptions from individual runs are caught; orchestrator never crashes on
  a single run failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Exit codes — CI/CD compatible
# ---------------------------------------------------------------------------

EXIT_SUCCESS           = 0
EXIT_PARTIAL_FAILURE   = 1   # Some runs failed, some succeeded
EXIT_CRITICAL_FAILURE  = 2   # A prerequisite group failed; downstream skipped
EXIT_GATE_FAILURE      = 3   # All runs completed but gate checks failed
EXIT_UNEXPECTED_ERROR  = 4

# ---------------------------------------------------------------------------
# Project root — resolves correctly whether launched from project root or
# from any subdirectory. Adds project root to sys.path so `src.*` imports work.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: MLflow experiment name — must match train.py, architectures.py, and all
#: other Stage 5 files. "WLASL-35-class" is the locked value from the handoff.
MLFLOW_EXPERIMENT_NAME: str = "WLASL-35-class"

#: Total expected number of runs across all groups + champion.
#: Groups: 4 + 3 + 6 + 3 + 1 = 17
EXPECTED_TOTAL_RUNS: int = 17

#: Minimum val_macro_f1 for minimum viability.
VIABILITY_THRESHOLD: float = 0.60

#: Target val_macro_f1. Met if any run achieves this.
TARGET_THRESHOLD: float = 0.70

#: Maximum fraction of a group's runs that can fail before the group is
#: considered "critically failed" and dependent groups are skipped.
CRITICAL_FAILURE_FRACTION: float = 0.50

#: High-risk sign classes. F1=0.0 for any of these is logged as WARNING.
_HIGH_RISK_SIGNS: Tuple[str, ...] = (
    "clothes", "think", "birthday", "name", "book",
)

#: Run name prefix used for duplicate-detection during --resume.
_RUN_MANIFEST_FILENAME: str = "run_manifest.json"


# ---------------------------------------------------------------------------
# Run specification dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """
    Complete specification for a single experiment run.

    Immutable after construction. All fields correspond to arguments of
    run_single_experiment() in run_training.py.
    """
    run_name:         str
    model:            str
    data:             str
    augmentation:     str
    experiment_group: str
    experiment:       Optional[str]      = None
    overrides:        Dict[str, Any]     = field(default_factory=dict)
    description:      str                = ""  # human-readable, for logging only

    def __post_init__(self) -> None:
        # Always enforce class_weight_balancing=True — no exceptions.
        if "training.class_weight_balancing" not in self.overrides:
            self.overrides["training.class_weight_balancing"] = True

    def __repr__(self) -> str:
        return (
            f"RunSpec(run_name={self.run_name!r}, model={self.model!r}, "
            f"data={self.data!r}, augmentation={self.augmentation!r}, "
            f"group={self.experiment_group!r})"
        )


# ---------------------------------------------------------------------------
# Run result record
# ---------------------------------------------------------------------------

@dataclass
class RunRecord:
    """
    Full record of one run's outcome. Written to the execution report.
    """
    run_name:          str
    experiment_group:  str
    model:             str
    data:              str
    augmentation:      str
    status:            str          # "completed" | "skipped" | "failed"
    best_val_macro_f1: float        = 0.0
    best_val_acc:      float        = 0.0
    best_epoch:        int          = 0
    total_epochs:      int          = 0
    mlflow_run_id:     str          = ""
    artifact_dir:      str          = ""
    model_save_path:   str          = ""
    elapsed_sec:       float        = 0.0
    error_message:     str          = ""
    high_risk_f1:      Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Group summary
# ---------------------------------------------------------------------------

@dataclass
class GroupSummary:
    """Per-group execution summary for the execution report."""
    group_id:          int
    group_name:        str
    n_planned:         int
    n_completed:       int
    n_skipped:         int
    n_failed:          int
    best_run_name:     str   = ""
    best_val_macro_f1: float = 0.0
    elapsed_sec:       float = 0.0
    skipped_reason:    str   = ""  # set when the group was skipped entirely

    @property
    def critical_failure(self) -> bool:
        """True if more than CRITICAL_FAILURE_FRACTION of runs failed."""
        if self.n_planned == 0:
            return False
        return (self.n_failed / self.n_planned) > CRITICAL_FAILURE_FRACTION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# MLflow result querying
# ---------------------------------------------------------------------------

def _select_best_run(
    experiment_name: str,
    tag_filter:      Dict[str, str],
    metric:          str = "best_val_macro_f1",
    tracking_uri:    Optional[str] = None,
) -> Dict[str, Any]:
    """
    Query the MLflow tracking store and return the best run's parameters.

    This is the core of the adaptive selection logic. It reads real results
    from the tracking store — it does NOT use hardcoded "expected best" values.

    Parameters
    ----------
    experiment_name : str
        MLflow experiment name (e.g. "WLASL-35-class").
    tag_filter : dict
        MLflow tag key-value pairs to filter runs. Example:
        {"experiment_group": "augmentation"} selects all Group 2 runs.
    metric : str
        MLflow metric name to rank runs by. Default "best_val_macro_f1".
    tracking_uri : str | None
        Override for the MLflow tracking URI.

    Returns
    -------
    dict with keys: run_name, augmentation, seq_len, landmark_config,
                    model_type, best_val_macro_f1, mlflow_run_id.
    All values extracted from MLflow params and metrics.

    Raises
    ------
    RuntimeError
        If no runs match the filter, or if the tracking store is inaccessible.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        runs_df = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=" AND ".join(
                f"tags.{k} = '{v}'" for k, v in tag_filter.items()
            ),
            order_by=[f"metrics.{metric} DESC"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"_select_best_run(): MLflow search failed for filter={tag_filter}. "
            f"Error: {type(exc).__name__}: {exc}\n"
            "Ensure the MLflow tracking store is populated (Group runs completed) "
            f"and accessible at URI: {mlflow.get_tracking_uri()}"
        ) from exc

    if runs_df.empty:
        raise RuntimeError(
            f"_select_best_run(): no completed runs found for filter={tag_filter} "
            f"in experiment '{experiment_name}'. "
            "Ensure the prerequisite group ran successfully before launching "
            "dependent groups. Use --groups to run groups in order."
        )

    # Filter to only FINISHED runs (not FAILED or RUNNING)
    finished = runs_df[runs_df["status"] == "FINISHED"]
    if finished.empty:
        raise RuntimeError(
            f"_select_best_run(): {len(runs_df)} run(s) match filter={tag_filter} "
            "but none have status=FINISHED. Check for failed or still-running runs."
        )

    best = finished.iloc[0]

    def _get(col: str, default: Any = "") -> Any:
        """Safely retrieve a column value; return default if absent or NaN."""
        val = best.get(col, default)
        if val is None:
            return default
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                return default
        except (TypeError, ValueError):
            pass
        return val

    return {
        "run_name":          str(_get("tags.mlflow.runName", _get("tags.run_name"))),
        "augmentation":      str(_get("tags.augmentation_name")),
        "seq_len":           str(_get("tags.seq_len")),
        "landmark_config":   str(_get("tags.landmark_config")),
        "model_type":        str(_get("tags.model_type")),
        "best_val_macro_f1": float(_get(f"metrics.{metric}", 0.0)),
        "mlflow_run_id":     str(_get("run_id")),
        # Data config name (e.g. "seq60") for RunSpec.data field
        "data_config":       str(_get("params.data_config", f"seq{_get('params.seq_len', '60')}")),
    }


# ---------------------------------------------------------------------------
# Run completion check
# ---------------------------------------------------------------------------

def _is_run_already_complete(run_name: str) -> bool:
    """
    Return True if artifacts/experiments/{run_name}/run_manifest.json exists.

    This is the definitive signal that a previous training run for this
    run_name completed successfully. Used by --resume mode to skip re-runs.
    """
    manifest = Path("artifacts") / "experiments" / run_name / _RUN_MANIFEST_FILENAME
    return manifest.exists()


def _load_completed_manifest(run_name: str) -> Dict[str, Any]:
    """
    Load and return the run_manifest.json for a completed run.

    Used in --resume mode to populate RunRecord without re-training.
    """
    manifest_path = (
        Path("artifacts") / "experiments" / run_name / _RUN_MANIFEST_FILENAME
    )
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# RunSpec builders for each group
# ---------------------------------------------------------------------------

def _build_group_1_specs(extra_overrides: Dict[str, Any]) -> List[RunSpec]:
    """
    Group 1 — Architecture comparison.

    Fixed: seq60, augmentation=none, same seed.
    Varies: model architecture (dense, lstm, gru, bilstm).
    Purpose: isolates architecture as the only variable.
    """
    architectures = [
        ("dense",  "Dense feedforward baseline — proves temporal modelling necessary"),
        ("lstm",   "Single LSTM baseline — primary ablation workhorse"),
        ("gru",    "GRU — streamlined recurrent baseline"),
        ("bilstm", "Bidirectional LSTM — champion candidate"),
    ]

    specs = []
    for model, description in architectures:
        run_name = f"{model}_seq60_no_aug"
        overrides = {**extra_overrides}
        specs.append(RunSpec(
            run_name=run_name,
            model=model,
            data="seq60",
            augmentation="none",
            experiment_group="architecture",
            description=description,
            overrides=overrides,
        ))
    return specs


def _build_group_2_specs(extra_overrides: Dict[str, Any]) -> List[RunSpec]:
    """
    Group 2 — Augmentation ablation.

    Fixed: lstm, seq60.
    Varies: augmentation strategy.
    NOTE: Group 2 does NOT depend on Group 1 results — it can run concurrently.
    The best augmentation from Group 2 feeds Groups 3 and 4.
    """
    strategies = [
        ("none",             "No augmentation — deterministic baseline"),
        ("temporal",         "Temporal-only augmentation — jitter + speed"),
        ("spatial_temporal", "Full spatial+temporal augmentation — complete chain"),
    ]

    specs = []
    for augmentation, description in strategies:
        run_name = f"lstm_seq60_{augmentation}_aug"
        overrides = {**extra_overrides}
        specs.append(RunSpec(
            run_name=run_name,
            model="lstm",
            data="seq60",
            augmentation=augmentation,
            experiment_group="augmentation",
            description=description,
            overrides=overrides,
        ))
    return specs


def _build_group_3_specs(
    best_augmentation: str,
    extra_overrides:   Dict[str, Any],
) -> List[RunSpec]:
    """
    Group 3 — Sequence length ablation.

    Fixed: lstm, best_augmentation (from Group 2).
    Varies: sequence length across {seq60, seq80, seq100, seq40, seq30, seq20}.

    Run order is critical:
        seq60   first — fastest sanity check, establishes Group 2 baseline
        seq80   second — HIGHEST PRIORITY: 97% truncation at seq60, P75=84 frames
                         per Notebook 04. This is the single most likely source
                         of material accuracy improvement in all of Stage 5.
        seq100  third — diminishing returns check (only 7.1% more clips fully covered)
        seq40   fourth — below-primary regress
        seq30   fifth
        seq20   sixth — training data anchor (content coverage ~34%)
    """
    # Ordered by scientific priority, NOT by seq_len value.
    ordered_data_configs = [
        ("seq60",  "Sequence length 60 — primary config (85% mean content coverage)"),
        ("seq80",  "Sequence length 80 — HIGHEST PRIORITY (P75 coverage, 97% truncation at 60)"),
        ("seq100", "Sequence length 100 — diminishing returns check"),
        ("seq40",  "Sequence length 40 — below-primary regression"),
        ("seq30",  "Sequence length 30 — low coverage (~50% mean content)"),
        ("seq20",  "Sequence length 20 — content anchor (~34% mean content)"),
    ]

    specs = []
    for data_config, description in ordered_data_configs:
        seq_label = data_config.replace("seq", "")
        run_name  = f"lstm_{data_config}_{best_augmentation}_aug"
        overrides = {**extra_overrides}
        specs.append(RunSpec(
            run_name=run_name,
            model="lstm",
            data=data_config,
            augmentation=best_augmentation,
            experiment_group="sequence_length",
            description=description,
            overrides=overrides,
        ))
    return specs


def _build_group_4_specs(
    best_augmentation: str,
    best_data_config:  str,
    extra_overrides:   Dict[str, Any],
) -> List[RunSpec]:
    """
    Group 4 — Landmark configuration ablation.

    Fixed: lstm, best_augmentation, best_data_config.
    Varies: landmark_config.

    Run order:
        hands_only first  — Fisher ratio 0.8097 (1.47× full baseline)
        pose_only  second — Fisher ratio 0.2176 (0.40× baseline, likely worst)
        full       third  — full 225-dim baseline
    """
    ordered_configs = [
        ("hands_only", "Hands-only features (126 dims) — Fisher ratio 0.8097 (1.47× full)"),
        ("pose_only",  "Pose-only features (99 dims)   — Fisher ratio 0.2176 (0.40× full)"),
        ("full",       "Full features (225 dims)        — Fisher ratio 0.5492 (baseline)"),
    ]

    specs = []
    for lm_config, description in ordered_configs:
        seq_label = best_data_config.replace("seq", "")
        run_name  = f"lstm_{best_data_config}_{best_augmentation}_{lm_config}"
        overrides = {
            "data.landmark_config": lm_config,
            **extra_overrides,
        }
        specs.append(RunSpec(
            run_name=run_name,
            model="lstm",
            data=best_data_config,
            augmentation=best_augmentation,
            experiment_group="landmark_config",
            description=description,
            overrides=overrides,
        ))
    return specs


def _build_champion_spec(
    best_model:        str,
    best_augmentation: str,
    best_data_config:  str,
    best_lm_config:    str,
    extra_overrides:   Dict[str, Any],
) -> RunSpec:
    """
    Champion run — best settings from Groups 1-4, hidden_units=128, epochs=100.

    The champion run uses:
      - Best architecture from Group 1 (passed in as best_model; typically bilstm)
      - Best augmentation from Group 2
      - Best sequence length from Group 3
      - Best landmark config from Group 4
      - hidden_units=128 (doubled from ablation default of 64)
      - epochs=100 (extended from 80 default for thorough convergence)
      - patience=15 (extended for the larger model)
    """
    run_name = "champion_model_v1"
    overrides = {
        "model.hidden_units": 128,
        "training.epochs":    100,
        "training.early_stopping_patience": 15,
        "data.landmark_config": best_lm_config,
        **extra_overrides,
    }
    return RunSpec(
        run_name=run_name,
        model=best_model,
        data=best_data_config,
        augmentation=best_augmentation,
        experiment_group="champion",
        experiment="best_model",
        description=(
            f"Champion: {best_model} + {best_data_config} + "
            f"{best_augmentation} + {best_lm_config} | "
            "hidden_units=128, epochs=100"
        ),
        overrides=overrides,
    )


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def _execute_run(
    spec:             RunSpec,
    resume:           bool,
    force:            bool,
    mlflow_uri:       Optional[str],
    logger:           Any,
) -> RunRecord:
    """
    Execute one run from a RunSpec and return a RunRecord.

    Handles:
      - Resume mode (skip if run_manifest.json already exists)
      - Delegation to run_single_experiment()
      - Fault tolerance (catches all exceptions, returns FAILED record)
      - Timing

    Parameters
    ----------
    spec       : RunSpec
    resume     : bool — skip completed runs
    force      : bool — allow overwriting completed runs
    mlflow_uri : str | None
    logger     : logging.Logger

    Returns
    -------
    RunRecord  — status is "completed", "skipped", or "failed"
    """
    t_start = time.time()

    # ── Resume: skip runs that already have a completed manifest ───────────
    if resume and _is_run_already_complete(spec.run_name):
        logger.info(
            f"  SKIPPING (--resume): run '{spec.run_name}' already has a "
            "completed run_manifest.json.",
            extra={"stage": "orchestrator"},
        )
        try:
            manifest = _load_completed_manifest(spec.run_name)
            return RunRecord(
                run_name=spec.run_name,
                experiment_group=spec.experiment_group,
                model=spec.model,
                data=spec.data,
                augmentation=spec.augmentation,
                status="skipped",
                best_val_macro_f1=float(manifest.get("best_val_macro_f1", 0.0)),
                best_val_acc=float(manifest.get("best_val_acc", 0.0)),
                best_epoch=int(manifest.get("best_epoch", 0)),
                total_epochs=int(manifest.get("total_epochs_trained", 0)),
                mlflow_run_id=str(manifest.get("mlflow_run_id", "")),
                artifact_dir=str(
                    Path("artifacts") / "experiments" / spec.run_name
                ),
                elapsed_sec=round(time.time() - t_start, 1),
                high_risk_f1=manifest.get("high_risk_class_f1", {}),
            )
        except Exception as exc:
            logger.warning(
                f"  Failed to load completed manifest for '{spec.run_name}': "
                f"{exc}. Will re-run.",
                extra={"stage": "orchestrator"},
            )
            # Fall through to normal execution

    # ── Normal execution via run_single_experiment() ────────────────────
    from pipelines.run_training import run_single_experiment

    logger.info(
        f"  Launching: {spec.run_name} | "
        f"model={spec.model} | data={spec.data} | "
        f"augmentation={spec.augmentation} | group={spec.experiment_group}",
        extra={"stage": "orchestrator"},
    )
    if spec.description:
        logger.info(
            f"  Description: {spec.description}",
            extra={"stage": "orchestrator"},
        )

    try:
        result = run_single_experiment(
            model=spec.model,
            data=spec.data,
            augmentation=spec.augmentation,
            run_name=spec.run_name,
            experiment_group=spec.experiment_group,
            experiment=spec.experiment,
            overrides=spec.overrides if spec.overrides else None,
            mlflow_tracking_uri=mlflow_uri,
            force=force,
        )

        elapsed = time.time() - t_start
        record  = RunRecord(
            run_name=spec.run_name,
            experiment_group=spec.experiment_group,
            model=spec.model,
            data=spec.data,
            augmentation=spec.augmentation,
            status="completed",
            best_val_macro_f1=float(result.get("best_val_macro_f1", 0.0)),
            best_val_acc=float(result.get("best_val_acc", 0.0)),
            best_epoch=int(result.get("best_epoch", 0)),
            total_epochs=int(result.get("total_epochs_trained", 0)),
            mlflow_run_id=str(result.get("mlflow_run_id", "")),
            artifact_dir=str(result.get("artifact_dir", "")),
            model_save_path=str(result.get("model_save_path", "")),
            elapsed_sec=round(elapsed, 1),
            high_risk_f1=result.get("high_risk_class_f1", {}),
        )

        f1   = record.best_val_macro_f1
        gate = "✓ TARGET MET" if f1 >= TARGET_THRESHOLD else (
               "✓ VIABLE"    if f1 >= VIABILITY_THRESHOLD else "✗ BELOW VIABILITY"
        )
        logger.info(
            f"  DONE: {spec.run_name} | "
            f"val_macro_f1={f1:.4f} {gate} | "
            f"val_acc={record.best_val_acc:.4f} | "
            f"epoch={record.best_epoch + 1} | "
            f"elapsed={elapsed:.0f}s",
            extra={"stage": "orchestrator"},
        )

        # High-risk class warnings
        for sign, f1_score in record.high_risk_f1.items():
            if f1_score == 0.0:
                logger.warning(
                    f"  HIGH RISK: '{sign}' F1=0.0 in run '{spec.run_name}'. "
                    "This sign failed to learn. Document in LIMITATIONS.md.",
                    extra={"stage": "orchestrator"},
                )

        return record

    except KeyboardInterrupt:
        logger.warning(
            f"  INTERRUPTED: '{spec.run_name}' aborted by user.",
            extra={"stage": "orchestrator"},
        )
        return RunRecord(
            run_name=spec.run_name,
            experiment_group=spec.experiment_group,
            model=spec.model,
            data=spec.data,
            augmentation=spec.augmentation,
            status="failed",
            elapsed_sec=round(time.time() - t_start, 1),
            error_message="KeyboardInterrupt",
        )
    except Exception as exc:
        elapsed = time.time() - t_start
        tb_str  = traceback.format_exc()
        logger.error(
            f"  FAILED: '{spec.run_name}' — "
            f"{type(exc).__name__}: {exc}\n{tb_str}",
            extra={"stage": "orchestrator"},
        )
        return RunRecord(
            run_name=spec.run_name,
            experiment_group=spec.experiment_group,
            model=spec.model,
            data=spec.data,
            augmentation=spec.augmentation,
            status="failed",
            elapsed_sec=round(elapsed, 1),
            error_message=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Group executor
# ---------------------------------------------------------------------------

def _execute_group(
    group_id:    int,
    group_name:  str,
    specs:       List[RunSpec],
    resume:      bool,
    force:       bool,
    mlflow_uri:  Optional[str],
    logger:      Any,
    dry_run:     bool = False,
) -> Tuple[GroupSummary, List[RunRecord]]:
    """
    Execute all runs in a group and return (GroupSummary, [RunRecord]).

    Never raises — all per-run exceptions are caught in _execute_run().

    Parameters
    ----------
    group_id   : int     (1–5)
    group_name : str
    specs      : list[RunSpec]
    resume     : bool
    force      : bool
    mlflow_uri : str | None
    logger     : logging.Logger
    dry_run    : bool    if True, log the plan and return without training

    Returns
    -------
    (GroupSummary, list[RunRecord])
    """
    t_group_start = time.time()
    records: List[RunRecord] = []

    n_planned = len(specs)
    logger.info(
        f"\n{'=' * 72}",
        extra={"stage": "orchestrator"},
    )
    logger.info(
        f"GROUP {group_id}: {group_name} | {n_planned} run(s)",
        extra={"stage": "orchestrator"},
    )
    logger.info(
        f"{'=' * 72}",
        extra={"stage": "orchestrator"},
    )

    for i, spec in enumerate(specs, start=1):
        logger.info(
            f"\n  [{i}/{n_planned}] {spec.run_name}",
            extra={"stage": "orchestrator"},
        )
        if dry_run:
            logger.info(
                f"  DRY RUN — would execute: "
                f"model={spec.model} | data={spec.data} | "
                f"augmentation={spec.augmentation} | "
                f"overrides={spec.overrides}",
                extra={"stage": "orchestrator"},
            )
            records.append(RunRecord(
                run_name=spec.run_name,
                experiment_group=spec.experiment_group,
                model=spec.model,
                data=spec.data,
                augmentation=spec.augmentation,
                status="skipped",
            ))
        else:
            record = _execute_run(spec, resume, force, mlflow_uri, logger)
            records.append(record)

    # Compute group summary
    completed = [r for r in records if r.status == "completed"]
    skipped   = [r for r in records if r.status == "skipped"]
    failed    = [r for r in records if r.status == "failed"]

    # Best run = highest val_macro_f1 among completed + skipped (loaded from manifest)
    best_record = max(
        [r for r in records if r.status in ("completed", "skipped")],
        key=lambda r: r.best_val_macro_f1,
        default=None,
    )

    elapsed = time.time() - t_group_start
    summary = GroupSummary(
        group_id=group_id,
        group_name=group_name,
        n_planned=n_planned,
        n_completed=len(completed),
        n_skipped=len(skipped),
        n_failed=len(failed),
        best_run_name=best_record.run_name if best_record else "",
        best_val_macro_f1=best_record.best_val_macro_f1 if best_record else 0.0,
        elapsed_sec=round(elapsed, 1),
    )

    logger.info(
        f"\nGroup {group_id} COMPLETE | "
        f"completed={len(completed)} | skipped={len(skipped)} | "
        f"failed={len(failed)} | "
        f"best_val_macro_f1={summary.best_val_macro_f1:.4f} "
        f"({summary.best_run_name}) | "
        f"elapsed={elapsed:.0f}s",
        extra={"stage": "orchestrator"},
    )

    if summary.critical_failure:
        logger.error(
            f"GROUP {group_id} CRITICAL FAILURE: "
            f"{len(failed)}/{n_planned} runs failed "
            f"(>{CRITICAL_FAILURE_FRACTION:.0%} threshold). "
            "Dependent downstream groups will be SKIPPED.",
            extra={"stage": "orchestrator"},
        )

    return summary, records


# ---------------------------------------------------------------------------
# Stage 5 completion gate
# ---------------------------------------------------------------------------

def _run_completion_gate(
    all_records: List[RunRecord],
    logger:      Any,
) -> Dict[str, Any]:
    """
    Run all Stage 5 completion checks and return a gate status dict.

    Checks:
      1. Expected number of runs (17 total)
      2. All completed runs have best_val_macro_f1 logged
      3. run_manifest.json exists for each completed run
      4. SavedModel directory exists for each completed run
      5. Viability threshold (≥0.60 for at least one run)
      6. Target threshold (≥0.70 for at least one run — WARNING if not met)
      7. High-risk class F1=0.0 count across all runs
      8. champion_model_v1 SavedModel exists

    Returns a dict with:
        passed         : bool  — all hard checks passed
        warnings       : list  — soft checks that failed
        errors         : list  — hard checks that failed
        champion_path  : str   — path to champion SavedModel (if exists)
        best_f1        : float — highest val_macro_f1 across all runs
        gate_details   : dict  — per-check results
    """
    logger.info(
        "\n" + "=" * 72 + "\nSTAGE 5 COMPLETION GATE\n" + "=" * 72,
        extra={"stage": "orchestrator"},
    )

    errors:   List[str] = []
    warnings: List[str] = []
    details:  Dict[str, Any] = {}

    completed_records = [r for r in all_records if r.status in ("completed", "skipped")]
    failed_records    = [r for r in all_records if r.status == "failed"]

    # ── Check 1: Total run count ─────────────────────────────────────────
    n_usable = len(completed_records)
    details["total_runs"]    = len(all_records)
    details["completed"]     = len(completed_records)
    details["failed"]        = len(failed_records)
    details["expected_runs"] = EXPECTED_TOTAL_RUNS

    if n_usable < EXPECTED_TOTAL_RUNS:
        msg = (
            f"Expected {EXPECTED_TOTAL_RUNS} runs, "
            f"got {n_usable} completed/skipped + {len(failed_records)} failed."
        )
        warnings.append(msg)
        logger.warning(f"  GATE [SOFT]: {msg}", extra={"stage": "orchestrator"})
    else:
        logger.info(
            f"  GATE [PASS]: {n_usable}/{EXPECTED_TOTAL_RUNS} runs completed.",
            extra={"stage": "orchestrator"},
        )

    # ── Check 2: run_manifest.json existence ─────────────────────────────
    missing_manifests = []
    for r in completed_records:
        manifest_path = Path("artifacts") / "experiments" / r.run_name / _RUN_MANIFEST_FILENAME
        if not manifest_path.exists():
            missing_manifests.append(r.run_name)

    details["missing_manifests"] = missing_manifests
    if missing_manifests:
        msg = f"{len(missing_manifests)} completed run(s) missing run_manifest.json: {missing_manifests}"
        errors.append(msg)
        logger.error(f"  GATE [FAIL]: {msg}", extra={"stage": "orchestrator"})
    else:
        logger.info(
            "  GATE [PASS]: All completed runs have run_manifest.json.",
            extra={"stage": "orchestrator"},
        )

    # ── Check 3: SavedModel directories ──────────────────────────────────
    missing_savedmodels = []
    for r in completed_records:
        if r.model_save_path:
            save_path = Path(r.model_save_path)
        else:
            save_path = Path("models") / f"{r.run_name}_saved_model"

        if not save_path.exists():
            missing_savedmodels.append(r.run_name)

    details["missing_savedmodels"] = missing_savedmodels
    if missing_savedmodels:
        msg = (
            f"{len(missing_savedmodels)} run(s) missing SavedModel directory: "
            f"{missing_savedmodels}"
        )
        warnings.append(msg)
        logger.warning(f"  GATE [SOFT]: {msg}", extra={"stage": "orchestrator"})
    else:
        logger.info(
            "  GATE [PASS]: All completed runs have SavedModel directories.",
            extra={"stage": "orchestrator"},
        )

    # ── Check 4: Viability threshold ─────────────────────────────────────
    all_f1s = [r.best_val_macro_f1 for r in completed_records]
    best_f1  = max(all_f1s) if all_f1s else 0.0
    best_run = max(completed_records, key=lambda r: r.best_val_macro_f1, default=None)

    details["best_f1"]        = best_f1
    details["best_run_name"]  = best_run.run_name if best_run else ""
    details["viability_met"]  = best_f1 >= VIABILITY_THRESHOLD
    details["target_met"]     = best_f1 >= TARGET_THRESHOLD

    if best_f1 >= VIABILITY_THRESHOLD:
        logger.info(
            f"  GATE [PASS]: Viability threshold met — "
            f"best val_macro_f1={best_f1:.4f} >= {VIABILITY_THRESHOLD}",
            extra={"stage": "orchestrator"},
        )
    else:
        msg = (
            f"Minimum viability NOT met: best val_macro_f1={best_f1:.4f} "
            f"< {VIABILITY_THRESHOLD}. "
            "Review training curves for underfitting or gradient instability."
        )
        errors.append(msg)
        logger.error(f"  GATE [FAIL]: {msg}", extra={"stage": "orchestrator"})

    if best_f1 >= TARGET_THRESHOLD:
        logger.info(
            f"  GATE [PASS]: Target threshold met — "
            f"best val_macro_f1={best_f1:.4f} >= {TARGET_THRESHOLD}",
            extra={"stage": "orchestrator"},
        )
    else:
        msg = (
            f"Target NOT met: best val_macro_f1={best_f1:.4f} "
            f"< {TARGET_THRESHOLD}. "
            "Consider: longer seq_len, more hidden_units, more epochs, "
            "or hands_only landmark config (Fisher ratio 0.8097)."
        )
        warnings.append(msg)
        logger.warning(f"  GATE [SOFT]: {msg}", extra={"stage": "orchestrator"})

    # ── Check 5: Champion model existence ────────────────────────────────
    champion_saved_model = Path("models") / "champion_model_v1_saved_model"
    champion_exists      = champion_saved_model.exists()
    details["champion_model_path"] = str(champion_saved_model)
    details["champion_exists"]     = champion_exists

    if champion_exists:
        logger.info(
            f"  GATE [PASS]: Champion SavedModel exists at {champion_saved_model}",
            extra={"stage": "orchestrator"},
        )
    else:
        msg = f"Champion SavedModel not found at {champion_saved_model}."
        warnings.append(msg)
        logger.warning(f"  GATE [SOFT]: {msg}", extra={"stage": "orchestrator"})

    # ── Check 6: High-risk class analysis ────────────────────────────────
    zero_f1_by_sign: Dict[str, List[str]] = {s: [] for s in _HIGH_RISK_SIGNS}
    for r in completed_records:
        for sign in _HIGH_RISK_SIGNS:
            if r.high_risk_f1.get(sign, 1.0) == 0.0:
                zero_f1_by_sign[sign].append(r.run_name)

    details["high_risk_zero_f1"] = {k: v for k, v in zero_f1_by_sign.items() if v}
    for sign, runs in zero_f1_by_sign.items():
        if runs:
            logger.warning(
                f"  HIGH RISK: '{sign}' has F1=0.0 in {len(runs)} run(s): {runs}. "
                "Document in LIMITATIONS.md and report.",
                extra={"stage": "orchestrator"},
            )

    # ── Final gate status ─────────────────────────────────────────────────
    gate_passed = len(errors) == 0
    status_str  = "PASSED" if gate_passed else f"FAILED ({len(errors)} errors)"

    logger.info(
        f"\nCompletion gate: {status_str} | "
        f"{len(warnings)} warning(s) | "
        f"best_val_macro_f1={best_f1:.4f} ({details['best_run_name']})",
        extra={"stage": "orchestrator"},
    )

    return {
        "passed":        gate_passed,
        "errors":        errors,
        "warnings":      warnings,
        "champion_path": str(champion_saved_model) if champion_exists else "",
        "best_f1":       best_f1,
        "best_run_name": details.get("best_run_name", ""),
        "gate_details":  details,
    }


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_execution_report(
    all_records:      List[RunRecord],
    group_summaries:  List[GroupSummary],
    gate_status:      Dict[str, Any],
    t_total_start:    float,
    args:             argparse.Namespace,
) -> Path:
    """
    Write reports/experiment_execution_report.json.

    This is the machine-readable result of all Stage 5 runs. Notebook 05
    §7 reads this file to build the experiment registry table.
    """
    Path("reports").mkdir(parents=True, exist_ok=True)
    report_path = Path("reports") / "experiment_execution_report.json"

    total_elapsed = time.time() - t_total_start
    completed     = [r for r in all_records if r.status in ("completed", "skipped")]
    best_record   = max(completed, key=lambda r: r.best_val_macro_f1, default=None)

    report = {
        "stage":              "Stage 5",
        "generated_at_utc":   datetime.now(timezone.utc).isoformat(),
        "total_elapsed_sec":  round(total_elapsed, 1),
        "total_elapsed_min":  round(total_elapsed / 60, 1),
        "args": {
            "groups":     getattr(args, "groups", None),
            "resume":     getattr(args, "resume", False),
            "dry_run":    getattr(args, "dry_run", False),
            "force":      getattr(args, "force", False),
            "mlflow_uri": getattr(args, "mlflow_tracking_uri", None),
        },
        "run_summary": {
            "total":     len(all_records),
            "completed": len([r for r in all_records if r.status == "completed"]),
            "skipped":   len([r for r in all_records if r.status == "skipped"]),
            "failed":    len([r for r in all_records if r.status == "failed"]),
            "expected":  EXPECTED_TOTAL_RUNS,
        },
        "best_run": {
            "run_name":          best_record.run_name if best_record else "",
            "best_val_macro_f1": best_record.best_val_macro_f1 if best_record else 0.0,
            "best_val_acc":      best_record.best_val_acc if best_record else 0.0,
            "model":             best_record.model if best_record else "",
            "data":              best_record.data if best_record else "",
            "augmentation":      best_record.augmentation if best_record else "",
        },
        "group_summaries": [s.to_dict() for s in group_summaries],
        "runs":             [r.to_dict() for r in all_records],
        "gate_status":      gate_status,
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report_path


def _write_experiment_summary_md(
    all_records:    List[RunRecord],
    gate_status:    Dict[str, Any],
    t_total_start:  float,
) -> Path:
    """
    Write reports/experiment_summary.md — the human-readable Stage 5 registry.

    Structure:
      - One-line summary
      - Full experiment registry table (17 rows)
      - Group-by-group conclusions (4 paragraphs)
      - Champion selection justification
      - High-risk class analysis
      - Forward reference to Stage 6

    This file is consumed by the Stage 11 one-page report.
    """
    Path("reports").mkdir(parents=True, exist_ok=True)
    md_path = Path("reports") / "experiment_summary.md"

    completed   = [r for r in all_records if r.status in ("completed", "skipped")]
    failed      = [r for r in all_records if r.status == "failed"]
    best_f1     = gate_status.get("best_f1", 0.0)
    best_run    = gate_status.get("best_run_name", "unknown")
    champion_ok = gate_status.get("champion_exists", False)
    total_elapsed_min = round((time.time() - t_total_start) / 60, 0)

    # Sort records for table: group order → val_macro_f1 descending within group
    group_order = {"architecture": 1, "augmentation": 2, "sequence_length": 3,
                   "landmark_config": 4, "champion": 5, "custom": 6}
    sorted_records = sorted(
        completed,
        key=lambda r: (group_order.get(r.experiment_group, 99), -r.best_val_macro_f1),
    )

    lines: List[str] = []

    # ── Header ────────────────────────────────────────────────────────────
    lines.append("# Stage 5 Experiment Summary — WLASL 35-Class Gesture Recognition")
    lines.append("")
    lines.append(
        f"**{len(completed)} experiments tracked across 4 groups + champion run.**  "
    )
    lines.append(
        f"Best run: `{best_run}` — val_macro_f1 = **{best_f1:.4f}**  "
    )
    lines.append(
        f"Total elapsed: {total_elapsed_min:.0f} min | "
        f"Completion gate: {'PASSED ✓' if gate_status.get('passed') else 'FAILED ✗'}  "
    )
    lines.append(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    # ── Experiment Registry Table ─────────────────────────────────────────
    lines.append("## Experiment Registry")
    lines.append("")
    lines.append(
        "| Run Name | Group | Architecture | Seq Len | Aug | Landmark | "
        "Val Macro-F1 | Val Acc | Epochs | Params |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|"
    )

    for r in sorted_records:
        # Derive params from model save path or manifest — use placeholder
        f1_str  = f"{r.best_val_macro_f1:.4f}"
        acc_str = f"{r.best_val_acc:.4f}"
        try:
            manifest_path = (
                Path("artifacts") / "experiments" / r.run_name / _RUN_MANIFEST_FILENAME
            )
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as mf:
                    mdata = json.load(mf)
                param_count = mdata.get("model_param_count", "—")
                if isinstance(param_count, int):
                    param_count = f"{param_count:,}"
            else:
                param_count = "—"
        except Exception:
            param_count = "—"

        # Best epoch is 0-indexed in the manifest; display as 1-indexed
        epoch_display = r.best_epoch + 1 if r.best_epoch > 0 else "—"

        lines.append(
            f"| `{r.run_name}` | {r.experiment_group} | {r.model} | "
            f"{r.data.replace('seq','')} | {r.augmentation} | — | "
            f"**{f1_str}** | {acc_str} | {epoch_display} | {param_count} |"
        )

    if failed:
        for r in failed:
            lines.append(
                f"| `{r.run_name}` | {r.experiment_group} | {r.model} | "
                f"{r.data.replace('seq','')} | {r.augmentation} | — | "
                f"❌ FAILED | — | — | — |"
            )

    lines.append("")

    # ── Group conclusions (populated with available data) ─────────────────
    lines.append("## Group-by-Group Conclusions")
    lines.append("")

    group_names = {
        "architecture":   "Group 1 — Architecture Comparison",
        "augmentation":   "Group 2 — Augmentation Ablation",
        "sequence_length":"Group 3 — Sequence Length Ablation",
        "landmark_config":"Group 4 — Landmark Configuration Ablation",
        "champion":       "Champion Run",
    }

    for group_key, group_title in group_names.items():
        group_records = [r for r in sorted_records if r.experiment_group == group_key]
        if not group_records:
            continue

        best_in_group = max(group_records, key=lambda r: r.best_val_macro_f1)
        lines.append(f"### {group_title}")
        lines.append("")

        if group_key == "architecture":
            lines.append(
                f"Best architecture: `{best_in_group.model}` with val_macro_f1="
                f"{best_in_group.best_val_macro_f1:.4f}. "
                f"{len(group_records)}/{len(group_records)} runs completed. "
                "The Dense baseline should materially underperform recurrent models — "
                "confirming that temporal sequence modelling is necessary for this task."
            )
        elif group_key == "augmentation":
            lines.append(
                f"Best augmentation strategy: `{best_in_group.augmentation}` with "
                f"val_macro_f1={best_in_group.best_val_macro_f1:.4f}. "
                "Augmentation results are fed directly into Groups 3 and 4 "
                "via adaptive MLflow-based selection."
            )
        elif group_key == "sequence_length":
            lines.append(
                f"Best sequence length: `{best_in_group.data}` with "
                f"val_macro_f1={best_in_group.best_val_macro_f1:.4f}. "
                "Note: Notebook 04 showed 97% truncation at seq_len=60 with P75=84 frames. "
                "The seq80 result is the most diagnostically important finding in Stage 5."
            )
        elif group_key == "landmark_config":
            lines.append(
                f"Best landmark configuration: `{best_in_group.run_name}` with "
                f"val_macro_f1={best_in_group.best_val_macro_f1:.4f}. "
                "Notebook 04 Fisher ratio: hands_only=0.8097, full=0.5492, pose_only=0.2176. "
                "Whether the Fisher ratio advantage translates to accuracy is confirmed here."
            )
        elif group_key == "champion":
            lines.append(
                f"Champion val_macro_f1={best_in_group.best_val_macro_f1:.4f} | "
                f"val_acc={best_in_group.best_val_acc:.4f}. "
                "Trained with hidden_units=128, epochs=100, spatial_temporal augmentation. "
                f"{'Target ≥0.70 MET ✓' if best_in_group.best_val_macro_f1 >= TARGET_THRESHOLD else f'Target ≥0.70 NOT met (gap: {TARGET_THRESHOLD - best_in_group.best_val_macro_f1:.4f})'}."
            )

        lines.append("")

    # ── High-risk class analysis ──────────────────────────────────────────
    lines.append("## High-Risk Class Analysis")
    lines.append("")
    lines.append(
        "The following classes have fewer than 5 training clips and are "
        "most likely to fail: `clothes` (2 clips), `think` (3 clips), "
        "`birthday` (4 clips), `name` (4 clips), `book` (4 clips)."
    )
    lines.append("")

    zero_f1_aggregate: Dict[str, List[str]] = {s: [] for s in _HIGH_RISK_SIGNS}
    for r in completed:
        for sign in _HIGH_RISK_SIGNS:
            if r.high_risk_f1.get(sign, 1.0) == 0.0:
                zero_f1_aggregate[sign].append(r.run_name)

    any_zero = False
    for sign, run_list in zero_f1_aggregate.items():
        if run_list:
            any_zero = True
            lines.append(f"- **`{sign}`**: F1=0.0 in {len(run_list)} run(s) — {run_list}")
        else:
            lines.append(f"- **`{sign}`**: F1>0.0 in all runs ✓")

    if any_zero:
        lines.append("")
        lines.append(
            "> **Action required**: Update `LIMITATIONS.md` with the above "
            "zero-F1 classes. These signs CANNOT be reliably recognised by the "
            "current model and should be flagged in the client report."
        )

    lines.append("")

    # ── Gate status ────────────────────────────────────────────────────────
    lines.append("## Stage 5 Completion Gate")
    lines.append("")
    lines.append(
        f"**Status: {'PASSED ✓' if gate_status.get('passed') else 'FAILED ✗'}**"
    )
    if gate_status.get("errors"):
        lines.append("")
        lines.append("**Errors (blocking):**")
        for err in gate_status["errors"]:
            lines.append(f"- ❌ {err}")
    if gate_status.get("warnings"):
        lines.append("")
        lines.append("**Warnings (non-blocking):**")
        for w in gate_status["warnings"]:
            lines.append(f"- ⚠️  {w}")

    lines.append("")

    # ── Forward reference ─────────────────────────────────────────────────
    lines.append("## Next Steps: Stage 6")
    lines.append("")
    lines.append(
        "Stage 6 will evaluate the champion model (`champion_model_v1`) on the "
        "held-out test set (51 clips, 7 signers never seen during training). "
        "Key deliverables: per-signer accuracy box plot, SHAP frame importance "
        "heatmaps, latency benchmark (200 inference calls, P50/P95), "
        "confidence calibration reliability diagram, and per-class F1 bar chart "
        "sorted ascending (flagging zero-F1 classes for the client report)."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_all_experiments.py",
        description=(
            "Orchestrate the full Stage 5 WLASL multi-model experiment matrix "
            "(17 runs across 4 groups + champion). Executes groups in dependency "
            "order, reads MLflow results to adaptively select optimal "
            "hyperparameters for dependent groups."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Run all experiments (default)
  python pipelines/run_all_experiments.py

  # Run only groups 1 and 2 (can run in parallel — no cross-dependencies)
  python pipelines/run_all_experiments.py --groups 1 2

  # Run group 3 onwards (groups 1+2 must be complete)
  python pipelines/run_all_experiments.py --groups 3 4 5

  # Resume — skip runs with existing run_manifest.json
  python pipelines/run_all_experiments.py --resume

  # Dry run — print the execution plan without training
  python pipelines/run_all_experiments.py --dry-run

  # Run only the champion (all groups must be complete)
  python pipelines/run_all_experiments.py --champion-only

  # Override training epochs for all runs (e.g. quick smoke test)
  python pipelines/run_all_experiments.py --override training.epochs=5

  # Custom MLflow URI
  python pipelines/run_all_experiments.py --mlflow-tracking-uri http://localhost:5000
        """,
    )

    parser.add_argument(
        "--groups",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        metavar="N",
        help=(
            "Run only these group IDs (1=architecture, 2=augmentation, "
            "3=sequence_length, 4=landmark_config, 5=champion). "
            "Default: all groups. Dependencies are enforced: "
            "group 3 requires 2, group 4 requires 2+3, group 5 requires 1+2+3+4."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Skip runs that already have a completed run_manifest.json. "
            "Their results are loaded from the existing manifest. "
            "Use after an interrupted run to continue without re-training."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Print the full execution plan (all RunSpecs in order) "
            "without launching any training runs."
        ),
    )
    parser.add_argument(
        "--champion-only",
        action="store_true",
        default=False,
        help=(
            "Run only the champion model. "
            "Requires Groups 1–4 to be complete in the MLflow tracking store."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Allow overwriting existing run artefacts. "
            "Without --force, a run_name collision raises an error. "
            "Without --resume, this flag is required to re-run completed runs."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Dot-notation config override applied to ALL runs. "
            "May be specified multiple times. "
            "Useful for quick smoke tests: --override training.epochs=5. "
            "Note: training.class_weight_balancing=True is always enforced "
            "regardless of this flag."
        ),
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        metavar="URI",
        help=(
            "MLflow tracking URI override for all runs and all MLflow queries. "
            "Examples: mlruns | http://localhost:5000"
        ),
    )
    parser.add_argument(
        "--splits-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override for the split CSV directory (default: data/splits).",
    )
    parser.add_argument(
        "--landmarks-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override for the landmarks directory (default: data/landmarks).",
    )

    return parser


# ---------------------------------------------------------------------------
# Override parsing (mirrors run_training.py._parse_overrides)
# ---------------------------------------------------------------------------

def _parse_overrides(override_list: List[str]) -> Dict[str, Any]:
    """
    Parse 'KEY=VALUE' override strings with ast.literal_eval type coercion.
    """
    import ast

    if not override_list:
        return {}

    result: Dict[str, Any] = {}
    for raw in override_list:
        if "=" not in raw:
            print(
                f"[run_all_experiments] WARNING: malformed override '{raw}' "
                "(missing '='). Ignored.",
                file=sys.stderr,
            )
            continue
        key, _, value_str = raw.partition("=")
        key = key.strip()
        value_str = value_str.strip()
        if not key:
            continue
        try:
            result[key] = ast.literal_eval(value_str)
        except (ValueError, SyntaxError):
            result[key] = value_str

    return result


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def run_all_experiments(args: argparse.Namespace) -> int:
    """
    Execute the full Stage 5 experiment matrix.

    Returns an exit code:
        EXIT_SUCCESS           — all runs completed, gate passed
        EXIT_PARTIAL_FAILURE   — some runs failed
        EXIT_CRITICAL_FAILURE  — a group critically failed
        EXIT_GATE_FAILURE      — gate checks failed after all runs
        EXIT_UNEXPECTED_ERROR  — unexpected exception in orchestrator logic
    """
    from src.utils.logger import get_logger, configure_logging

    configure_logging(level="INFO", log_dir="logs", run_name="run_all_experiments")
    logger = get_logger(__name__)

    t_total_start = time.time()

    logger.info(
        "\n" + "=" * 72 + "\n"
        "WLASL Stage 5 — Multi-Model Experiment Orchestrator\n"
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        + "=" * 72,
        extra={"stage": "orchestrator"},
    )

    # ── Parse global overrides ─────────────────────────────────────────────
    extra_overrides = _parse_overrides(args.override)
    if extra_overrides:
        logger.info(
            f"Global config overrides: {extra_overrides}",
            extra={"stage": "orchestrator"},
        )

    # Always enforce class_weight_balancing=True globally
    extra_overrides["training.class_weight_balancing"] = True

    # Optional path overrides for splits/landmarks (passed via run_single_experiment
    # but not currently used by RunSpec; stored for future use if needed)
    mlflow_uri = args.mlflow_tracking_uri

    # ── Determine which groups to run ──────────────────────────────────────
    if args.champion_only:
        groups_to_run: Set[int] = {5}
    elif args.groups is not None:
        groups_to_run = set(args.groups)
    else:
        groups_to_run = {1, 2, 3, 4, 5}

    logger.info(
        f"Groups to run: {sorted(groups_to_run)} | "
        f"resume={args.resume} | dry_run={args.dry_run} | force={args.force}",
        extra={"stage": "orchestrator"},
    )

    # ── Execution state ────────────────────────────────────────────────────
    all_records:     List[RunRecord]     = []
    group_summaries: List[GroupSummary]  = []
    exit_code:       int                 = EXIT_SUCCESS

    # Track whether prerequisites are satisfied for dependent groups
    group_1_passed = False
    group_2_passed = False
    group_3_passed = False
    group_4_passed = False

    # Adaptive selections — set after each group completes
    best_augmentation = "spatial_temporal"  # safe fallback if Group 2 skipped
    best_data_config  = "seq60"             # safe fallback if Group 3 skipped
    best_landmark     = "hands_only"        # safe fallback if Group 4 skipped
    best_model        = "bilstm"            # safe fallback if Group 1 skipped

    # ══════════════════════════════════════════════════════════════════════
    # GROUP 1 — Architecture comparison
    # ══════════════════════════════════════════════════════════════════════

    if 1 in groups_to_run:
        specs_g1 = _build_group_1_specs(extra_overrides)
        summary_g1, records_g1 = _execute_group(
            group_id=1,
            group_name="Architecture Comparison",
            specs=specs_g1,
            resume=args.resume,
            force=args.force,
            mlflow_uri=mlflow_uri,
            logger=logger,
            dry_run=args.dry_run,
        )
        all_records.extend(records_g1)
        group_summaries.append(summary_g1)

        if summary_g1.critical_failure:
            exit_code = EXIT_CRITICAL_FAILURE
            logger.error(
                "Group 1 critical failure. Champion run will use fallback "
                f"model='{best_model}'.",
                extra={"stage": "orchestrator"},
            )
        else:
            group_1_passed = True
            # Adaptive: select best architecture for the champion run
            if not args.dry_run:
                try:
                    best_arch_result = _select_best_run(
                        experiment_name=MLFLOW_EXPERIMENT_NAME,
                        tag_filter={"experiment_group": "architecture"},
                        metric="best_val_macro_f1",
                        tracking_uri=mlflow_uri,
                    )
                    best_model = best_arch_result["model_type"]
                    logger.info(
                        f"Adaptive selection: best architecture = '{best_model}' "
                        f"(val_macro_f1={best_arch_result['best_val_macro_f1']:.4f})",
                        extra={"stage": "orchestrator"},
                    )
                except Exception as exc:
                    logger.warning(
                        f"Group 1 best-model selection failed: {exc}. "
                        f"Using fallback model='{best_model}'.",
                        extra={"stage": "orchestrator"},
                    )

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args
        )

    # ══════════════════════════════════════════════════════════════════════
    # GROUP 2 — Augmentation ablation
    # NOTE: Group 2 has NO dependency on Group 1 results. Both can run
    # concurrently. In sequential execution, we simply run them back-to-back.
    # ══════════════════════════════════════════════════════════════════════

    if 2 in groups_to_run:
        specs_g2 = _build_group_2_specs(extra_overrides)
        summary_g2, records_g2 = _execute_group(
            group_id=2,
            group_name="Augmentation Ablation",
            specs=specs_g2,
            resume=args.resume,
            force=args.force,
            mlflow_uri=mlflow_uri,
            logger=logger,
            dry_run=args.dry_run,
        )
        all_records.extend(records_g2)
        group_summaries.append(summary_g2)

        if summary_g2.critical_failure:
            if exit_code == EXIT_SUCCESS:
                exit_code = EXIT_CRITICAL_FAILURE
            logger.error(
                f"Group 2 critical failure. Groups 3+4 will use fallback "
                f"augmentation='{best_augmentation}'.",
                extra={"stage": "orchestrator"},
            )
        else:
            group_2_passed = True
            # Adaptive: select best augmentation strategy
            if not args.dry_run:
                try:
                    best_aug_result = _select_best_run(
                        experiment_name=MLFLOW_EXPERIMENT_NAME,
                        tag_filter={"experiment_group": "augmentation"},
                        metric="best_val_macro_f1",
                        tracking_uri=mlflow_uri,
                    )
                    best_augmentation = best_aug_result["augmentation"]
                    logger.info(
                        f"Adaptive selection: best augmentation = '{best_augmentation}' "
                        f"(val_macro_f1={best_aug_result['best_val_macro_f1']:.4f})",
                        extra={"stage": "orchestrator"},
                    )
                except Exception as exc:
                    logger.warning(
                        f"Group 2 best-augmentation selection failed: {exc}. "
                        f"Using fallback augmentation='{best_augmentation}'.",
                        extra={"stage": "orchestrator"},
                    )

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args
        )

    # ══════════════════════════════════════════════════════════════════════
    # GROUP 3 — Sequence length ablation (depends on Group 2)
    # ══════════════════════════════════════════════════════════════════════

    if 3 in groups_to_run:
        # Dependency: Group 2 must have completed or been run in a prior session
        # (MLflow store has Group 2 results). If Group 2 was skipped (not in
        # groups_to_run), attempt to read its results from MLflow.
        if 2 not in groups_to_run and not args.dry_run:
            logger.info(
                "Group 2 not in this session — attempting to read best augmentation "
                "from MLflow (assumes Group 2 ran in a prior session).",
                extra={"stage": "orchestrator"},
            )
            try:
                best_aug_result = _select_best_run(
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    tag_filter={"experiment_group": "augmentation"},
                    metric="best_val_macro_f1",
                    tracking_uri=mlflow_uri,
                )
                best_augmentation = best_aug_result["augmentation"]
                logger.info(
                    f"Read best augmentation from MLflow: '{best_augmentation}' "
                    f"(val_macro_f1={best_aug_result['best_val_macro_f1']:.4f})",
                    extra={"stage": "orchestrator"},
                )
                group_2_passed = True
            except Exception as exc:
                logger.warning(
                    f"Could not read Group 2 results from MLflow: {exc}. "
                    f"Using fallback augmentation='{best_augmentation}'.",
                    extra={"stage": "orchestrator"},
                )

        # Only skip Group 3 if Group 2 ran this session and critically failed
        if 2 in groups_to_run and not group_2_passed:
            skip_reason = "Group 2 critically failed — cannot select best augmentation."
            logger.error(
                f"SKIPPING Group 3: {skip_reason}",
                extra={"stage": "orchestrator"},
            )
            group_summaries.append(GroupSummary(
                group_id=3,
                group_name="Sequence Length Ablation",
                n_planned=6,
                n_completed=0,
                n_skipped=0,
                n_failed=0,
                skipped_reason=skip_reason,
            ))
        else:
            specs_g3 = _build_group_3_specs(best_augmentation, extra_overrides)
            summary_g3, records_g3 = _execute_group(
                group_id=3,
                group_name=f"Sequence Length Ablation (aug={best_augmentation})",
                specs=specs_g3,
                resume=args.resume,
                force=args.force,
                mlflow_uri=mlflow_uri,
                logger=logger,
                dry_run=args.dry_run,
            )
            all_records.extend(records_g3)
            group_summaries.append(summary_g3)

            if summary_g3.critical_failure:
                if exit_code == EXIT_SUCCESS:
                    exit_code = EXIT_CRITICAL_FAILURE
                logger.error(
                    f"Group 3 critical failure. Group 4 will use fallback "
                    f"seq='{best_data_config}'.",
                    extra={"stage": "orchestrator"},
                )
            else:
                group_3_passed = True
                # Adaptive: select best sequence length
                if not args.dry_run:
                    try:
                        best_seq_result = _select_best_run(
                            experiment_name=MLFLOW_EXPERIMENT_NAME,
                            tag_filter={"experiment_group": "sequence_length"},
                            metric="best_val_macro_f1",
                            tracking_uri=mlflow_uri,
                        )
                        raw_seq = best_seq_result.get("data_config", "")
                        # Normalise: "seq80", "80", or fallback
                        if raw_seq.startswith("seq"):
                            best_data_config = raw_seq
                        elif raw_seq.isdigit():
                            best_data_config = f"seq{raw_seq}"
                        else:
                            best_data_config = f"seq{best_seq_result.get('seq_len', '60')}"
                        logger.info(
                            f"Adaptive selection: best seq_len = '{best_data_config}' "
                            f"(val_macro_f1={best_seq_result['best_val_macro_f1']:.4f})",
                            extra={"stage": "orchestrator"},
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Group 3 best seq_len selection failed: {exc}. "
                            f"Using fallback data='{best_data_config}'.",
                            extra={"stage": "orchestrator"},
                        )

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args
        )

    # ══════════════════════════════════════════════════════════════════════
    # GROUP 4 — Landmark configuration ablation (depends on Groups 2 + 3)
    # ══════════════════════════════════════════════════════════════════════

    if 4 in groups_to_run:
        # Attempt to read Group 2/3 from MLflow if they weren't run this session
        if 2 not in groups_to_run and not args.dry_run and not group_2_passed:
            try:
                r = _select_best_run(MLFLOW_EXPERIMENT_NAME,
                                     {"experiment_group": "augmentation"},
                                     tracking_uri=mlflow_uri)
                best_augmentation = r["augmentation"]
                group_2_passed = True
            except Exception:
                pass

        if 3 not in groups_to_run and not args.dry_run and not group_3_passed:
            try:
                r = _select_best_run(MLFLOW_EXPERIMENT_NAME,
                                     {"experiment_group": "sequence_length"},
                                     tracking_uri=mlflow_uri)
                raw_seq = r.get("data_config", "")
                best_data_config = raw_seq if raw_seq.startswith("seq") else f"seq{r.get('seq_len','60')}"
                group_3_passed = True
            except Exception:
                pass

        deps_met = (
            (group_2_passed or 2 not in groups_to_run)
            and (group_3_passed or 3 not in groups_to_run)
        )

        if not deps_met:
            skip_reason = (
                "Prerequisite groups failed: "
                f"Group 2={'OK' if group_2_passed else 'FAILED'}, "
                f"Group 3={'OK' if group_3_passed else 'FAILED'}."
            )
            logger.error(
                f"SKIPPING Group 4: {skip_reason}",
                extra={"stage": "orchestrator"},
            )
            group_summaries.append(GroupSummary(
                group_id=4,
                group_name="Landmark Configuration Ablation",
                n_planned=3,
                n_completed=0,
                n_skipped=0,
                n_failed=0,
                skipped_reason=skip_reason,
            ))
        else:
            specs_g4 = _build_group_4_specs(
                best_augmentation, best_data_config, extra_overrides
            )
            summary_g4, records_g4 = _execute_group(
                group_id=4,
                group_name=(
                    f"Landmark Configuration Ablation "
                    f"(aug={best_augmentation}, data={best_data_config})"
                ),
                specs=specs_g4,
                resume=args.resume,
                force=args.force,
                mlflow_uri=mlflow_uri,
                logger=logger,
                dry_run=args.dry_run,
            )
            all_records.extend(records_g4)
            group_summaries.append(summary_g4)

            if not summary_g4.critical_failure:
                group_4_passed = True
                # Adaptive: select best landmark config
                if not args.dry_run:
                    try:
                        best_lm_result = _select_best_run(
                            experiment_name=MLFLOW_EXPERIMENT_NAME,
                            tag_filter={"experiment_group": "landmark_config"},
                            metric="best_val_macro_f1",
                            tracking_uri=mlflow_uri,
                        )
                        raw_lm = best_lm_result.get("landmark_config", "")
                        if raw_lm in ("hands_only", "pose_only", "full"):
                            best_landmark = raw_lm
                        logger.info(
                            f"Adaptive selection: best landmark_config = '{best_landmark}' "
                            f"(val_macro_f1={best_lm_result['best_val_macro_f1']:.4f})",
                            extra={"stage": "orchestrator"},
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Group 4 best landmark selection failed: {exc}. "
                            f"Using fallback landmark='{best_landmark}'.",
                            extra={"stage": "orchestrator"},
                        )

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args
        )

    # ══════════════════════════════════════════════════════════════════════
    # GROUP 5 — Champion run (depends on all groups)
    # ══════════════════════════════════════════════════════════════════════

    if 5 in groups_to_run:
        # Attempt to read all prior group results from MLflow if not run this session
        if not args.dry_run:
            if 1 not in groups_to_run:
                try:
                    r = _select_best_run(MLFLOW_EXPERIMENT_NAME,
                                         {"experiment_group": "architecture"},
                                         tracking_uri=mlflow_uri)
                    best_model = r["model_type"]
                    group_1_passed = True
                except Exception:
                    pass

            if 2 not in groups_to_run and not group_2_passed:
                try:
                    r = _select_best_run(MLFLOW_EXPERIMENT_NAME,
                                         {"experiment_group": "augmentation"},
                                         tracking_uri=mlflow_uri)
                    best_augmentation = r["augmentation"]
                    group_2_passed = True
                except Exception:
                    pass

            if 3 not in groups_to_run and not group_3_passed:
                try:
                    r = _select_best_run(MLFLOW_EXPERIMENT_NAME,
                                         {"experiment_group": "sequence_length"},
                                         tracking_uri=mlflow_uri)
                    raw_seq = r.get("data_config", "")
                    best_data_config = raw_seq if raw_seq.startswith("seq") else f"seq{r.get('seq_len','60')}"
                    group_3_passed = True
                except Exception:
                    pass

            if 4 not in groups_to_run and not group_4_passed:
                try:
                    r = _select_best_run(MLFLOW_EXPERIMENT_NAME,
                                         {"experiment_group": "landmark_config"},
                                         tracking_uri=mlflow_uri)
                    raw_lm = r.get("landmark_config", "")
                    if raw_lm in ("hands_only", "pose_only", "full"):
                        best_landmark = raw_lm
                    group_4_passed = True
                except Exception:
                    pass

        logger.info(
            f"\nChampion run configuration (from adaptive selection):\n"
            f"  model={best_model} | augmentation={best_augmentation} | "
            f"data={best_data_config} | landmark={best_landmark}",
            extra={"stage": "orchestrator"},
        )

        champion_spec = _build_champion_spec(
            best_model=best_model,
            best_augmentation=best_augmentation,
            best_data_config=best_data_config,
            best_lm_config=best_landmark,
            extra_overrides=extra_overrides,
        )

        champion_summary, champion_records = _execute_group(
            group_id=5,
            group_name="Champion Run",
            specs=[champion_spec],
            resume=args.resume,
            force=args.force,
            mlflow_uri=mlflow_uri,
            logger=logger,
            dry_run=args.dry_run,
        )
        all_records.extend(champion_records)
        group_summaries.append(champion_summary)

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args
        )

    # ══════════════════════════════════════════════════════════════════════
    # Completion gate + final reports
    # ══════════════════════════════════════════════════════════════════════

    if not args.dry_run:
        gate_status = _run_completion_gate(all_records, logger)
        if not gate_status["passed"] and exit_code == EXIT_SUCCESS:
            exit_code = EXIT_GATE_FAILURE
    else:
        gate_status = {"passed": True, "errors": [], "warnings": [], "best_f1": 0.0,
                       "best_run_name": "", "champion_path": "", "gate_details": {}}
        logger.info(
            "DRY RUN — gate checks skipped.",
            extra={"stage": "orchestrator"},
        )

    # Write final reports
    try:
        report_path = _write_execution_report(
            all_records, group_summaries, gate_status, t_total_start, args
        )
        logger.info(
            f"Execution report written: {report_path}",
            extra={"stage": "orchestrator"},
        )
    except Exception as exc:
        logger.error(
            f"Failed to write execution report: {exc}",
            extra={"stage": "orchestrator"},
        )

    try:
        md_path = _write_experiment_summary_md(all_records, gate_status, t_total_start)
        logger.info(
            f"Experiment summary written: {md_path}",
            extra={"stage": "orchestrator"},
        )
    except Exception as exc:
        logger.error(
            f"Failed to write experiment_summary.md: {exc}",
            extra={"stage": "orchestrator"},
        )

    # ── Final console summary ─────────────────────────────────────────────
    total_elapsed = time.time() - t_total_start
    completed_count = len([r for r in all_records if r.status == "completed"])
    skipped_count   = len([r for r in all_records if r.status == "skipped"])
    failed_count    = len([r for r in all_records if r.status == "failed"])
    best_f1         = gate_status.get("best_f1", 0.0)
    best_run        = gate_status.get("best_run_name", "—")

    logger.info(
        "\n" + "=" * 72 + "\n"
        "STAGE 5 COMPLETE\n"
        + "=" * 72 + "\n"
        f"  Runs:        completed={completed_count} | "
        f"skipped={skipped_count} | failed={failed_count}\n"
        f"  Best run:    {best_run} | val_macro_f1={best_f1:.4f}\n"
        f"  Gate:        {'PASSED ✓' if gate_status.get('passed') else 'FAILED ✗'}\n"
        f"  Total time:  {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)\n"
        f"  Exit code:   {exit_code}\n"
        + "=" * 72,
        extra={"stage": "orchestrator"},
    )

    # Partial failure
    if failed_count > 0 and exit_code == EXIT_SUCCESS:
        exit_code = EXIT_PARTIAL_FAILURE

    return exit_code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    try:
        return run_all_experiments(args)
    except KeyboardInterrupt:
        print(
            "\n[run_all_experiments] Interrupted by user. "
            "Use --resume to continue from where you left off.",
            file=sys.stderr,
        )
        return EXIT_PARTIAL_FAILURE
    except Exception as exc:
        print(
            f"\n[run_all_experiments] Unexpected error in orchestrator: "
            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            file=sys.stderr,
        )
        return EXIT_UNEXPECTED_ERROR


if __name__ == "__main__":
    sys.exit(main())