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
preceding groups.  All adaptive selection is filtered by the session
``batch_id`` tag (a UUID4 stamped at orchestrator start) so that results
from prior experimental sessions are never accidentally mixed in.  This
is the key fix for Issues #1 and #17 in the critical review.

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
        python pipelines/run_all_experiments.py --champion-only --batch-id <uuid>

    Custom MLflow tracking URI:
        python pipelines/run_all_experiments.py --mlflow-tracking-uri http://localhost:5000

    Override config values for all runs:
        python pipelines/run_all_experiments.py --override training.epochs=40

    Require all groups to pass before champion:
        python pipelines/run_all_experiments.py --require-all-groups

Fault tolerance
-----------------
Each run is individually wrapped in a try/except. A single run failing (NaN
loss, OOM, corrupt data) does NOT abort the orchestrator. The failure is
logged, the run is recorded in the execution report as FAILED, and execution
continues with the next run.

If a critical group fails (> 50% of runs in the group fail), subsequent
dependent groups are skipped and a clear diagnostic is logged.

Run collision avoidance
--------------------------
Before each run, ``_is_run_already_complete()`` checks for an existing
``run_manifest.json`` AND a SavedModel directory for that run_name.  Both
must exist for a run to be skipped under ``--resume``.  This prevents a
crashed-after-manifest run from being silently skipped.

MLflow result reading
-----------------------
``_select_best_run()`` always filters by the session ``batch_id`` tag so
only runs from the current execution session are considered.  Pass
``--batch-id <uuid>`` on the CLI to continue a prior session (e.g. after
a crash) using ``--resume``.

Stage 5 completion gate
------------------------
Two gates are run:
  1. Pre-champion dependency gate — verifies Groups 1–4 are sufficiently
     complete before the champion run begins.  Fails hard if critical groups
     are missing and ``--require-all-groups`` is set.
  2. Final completion gate — verifies all 17 MLflow runs, manifests,
     SavedModels, viability/target thresholds, and high-risk class F1s.

Execution report
-----------------
``reports/experiment_execution_report.json`` is written atomically (temp-file
then os.replace) on completion and on partial completion if interrupted.
``reports/experiment_summary.md`` is written after the gate check with
data-driven (not hardcoded) narrative conclusions.

Design constraints (all non-negotiable)
-----------------------------------------
- run_single_experiment() from run_training.py is the sole training entry point.
- Primary metric is val_macro_f1 everywhere. val_accuracy is secondary.
- class_weight_balancing=True for every run (enforced in _build_run_kwargs).
- Masking=True for all recurrent models (enforced in architectures.py).
- seq80 runs SECOND within Group 3 (highest expected accuracy gain).
- hands_only runs FIRST within Group 4 (highest Fisher ratio: 0.8097).
- No nested MLflow runs.
- All exceptions from individual runs are caught; orchestrator never crashes on
  a single run failure.
- All adaptive selections filtered by batch_id (Issues #1, #17).
- Reports written atomically (Issue #14).
- Landmark config tracked per run in RunRecord (Issue #8).
- Data-driven narrative in experiment_summary.md (Issue #7).
- Pre-champion dependency gate (Issues #2, #3).
- All exception handlers log at WARNING or above (Issue #4).
- Metric finiteness validated before ranking (Issue #5).
- best_epoch sentinel is -1 / displayed as 1-indexed correctly (Issue #6).
- High-risk F1 threshold is configurable (Issue #9).
- Checkpoint writes are throttled; final write is fully populated (Issue #10).
- Dry-run reports are watermarked and not misleading (Issue #11).
- Resume validates both manifest and SavedModel (Issue #12).
- Override hash appended to run_name when non-default overrides present (Issue #13).
- Fallback champion is loudly warned with --require-all-groups guard (Issue #15).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Exit codes — CI/CD compatible
# ---------------------------------------------------------------------------

EXIT_SUCCESS           = 0
EXIT_PARTIAL_FAILURE   = 1
EXIT_CRITICAL_FAILURE  = 2
EXIT_GATE_FAILURE      = 3
EXIT_UNEXPECTED_ERROR  = 4

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: MLflow experiment name — locked value from the handoff document.
MLFLOW_EXPERIMENT_NAME: str = "WLASL-35-class"

#: Total expected number of runs: Groups 4+3+6+3 + champion = 17.
EXPECTED_TOTAL_RUNS: int = 17

#: Minimum val_macro_f1 for minimum viability (hard gate check).
VIABILITY_THRESHOLD: float = 0.60

#: Target val_macro_f1 (soft gate warning if not met).
TARGET_THRESHOLD: float = 0.70

#: Maximum fraction of a group's runs that can fail before the group is
#: considered "critically failed" and dependent groups are skipped.
CRITICAL_FAILURE_FRACTION: float = 0.50

#: High-risk sign classes requiring explicit F1 logging after every run.
_HIGH_RISK_SIGNS: Tuple[str, ...] = (
    "clothes", "think", "birthday", "name", "book",
)

#: F1 threshold below which a high-risk class is flagged (Issue #9).
#: Using 0.01 rather than exact 0.0 guards against floating-point noise
#: while still catching genuinely failed classes.
HIGH_RISK_F1_ALERT_THRESHOLD: float = 0.01

#: Filename for run completion manifest.
_RUN_MANIFEST_FILENAME: str = "run_manifest.json"

#: MLflow tag key for batch/session identity (Issues #1, #17).
_BATCH_ID_TAG: str = "stage5_batch_id"

#: MLflow tag key for dataset version — for provenance verification.
_DATASET_VERSION_TAG: str = "dataset_split_version"

#: Assumed dataset split version for this experiment matrix.
_DATASET_VERSION_VALUE: str = "v1"

#: Minimum number of completed runs in a dependency group before its results
#: can be trusted for adaptive selection.
_MIN_GROUP_RUNS_FOR_SELECTION: int = 1


# ---------------------------------------------------------------------------
# Override fingerprinting (Issue #13)
# ---------------------------------------------------------------------------

def _override_fingerprint(overrides: Dict[str, Any]) -> str:
    """
    Return an 8-char hex fingerprint of the non-default override dict.

    Used to append a suffix to run_names when user-supplied overrides
    change the experiment semantics (e.g. different epoch counts).  The
    fingerprint is stable across platforms (MD5 of sorted JSON serialisation).

    Parameters
    ----------
    overrides : dict
        The parsed override dict from _parse_overrides().  Must not include
        the always-enforced ``training.class_weight_balancing=True`` entry,
        as that is not a user-specified override.

    Returns
    -------
    str  8-character lowercase hex string, or "" if overrides is empty.
    """
    # Strip the always-enforced key so it does not affect the fingerprint.
    meaningful = {
        k: v for k, v in overrides.items()
        if k != "training.class_weight_balancing"
    }
    if not meaningful:
        return ""
    serialised = json.dumps(meaningful, sort_keys=True, default=str)
    return hashlib.md5(serialised.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Run specification dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunSpec:
    """
    Complete specification for a single experiment run.

    All fields correspond to arguments of run_single_experiment() in
    run_training.py.  Immutable after construction.
    """
    run_name:         str
    model:            str
    data:             str
    augmentation:     str
    experiment_group: str
    experiment:       Optional[str]      = None
    overrides:        Dict[str, Any]     = field(default_factory=dict)
    description:      str                = ""
    landmark_config:  str                = "full"   # Issue #8: track explicitly

    def __post_init__(self) -> None:
        # Always enforce class_weight_balancing=True — no exceptions.
        if "training.class_weight_balancing" not in self.overrides:
            self.overrides["training.class_weight_balancing"] = True

    def __repr__(self) -> str:
        return (
            f"RunSpec(run_name={self.run_name!r}, model={self.model!r}, "
            f"data={self.data!r}, augmentation={self.augmentation!r}, "
            f"group={self.experiment_group!r}, landmark={self.landmark_config!r})"
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
    status:            str               # "completed" | "skipped" | "failed"
    landmark_config:   str               = "full"   # Issue #8: track per run
    best_val_macro_f1: float             = 0.0
    best_val_acc:      float             = 0.0
    #: Sentinel value -1 means "not yet trained" (Issue #6).
    best_epoch:        int               = -1
    total_epochs:      int               = 0
    mlflow_run_id:     str               = ""
    artifact_dir:      str               = ""
    model_save_path:   str               = ""
    elapsed_sec:       float             = 0.0
    error_message:     str               = ""
    high_risk_f1:      Dict[str, float]  = field(default_factory=dict)
    batch_id:          str               = ""

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
    best_run_name:     str    = ""
    best_val_macro_f1: float  = 0.0
    elapsed_sec:       float  = 0.0
    skipped_reason:    str    = ""

    @property
    def critical_failure(self) -> bool:
        """True if more than CRITICAL_FAILURE_FRACTION of runs failed."""
        if self.n_planned == 0:
            return False
        return (self.n_failed / self.n_planned) > CRITICAL_FAILURE_FRACTION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Atomic file write helper (Issue #14)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Any) -> None:
    """
    Write ``data`` as JSON to ``path`` atomically via a temp file + os.replace.

    A process kill mid-write can only corrupt the temp file, never the
    target.  The target is only updated when the full write succeeds.

    Parameters
    ----------
    path : Path
    data : Any  JSON-serialisable object.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file in the same directory so os.replace is atomic
    # (same filesystem).  tempfile.NamedTemporaryFile with delete=False
    # is used so the name is known and can be explicitly replaced.
    fd, tmp_path_str = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp_path_str, str(path))
    except Exception:
        # Clean up the temp file if write or replace failed
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a text file atomically via a temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path_str = tempfile.mkstemp(
        dir=path.parent, prefix=".tmp_", suffix=".md"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp_path_str, str(path))
    except Exception:
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# MLflow result querying (Issues #1, #4, #5, #17)
# ---------------------------------------------------------------------------

def _select_best_run(
    experiment_name: str,
    tag_filter:      Dict[str, str],
    metric:          str = "best_val_macro_f1",
    tracking_uri:    Optional[str] = None,
    batch_id:        Optional[str] = None,
    logger:          Any = None,
) -> Dict[str, Any]:
    """
    Query the MLflow tracking store and return the best run's parameters.

    This is the core of the adaptive selection logic. It reads real results
    from the tracking store filtered by the current session's ``batch_id``
    so that runs from prior experimental sessions are never accidentally
    mixed in (Issues #1 and #17).

    Parameters
    ----------
    experiment_name : str
    tag_filter : dict
        MLflow tag key-value pairs to filter runs (beyond batch_id).
    metric : str
        MLflow metric name to rank runs by.
    tracking_uri : str | None
    batch_id : str | None
        Session batch_id. When provided, added to tag_filter to restrict
        selection to the current session's runs (Issues #1, #17).
    logger : logging.Logger | None

    Returns
    -------
    dict with keys: run_name, augmentation, seq_len, landmark_config,
                    model_type, best_val_macro_f1, mlflow_run_id, data_config.

    Raises
    ------
    RuntimeError
        If no runs match the filter, tracking store is inaccessible, or
        no run has a finite metric value above zero (Issue #5).
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Always inject batch_id filter so we only select from THIS session
    # (Issues #1 and #17).
    effective_filter = dict(tag_filter)
    if batch_id:
        effective_filter[_BATCH_ID_TAG] = batch_id

    filter_parts = [
        f"tags.{k} = '{v}'" for k, v in effective_filter.items()
    ]
    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    try:
        runs_df = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=filter_string,
            order_by=[f"metrics.{metric} DESC"],
        )
    except Exception as exc:
        raise RuntimeError(
            f"_select_best_run(): MLflow search failed for filter={effective_filter}. "
            f"Error: {type(exc).__name__}: {exc}\n"
            "Ensure the MLflow tracking store is populated and accessible at "
            f"URI: {mlflow.get_tracking_uri()}"
        ) from exc

    if runs_df.empty:
        raise RuntimeError(
            f"_select_best_run(): no completed runs found for filter={effective_filter} "
            f"in experiment '{experiment_name}'. "
            "Ensure the prerequisite group ran successfully before launching "
            "dependent groups."
        )

    # Filter to FINISHED runs only
    finished = runs_df[runs_df["status"] == "FINISHED"]
    if finished.empty:
        raise RuntimeError(
            f"_select_best_run(): {len(runs_df)} run(s) match filter={effective_filter} "
            "but none have status=FINISHED."
        )

    def _get(row: Any, col: str, default: Any = "") -> Any:
        """Safely retrieve a column value, returning default for NaN."""
        val = row.get(col, default)
        if val is None:
            return default
        try:
            if isinstance(val, float) and math.isnan(val):
                return default
        except (TypeError, ValueError):
            pass
        return val

    # Issue #5: post-sort validation — ensure the best metric is finite and > 0.
    # MLflow sorts NaN as -inf in some backends, meaning a NaN-metric run
    # could appear first; we filter them out explicitly.
    metric_col = f"metrics.{metric}"
    valid_runs = finished[
        finished[metric_col].apply(
            lambda v: isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0
        )
    ] if metric_col in finished.columns else finished.iloc[0:0]

    if valid_runs.empty:
        raw_values = (
            finished[metric_col].tolist()
            if metric_col in finished.columns
            else []
        )
        raise RuntimeError(
            f"_select_best_run(): no FINISHED run in filter={effective_filter} "
            f"has a finite {metric} > 0.0. "
            f"Raw metric values found: {raw_values}. "
            "This indicates all matching runs produced degenerate metrics "
            "(NaN, Inf, or 0.0). Check training logs for data pipeline errors."
        )

    best = valid_runs.iloc[0]

    return {
        "run_name":          str(_get(best, "tags.mlflow.runName", _get(best, "tags.run_name"))),
        "augmentation":      str(_get(best, "tags.augmentation_name")),
        "seq_len":           str(_get(best, "params.seq_len", _get(best, "tags.seq_len"))),
        "landmark_config":   str(_get(best, "tags.landmark_config", "full")),
        "model_type":        str(_get(best, "tags.model_type")),
        "best_val_macro_f1": float(_get(best, metric_col, 0.0)),
        "mlflow_run_id":     str(_get(best, "run_id")),
        "data_config":       str(_get(best, "params.data_config",
                                    f"seq{_get(best, 'params.seq_len', '60')}")),
    }


# ---------------------------------------------------------------------------
# Run completion check (Issue #12)
# ---------------------------------------------------------------------------

def _is_run_already_complete(run_name: str) -> bool:
    """
    Return True only if BOTH run_manifest.json AND the SavedModel directory
    exist for this run_name (Issue #12).

    Checking both prevents a crashed run (manifest written, model not saved)
    from being silently skipped under ``--resume``.

    Parameters
    ----------
    run_name : str

    Returns
    -------
    bool
    """
    manifest = (
        Path("artifacts") / "experiments" / run_name / _RUN_MANIFEST_FILENAME
    )
    saved_model = Path("models") / f"{run_name}_saved_model"
    return manifest.exists() and saved_model.exists()


def _load_completed_manifest(run_name: str) -> Dict[str, Any]:
    """Load and return the run_manifest.json for a completed run."""
    manifest_path = (
        Path("artifacts") / "experiments" / run_name / _RUN_MANIFEST_FILENAME
    )
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# RunSpec builders for each group
# ---------------------------------------------------------------------------

def _build_group_1_specs(
    extra_overrides:    Dict[str, Any],
    override_suffix:    str,
) -> List[RunSpec]:
    """
    Group 1 — Architecture comparison.

    Fixed: seq60, augmentation=none, same seed.
    Varies: model architecture (dense, lstm, gru, bilstm).
    """
    architectures = [
        ("dense",  "Dense feedforward baseline — proves temporal modelling necessary"),
        ("lstm",   "Single LSTM baseline — primary ablation workhorse"),
        ("gru",    "GRU — streamlined recurrent baseline"),
        ("bilstm", "Bidirectional LSTM — champion candidate"),
    ]

    specs = []
    for model, description in architectures:
        base_name = f"{model}_seq60_no_aug"
        run_name  = f"{base_name}_{override_suffix}" if override_suffix else base_name
        specs.append(RunSpec(
            run_name=run_name,
            model=model,
            data="seq60",
            augmentation="none",
            experiment_group="architecture",
            landmark_config="full",
            description=description,
            overrides=dict(extra_overrides),
        ))
    return specs


def _build_group_2_specs(
    extra_overrides: Dict[str, Any],
    override_suffix: str,
) -> List[RunSpec]:
    """
    Group 2 — Augmentation ablation.

    Fixed: lstm, seq60.
    Varies: augmentation strategy.
    """
    strategies = [
        ("none",             "No augmentation — deterministic baseline"),
        ("temporal",         "Temporal-only augmentation — jitter + speed"),
        ("spatial_temporal", "Full spatial+temporal augmentation — complete chain"),
    ]

    specs = []
    for augmentation, description in strategies:
        base_name = f"lstm_seq60_{augmentation}_aug"
        run_name  = f"{base_name}_{override_suffix}" if override_suffix else base_name
        specs.append(RunSpec(
            run_name=run_name,
            model="lstm",
            data="seq60",
            augmentation=augmentation,
            experiment_group="augmentation",
            landmark_config="full",
            description=description,
            overrides=dict(extra_overrides),
        ))
    return specs


def _build_group_3_specs(
    best_augmentation: str,
    extra_overrides:   Dict[str, Any],
    override_suffix:   str,
) -> List[RunSpec]:
    """
    Group 3 — Sequence length ablation.

    Fixed: lstm, best_augmentation (from Group 2).
    Varies: sequence length.

    Run order (critical):
        seq60   first  — fast sanity check / Group 2 baseline
        seq80   second — HIGHEST PRIORITY: 97% truncation at seq60, P75=84 frames
        seq100  third  — diminishing returns check (7.1% more coverage)
        seq40   fourth — below-primary regression
        seq30   fifth
        seq20   sixth  — training data anchor (~34% mean content)
    """
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
        base_name = f"lstm_{data_config}_{best_augmentation}_aug"
        run_name  = f"{base_name}_{override_suffix}" if override_suffix else base_name
        specs.append(RunSpec(
            run_name=run_name,
            model="lstm",
            data=data_config,
            augmentation=best_augmentation,
            experiment_group="sequence_length",
            landmark_config="full",
            description=description,
            overrides=dict(extra_overrides),
        ))
    return specs


def _build_group_4_specs(
    best_augmentation: str,
    best_data_config:  str,
    extra_overrides:   Dict[str, Any],
    override_suffix:   str,
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
        base_name = f"lstm_{best_data_config}_{best_augmentation}_{lm_config}"
        run_name  = f"{base_name}_{override_suffix}" if override_suffix else base_name
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
            landmark_config=lm_config,   # Issue #8: track explicitly
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
    override_suffix:   str,
    using_fallbacks:   bool = False,
) -> RunSpec:
    """
    Champion run — best settings from Groups 1-4, hidden_units=128, epochs=100.

    Parameters
    ----------
    using_fallbacks : bool
        If True, at least one group failed and the champion uses fallback
        hyperparameters.  Logged prominently and tracked in the manifest.
    """
    base_name = "champion_model_v1"
    run_name  = f"{base_name}_{override_suffix}" if override_suffix else base_name

    overrides = {
        "model.hidden_units":               128,
        "training.epochs":                  100,
        "training.early_stopping_patience": 15,
        "data.landmark_config":             best_lm_config,
        **extra_overrides,
    }
    if using_fallbacks:
        # Tag the run so it is easily identifiable in MLflow
        overrides["_champion_uses_fallbacks"] = True

    return RunSpec(
        run_name=run_name,
        model=best_model,
        data=best_data_config,
        augmentation=best_augmentation,
        experiment_group="champion",
        experiment="best_model",
        landmark_config=best_lm_config,
        description=(
            f"Champion: {best_model} + {best_data_config} + "
            f"{best_augmentation} + {best_lm_config}"
            + (" [FALLBACK PARAMS]" if using_fallbacks else "")
            + " | hidden_units=128, epochs=100"
        ),
        overrides=overrides,
    )


# ---------------------------------------------------------------------------
# Single-run executor
# ---------------------------------------------------------------------------

def _execute_run(
    spec:       RunSpec,
    resume:     bool,
    force:      bool,
    mlflow_uri: Optional[str],
    batch_id:   str,
    logger:     Any,
) -> RunRecord:
    """
    Execute one run from a RunSpec and return a RunRecord.

    Handles resume mode, delegation to run_single_experiment(), fault
    tolerance, and timing.

    The ``batch_id`` is injected as an MLflow tag (via the ``_tag_*`` key
    convention) so every run in this session is tagged for provenance
    filtering (Issues #1, #17).

    Keys prefixed with ``_tag_`` are extracted from the overrides dict and
    passed through the overrides as-is; run_training.py's _execute_run()
    strips them before load_config() and applies them as MLflow tags after
    the run context is opened. Keys prefixed with ``_`` but not ``_tag_``
    (e.g. ``_champion_uses_fallbacks``) are dropped entirely here — they are
    informational markers that have no corresponding config field and no
    MLflow destination.

    Parameters
    ----------
    spec       : RunSpec
    resume     : bool
    force      : bool
    mlflow_uri : str | None
    batch_id   : str   UUID4 for this session.
    logger     : logging.Logger

    Returns
    -------
    RunRecord  — status is "completed", "skipped", or "failed"
    """
    t_start = time.time()

    # ── Resume: skip runs that have BOTH manifest AND SavedModel (Issue #12) ──
    if resume and _is_run_already_complete(spec.run_name):
        logger.info(
            f"  SKIPPING (--resume): run '{spec.run_name}' — "
            "both run_manifest.json and SavedModel directory exist.",
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
                landmark_config=manifest.get(
                    "landmark_config", spec.landmark_config
                ),
                status="skipped",
                best_val_macro_f1=float(manifest.get("best_val_macro_f1", 0.0)),
                best_val_acc=float(manifest.get("best_val_acc", 0.0)),
                best_epoch=int(manifest.get("best_epoch", -1)),
                total_epochs=int(manifest.get("total_epochs_trained", 0)),
                mlflow_run_id=str(manifest.get("mlflow_run_id", "")),
                artifact_dir=str(
                    Path("artifacts") / "experiments" / spec.run_name
                ),
                model_save_path=str(
                    Path("models") / f"{spec.run_name}_saved_model"
                ),
                elapsed_sec=round(time.time() - t_start, 1),
                high_risk_f1=manifest.get("high_risk_class_f1", {}),
                batch_id=batch_id,
            )
        except Exception as exc:
            logger.warning(
                f"  Failed to load completed manifest for '{spec.run_name}': "
                f"{type(exc).__name__}: {exc}. Will re-run.",
                extra={"stage": "orchestrator"},
            )
            # Fall through to normal execution

    # ── Normal execution via run_single_experiment() ──────────────────────
    from pipelines.run_training import run_single_experiment

    logger.info(
        f"  Launching: {spec.run_name} | "
        f"model={spec.model} | data={spec.data} | "
        f"augmentation={spec.augmentation} | "
        f"landmark={spec.landmark_config} | "
        f"group={spec.experiment_group}",
        extra={"stage": "orchestrator"},
    )
    if spec.description:
        logger.info(
            f"  Description: {spec.description}",
            extra={"stage": "orchestrator"},
        )

    # ── Build overrides — separating private keys from real config keys ────
    #
    # Three categories of keys arrive in spec.overrides:
    #
    #   (a) Real config keys  e.g. "data.landmark_config", "model.hidden_units"
    #       → passed to run_single_experiment() → load_config() → OmegaConf merge
    #       → Pydantic validation. These MUST be valid ExperimentConfig fields.
    #
    #   (b) _tag_* keys  e.g. "_tag_stage5_batch_id", "_tag_dataset_split_version"
    #       → injected below to carry provenance metadata.
    #       → run_training.py._execute_run() strips the "_tag_" prefix and applies
    #         them as mlflow.set_tags() AFTER the run context opens, so they never
    #         touch load_config() / OmegaConf / Pydantic.
    #       → MUST NOT be passed directly to OmegaConf — ExperimentConfig uses
    #         extra="forbid" and would raise ValidationError immediately.
    #
    #   (c) Other _* keys  e.g. "_champion_uses_fallbacks"
    #       → Informational markers set by _build_champion_spec().
    #       → No corresponding config field, no MLflow destination.
    #       → Dropped here. Logged at DEBUG so the drop is traceable.
    #
    # run_training.py is responsible for the actual split (see its _execute_run
    # docstring). We keep the separation clean here by only injecting _tag_* keys
    # into run_overrides; real config keys flow through from spec.overrides.

    run_overrides: Dict[str, Any] = {}
    dropped_keys: list = []

    for k, v in spec.overrides.items():
        if k.startswith("_tag_") or k.startswith("_"):
            # _tag_* → kept for run_training.py to apply as MLflow tags.
            # Other _* → informational only, dropped with a debug log.
            if k.startswith("_tag_"):
                run_overrides[k] = v          # pass through to run_training.py
            else:
                dropped_keys.append(k)        # drop silently (logged below)
        else:
            run_overrides[k] = v              # real config key — pass through

    if dropped_keys:
        logger.debug(
            f"  Dropped {len(dropped_keys)} informational private override key(s) "
            f"for run '{spec.run_name}': {dropped_keys}. "
            "These have no config field or MLflow destination.",
            extra={"stage": "orchestrator"},
        )

    # Inject session provenance tags as _tag_* keys.
    # run_training.py strips the "_tag_" prefix and applies them via
    # mlflow.set_tags() after the run context opens — they never reach
    # load_config() or Pydantic validation.
    run_overrides[f"_tag_{_BATCH_ID_TAG}"] = batch_id
    run_overrides[f"_tag_{_DATASET_VERSION_TAG}"] = _DATASET_VERSION_VALUE

    logger.debug(
        f"  Run overrides for '{spec.run_name}': "
        f"config_keys={[k for k in run_overrides if not k.startswith('_')]} | "
        f"tag_keys={[k for k in run_overrides if k.startswith('_tag_')]}",
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
            overrides=run_overrides,
            mlflow_tracking_uri=mlflow_uri,
            force=force,
        )

        elapsed = time.time() - t_start
        f1      = float(result.get("best_val_macro_f1", 0.0))

        gate = (
            "✓ TARGET MET" if f1 >= TARGET_THRESHOLD else
            ("✓ VIABLE"    if f1 >= VIABILITY_THRESHOLD else "✗ BELOW VIABILITY")
        )
        logger.info(
            f"  DONE: {spec.run_name} | "
            f"val_macro_f1={f1:.4f} {gate} | "
            f"val_acc={result.get('best_val_acc', 0.0):.4f} | "
            f"epoch={result.get('best_epoch', -1) + 1} | "
            f"elapsed={elapsed:.0f}s",
            extra={"stage": "orchestrator"},
        )

        record = RunRecord(
            run_name=spec.run_name,
            experiment_group=spec.experiment_group,
            model=spec.model,
            data=spec.data,
            augmentation=spec.augmentation,
            landmark_config=str(result.get("landmark_config", spec.landmark_config)),
            status="completed",
            best_val_macro_f1=f1,
            best_val_acc=float(result.get("best_val_acc", 0.0)),
            best_epoch=int(result.get("best_epoch", -1)),
            total_epochs=int(result.get("total_epochs_trained", 0)),
            mlflow_run_id=str(result.get("mlflow_run_id", "")),
            artifact_dir=str(result.get("artifact_dir", "")),
            model_save_path=str(result.get("model_save_path", "")),
            elapsed_sec=round(elapsed, 1),
            high_risk_f1=result.get("high_risk_class_f1", {}),
            batch_id=batch_id,
        )

        # High-risk class warnings (Issue #9: threshold rather than == 0.0)
        for sign, f1_score in record.high_risk_f1.items():
            if f1_score < HIGH_RISK_F1_ALERT_THRESHOLD:
                logger.warning(
                    f"  HIGH RISK: '{sign}' F1={f1_score:.4f} "
                    f"(< threshold {HIGH_RISK_F1_ALERT_THRESHOLD}) "
                    f"in run '{spec.run_name}'. "
                    "Document in LIMITATIONS.md.",
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
            landmark_config=spec.landmark_config,
            status="failed",
            elapsed_sec=round(time.time() - t_start, 1),
            error_message="KeyboardInterrupt",
            batch_id=batch_id,
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
            landmark_config=spec.landmark_config,
            status="failed",
            elapsed_sec=round(elapsed, 1),
            error_message=f"{type(exc).__name__}: {exc}",
            batch_id=batch_id,
        )

# ---------------------------------------------------------------------------
# Group executor
# ---------------------------------------------------------------------------

def _execute_group(
    group_id:   int,
    group_name: str,
    specs:      List[RunSpec],
    resume:     bool,
    force:      bool,
    mlflow_uri: Optional[str],
    batch_id:   str,
    logger:     Any,
    dry_run:    bool = False,
) -> Tuple[GroupSummary, List[RunRecord]]:
    """
    Execute all runs in a group and return (GroupSummary, [RunRecord]).

    Never raises — all per-run exceptions are caught in _execute_run().
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
                f"landmark={spec.landmark_config} | "
                f"overrides={spec.overrides}",
                extra={"stage": "orchestrator"},
            )
            records.append(RunRecord(
                run_name=spec.run_name,
                experiment_group=spec.experiment_group,
                model=spec.model,
                data=spec.data,
                augmentation=spec.augmentation,
                landmark_config=spec.landmark_config,
                status="skipped",
                batch_id=batch_id,
            ))
        else:
            record = _execute_run(spec, resume, force, mlflow_uri, batch_id, logger)
            records.append(record)

    completed = [r for r in records if r.status == "completed"]
    skipped   = [r for r in records if r.status == "skipped"]
    failed    = [r for r in records if r.status == "failed"]

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
            f"(> {CRITICAL_FAILURE_FRACTION:.0%} threshold). "
            "Dependent downstream groups will be SKIPPED.",
            extra={"stage": "orchestrator"},
        )

    return summary, records


# ---------------------------------------------------------------------------
# Pre-champion dependency gate (Issues #2, #3)
# ---------------------------------------------------------------------------

def _dependency_gate(
    group_1_passed: bool,
    group_2_passed: bool,
    group_3_passed: bool,
    group_4_passed: bool,
    using_fallbacks: bool,
    require_all:     bool,
    logger:          Any,
) -> bool:
    """
    Verify that all dependency groups have passed before running the champion.

    This is a mid-execution gate inserted between Group 4 and the Champion
    run.  It gives the operator a chance to see exactly which groups failed
    before the champion consumes potentially unreliable fallback parameters.

    Parameters
    ----------
    require_all : bool
        If True, the gate is hard — champion is skipped if any group failed.
        If False, champion runs with a prominent WARNING and fallback values.

    Returns
    -------
    bool  True if champion should proceed.
    """
    all_passed = group_1_passed and group_2_passed and group_3_passed and group_4_passed
    status = {
        1: "OK" if group_1_passed else "FAILED",
        2: "OK" if group_2_passed else "FAILED",
        3: "OK" if group_3_passed else "FAILED",
        4: "OK" if group_4_passed else "FAILED",
    }
    logger.info(
        f"\nPre-champion dependency gate | "
        f"G1={status[1]} G2={status[2]} G3={status[3]} G4={status[4]} | "
        f"all_passed={all_passed} | "
        f"using_fallbacks={using_fallbacks} | "
        f"require_all={require_all}",
        extra={"stage": "orchestrator"},
    )

    if not all_passed:
        failed_groups = [g for g, ok in zip([1, 2, 3, 4],
                                            [group_1_passed, group_2_passed,
                                             group_3_passed, group_4_passed])
                         if not ok]
        if require_all:
            logger.error(
                f"DEPENDENCY GATE FAILED: Groups {failed_groups} did not pass. "
                "Champion run will be SKIPPED (--require-all-groups is set). "
                "Fix the failing groups and use --resume to continue.",
                extra={"stage": "orchestrator"},
            )
            return False
        else:
            logger.warning(
                f"DEPENDENCY GATE WARNING: Groups {failed_groups} did not pass. "
                "Champion will run with FALLBACK hyperparameters. "
                "The champion result may not reflect optimised settings. "
                "Use --require-all-groups to prevent this behaviour.",
                extra={"stage": "orchestrator"},
            )

    return True


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
      1. Expected run count (17 total)
      2. run_manifest.json existence for each completed run
      3. SavedModel directory existence for each completed run
      4. Viability threshold (≥ 0.60 for at least one run) — HARD
      5. Target threshold (≥ 0.70 for at least one run) — SOFT WARNING
      6. Champion SavedModel existence
      7. High-risk class F1 < threshold analysis across all runs (Issue #9)
    """
    logger.info(
        "\n" + "=" * 72 + "\nSTAGE 5 FINAL COMPLETION GATE\n" + "=" * 72,
        extra={"stage": "orchestrator"},
    )

    errors:   List[str] = []
    warnings: List[str] = []
    details:  Dict[str, Any] = {}

    completed_records = [r for r in all_records if r.status in ("completed", "skipped")]
    failed_records    = [r for r in all_records if r.status == "failed"]

    # Check 1: total run count
    n_usable = len(completed_records)
    details["total_runs"]    = len(all_records)
    details["completed"]     = len(completed_records)
    details["failed"]        = len(failed_records)
    details["expected_runs"] = EXPECTED_TOTAL_RUNS

    if n_usable < EXPECTED_TOTAL_RUNS:
        msg = (
            f"Expected {EXPECTED_TOTAL_RUNS} completed runs, "
            f"got {n_usable} (+ {len(failed_records)} failed)."
        )
        warnings.append(msg)
        logger.warning(f"  GATE [SOFT]: {msg}", extra={"stage": "orchestrator"})
    else:
        logger.info(
            f"  GATE [PASS]: {n_usable}/{EXPECTED_TOTAL_RUNS} runs completed.",
            extra={"stage": "orchestrator"},
        )

    # Check 2: run_manifest.json
    missing_manifests = [
        r.run_name for r in completed_records
        if not (Path("artifacts") / "experiments" / r.run_name / _RUN_MANIFEST_FILENAME).exists()
    ]
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

    # Check 3: SavedModel directories
    missing_savedmodels = []
    for r in completed_records:
        sp = Path(r.model_save_path) if r.model_save_path else (
            Path("models") / f"{r.run_name}_saved_model"
        )
        if not sp.exists():
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

    # Check 4 & 5: thresholds
    all_f1s = [r.best_val_macro_f1 for r in completed_records]
    best_f1  = max(all_f1s) if all_f1s else 0.0
    best_run = max(completed_records, key=lambda r: r.best_val_macro_f1, default=None)

    details["best_f1"]        = best_f1
    details["best_run_name"]  = best_run.run_name if best_run else ""
    details["viability_met"]  = best_f1 >= VIABILITY_THRESHOLD
    details["target_met"]     = best_f1 >= TARGET_THRESHOLD

    if best_f1 >= VIABILITY_THRESHOLD:
        logger.info(
            f"  GATE [PASS]: Viability met — best val_macro_f1={best_f1:.4f} "
            f">= {VIABILITY_THRESHOLD}",
            extra={"stage": "orchestrator"},
        )
    else:
        msg = (
            f"Minimum viability NOT met: best val_macro_f1={best_f1:.4f} "
            f"< {VIABILITY_THRESHOLD}. Review training curves."
        )
        errors.append(msg)
        logger.error(f"  GATE [FAIL]: {msg}", extra={"stage": "orchestrator"})

    if best_f1 >= TARGET_THRESHOLD:
        logger.info(
            f"  GATE [PASS]: Target met — best val_macro_f1={best_f1:.4f} "
            f">= {TARGET_THRESHOLD}",
            extra={"stage": "orchestrator"},
        )
    else:
        msg = (
            f"Target NOT met: best val_macro_f1={best_f1:.4f} < {TARGET_THRESHOLD}. "
            "Consider: longer seq_len, more hidden_units, hands_only landmarks."
        )
        warnings.append(msg)
        logger.warning(f"  GATE [SOFT]: {msg}", extra={"stage": "orchestrator"})

    # Check 6: champion SavedModel
    champion_path = Path("models") / "champion_model_v1_saved_model"
    champion_exists = champion_path.exists()
    details["champion_model_path"] = str(champion_path)
    details["champion_exists"]     = champion_exists
    if champion_exists:
        logger.info(
            f"  GATE [PASS]: Champion SavedModel exists at {champion_path}",
            extra={"stage": "orchestrator"},
        )
    else:
        msg = f"Champion SavedModel not found at {champion_path}."
        warnings.append(msg)
        logger.warning(f"  GATE [SOFT]: {msg}", extra={"stage": "orchestrator"})

    # Check 7: high-risk class analysis (Issue #9: threshold-based)
    below_threshold_by_sign: Dict[str, List[str]] = {s: [] for s in _HIGH_RISK_SIGNS}
    for r in completed_records:
        for sign in _HIGH_RISK_SIGNS:
            score = r.high_risk_f1.get(sign, 1.0)
            if score < HIGH_RISK_F1_ALERT_THRESHOLD:
                below_threshold_by_sign[sign].append(r.run_name)

    details["high_risk_below_threshold"] = {
        k: v for k, v in below_threshold_by_sign.items() if v
    }
    for sign, runs in below_threshold_by_sign.items():
        if runs:
            logger.warning(
                f"  HIGH RISK: '{sign}' F1 < {HIGH_RISK_F1_ALERT_THRESHOLD} "
                f"in {len(runs)} run(s): {runs}. Document in LIMITATIONS.md.",
                extra={"stage": "orchestrator"},
            )

    gate_passed = len(errors) == 0
    status_str  = "PASSED" if gate_passed else f"FAILED ({len(errors)} error(s))"
    logger.info(
        f"\nFinal gate: {status_str} | "
        f"{len(warnings)} warning(s) | "
        f"best_val_macro_f1={best_f1:.4f} ({details['best_run_name']})",
        extra={"stage": "orchestrator"},
    )

    return {
        "passed":        gate_passed,
        "errors":        errors,
        "warnings":      warnings,
        "champion_path": str(champion_path) if champion_exists else "",
        "best_f1":       best_f1,
        "best_run_name": details.get("best_run_name", ""),
        "gate_details":  details,
        "dry_run":       False,
    }


# ---------------------------------------------------------------------------
# Report writers (Issues #7, #8, #10, #11, #14)
# ---------------------------------------------------------------------------

def _write_execution_report(
    all_records:      List[RunRecord],
    group_summaries:  List[GroupSummary],
    gate_status:      Dict[str, Any],
    t_total_start:    float,
    args:             argparse.Namespace,
    batch_id:         str,
    is_checkpoint:    bool = False,
) -> Path:
    """
    Write reports/experiment_execution_report.json atomically (Issue #14).

    Parameters
    ----------
    is_checkpoint : bool
        If True, this is an intermediate write (group just completed) and
        the gate_status section will be marked as "pending".  Only the
        final call passes the real gate_status.  (Issue #10).
    """
    Path("reports").mkdir(parents=True, exist_ok=True)
    report_path = Path("reports") / "experiment_execution_report.json"

    total_elapsed   = time.time() - t_total_start
    completed       = [r for r in all_records if r.status in ("completed", "skipped")]
    best_record     = max(completed, key=lambda r: r.best_val_macro_f1, default=None)
    effective_gate  = gate_status if not is_checkpoint else {
        "status": "pending — final gate not yet run",
        "is_checkpoint": True,
    }

    report = {
        "stage":              "Stage 5",
        "batch_id":           batch_id,
        "generated_at_utc":   datetime.now(timezone.utc).isoformat(),
        "is_checkpoint":      is_checkpoint,
        "total_elapsed_sec":  round(total_elapsed, 1),
        "total_elapsed_min":  round(total_elapsed / 60, 1),
        "args": {
            "groups":     getattr(args, "groups", None),
            "resume":     getattr(args, "resume", False),
            "dry_run":    getattr(args, "dry_run", False),
            "force":      getattr(args, "force", False),
            "mlflow_uri": getattr(args, "mlflow_tracking_uri", None),
            "require_all_groups": getattr(args, "require_all_groups", False),
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
            "landmark_config":   best_record.landmark_config if best_record else "",
        },
        "group_summaries": [s.to_dict() for s in group_summaries],
        "runs":             [r.to_dict() for r in all_records],
        "gate_status":      effective_gate,
    }

    _atomic_write_json(report_path, report)
    return report_path


def _write_experiment_summary_md(
    all_records:   List[RunRecord],
    gate_status:   Dict[str, Any],
    t_total_start: float,
    batch_id:      str,
    dry_run:       bool = False,
) -> Path:
    """
    Write reports/experiment_summary.md atomically (Issues #7, #8, #11, #14).

    Data-driven conclusions (Issue #7):
        The narrative sections compare actual measured F1 values rather than
        using pre-baked assertions like "Dense should underperform".

    Landmark config populated (Issue #8):
        RunRecord.landmark_config is now tracked and displayed in the table.

    Dry-run watermark (Issue #11):
        If dry_run=True, the summary is clearly watermarked as a plan-only
        document with no experimental data.
    """
    Path("reports").mkdir(parents=True, exist_ok=True)
    md_path = Path("reports") / "experiment_summary.md"

    completed   = [r for r in all_records if r.status in ("completed", "skipped")]
    failed      = [r for r in all_records if r.status == "failed"]
    best_f1     = gate_status.get("best_f1", 0.0)
    best_run    = gate_status.get("best_run_name", "unknown")
    total_elapsed_min = round((time.time() - t_total_start) / 60, 0)

    group_order = {
        "architecture": 1, "augmentation": 2, "sequence_length": 3,
        "landmark_config": 4, "champion": 5, "custom": 6,
    }
    sorted_records = sorted(
        completed,
        key=lambda r: (group_order.get(r.experiment_group, 99), -r.best_val_macro_f1),
    )

    lines: List[str] = []

    # ── Header (Issue #11: dry-run watermark) ─────────────────────────────
    lines.append("# Stage 5 Experiment Summary — WLASL 35-Class Gesture Recognition")
    lines.append("")

    if dry_run:
        lines.append(
            "> ⚠️  **DRY RUN** — This document reflects a planned execution only. "
            "No experiments have been trained.  All metrics are placeholder zeros."
        )
        lines.append("")

    lines.append(
        f"**{len(completed)} experiments tracked across 4 groups + champion.**  "
    )
    lines.append(
        f"Best run: `{best_run}` — val_macro_f1 = **{best_f1:.4f}**  "
    )
    lines.append(
        f"Total elapsed: {total_elapsed_min:.0f} min | "
        f"Session batch_id: `{batch_id}` | "
        f"Completion gate: {'PASSED ✓' if gate_status.get('passed') else 'FAILED ✗'}  "
    )
    lines.append(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    lines.append("")

    # ── Experiment Registry Table (Issue #8: landmark column populated) ───
    lines.append("## Experiment Registry")
    lines.append("")
    lines.append(
        "| Run Name | Group | Architecture | Seq Len | Aug | Landmark | "
        "Val Macro-F1 | Val Acc | Best Epoch | Params |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for r in sorted_records:
        f1_str  = f"{r.best_val_macro_f1:.4f}"
        acc_str = f"{r.best_val_acc:.4f}"
        # Issue #6: best_epoch sentinel is -1; display as 1-indexed
        epoch_display = str(r.best_epoch + 1) if r.best_epoch >= 0 else "—"

        param_count = "—"
        try:
            manifest_path = (
                Path("artifacts") / "experiments" / r.run_name / _RUN_MANIFEST_FILENAME
            )
            if manifest_path.exists():
                with open(manifest_path, encoding="utf-8") as mf:
                    mdata = json.load(mf)
                pc = mdata.get("model_param_count", None)
                if isinstance(pc, int):
                    param_count = f"{pc:,}"
        except Exception:
            pass

        lines.append(
            f"| `{r.run_name}` | {r.experiment_group} | {r.model} | "
            f"{r.data.replace('seq', '')} | {r.augmentation} | "
            f"{r.landmark_config} | "           # Issue #8: real value, not "—"
            f"**{f1_str}** | {acc_str} | {epoch_display} | {param_count} |"
        )

    for r in failed:
        lines.append(
            f"| `{r.run_name}` | {r.experiment_group} | {r.model} | "
            f"{r.data.replace('seq', '')} | {r.augmentation} | "
            f"{r.landmark_config} | ❌ FAILED | — | — | — |"
        )
    lines.append("")

    # ── Group-by-group conclusions (Issue #7: data-driven, not pre-baked) ─
    lines.append("## Group-by-Group Conclusions")
    lines.append("")

    group_meta = {
        "architecture": ("Group 1 — Architecture Comparison",
                         "Compares Dense, LSTM, GRU, BiLSTM on seq60 with no augmentation."),
        "augmentation": ("Group 2 — Augmentation Ablation",
                         "Compares none / temporal / spatial_temporal augmentation on LSTM+seq60."),
        "sequence_length": ("Group 3 — Sequence Length Ablation",
                            "Compares seq20–seq100 with optimal augmentation."),
        "landmark_config": ("Group 4 — Landmark Configuration Ablation",
                            "Compares hands_only / pose_only / full with optimal aug+seq."),
        "champion":       ("Champion Run",
                           "Best architecture + settings, hidden_units=128, up to 100 epochs."),
    }

    for group_key, (group_title, group_desc) in group_meta.items():
        group_records = [r for r in sorted_records if r.experiment_group == group_key]
        if not group_records:
            continue

        # Sort by F1 descending for data-driven narrative
        ranked = sorted(group_records, key=lambda r: r.best_val_macro_f1, reverse=True)
        winner = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        gap = (winner.best_val_macro_f1 - runner_up.best_val_macro_f1) if runner_up else 0.0

        lines.append(f"### {group_title}")
        lines.append(f"*{group_desc}*")
        lines.append("")

        if group_key == "architecture":
            # Issue #7: data-driven — compare actual Dense vs recurrent results
            dense_records = [r for r in group_records if r.model == "dense"]
            recurrent = [r for r in group_records if r.model != "dense"]
            dense_f1 = dense_records[0].best_val_macro_f1 if dense_records else None
            best_recurrent = max(recurrent, key=lambda r: r.best_val_macro_f1, default=None)
            recurrent_f1 = best_recurrent.best_val_macro_f1 if best_recurrent else None

            lines.append(
                f"**Winner:** `{winner.model}` with val_macro_f1 = **{winner.best_val_macro_f1:.4f}**"
            )
            if runner_up:
                lines.append(
                    f"**Runner-up:** `{runner_up.model}` at {runner_up.best_val_macro_f1:.4f} "
                    f"(gap: {gap:.4f})"
                )
            if dense_f1 is not None and recurrent_f1 is not None:
                temporal_gain = recurrent_f1 - dense_f1
                verdict = (
                    "Temporal modelling IS necessary"
                    if temporal_gain > 0.02 else
                    ("Temporal modelling provides marginal benefit"
                     if temporal_gain > 0.0 else
                     "Dense unexpectedly matches or exceeds recurrent models — "
                     "investigate data quality and sequence length")
                )
                lines.append(
                    f"**Temporal modelling gain:** {temporal_gain:+.4f} "
                    f"(best recurrent {recurrent_f1:.4f} vs Dense {dense_f1:.4f}). "
                    f"**Verdict:** {verdict}."
                )
            lines.append(
                f"Best architecture for champion run: **`{winner.model}`**"
            )

        elif group_key == "augmentation":
            lines.append(
                f"**Best augmentation:** `{winner.augmentation}` with val_macro_f1 = "
                f"**{winner.best_val_macro_f1:.4f}**"
            )
            if runner_up:
                lines.append(
                    f"Runner-up: `{runner_up.augmentation}` at "
                    f"{runner_up.best_val_macro_f1:.4f} (gap: {gap:.4f})"
                )
            no_aug = next(
                (r for r in group_records if r.augmentation == "none"), None
            )
            best_aug_record = winner if winner.augmentation != "none" else runner_up
            if no_aug and best_aug_record and best_aug_record.augmentation != "none":
                aug_gain = best_aug_record.best_val_macro_f1 - no_aug.best_val_macro_f1
                verdict = (
                    "Augmentation clearly helps generalisation"
                    if aug_gain > 0.02 else
                    ("Augmentation provides minor benefit"
                     if aug_gain > 0.0 else
                     "Augmentation does not help — no-aug baseline matches or wins")
                )
                lines.append(
                    f"**Augmentation gain:** {aug_gain:+.4f} "
                    f"(`{best_aug_record.augmentation}` vs none). "
                    f"**Verdict:** {verdict}."
                )
            lines.append(
                f"Best augmentation for Groups 3/4/Champion: **`{winner.augmentation}`**"
            )

        elif group_key == "sequence_length":
            seq60_record = next(
                (r for r in group_records if r.data == "seq60"), None
            )
            seq80_record = next(
                (r for r in group_records if r.data == "seq80"), None
            )
            lines.append(
                f"**Best seq_len:** `{winner.data}` with val_macro_f1 = "
                f"**{winner.best_val_macro_f1:.4f}**"
            )
            if runner_up:
                lines.append(
                    f"Runner-up: `{runner_up.data}` at {runner_up.best_val_macro_f1:.4f}"
                )
            if seq80_record and seq60_record:
                seq80_gain = seq80_record.best_val_macro_f1 - seq60_record.best_val_macro_f1
                verdict = (
                    f"seq80 gains {seq80_gain:+.4f} over seq60 — extending sequence "
                    "length meaningfully reduces 97% truncation loss"
                    if seq80_gain > 0.01 else
                    f"seq80 gain is marginal ({seq80_gain:+.4f}) — truncation at seq60 "
                    "is less damaging than Notebook 04 coverage analysis suggested"
                )
                lines.append(f"**Notebook 04 seq80 hypothesis:** {verdict}.")
            if winner.data != "seq80":
                lines.append(
                    f"Note: optimal seq_len={winner.data} rather than the a priori "
                    f"expected seq80 — results are data-driven."
                )
            lines.append(
                f"Best seq_len for Group 4/Champion: **`{winner.data}`**"
            )

        elif group_key == "landmark_config":
            hands_only = next(
                (r for r in group_records if r.landmark_config == "hands_only"), None
            )
            full_record = next(
                (r for r in group_records if r.landmark_config == "full"), None
            )
            lines.append(
                f"**Best landmark config:** `{winner.landmark_config}` "
                f"with val_macro_f1 = **{winner.best_val_macro_f1:.4f}** "
                f"(Fisher ratio: "
                f"{'0.8097' if winner.landmark_config == 'hands_only' else '0.5492' if winner.landmark_config == 'full' else '0.2176'})"
            )
            if hands_only and full_record:
                fisher_gain = hands_only.best_val_macro_f1 - full_record.best_val_macro_f1
                verdict = (
                    f"Fisher ratio advantage (1.47×) translates to "
                    f"{fisher_gain:+.4f} accuracy gain — hands_only is recommended"
                    if fisher_gain > 0.01 else
                    f"Fisher ratio advantage does NOT translate (gain: {fisher_gain:+.4f}) "
                    "— full features are competitive"
                )
                lines.append(f"**Fisher ratio hypothesis:** {verdict}.")
            lines.append(
                f"Best landmark config for Champion: **`{winner.landmark_config}`**"
            )

        elif group_key == "champion":
            lines.append(
                f"Champion val_macro_f1 = **{winner.best_val_macro_f1:.4f}** | "
                f"val_acc = {winner.best_val_acc:.4f}"
            )
            verdict = (
                f"✓ Target ≥ {TARGET_THRESHOLD} MET"
                if winner.best_val_macro_f1 >= TARGET_THRESHOLD else
                (f"✓ Viability ≥ {VIABILITY_THRESHOLD} met, target not met "
                 f"(gap: {TARGET_THRESHOLD - winner.best_val_macro_f1:.4f})")
                if winner.best_val_macro_f1 >= VIABILITY_THRESHOLD else
                f"✗ Below viability threshold {VIABILITY_THRESHOLD} — "
                "inspect training curves for underfitting"
            )
            lines.append(f"**Target assessment:** {verdict}.")

        lines.append("")

    # ── High-risk class analysis (Issue #9: threshold-based) ──────────────
    lines.append("## High-Risk Class Analysis")
    lines.append("")
    lines.append(
        f"Classes with < 5 training clips. F1 < {HIGH_RISK_F1_ALERT_THRESHOLD} "
        f"flagged as failed: "
        "`clothes` (2 clips), `think` (3 clips), `birthday` (4 clips), "
        "`name` (4 clips), `book` (4 clips)."
    )
    lines.append("")

    alert_aggregate: Dict[str, List[str]] = {s: [] for s in _HIGH_RISK_SIGNS}
    for r in completed:
        for sign in _HIGH_RISK_SIGNS:
            score = r.high_risk_f1.get(sign, 1.0)
            if score < HIGH_RISK_F1_ALERT_THRESHOLD:
                alert_aggregate[sign].append(r.run_name)

    any_alert = False
    for sign, run_list in alert_aggregate.items():
        if run_list:
            any_alert = True
            lines.append(
                f"- **`{sign}`**: F1 < {HIGH_RISK_F1_ALERT_THRESHOLD} "
                f"in {len(run_list)} run(s) — {run_list}"
            )
        else:
            lines.append(f"- **`{sign}`**: F1 above threshold in all runs ✓")

    if any_alert:
        lines.append("")
        lines.append(
            "> **Action required**: Update `LIMITATIONS.md` with the above "
            "flagged classes. These signs cannot be reliably recognised and "
            "should be disclosed in the client report."
        )
    lines.append("")

    # ── Gate status ────────────────────────────────────────────────────────
    lines.append("## Stage 5 Completion Gate")
    lines.append("")
    if dry_run:
        lines.append("**Status: DRY RUN — gate not evaluated**")
    else:
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
        "Stage 6 will evaluate `champion_model_v1` on the held-out test set "
        "(51 clips, 7 signers never seen during training). "
        "Key deliverables: per-signer accuracy box plot, SHAP frame importance "
        "heatmaps, latency benchmark (200 inference calls, P50/P95), "
        "confidence calibration reliability diagram, and per-class F1 bar chart "
        "sorted ascending."
    )
    lines.append("")

    _atomic_write_text(md_path, "\n".join(lines))
    return md_path


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_all_experiments.py",
        description=(
            "Orchestrate the full Stage 5 WLASL multi-model experiment matrix "
            "(17 runs across 4 groups + champion).  Executes groups in "
            "dependency order, reads MLflow results filtered by session "
            "batch_id to adaptively select optimal hyperparameters."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Run all experiments (default — generates a new batch_id)
  python pipelines/run_all_experiments.py

  # Run only groups 1 and 2
  python pipelines/run_all_experiments.py --groups 1 2

  # Resume a prior session (reuse batch_id from the execution report)
  python pipelines/run_all_experiments.py --resume --batch-id <uuid>

  # Dry run — print the execution plan without training
  python pipelines/run_all_experiments.py --dry-run

  # Champion only (prior groups must be complete in the same batch_id)
  python pipelines/run_all_experiments.py --champion-only --batch-id <uuid>

  # Hard dependency gate before champion
  python pipelines/run_all_experiments.py --require-all-groups

  # Quick smoke test (5 epochs per run)
  python pipelines/run_all_experiments.py --override training.epochs=5

  # Custom MLflow URI
  python pipelines/run_all_experiments.py --mlflow-tracking-uri http://localhost:5000
        """,
    )

    parser.add_argument(
        "--groups", nargs="+", type=int,
        choices=[1, 2, 3, 4, 5], default=None, metavar="N",
        help=(
            "Run only these group IDs (1=architecture, 2=augmentation, "
            "3=sequence_length, 4=landmark_config, 5=champion). "
            "Default: all groups."
        ),
    )
    parser.add_argument(
        "--batch-id", type=str, default=None, metavar="UUID",
        help=(
            "Session batch_id UUID.  All MLflow runs are tagged with this "
            "value and adaptive selection queries are filtered to it "
            "(Issues #1, #17).  Auto-generated if not supplied.  "
            "Supply the same UUID with --resume to continue a prior session."
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", default=False,
        help=(
            "Skip runs that already have BOTH a run_manifest.json AND a "
            "SavedModel directory (Issue #12).  Always pair with --batch-id "
            "to continue the same session."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print the full execution plan without launching any training runs.",
    )
    parser.add_argument(
        "--champion-only", action="store_true", default=False,
        help=(
            "Run only the champion model.  Requires Groups 1–4 to be complete "
            "in the MLflow tracking store under the given --batch-id."
        ),
    )
    parser.add_argument(
        "--require-all-groups", action="store_true", default=False,
        help=(
            "Hard dependency gate: skip the champion run if any of Groups 1–4 "
            "critically failed (Issue #2).  Without this flag, champion runs "
            "with a WARNING using fallback hyperparameters."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", default=False,
        help="Allow overwriting existing run artefacts.",
    )
    parser.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Dot-notation config override applied to ALL runs. "
            "May be specified multiple times.  "
            "A fingerprint suffix is appended to run_names when overrides "
            "are present so results remain distinguishable (Issue #13)."
        ),
    )
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default=None, metavar="URI",
        help="MLflow tracking URI override for all runs.",
    )
    parser.add_argument(
        "--splits-dir", type=str, default=None, metavar="PATH",
        help="Override for the split CSV directory (default: data/splits).",
    )
    parser.add_argument(
        "--landmarks-dir", type=str, default=None, metavar="PATH",
        help="Override for the landmarks directory (default: data/landmarks).",
    )

    return parser


# ---------------------------------------------------------------------------
# Override parsing
# ---------------------------------------------------------------------------

def _parse_overrides(override_list: List[str]) -> Dict[str, Any]:
    """Parse 'KEY=VALUE' override strings with ast.literal_eval type coercion."""
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
# Adaptive selection helper with full logging (Issue #4)
# ---------------------------------------------------------------------------

def _adaptive_select(
    description:    str,
    experiment_name: str,
    tag_filter:     Dict[str, str],
    batch_id:       str,
    mlflow_uri:     Optional[str],
    logger:         Any,
    fallback:       str,
    result_key:     str = "augmentation",
) -> Tuple[str, bool]:
    """
    Attempt to read the best hyperparameter from MLflow; return fallback on failure.

    All exceptions are logged at WARNING level with full context (Issue #4).

    Parameters
    ----------
    description  : str   Human-readable description for log messages.
    tag_filter   : dict  MLflow tag filter (beyond batch_id).
    batch_id     : str   Session batch_id.
    fallback     : str   Value to return if selection fails.
    result_key   : str   Key to read from _select_best_run() result dict.

    Returns
    -------
    Tuple[str, bool]  (selected_value, is_from_mlflow)
        is_from_mlflow=False means the fallback was used.
    """
    try:
        result = _select_best_run(
            experiment_name=experiment_name,
            tag_filter=tag_filter,
            metric="best_val_macro_f1",
            tracking_uri=mlflow_uri,
            batch_id=batch_id,
            logger=logger,
        )
        value = str(result.get(result_key, "")).strip()
        if not value:
            raise ValueError(
                f"_select_best_run() returned empty '{result_key}' field "
                f"for filter={tag_filter}"
            )
        logger.info(
            f"Adaptive selection [{description}]: "
            f"'{result_key}'='{value}' "
            f"(val_macro_f1={result['best_val_macro_f1']:.4f}) "
            f"from run '{result['run_name']}'",
            extra={"stage": "orchestrator"},
        )
        return value, True
    except Exception as exc:
        # Issue #4: never swallow silently — always log at WARNING minimum
        logger.warning(
            f"Adaptive selection [{description}] FAILED — using fallback "
            f"'{result_key}'='{fallback}'. "
            f"Reason: {type(exc).__name__}: {exc}",
            extra={"stage": "orchestrator"},
        )
        return fallback, False


# ---------------------------------------------------------------------------
# Main orchestration function
# ---------------------------------------------------------------------------

def run_all_experiments(args: argparse.Namespace) -> int:
    """
    Execute the full Stage 5 experiment matrix.

    Returns
    -------
    int  Exit code (EXIT_SUCCESS, EXIT_PARTIAL_FAILURE, EXIT_CRITICAL_FAILURE,
         EXIT_GATE_FAILURE, or EXIT_UNEXPECTED_ERROR).
    """
    from src.utils.logger import get_logger, configure_logging

    configure_logging(level="INFO", log_dir="logs", run_name="run_all_experiments")
    logger = get_logger(__name__)

    t_total_start = time.time()

    # ── Batch ID: either user-supplied (resume) or freshly generated ──────
    # Issues #1, #13, #17: all MLflow runs and adaptive selections are
    # filtered by this UUID so results from prior sessions are never mixed in.
    batch_id: str = args.batch_id if args.batch_id else str(uuid.uuid4())

    # ── Parse overrides and compute fingerprint (Issue #13) ──────────────
    extra_overrides = _parse_overrides(args.override)
    extra_overrides["training.class_weight_balancing"] = True  # always enforced
    override_suffix = _override_fingerprint(extra_overrides)

    mlflow_uri = args.mlflow_tracking_uri

    logger.info(
        "\n" + "=" * 72 + "\n"
        "WLASL Stage 5 — Multi-Model Experiment Orchestrator\n"
        f"batch_id:    {batch_id}\n"
        f"Generated:   {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
        f"overrides:   {extra_overrides}\n"
        f"ovr_suffix:  '{override_suffix}' "
        f"{'(appended to run_names)' if override_suffix else '(none)'}\n"
        + "=" * 72,
        extra={"stage": "orchestrator"},
    )

    if args.resume and not args.batch_id:
        logger.warning(
            "--resume specified without --batch-id.  A new batch_id has been "
            f"generated ({batch_id}) which will NOT match prior session MLflow "
            "runs.  Pass --batch-id <uuid> from the previous execution report "
            "to correctly resume adaptive selection.",
            extra={"stage": "orchestrator"},
        )

    # ── Determine which groups to run ─────────────────────────────────────
    if args.champion_only:
        groups_to_run: Set[int] = {5}
    elif args.groups is not None:
        groups_to_run = set(args.groups)
    else:
        groups_to_run = {1, 2, 3, 4, 5}

    logger.info(
        f"Groups to run: {sorted(groups_to_run)} | "
        f"resume={args.resume} | dry_run={args.dry_run} | "
        f"force={args.force} | require_all={args.require_all_groups}",
        extra={"stage": "orchestrator"},
    )

    # ── Execution state ───────────────────────────────────────────────────
    all_records:     List[RunRecord]    = []
    group_summaries: List[GroupSummary] = []
    exit_code:       int                = EXIT_SUCCESS

    group_1_passed = False
    group_2_passed = False
    group_3_passed = False
    group_4_passed = False

    # Adaptive selection values — safe fallbacks (Issue #15: loudly flagged)
    best_augmentation = "spatial_temporal"
    best_data_config  = "seq60"
    best_landmark     = "hands_only"
    best_model        = "bilstm"

    # Track which selections came from real MLflow data vs fallback
    aug_from_mlflow      = False
    seq_from_mlflow      = False
    landmark_from_mlflow = False
    model_from_mlflow    = False

    # ════════════════════════════════════════════════════════════════════
    # GROUP 1 — Architecture comparison
    # ════════════════════════════════════════════════════════════════════

    if 1 in groups_to_run:
        specs_g1 = _build_group_1_specs(extra_overrides, override_suffix)
        summary_g1, records_g1 = _execute_group(
            group_id=1, group_name="Architecture Comparison",
            specs=specs_g1, resume=args.resume, force=args.force,
            mlflow_uri=mlflow_uri, batch_id=batch_id,
            logger=logger, dry_run=args.dry_run,
        )
        all_records.extend(records_g1)
        group_summaries.append(summary_g1)

        if summary_g1.critical_failure:
            exit_code = EXIT_CRITICAL_FAILURE
            logger.error(
                "Group 1 critical failure. Champion will use fallback "
                f"model='{best_model}' (bilstm). Use --require-all-groups to "
                "prevent champion from running on unvalidated inputs.",
                extra={"stage": "orchestrator"},
            )
        else:
            group_1_passed = True
            if not args.dry_run:
                best_model, model_from_mlflow = _adaptive_select(
                    description="Group 1 best architecture",
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    tag_filter={"experiment_group": "architecture"},
                    batch_id=batch_id,
                    mlflow_uri=mlflow_uri,
                    logger=logger,
                    fallback=best_model,
                    result_key="model_type",
                )

        # Issue #10: checkpoint write — marked is_checkpoint=True
        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args,
            batch_id=batch_id, is_checkpoint=True,
        )

    # ════════════════════════════════════════════════════════════════════
    # GROUP 2 — Augmentation ablation
    # (No dependency on Group 1 — can run concurrently in theory)
    # ════════════════════════════════════════════════════════════════════

    if 2 in groups_to_run:
        specs_g2 = _build_group_2_specs(extra_overrides, override_suffix)
        summary_g2, records_g2 = _execute_group(
            group_id=2, group_name="Augmentation Ablation",
            specs=specs_g2, resume=args.resume, force=args.force,
            mlflow_uri=mlflow_uri, batch_id=batch_id,
            logger=logger, dry_run=args.dry_run,
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
            if not args.dry_run:
                best_augmentation, aug_from_mlflow = _adaptive_select(
                    description="Group 2 best augmentation",
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    tag_filter={"experiment_group": "augmentation"},
                    batch_id=batch_id,
                    mlflow_uri=mlflow_uri,
                    logger=logger,
                    fallback=best_augmentation,
                    result_key="augmentation",
                )

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args,
            batch_id=batch_id, is_checkpoint=True,
        )

    # ════════════════════════════════════════════════════════════════════
    # GROUP 3 — Sequence length ablation (depends on Group 2)
    # ════════════════════════════════════════════════════════════════════

    if 3 in groups_to_run:
        # If Group 2 was not run this session, try to read from MLflow
        if 2 not in groups_to_run and not args.dry_run:
            logger.info(
                "Group 2 not in this session — attempting to read best "
                "augmentation from MLflow (batch_id filter applied).",
                extra={"stage": "orchestrator"},
            )
            best_augmentation, aug_from_mlflow = _adaptive_select(
                description="Group 2 best augmentation (prior session)",
                experiment_name=MLFLOW_EXPERIMENT_NAME,
                tag_filter={"experiment_group": "augmentation"},
                batch_id=batch_id,
                mlflow_uri=mlflow_uri,
                logger=logger,
                fallback=best_augmentation,
                result_key="augmentation",
            )
            if aug_from_mlflow:
                group_2_passed = True

        # Only skip if Group 2 ran this session AND critically failed
        if 2 in groups_to_run and not group_2_passed:
            skip_reason = "Group 2 critically failed — cannot select best augmentation."
            logger.error(
                f"SKIPPING Group 3: {skip_reason}",
                extra={"stage": "orchestrator"},
            )
            group_summaries.append(GroupSummary(
                group_id=3, group_name="Sequence Length Ablation",
                n_planned=6, n_completed=0, n_skipped=0, n_failed=0,
                skipped_reason=skip_reason,
            ))
        else:
            specs_g3 = _build_group_3_specs(
                best_augmentation, extra_overrides, override_suffix
            )
            summary_g3, records_g3 = _execute_group(
                group_id=3,
                group_name=f"Sequence Length Ablation (aug={best_augmentation})",
                specs=specs_g3, resume=args.resume, force=args.force,
                mlflow_uri=mlflow_uri, batch_id=batch_id,
                logger=logger, dry_run=args.dry_run,
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
                if not args.dry_run:
                    raw_seq, seq_from_mlflow = _adaptive_select(
                        description="Group 3 best sequence length",
                        experiment_name=MLFLOW_EXPERIMENT_NAME,
                        tag_filter={"experiment_group": "sequence_length"},
                        batch_id=batch_id,
                        mlflow_uri=mlflow_uri,
                        logger=logger,
                        fallback=best_data_config,
                        result_key="data_config",
                    )
                    # Normalise "seq80", "80", etc.
                    if raw_seq.startswith("seq"):
                        best_data_config = raw_seq
                    elif raw_seq.isdigit():
                        best_data_config = f"seq{raw_seq}"
                    else:
                        best_data_config = raw_seq  # keep as-is, log may warn

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args,
            batch_id=batch_id, is_checkpoint=True,
        )

    # ════════════════════════════════════════════════════════════════════
    # GROUP 4 — Landmark configuration ablation (depends on Groups 2 + 3)
    # ════════════════════════════════════════════════════════════════════

    if 4 in groups_to_run:
        # Read missing prerequisite results from MLflow if not run this session
        if 2 not in groups_to_run and not args.dry_run and not group_2_passed:
            best_augmentation, aug_from_mlflow = _adaptive_select(
                description="Group 2 best aug (prior session, for Group 4)",
                experiment_name=MLFLOW_EXPERIMENT_NAME,
                tag_filter={"experiment_group": "augmentation"},
                batch_id=batch_id, mlflow_uri=mlflow_uri,
                logger=logger, fallback=best_augmentation,
                result_key="augmentation",
            )
            if aug_from_mlflow:
                group_2_passed = True

        if 3 not in groups_to_run and not args.dry_run and not group_3_passed:
            raw_seq, seq_from_mlflow = _adaptive_select(
                description="Group 3 best seq (prior session, for Group 4)",
                experiment_name=MLFLOW_EXPERIMENT_NAME,
                tag_filter={"experiment_group": "sequence_length"},
                batch_id=batch_id, mlflow_uri=mlflow_uri,
                logger=logger, fallback=best_data_config,
                result_key="data_config",
            )
            if seq_from_mlflow:
                best_data_config = (
                    raw_seq if raw_seq.startswith("seq") else
                    f"seq{raw_seq}" if raw_seq.isdigit() else raw_seq
                )
                group_3_passed = True

        deps_met = (
            (group_2_passed or 2 not in groups_to_run)
            and (group_3_passed or 3 not in groups_to_run)
        )

        if not deps_met:
            skip_reason = (
                f"Prerequisite groups failed: "
                f"Group 2={'OK' if group_2_passed else 'FAILED'}, "
                f"Group 3={'OK' if group_3_passed else 'FAILED'}."
            )
            logger.error(
                f"SKIPPING Group 4: {skip_reason}",
                extra={"stage": "orchestrator"},
            )
            group_summaries.append(GroupSummary(
                group_id=4, group_name="Landmark Configuration Ablation",
                n_planned=3, n_completed=0, n_skipped=0, n_failed=0,
                skipped_reason=skip_reason,
            ))
        else:
            specs_g4 = _build_group_4_specs(
                best_augmentation, best_data_config, extra_overrides, override_suffix
            )
            summary_g4, records_g4 = _execute_group(
                group_id=4,
                group_name=(
                    f"Landmark Config Ablation "
                    f"(aug={best_augmentation}, data={best_data_config})"
                ),
                specs=specs_g4, resume=args.resume, force=args.force,
                mlflow_uri=mlflow_uri, batch_id=batch_id,
                logger=logger, dry_run=args.dry_run,
            )
            all_records.extend(records_g4)
            group_summaries.append(summary_g4)

            if not summary_g4.critical_failure:
                group_4_passed = True
                if not args.dry_run:
                    best_landmark, landmark_from_mlflow = _adaptive_select(
                        description="Group 4 best landmark config",
                        experiment_name=MLFLOW_EXPERIMENT_NAME,
                        tag_filter={"experiment_group": "landmark_config"},
                        batch_id=batch_id, mlflow_uri=mlflow_uri,
                        logger=logger, fallback=best_landmark,
                        result_key="landmark_config",
                    )
                    if best_landmark not in ("hands_only", "pose_only", "full"):
                        logger.warning(
                            f"Landmark config '{best_landmark}' from MLflow is not "
                            "a known value. Falling back to 'hands_only'.",
                            extra={"stage": "orchestrator"},
                        )
                        best_landmark = "hands_only"
            else:
                if exit_code == EXIT_SUCCESS:
                    exit_code = EXIT_CRITICAL_FAILURE

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args,
            batch_id=batch_id, is_checkpoint=True,
        )

    # ════════════════════════════════════════════════════════════════════
    # GROUP 5 — Champion run
    # ════════════════════════════════════════════════════════════════════

    if 5 in groups_to_run:
        # For champion-only mode: read all prior groups from MLflow
        if not args.dry_run:
            if 1 not in groups_to_run and not group_1_passed:
                best_model, model_from_mlflow = _adaptive_select(
                    description="Group 1 best arch (prior session, for Champion)",
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    tag_filter={"experiment_group": "architecture"},
                    batch_id=batch_id, mlflow_uri=mlflow_uri,
                    logger=logger, fallback=best_model, result_key="model_type",
                )
                if model_from_mlflow:
                    group_1_passed = True

            if 2 not in groups_to_run and not group_2_passed:
                best_augmentation, aug_from_mlflow = _adaptive_select(
                    description="Group 2 best aug (prior session, for Champion)",
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    tag_filter={"experiment_group": "augmentation"},
                    batch_id=batch_id, mlflow_uri=mlflow_uri,
                    logger=logger, fallback=best_augmentation, result_key="augmentation",
                )
                if aug_from_mlflow:
                    group_2_passed = True

            if 3 not in groups_to_run and not group_3_passed:
                raw_seq, seq_from_mlflow = _adaptive_select(
                    description="Group 3 best seq (prior session, for Champion)",
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    tag_filter={"experiment_group": "sequence_length"},
                    batch_id=batch_id, mlflow_uri=mlflow_uri,
                    logger=logger, fallback=best_data_config, result_key="data_config",
                )
                if seq_from_mlflow:
                    best_data_config = (
                        raw_seq if raw_seq.startswith("seq") else
                        f"seq{raw_seq}" if raw_seq.isdigit() else raw_seq
                    )
                    group_3_passed = True

            if 4 not in groups_to_run and not group_4_passed:
                best_landmark, landmark_from_mlflow = _adaptive_select(
                    description="Group 4 best landmark (prior session, for Champion)",
                    experiment_name=MLFLOW_EXPERIMENT_NAME,
                    tag_filter={"experiment_group": "landmark_config"},
                    batch_id=batch_id, mlflow_uri=mlflow_uri,
                    logger=logger, fallback=best_landmark, result_key="landmark_config",
                )
                if landmark_from_mlflow:
                    group_4_passed = True

        # Determine if any fallbacks are in use (Issue #15)
        using_fallbacks = not (
            model_from_mlflow and aug_from_mlflow
            and seq_from_mlflow and landmark_from_mlflow
        )

        # ── Pre-champion dependency gate (Issues #2, #3) ──────────────────
        champion_should_run = _dependency_gate(
            group_1_passed=group_1_passed,
            group_2_passed=group_2_passed,
            group_3_passed=group_3_passed,
            group_4_passed=group_4_passed,
            using_fallbacks=using_fallbacks,
            require_all=args.require_all_groups,
            logger=logger,
        )

        if not champion_should_run:
            skip_reason = (
                "Pre-champion dependency gate failed (--require-all-groups set). "
                "One or more prerequisite groups critically failed."
            )
            group_summaries.append(GroupSummary(
                group_id=5, group_name="Champion Run",
                n_planned=1, n_completed=0, n_skipped=0, n_failed=0,
                skipped_reason=skip_reason,
            ))
            if exit_code == EXIT_SUCCESS:
                exit_code = EXIT_CRITICAL_FAILURE
        else:
            logger.info(
                f"\nChampion configuration (adaptive selection):\n"
                f"  model        = {best_model}"
                f"{'  [FALLBACK]' if not model_from_mlflow else ''}\n"
                f"  augmentation = {best_augmentation}"
                f"{'  [FALLBACK]' if not aug_from_mlflow else ''}\n"
                f"  data         = {best_data_config}"
                f"{'  [FALLBACK]' if not seq_from_mlflow else ''}\n"
                f"  landmark     = {best_landmark}"
                f"{'  [FALLBACK]' if not landmark_from_mlflow else ''}",
                extra={"stage": "orchestrator"},
            )

            champion_spec = _build_champion_spec(
                best_model=best_model,
                best_augmentation=best_augmentation,
                best_data_config=best_data_config,
                best_lm_config=best_landmark,
                extra_overrides=extra_overrides,
                override_suffix=override_suffix,
                using_fallbacks=using_fallbacks,
            )
            champion_summary, champion_records = _execute_group(
                group_id=5, group_name="Champion Run",
                specs=[champion_spec], resume=args.resume, force=args.force,
                mlflow_uri=mlflow_uri, batch_id=batch_id,
                logger=logger, dry_run=args.dry_run,
            )
            all_records.extend(champion_records)
            group_summaries.append(champion_summary)

        _write_execution_report(
            all_records, group_summaries, {}, t_total_start, args,
            batch_id=batch_id, is_checkpoint=True,
        )

    # ════════════════════════════════════════════════════════════════════
    # Final completion gate + reports
    # ════════════════════════════════════════════════════════════════════

    if not args.dry_run:
        gate_status = _run_completion_gate(all_records, logger)
        if not gate_status["passed"] and exit_code == EXIT_SUCCESS:
            exit_code = EXIT_GATE_FAILURE
    else:
        gate_status = {
            "passed":        True,
            "errors":        [],
            "warnings":      [],
            "best_f1":       0.0,
            "best_run_name": "",
            "champion_path": "",
            "gate_details":  {},
            "dry_run":       True,
        }
        logger.info(
            "DRY RUN — final gate checks skipped.",
            extra={"stage": "orchestrator"},
        )

    # ── Write final reports atomically (Issues #10, #14) ─────────────────
    try:
        report_path = _write_execution_report(
            all_records, group_summaries, gate_status, t_total_start, args,
            batch_id=batch_id, is_checkpoint=False,   # final write with real gate
        )
        logger.info(
            f"Execution report written: {report_path}",
            extra={"stage": "orchestrator"},
        )
    except Exception as exc:
        logger.error(
            f"Failed to write execution report: {type(exc).__name__}: {exc}",
            extra={"stage": "orchestrator"},
        )

    try:
        md_path = _write_experiment_summary_md(
            all_records, gate_status, t_total_start,
            batch_id=batch_id, dry_run=args.dry_run,
        )
        logger.info(
            f"Experiment summary written: {md_path}",
            extra={"stage": "orchestrator"},
        )
    except Exception as exc:
        logger.error(
            f"Failed to write experiment_summary.md: {type(exc).__name__}: {exc}",
            extra={"stage": "orchestrator"},
        )

    # ── Final console summary ─────────────────────────────────────────────
    total_elapsed    = time.time() - t_total_start
    completed_count  = len([r for r in all_records if r.status == "completed"])
    skipped_count    = len([r for r in all_records if r.status == "skipped"])
    failed_count     = len([r for r in all_records if r.status == "failed"])
    best_f1          = gate_status.get("best_f1", 0.0)
    best_run_name    = gate_status.get("best_run_name", "—")

    logger.info(
        "\n" + "=" * 72 + "\n"
        "STAGE 5 COMPLETE\n"
        + "=" * 72 + "\n"
        f"  batch_id:    {batch_id}\n"
        f"  Runs:        completed={completed_count} | "
        f"skipped={skipped_count} | failed={failed_count}\n"
        f"  Best run:    {best_run_name} | val_macro_f1={best_f1:.4f}\n"
        f"  Gate:        {'PASSED ✓' if gate_status.get('passed') else 'FAILED ✗'}\n"
        f"  Total time:  {total_elapsed:.0f}s ({total_elapsed / 60:.1f}min)\n"
        f"  Exit code:   {exit_code}\n"
        + "=" * 72,
        extra={"stage": "orchestrator"},
    )

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
            f"Use --resume --batch-id <uuid> to continue.",
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