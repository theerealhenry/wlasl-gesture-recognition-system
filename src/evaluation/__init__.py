"""
src/evaluation
===============
Stage 6 — Evaluation, Benchmarking, and Interpretability.

Currently implemented:
    metrics.py — core metric primitives (macro-F1, confusion matrix,
                 per-class breakdown, bootstrap CI). See metrics.py's module
                 docstring for full design rationale, including the
                 post-review revision history (signer-aware bootstrap
                 caveat, duplicate sign-name validation, metadata
                 passthrough, and ten other fixes).

Not yet implemented (tracked in the Stage 6 plan, Phase A2-A4):
    benchmark.py        — latency / model-size benchmarking
    calibration.py       — reliability diagrams, confidence-threshold curves
    signer_analysis.py   — per-signer generalisation breakdown

Import convention
------------------
This package uses absolute imports (``from src.evaluation.metrics import
...``) rather than relative imports (``from .metrics import ...``) for
consistency with the rest of this codebase: ``src/models/train.py``,
``src/models/factory.py``, and ``src/models/architectures.py`` all import
across the ``src.*`` namespace absolutely, and ``pipelines/run_training.py``
imports every project module the same way (e.g.
``from src.features.dataset import GestureDataset``). Switching only this
package to relative imports would make it the one inconsistent corner of an
otherwise uniform import style, for a portability benefit this project does
not currently need (it is not distributed as an installable package; it is
run via ``PYTHONPATH``/working-directory convention from the repository
root, same as every other ``src.*`` import in the codebase).

If an IDE's import resolver (e.g. Pylance) underlines these imports, that is
almost always a workspace-root configuration issue rather than a problem
with the import itself — open the editor at the repository root (the
directory containing ``src/``, ``pipelines/``, ``configs/``), or add
``"python.analysis.extraPaths": ["./src"]`` to ``.vscode/settings.json``.
It is not a sign that the absolute-import convention itself needs changing.

Re-exported public API
------------------------
Importing from this package re-exports the ``metrics.py`` public API
directly:

    from src.evaluation import (
        compute_macro_f1,
        compute_accuracy,
        compute_per_class_metrics,
        compute_confusion_matrix_from_predictions,
        bootstrap_macro_f1_ci,
        compute_evaluation_summary,
    )

``__all__`` below is kept explicit (not auto-derived from ``globals()``):
explicit exports are slightly more maintenance overhead when a new function
is added to ``metrics.py`` (it must be added in two places — the import
list and ``__all__``), but they guarantee ``from src.evaluation import *``
never accidentally exposes an internal helper. This trade-off is
intentional and should not be "simplified" away.

Maintenance note for benchmark.py / calibration.py / signer_analysis.py
----------------------------------------------------------------------------
When those modules are implemented, their public functions should be added
to both the import block below and ``__all__`` at the same time, in the same
commit that adds the module — not deferred. A package ``__init__.py`` whose
exports lag behind its on-disk modules is a worse failure mode than one that
exports nothing yet, because it silently breaks the expectation set by this
very docstring (`from src.evaluation import benchmark_inference` failing
with an ``ImportError`` long after `benchmark.py` exists is a confusing
debugging experience for whoever hits it first).
"""

from src.evaluation.metrics import (
    DEFAULT_BOOTSTRAP_CI,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_SEED,
    HIGH_RISK_SIGNS,
    N_CLASSES,
    bootstrap_macro_f1_ci,
    compute_accuracy,
    compute_confusion_matrix,
    compute_confusion_matrix_from_predictions,
    compute_evaluation_summary,
    compute_macro_f1,
    compute_per_class_metrics,
    compute_support_counts,
    get_predictions,
    get_val_predictions,
    rank_classes_by_f1,
)

__all__ = [
    "N_CLASSES",
    "HIGH_RISK_SIGNS",
    "DEFAULT_N_BOOTSTRAP",
    "DEFAULT_BOOTSTRAP_CI",
    "DEFAULT_SEED",
    "get_predictions",
    "get_val_predictions",
    "compute_macro_f1",
    "compute_accuracy",
    "compute_confusion_matrix",
    "compute_confusion_matrix_from_predictions",
    "compute_support_counts",
    "compute_per_class_metrics",
    "rank_classes_by_f1",
    "bootstrap_macro_f1_ci",
    "compute_evaluation_summary",
]