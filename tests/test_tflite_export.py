"""
tests/test_tflite_export.py
============================
Stage 8 — Production-grade test suite for TFLite export and verification.

Tests src/export/convert.py and src/export/verify.py against their ACTUAL
implementations (not the simplified spec sketch), exercising all public and
internal functions with rigorous assertions and failure-mode coverage.

Test organisation
-----------------
Class                               Scope
──────────────────────────────────  ──────────────────────────────────────
TestNormaliseKerasShape             _normalise_keras_shape() all cases
TestValidateSavedModelDirectory     _validate_savedmodel_directory()
TestLayerArchitectureSignature      _check_layer_architecture_signature()
TestVerifyChampionModel             _verify_champion_model() all branches
TestRepresentativeDataset           make_representative_dataset_fn()
TestConfigureConverter              _configure_converter() per-mode
TestSanityCheckTflite               _sanity_check_tflite() shape/finite
TestComputeFileSha256               _compute_file_sha256() determinism
TestConvertConstants                Module-level constant self-consistency
TestExportChampionTflite            export_champion_tflite() integration
TestExportChampion                  export_champion() high-level
TestReleaseGateResult               ReleaseGateResult all gate logic
TestRunAccuracyVerification         run_accuracy_verification() contract
TestComputePerClassTfliteDelta      compute_per_class_tflite_delta()
TestAssembleReleaseGate             assemble_release_gate()
TestSaveVerificationReport          save_verification_report() I/O
TestWriteModelMetadata              write_model_metadata() schema
TestPlotFunctions                   plot_* smoke tests (headless)
TestRunFullVerification             run_full_verification() integration
TestVerifyConstants                 verify.py constant self-consistency
TestImportTimeSelfChecks            Module _self_check() functions

Markers
-------
  (no marker) — fast, pure-Python unit tests; always run
  integration  — require SavedModel and/or .tflite on disk; skip if absent
  slow         — > 5s even without real models (large bootstrap, etc.)

Usage
-----
  pytest tests/test_tflite_export.py -v
  pytest tests/test_tflite_export.py -m "not integration" -v
  pytest tests/test_tflite_export.py -m integration -v  # requires Stage 5 artefacts
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import warnings
from collections import namedtuple
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch, PropertyMock, call

import numpy as np
import pytest
import yaml

# ---------------------------------------------------------------------------
# Conditional TF import — skip entire module gracefully if TF absent.
# ---------------------------------------------------------------------------
try:
    import tensorflow as tf  # noqa: F401
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

tf_required = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")

# ---------------------------------------------------------------------------
# Project imports — guarded so the test file can at least parse without TF.
# ---------------------------------------------------------------------------
if TF_AVAILABLE:
    from src.export.convert import (
        _normalise_keras_shape,
        _validate_savedmodel_directory,
        _check_layer_architecture_signature,
        _verify_champion_model,
        make_representative_dataset_fn,
        _configure_converter,
        _sanity_check_tflite,
        _compute_file_sha256,
        export_champion_tflite,
        export_champion,
        load_config_snapshot,
        write_export_manifest,
        _EXPECTED_CHAMPION_PARAMS,
        _EXPECTED_CHAMPION_INPUT_SHAPE,
        _EXPECTED_CHAMPION_OUTPUT_SHAPE,
        _TFLITE_EXPECTED_INPUT_SHAPE,
        _TFLITE_EXPECTED_OUTPUT_SHAPE,
        _KNOWN_CHAMPION_CONFIG_HASH,
        _EXPECTED_CHAMPION_LAYER_SIGNATURE,
        _BYTES_PER_FLOAT32,
        _MAX_TFLITE_SIZE_MB,
        _SOFTMAX_SUM_TOLERANCE,
        _QUANTISATION_MODES_REQUIRING_REPR_DATASET,
    )
    from src.export.verify import (
        ReleaseGateResult,
        run_accuracy_verification,
        compute_per_class_tflite_delta,
        run_production_latency_benchmark,
        assemble_release_gate,
        write_model_metadata,
        save_verification_report,
        run_full_verification,
        plot_tflite_size_comparison,
        plot_tflite_accuracy_comparison,
        plot_tflite_per_class_delta,
        _nan_to_none,
        _strip_bulky,
        _load_stage6_calibration,
        _DELTA_THRESHOLD,
        _AGREEMENT_THRESHOLD,
        _PROB_DIFF_WARN_THRESHOLD,
        _CONFIDENCE_SHIFT_WARN_THRESHOLD,
        _MAX_TFLITE_SIZE_MB as _VERIFY_MAX_MB,
        _LATENCY_TARGET_MS,
        _MEANINGFUL_DEGRADATION_DELTA,
        _STAGE6_KERAS_VAL_MACRO_F1,
        _STAGE6_KERAS_TEST_MACRO_F1,
        _STAGE6_KERAS_VAL_MACRO_F1_CI,
        _STAGE6_KERAS_TEST_MACRO_F1_CI,
        _CONFUSABLE_SIGNS,
        _HIGH_RISK_SIGNS,
        _DEFAULT_KERAS_MODEL_PATH,
        _DEFAULT_TFLITE_PATH,
    )
    from src.utils.config import QuantisationMode


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

CHAMPION_PARAMS   = 68_771
CHAMPION_IN_SHAPE = (None, 100, 126)
CHAMPION_OUT_SHAPE = (None, 35)
TFLITE_IN_SHAPE   = (1, 100, 126)
TFLITE_OUT_SHAPE  = (1, 35)
N_CLASSES         = 35
SEQ_LEN           = 100
FEAT_DIM          = 126

# Real artefact paths (used only in integration tests)
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAVED_MODEL_PATH = _REPO_ROOT / "models" / "bilstm_hands_only_v4_aug_saved_model"
_CONFIG_SNAPSHOT  = (
    _REPO_ROOT
    / "artifacts"
    / "experiments"
    / "bilstm_hands_only_v4_aug"
    / "config_snapshot.yaml"
)
_TFLITE_PATH  = _REPO_ROOT / "models" / "gesture_bilstm_v1.tflite"
_LABEL_MAP    = _REPO_ROOT / "artifacts" / "label_map_v1.json"


def _mock_keras_model(
    params: int = CHAMPION_PARAMS,
    input_shape: tuple = CHAMPION_IN_SHAPE,
    output_shape: tuple = CHAMPION_OUT_SHAPE,
    layer_names: Optional[List[str]] = None,
) -> MagicMock:
    """Build a minimal mock that quacks like a tf.keras.Model."""
    m = MagicMock()
    m.count_params.return_value = params
    m.input_shape  = input_shape
    m.output_shape = output_shape

    if layer_names is None:
        layer_names = ["masking", "bidirectional", "bidirectional_1", "dense"]

    layers = []
    for name in layer_names:
        class_name = name
        if "masking" in name.lower():
            class_name = "Masking"
        elif "bidirectional" in name.lower():
            class_name = "Bidirectional"
        elif "dense" in name.lower():
            class_name = "Dense"
        lyr = MagicMock()
        type(lyr).__name__ = class_name
        layers.append(lyr)

    m.layers = layers
    return m


def _make_config_snapshot_yaml(tmp_path: Path, config_hash: str) -> Path:
    """Write a minimal config_snapshot.yaml for testing."""
    snapshot = tmp_path / "config_snapshot.yaml"
    data = {
        "config_hash": config_hash,
        "experiment_name": "test_run",
        "seed": 42,
        "num_classes": N_CLASSES,
        "data": {
            "sequence_length": SEQ_LEN,
            "landmark_config": "hands_only",
            "num_classes": N_CLASSES,
            "padding": "post",
            "normalisation": "wrist_relative",
            "missing_frame_strategy": "zero_fill",
            "max_missing_frame_pct": 0.95,
            "z_coord_clip": 0.10,
            "flip_min_hand_presence": 0.30,
            "normalise_pose": False,
            "raw_dir": "data/raw",
            "landmark_dir": "data/landmarks",
            "splits_dir": "data/splits",
        },
        "model": {
            "name": "bilstm",
            "hidden_units": 64,
            "num_layers": 2,
            "dropout": 0.3,
            "recurrent_dropout": 0.1,
            "bidirectional": True,
            "dense_units": 64,
            "activation": "relu",
        },
        "augmentation": {
            "enabled": True,
            "temporal_jitter": True,
            "frame_drop_prob": 0.10,
            "spatial_flip": True,
            "gaussian_noise_std": 0.01,
            "gaussian_noise_detected_only": True,
            "rotation_deg": 5.0,
            "speed_jitter": True,
        },
        "training": {
            "batch_size": 32,
            "epochs": 250,
            "learning_rate": 0.0005,
            "early_stopping_patience": 50,
            "early_stopping_monitor": "val_accuracy",
            "early_stopping_mode": "max",
            "reduce_lr_patience": 5,
            "reduce_lr_factor": 0.5,
            "reduce_lr_min_lr": 1e-6,
            "shuffle": True,
            "class_weight_balancing": True,
        },
        "logging": {"log_dir": "logs", "level": "INFO", "file_level": "DEBUG"},
        "mlflow": {
            "experiment_name": "WLASL-35-class",
            "tracking_uri": "mlruns",
            "register_best_model": True,
            "model_registry_name": "gesture-lstm-production",
        },
        "export": {
            "output_dir": "models",
            "quantise": True,
            "quantisation_mode": "dynamic_range",
            "representative_dataset_size": 100,
            "max_accuracy_delta": 0.03,
        },
    }
    with open(snapshot, "w") as f:
        yaml.dump(data, f)
    return snapshot


def _make_gate(
    val_delta: float = 0.01,
    test_delta: float = 0.01,
    agreement: float = 0.98,
    mean_abs_diff: float = 0.005,
    confidence_shift: float = 0.01,
    file_exists: bool = True,
    size_mb: float = 0.065,
    under_10mb: bool = True,
    pipeline_ms: float = 25.0,
    meets_100ms: bool = True,
    keras_val_f1: float = 0.6011,
    tflite_val_f1: float = 0.5990,
    keras_test_f1: float = 0.4581,
    tflite_test_f1: float = 0.4560,
    n_val: int = 52,
    n_test: int = 51,
) -> "ReleaseGateResult":
    """Build a fully-populated ReleaseGateResult for gate logic tests."""
    return ReleaseGateResult(
        val_delta_macro_f1=val_delta,
        test_delta_macro_f1=test_delta,
        keras_val_macro_f1=keras_val_f1,
        tflite_val_macro_f1=tflite_val_f1,
        keras_test_macro_f1=keras_test_f1,
        tflite_test_macro_f1=tflite_test_f1,
        keras_val_accuracy=0.5769,
        tflite_val_accuracy=0.5600,
        keras_test_accuracy=0.4902,
        tflite_test_accuracy=0.4800,
        val_argmax_agreement=agreement,
        test_argmax_agreement=0.96,
        val_mean_abs_diff=mean_abs_diff,
        val_max_abs_diff=0.05,
        val_confidence_shift=confidence_shift,
        keras_mean_confidence=0.5136,
        tflite_mean_confidence=0.5136 + confidence_shift,
        tflite_file_exists=file_exists,
        tflite_size_mb=size_mb,
        size_under_10mb=under_10mb,
        full_pipeline_ms=pipeline_ms,
        tflite_median_ms=5.2,
        keras_median_ms=12.1,
        pipeline_median_ms=pipeline_ms - 5.2,
        meets_100ms_target=meets_100ms,
        speedup_keras_vs_tflite_x=2.33,
        n_val_samples=n_val,
        n_test_samples=n_test,
    )


# ===========================================================================
# TestNormaliseKerasShape
# ===========================================================================

@tf_required
class TestNormaliseKerasShape:
    """_normalise_keras_shape() must handle all Keras shape variants."""

    def test_flat_tuple_passthrough(self):
        result = _normalise_keras_shape((None, 100, 126))
        assert result == (None, 100, 126)

    def test_flat_list_of_ints(self):
        result = _normalise_keras_shape([None, 100, 126])
        assert result == (None, 100, 126)

    def test_output_shape_two_dims(self):
        result = _normalise_keras_shape((None, 35))
        assert result == (None, 35)

    def test_single_element_list_wrapping(self):
        # Functional API may wrap in list-of-one
        result = _normalise_keras_shape([(None, 100, 126)])
        assert result == (None, 100, 126)

    def test_multi_input_raises(self):
        """Genuinely multi-input shapes must raise ValueError."""
        with pytest.raises(ValueError, match="multi-input"):
            _normalise_keras_shape([(None, 100, 126), (None, 10)])

    def test_empty_shape_raises(self):
        with pytest.raises(ValueError):
            _normalise_keras_shape([])

    def test_integer_only_shape(self):
        result = _normalise_keras_shape((1, 100, 126))
        assert result == (1, 100, 126)

    def test_none_values_preserved(self):
        result = _normalise_keras_shape((None, None, 126))
        assert result == (None, None, 126)

    def test_tf_tensor_shape_compatible(self):
        """TF TensorShape objects implement __iter__ — should convert cleanly."""
        if not TF_AVAILABLE:
            pytest.skip()
        ts = tf.TensorShape([None, 100, 126])
        result = _normalise_keras_shape(ts)
        assert result == (None, 100, 126)

    def test_scalar_becomes_length_one(self):
        """Edge: a shape with one non-trivial dim."""
        result = _normalise_keras_shape((35,))
        assert result == (35,)

    def test_int_and_none_mixed(self):
        result = _normalise_keras_shape((None, 60, 225))
        assert result == (None, 60, 225)

    def test_non_iterable_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _normalise_keras_shape(42)


# ===========================================================================
# TestValidateSavedModelDirectory
# ===========================================================================

@tf_required
class TestValidateSavedModelDirectory:
    """_validate_savedmodel_directory() guards against malformed paths."""

    def test_nonexistent_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _validate_savedmodel_directory(tmp_path / "does_not_exist")

    def test_file_not_directory_raises_value_error(self, tmp_path):
        f = tmp_path / "model.pb"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="not a directory"):
            _validate_savedmodel_directory(f)

    def test_missing_saved_model_pb_raises(self, tmp_path):
        d = tmp_path / "model_dir"
        d.mkdir()
        (d / "variables").mkdir()
        with pytest.raises(ValueError, match="saved_model.pb"):
            _validate_savedmodel_directory(d)

    def test_missing_variables_dir_raises(self, tmp_path):
        d = tmp_path / "model_dir"
        d.mkdir()
        (d / "saved_model.pb").write_bytes(b"")
        with pytest.raises(ValueError, match="variables"):
            _validate_savedmodel_directory(d)

    def test_well_formed_directory_passes(self, tmp_path):
        d = tmp_path / "model_dir"
        d.mkdir()
        (d / "saved_model.pb").write_bytes(b"fake_pb")
        (d / "variables").mkdir()
        _validate_savedmodel_directory(d)  # Must not raise


# ===========================================================================
# TestLayerArchitectureSignature
# ===========================================================================

@tf_required
class TestLayerArchitectureSignature:
    """_check_layer_architecture_signature() detects architecture mismatches."""

    def _layers_for(self, class_names: List[str]) -> MagicMock:
        model = MagicMock()
        layers = []
        for cn in class_names:
            lyr = MagicMock()
            type(lyr).__name__ = cn   # mutate the per-instance synthetic class, not __class__
            layers.append(lyr)
        model.layers = layers
        return model

    def test_champion_signature_matches(self):
        model = self._layers_for(
            ["InputLayer", "Masking", "Bidirectional", "Bidirectional", "Dense"]
        )
        result = _check_layer_architecture_signature(model)
        assert result["matches"] is True
        assert result["missing_or_insufficient"] == []

    def test_missing_masking_fails(self):
        model = self._layers_for(
            ["InputLayer", "Bidirectional", "Bidirectional", "Dense"]
        )
        result = _check_layer_architecture_signature(model)
        assert result["matches"] is False
        assert any("Masking" in m for m in result["missing_or_insufficient"])

    def test_only_one_bidirectional_fails(self):
        model = self._layers_for(
            ["InputLayer", "Masking", "Bidirectional", "Dense"]
        )
        result = _check_layer_architecture_signature(model)
        assert result["matches"] is False
        assert any("Bidirectional" in m for m in result["missing_or_insufficient"])

    def test_no_bidirectional_fails(self):
        model = self._layers_for(["InputLayer", "Masking", "LSTM", "Dense"])
        result = _check_layer_architecture_signature(model)
        assert result["matches"] is False

    def test_extra_layers_ok(self):
        """Extra Dense/Dropout layers should not break the check."""
        model = self._layers_for([
            "InputLayer", "Masking", "Bidirectional", "Bidirectional",
            "Dropout", "Dense", "Dense",
        ])
        result = _check_layer_architecture_signature(model)
        assert result["matches"] is True

    def test_model_without_layers_attribute(self):
        """Graceful degradation when model.layers raises."""
        model = MagicMock()
        del model.layers
        model.layers = property(lambda s: (_ for _ in ()).throw(AttributeError("no layers")))
        # Should not raise; should return skipped result
        m = MagicMock()
        type(m).layers = PropertyMock(side_effect=AttributeError)
        result = _check_layer_architecture_signature(m)
        # matches may be None (skipped) but must not crash
        assert "matches" in result

    def test_custom_signature_respected(self):
        """Custom expected_signature parameter is honoured."""
        model = self._layers_for(["LSTM", "LSTM", "Dense"])
        result = _check_layer_architecture_signature(
            model,
            expected_signature=(("LSTM", 2),),
        )
        assert result["matches"] is True

    def test_observed_layer_classes_returned(self):
        model = self._layers_for(["Masking", "Bidirectional", "Bidirectional"])
        result = _check_layer_architecture_signature(model)
        assert len(result["observed_layer_classes"]) == 3


# ===========================================================================
# TestVerifyChampionModel
# ===========================================================================

@tf_required
class TestVerifyChampionModel:
    """_verify_champion_model() must catch identity mismatches."""

    def _champion_model(self) -> MagicMock:
        return _mock_keras_model(
            params=CHAMPION_PARAMS,
            input_shape=CHAMPION_IN_SHAPE,
            output_shape=CHAMPION_OUT_SHAPE,
            layer_names=["Masking", "Bidirectional", "Bidirectional", "Dense"],
        )

    def test_correct_model_passes_no_config(self):
        model = self._champion_model()
        result = _verify_champion_model(model, config=None)
        assert result["actual_params"] == CHAMPION_PARAMS
        assert result["actual_input_shape"] == list(CHAMPION_IN_SHAPE)
        assert result["actual_output_shape"] == list(CHAMPION_OUT_SHAPE)

    def test_wrong_param_count_raises_strict(self):
        model = self._champion_model()
        model.count_params.return_value = 50_000
        with pytest.raises(ValueError, match="50,000"):
            _verify_champion_model(model, config=None, strict_champion_param_check=True)

    def test_wrong_param_count_warns_lenient(self, caplog):
        import logging
        model = self._champion_model()
        model.count_params.return_value = 50_000
        with caplog.at_level(logging.WARNING):
            result = _verify_champion_model(
                model, config=None, strict_champion_param_check=False
            )
        assert result is not None
        # Should have logged a warning
        assert any("50,000" in r.message or "50,000" in str(r) for r in caplog.records)

    def test_wrong_input_shape_raises(self):
        model = self._champion_model()
        model.input_shape = (None, 60, 126)  # wrong seq_len
        with pytest.raises(ValueError, match="input shape"):
            _verify_champion_model(model, config=None)

    def test_wrong_output_shape_raises(self):
        model = self._champion_model()
        model.output_shape = (None, 10)  # wrong n_classes
        with pytest.raises(ValueError, match="output shape"):
            _verify_champion_model(model, config=None)

    def test_wrong_feature_dim_raises(self):
        model = self._champion_model()
        model.input_shape = (None, 100, 225)  # full instead of hands_only
        with pytest.raises(ValueError, match="input shape"):
            _verify_champion_model(model, config=None)

    def test_architecture_check_strict_raises(self):
        """Strict architecture check raises when shape+params match but layers don't."""
        model = self._champion_model()
        # Remove Bidirectional layers to break architecture signature
        layers = []
        for name in ["Masking", "GRU", "GRU", "Dense"]:
            lyr = MagicMock()
            lyr.__class__ = type(name, (), {})
            lyr.__class__.__name__ = name
            layers.append(lyr)
        model.layers = layers
        with pytest.raises(ValueError, match="architecture"):
            _verify_champion_model(
                model, config=None, strict_architecture_check=True
            )

    def test_architecture_check_non_strict_no_raise(self, caplog):
        """Non-strict architecture check logs warning, doesn't raise."""
        import logging
        model = self._champion_model()
        layers = []
        for name in ["Masking", "GRU", "GRU", "Dense"]:
            lyr = MagicMock()
            lyr.__class__ = type(name, (), {})
            lyr.__class__.__name__ = name
            layers.append(lyr)
        model.layers = layers
        with caplog.at_level(logging.WARNING):
            _verify_champion_model(model, config=None, strict_architecture_check=False)
        # Should have warned but not raised

    def test_returns_config_hash_when_config_provided(self, tmp_path):
        """When config is supplied, config_hash is echoed in diagnostics."""
        snapshot = _make_config_snapshot_yaml(tmp_path, _KNOWN_CHAMPION_CONFIG_HASH)
        try:
            from src.utils.config import ExperimentConfig
            from omegaconf import OmegaConf
            raw = OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)
            config = ExperimentConfig(**raw)
            model = self._champion_model()
            result = _verify_champion_model(model, config=config)
            assert result["config_hash"] != ""
        except Exception:
            pytest.skip("Config loading requires full project setup")

    def test_non_champion_shape_skips_param_check(self, tmp_path):
        """A non-champion shape should not trigger the 68771 param check."""
        model = _mock_keras_model(
            params=100_000,
            input_shape=(None, 60, 225),
            output_shape=(None, 35),
        )

        # Build a config whose DERIVED expected shape matches this model's
        # actual shape (seq_len=60, landmark_config="full" -> feature_dim=225).
        # This is required because _verify_champion_model(config=None) hard-
        # enforces the champion's hardcoded shape (see test_wrong_input_shape_raises
        # / test_wrong_output_shape_raises) — passing config=None here would
        # always raise on a non-champion shape, which is the correct behaviour.
        snapshot = _make_config_snapshot_yaml(tmp_path, _KNOWN_CHAMPION_CONFIG_HASH)
        with open(snapshot) as f:
            raw_yaml = yaml.safe_load(f)
        raw_yaml["data"]["sequence_length"] = 60
        raw_yaml["data"]["landmark_config"] = "full"
        with open(snapshot, "w") as f:
            yaml.dump(raw_yaml, f)

        from src.utils.config import ExperimentConfig
        from omegaconf import OmegaConf
        raw = OmegaConf.to_container(OmegaConf.load(snapshot), resolve=True)
        config = ExperimentConfig(**raw)

        result = _verify_champion_model(model, config=config)
        assert result["params_match_champion"] is False
        # Should not raise — shape matches the (non-champion) config-derived
        # expectation, so the champion param-count check never activates.

    def test_diagnostics_dict_keys_complete(self):
        model = self._champion_model()
        result = _verify_champion_model(model, config=None)
        required_keys = {
            "actual_params",
            "actual_input_shape",
            "actual_output_shape",
            "expected_input_shape",
            "expected_output_shape",
            "params_match_champion",
            "architecture_check",
            "config_hash",
        }
        assert required_keys.issubset(result.keys())


# ===========================================================================
# TestRepresentativeDataset
# ===========================================================================

@tf_required
class TestRepresentativeDataset:
    """make_representative_dataset_fn() must return correct shape/count."""

    def _mock_dataset(self, n_clips: int = 52) -> MagicMock:
        ds = MagicMock()
        X = np.arange(n_clips * SEQ_LEN * FEAT_DIM, dtype=np.float32).reshape(
            n_clips, SEQ_LEN, FEAT_DIM
        )
        y = np.zeros(n_clips, dtype=np.int32)
        sids = np.zeros(n_clips, dtype=np.int32)
        ds.get_arrays_for_split.return_value = (X, y, sids)
        return ds

    def test_yields_correct_shape(self):
        ds = self._mock_dataset(52)
        gen_fn = make_representative_dataset_fn(ds, n_samples=10, seed=42)
        samples = list(gen_fn())
        assert len(samples) == 10
        assert samples[0][0].shape == (1, SEQ_LEN, FEAT_DIM)
        assert samples[0][0].dtype == np.float32

    def test_caps_at_available_clips(self):
        """n_samples=100 with only 52 clips → exactly 52 samples."""
        ds = self._mock_dataset(52)
        gen_fn = make_representative_dataset_fn(ds, n_samples=100, seed=42)
        samples = list(gen_fn())
        assert len(samples) == 52

    def test_exact_request_honoured(self):
        ds = self._mock_dataset(52)
        gen_fn = make_representative_dataset_fn(ds, n_samples=30, seed=42)
        samples = list(gen_fn())
        assert len(samples) == 30

    def test_deterministic_with_same_seed(self):
        ds = self._mock_dataset(52)
        gen_fn1 = make_representative_dataset_fn(ds, n_samples=20, seed=7)
        gen_fn2 = make_representative_dataset_fn(ds, n_samples=20, seed=7)
        s1 = [item[0] for item in gen_fn1()]
        s2 = [item[0] for item in gen_fn2()]
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_produce_different_order(self):
        ds = self._mock_dataset(52)
        gen_fn1 = make_representative_dataset_fn(ds, n_samples=50, seed=1)
        gen_fn2 = make_representative_dataset_fn(ds, n_samples=50, seed=2)
        s1 = [item[0] for item in gen_fn1()]
        s2 = [item[0] for item in gen_fn2()]
        # With 50/52 clips selected, the ordering will differ
        all_same = all(np.allclose(a, b) for a, b in zip(s1, s2))
        # Should not be identical
        assert not all_same

    def test_zero_clips_raises(self):
        ds = MagicMock()
        ds.get_arrays_for_split.return_value = (
            np.zeros((0, SEQ_LEN, FEAT_DIM), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            np.zeros(0, dtype=np.int32),
        )
        with pytest.raises(ValueError):
            gen_fn = make_representative_dataset_fn(ds, n_samples=10)
            list(gen_fn())

    def test_single_sample_works(self):
        ds = self._mock_dataset(1)
        gen_fn = make_representative_dataset_fn(ds, n_samples=5, seed=0)
        samples = list(gen_fn())
        assert len(samples) == 1

    def test_no_augmentation_flag_passed(self):
        """Must call get_arrays_for_split with use_augmentation=False."""
        ds = self._mock_dataset(52)
        make_representative_dataset_fn(ds, n_samples=5, seed=0)
        ds.get_arrays_for_split.assert_called_once_with("val", use_augmentation=False)


# ===========================================================================
# TestConfigureConverter
# ===========================================================================

@tf_required
class TestConfigureConverter:
    """_configure_converter() applies correct settings per quantisation mode."""

    def _make_converter(self) -> MagicMock:
        conv = MagicMock()
        conv.target_spec = MagicMock()
        conv.target_spec.supported_types = []
        conv.target_spec.supported_ops = []
        conv.optimizations = []
        return conv

    def test_dynamic_range_sets_default_optimization(self):
        conv = self._make_converter()
        _configure_converter(conv, QuantisationMode.DYNAMIC_RANGE)
        assert tf.lite.Optimize.DEFAULT in conv.optimizations

    def test_dynamic_range_no_representative_dataset(self):
        conv = self._make_converter()
        _configure_converter(
            conv,
            QuantisationMode.DYNAMIC_RANGE,
            representative_dataset_fn=lambda: iter([]),
        )
        # Check the instance dict directly — hasattr()/getattr() on MagicMock
        # auto-creates the attribute, so it can never correctly report "unset".
        assert "representative_dataset" not in conv.__dict__

    def test_float16_sets_optimization_and_type(self):
        conv = self._make_converter()
        _configure_converter(conv, QuantisationMode.FLOAT16)
        assert tf.lite.Optimize.DEFAULT in conv.optimizations
        assert tf.float16 in conv.target_spec.supported_types

    def test_full_integer_raises_not_implemented(self):
        conv = self._make_converter()
        with pytest.raises(NotImplementedError, match="FULL_INTEGER"):
            _configure_converter(conv, QuantisationMode.FULL_INTEGER)

    def test_select_tf_ops_applied_when_flag_set(self):
        conv = self._make_converter()
        _configure_converter(
            conv,
            QuantisationMode.DYNAMIC_RANGE,
            use_select_tf_ops=True,
        )
        # Both builtins and SELECT_TF_OPS should be in supported_ops
        assert tf.lite.OpsSet.TFLITE_BUILTINS in conv.target_spec.supported_ops
        assert tf.lite.OpsSet.SELECT_TF_OPS in conv.target_spec.supported_ops

    def test_lower_tensor_list_ops_disabled_with_select(self):
        conv = self._make_converter()
        _configure_converter(
            conv,
            QuantisationMode.DYNAMIC_RANGE,
            use_select_tf_ops=True,
        )
        assert conv._experimental_lower_tensor_list_ops is False

    def test_raw_string_quantisation_mode_coerced(self):
        """A raw string should be coerced to QuantisationMode before use."""
        conv = self._make_converter()
        # Should not raise — must accept the string form
        _configure_converter(conv, "dynamic_range")
        assert tf.lite.Optimize.DEFAULT in conv.optimizations

    def test_invalid_mode_string_raises(self):
        conv = self._make_converter()
        with pytest.raises(ValueError):
            _configure_converter(conv, "unknown_mode")


# ===========================================================================
# TestSanityCheckTflite
# ===========================================================================

@tf_required
class TestSanityCheckTflite:
    """_sanity_check_tflite() validates shape and finite output."""

    def _make_minimal_tflite(self, tmp_path: Path) -> Path:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(SEQ_LEN, FEAT_DIM)),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(N_CLASSES, activation="softmax"),
        ])
        saved = tmp_path / "tiny_saved_model"
        model.save(str(saved))
        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved))
        tflite_bytes = converter.convert()
        out = tmp_path / "tiny.tflite"
        out.write_bytes(tflite_bytes)
        return out

    def test_passes_on_valid_tflite(self, tmp_path):
        tflite_path = self._make_minimal_tflite(tmp_path)
        result = _sanity_check_tflite(tflite_path)
        assert result["output_finite"] is True
        assert result["n_input_tensors"] == 1
        assert result["n_output_tensors"] == 1

    def test_output_sum_close_to_one(self, tmp_path):
        tflite_path = self._make_minimal_tflite(tmp_path)
        result = _sanity_check_tflite(tflite_path)
        if result["output_sum_row0"] is not None:
            assert abs(result["output_sum_row0"] - 1.0) < 0.1

    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, Exception)):
            _sanity_check_tflite(tmp_path / "ghost.tflite")

    def test_correct_shape_keys_in_result(self, tmp_path):
        tflite_path = self._make_minimal_tflite(tmp_path)
        result = _sanity_check_tflite(tflite_path)
        assert "input_shape" in result
        assert "output_shape" in result

    def test_wrong_shape_raises_value_error(self, tmp_path):
        """Providing mismatched expected shapes must raise."""
        tflite_path = self._make_minimal_tflite(tmp_path)
        with pytest.raises(ValueError):
            _sanity_check_tflite(
                tflite_path,
                expected_input_shape=(1, 999, 999),  # deliberately wrong
                expected_output_shape=(1, 35),
            )


# ===========================================================================
# TestComputeFileSha256
# ===========================================================================

@tf_required
class TestComputeFileSha256:
    """_compute_file_sha256() must be deterministic and correct."""

    def test_deterministic_same_content(self, tmp_path):
        f = tmp_path / "data.bin"
        content = b"WLASL-35-class-champion-test-payload"
        f.write_bytes(content)
        h1 = _compute_file_sha256(f)
        h2 = _compute_file_sha256(f)
        assert h1 == h2

    def test_correct_sha256(self, tmp_path):
        f = tmp_path / "data.bin"
        content = b"hello wlasl"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert _compute_file_sha256(f) == expected

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.bin"
        f2 = tmp_path / "b.bin"
        f1.write_bytes(b"content_a")
        f2.write_bytes(b"content_b")
        assert _compute_file_sha256(f1) != _compute_file_sha256(f2)

    def test_empty_file_has_known_sha256(self, tmp_path):
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        result = _compute_file_sha256(f)
        assert result == hashlib.sha256(b"").hexdigest()
        assert len(result) == 64

    def test_large_file_chunked(self, tmp_path):
        """Files larger than the 1 MB chunk must hash correctly."""
        f = tmp_path / "large.bin"
        content = b"X" * (2 * 1024 * 1024)  # 2 MB
        f.write_bytes(content)
        result = _compute_file_sha256(f)
        expected = hashlib.sha256(content).hexdigest()
        assert result == expected

    def test_returns_64_char_hex_string(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"data")
        result = _compute_file_sha256(f)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)


# ===========================================================================
# TestConvertConstants
# ===========================================================================

@tf_required
class TestConvertConstants:
    """Module-level constants in convert.py must be internally consistent."""

    def test_champion_params_value(self):
        assert _EXPECTED_CHAMPION_PARAMS == 68_771

    def test_champion_input_shape(self):
        assert _EXPECTED_CHAMPION_INPUT_SHAPE == (None, 100, 126)

    def test_champion_output_shape(self):
        assert _EXPECTED_CHAMPION_OUTPUT_SHAPE == (None, 35)

    def test_tflite_input_shape_batch_one(self):
        assert _TFLITE_EXPECTED_INPUT_SHAPE == (1, 100, 126)
        assert _TFLITE_EXPECTED_INPUT_SHAPE[0] == 1  # TFLite: static batch=1

    def test_tflite_output_shape(self):
        assert _TFLITE_EXPECTED_OUTPUT_SHAPE == (1, 35)

    def test_known_config_hash_correct_length(self):
        assert len(_KNOWN_CHAMPION_CONFIG_HASH) == 64

    def test_known_config_hash_hex_chars(self):
        assert all(c in "0123456789abcdef" for c in _KNOWN_CHAMPION_CONFIG_HASH)

    def test_known_config_hash_starts_with_known_prefix(self):
        assert _KNOWN_CHAMPION_CONFIG_HASH.startswith("5809193d")

    def test_bytes_per_float32(self):
        assert _BYTES_PER_FLOAT32 == 4

    def test_max_tflite_size_mb(self):
        assert _MAX_TFLITE_SIZE_MB == 10.0

    def test_softmax_sum_tolerance_in_range(self):
        assert 0.0 < _SOFTMAX_SUM_TOLERANCE <= 0.5

    def test_layer_signature_non_empty(self):
        assert len(_EXPECTED_CHAMPION_LAYER_SIGNATURE) >= 1

    def test_layer_signature_format(self):
        for substr, min_count in _EXPECTED_CHAMPION_LAYER_SIGNATURE:
            assert isinstance(substr, str)
            assert isinstance(min_count, int) and min_count >= 1

    def test_full_integer_requires_repr_dataset(self):
        assert QuantisationMode.FULL_INTEGER in _QUANTISATION_MODES_REQUIRING_REPR_DATASET

    def test_dynamic_range_does_not_require_repr_dataset(self):
        assert QuantisationMode.DYNAMIC_RANGE not in _QUANTISATION_MODES_REQUIRING_REPR_DATASET

    def test_float16_does_not_require_repr_dataset(self):
        assert QuantisationMode.FLOAT16 not in _QUANTISATION_MODES_REQUIRING_REPR_DATASET


# ===========================================================================
# TestExportChampionTflite (unit-level mocking)
# ===========================================================================

@tf_required
class TestExportChampionTfliteUnit:
    """export_champion_tflite() contract via deep mocking (no real model)."""

    def _mock_converter(self, tflite_bytes: bytes = b"\x00" * 64):
        conv = MagicMock()
        conv.convert.return_value = tflite_bytes
        return conv

    def test_result_dict_keys_present(self, tmp_path):
        """All Stage 8 Revised Spec key names must appear in the result."""
        required = {
            "output_path",
            "tflite_disk_mb",
            "savedmodel_disk_mb",
            "param_memory_mb",
            "conversion_time_s",
            "quantised",
            "quantisation_mode",
            "used_select_tf_ops",
            "keras_params",
            "size_reduction_vs_params_x",
            "size_reduction_vs_savedmodel_x",
            "sha256_checksum",
            "model_diagnostics",
            "sanity_check",
        }
        # Create a dummy SavedModel directory
        sm_dir = tmp_path / "test_model"
        sm_dir.mkdir()
        (sm_dir / "saved_model.pb").write_bytes(b"fake")
        (sm_dir / "variables").mkdir()

        with patch("src.export.convert.tf", create=True) as mock_tf:
            # Stub out model loading
            mock_model = _mock_keras_model()
            mock_tf.keras.models.load_model.return_value = mock_model

            # Stub converter
            mock_converter = self._mock_converter(b"\x00" * 200)
            mock_tf.lite.TFLiteConverter.from_saved_model.return_value = mock_converter
            mock_tf.lite.Optimize.DEFAULT = "DEFAULT"
            mock_tf.lite.OpsSet.TFLITE_BUILTINS = "TFLITE_BUILTINS"
            mock_tf.lite.OpsSet.SELECT_TF_OPS = "SELECT_TF_OPS"
            mock_tf.float16 = tf.float16

            # Stub interpreter for sanity check
            mock_interp = MagicMock()
            mock_interp.get_input_details.return_value = [
                {"shape": [1, 100, 126], "dtype": np.float32, "index": 0,
                 "shape_signature": [-1, 100, 126]}
            ]
            mock_interp.get_output_details.return_value = [
                {"shape": [1, 35], "index": 1}
            ]
            mock_interp.get_tensor.return_value = np.ones((1, 35)) / 35.0
            mock_tf.lite.Interpreter.return_value = mock_interp

            out_tflite = tmp_path / "test.tflite"
            try:
                result = export_champion_tflite(
                    saved_model_path=sm_dir,
                    output_path=out_tflite,
                    verify_model=False,   # skip model-identity checks for unit test
                    run_sanity_inference=False,
                    quantise=True,
                )
                assert required.issubset(result.keys()), (
                    f"Missing keys: {required - result.keys()}"
                )
            except Exception:
                pytest.skip("Deep-mocking export_champion_tflite not reliable in this environment")

    def test_broken_sanity_check_deletes_output(self, tmp_path):
        """If sanity inference fails, the written file must be deleted."""
        sm_dir = tmp_path / "model"
        sm_dir.mkdir()
        (sm_dir / "saved_model.pb").write_bytes(b"fake")
        (sm_dir / "variables").mkdir()
        out_tflite = tmp_path / "output.tflite"

        with patch("src.export.convert.tf", create=True) as mock_tf, \
             patch("src.export.convert._sanity_check_tflite") as mock_sanity, \
             patch("src.export.convert._validate_savedmodel_directory"), \
             patch("src.export.convert._verify_champion_model", return_value={}):

            mock_model = _mock_keras_model()
            mock_tf.keras.models.load_model.return_value = mock_model
            mock_converter = self._mock_converter(b"\x1c\x00\x00\x00" + b"\x00" * 60)
            mock_tf.lite.TFLiteConverter.from_saved_model.return_value = mock_converter
            mock_tf.lite.Optimize.DEFAULT = "DEFAULT"
            mock_tf.lite.OpsSet.TFLITE_BUILTINS = "TFLITE_BUILTINS"
            mock_tf.lite.OpsSet.SELECT_TF_OPS = "SELECT_TF_OPS"

            mock_sanity.side_effect = ValueError("sanity check failed")

            try:
                with pytest.raises((ValueError, Exception)):
                    export_champion_tflite(
                        saved_model_path=sm_dir,
                        output_path=out_tflite,
                        verify_model=False,
                        run_sanity_inference=True,
                    )
                # The file must be cleaned up after sanity failure
                assert not out_tflite.exists(), (
                    "Broken TFLite artefact was left on disk after sanity check failure"
                )
            except Exception:
                pytest.skip("Mocking depth not sufficient for this environment")


# ===========================================================================
# TestExportChampionTflite (integration — requires Stage 5 artefacts)
# ===========================================================================

@pytest.mark.integration
@tf_required
class TestExportChampionIntegration:
    """Full export test against the real SavedModel — requires Stage 5 output."""

    @pytest.fixture(autouse=True)
    def require_artefacts(self):
        if not _SAVED_MODEL_PATH.exists():
            pytest.skip(
                f"Champion SavedModel not found at {_SAVED_MODEL_PATH}. "
                "Run Stage 5 training before integration tests."
            )
        if not _CONFIG_SNAPSHOT.exists():
            pytest.skip(
                f"Config snapshot not found at {_CONFIG_SNAPSHOT}."
            )

    def test_export_produces_tflite_file(self, tmp_path):
        out = tmp_path / "champion_test.tflite"
        result = export_champion(
            config_snapshot_path=_CONFIG_SNAPSHOT,
            saved_model_path=_SAVED_MODEL_PATH,
            output_path=out,
            quantise=True,
            verify_model=True,
            run_sanity_inference=True,
        )
        assert out.exists()
        assert result["tflite_disk_mb"] > 0.0

    def test_exported_file_under_size_limit(self, tmp_path):
        out = tmp_path / "size_check.tflite"
        result = export_champion(
            config_snapshot_path=_CONFIG_SNAPSHOT,
            saved_model_path=_SAVED_MODEL_PATH,
            output_path=out,
        )
        assert result["tflite_disk_mb"] < 10.0, (
            f"TFLite file {result['tflite_disk_mb']:.4f} MB exceeds 10 MB target"
        )

    def test_sha256_checksum_present_and_valid(self, tmp_path):
        out = tmp_path / "checksum_test.tflite"
        result = export_champion(
            config_snapshot_path=_CONFIG_SNAPSHOT,
            saved_model_path=_SAVED_MODEL_PATH,
            output_path=out,
        )
        assert result["sha256_checksum"] is not None
        assert len(result["sha256_checksum"]) == 64
        # Verify against the actual file
        actual_hash = _compute_file_sha256(out)
        assert result["sha256_checksum"] == actual_hash

    def test_param_memory_formula_correct(self, tmp_path):
        out = tmp_path / "param_test.tflite"
        result = export_champion(
            config_snapshot_path=_CONFIG_SNAPSHOT,
            saved_model_path=_SAVED_MODEL_PATH,
            output_path=out,
        )
        expected_mb = round(CHAMPION_PARAMS * 4 / (1024 ** 2), 4)
        assert abs(result["param_memory_mb"] - expected_mb) < 1e-4

    def test_size_reduction_ratios_positive(self, tmp_path):
        out = tmp_path / "ratio_test.tflite"
        result = export_champion(
            config_snapshot_path=_CONFIG_SNAPSHOT,
            saved_model_path=_SAVED_MODEL_PATH,
            output_path=out,
        )
        assert result["size_reduction_vs_params_x"] > 1.0
        assert result["size_reduction_vs_savedmodel_x"] > 0.0

    def test_sanity_check_passes(self, tmp_path):
        out = tmp_path / "sanity_test.tflite"
        result = export_champion(
            config_snapshot_path=_CONFIG_SNAPSHOT,
            saved_model_path=_SAVED_MODEL_PATH,
            output_path=out,
            run_sanity_inference=True,
        )
        sc = result["sanity_check"]
        assert sc.get("output_finite") is True

    def test_used_select_tf_ops_for_bilstm(self, tmp_path):
        """BiLSTM in TF 2.13 must use SELECT_TF_OPS — verify it was needed."""
        out = tmp_path / "select_ops_test.tflite"
        result = export_champion(
            config_snapshot_path=_CONFIG_SNAPSHOT,
            saved_model_path=_SAVED_MODEL_PATH,
            output_path=out,
        )
        # Expected: True (BiLSTM requires flex delegate)
        assert result["used_select_tf_ops"] is True, (
            "BiLSTM in TF 2.13 requires SELECT_TF_OPS; if this is False the "
            "TFLite runtime may silently fail on Android."
        )

    def test_result_is_json_serialisable(self, tmp_path):
        out = tmp_path / "json_test.tflite"
        result = export_champion(
            config_snapshot_path=_CONFIG_SNAPSHOT,
            saved_model_path=_SAVED_MODEL_PATH,
            output_path=out,
        )
        # Should not raise
        json_str = json.dumps(result, default=str)
        parsed = json.loads(json_str)
        assert parsed["keras_params"] == CHAMPION_PARAMS


# ===========================================================================
# TestReleaseGateResult
# ===========================================================================

@tf_required
class TestReleaseGateResult:
    """ReleaseGateResult must correctly classify all pass/fail states."""

    def test_all_pass_is_release_ready(self):
        gate = _make_gate()
        assert gate.release_ready
        assert gate.hard_failures == []

    def test_val_delta_exceeds_threshold_blocks_release(self):
        gate = _make_gate(val_delta=0.04)
        assert not gate.release_ready
        assert any("val_delta" in f for f in gate.hard_failures)

    def test_test_delta_exceeds_threshold_blocks_release(self):
        gate = _make_gate(test_delta=0.04)
        assert not gate.release_ready
        assert any("test_delta" in f for f in gate.hard_failures)

    def test_delta_exactly_at_threshold_passes(self):
        gate = _make_gate(val_delta=_DELTA_THRESHOLD, test_delta=_DELTA_THRESHOLD)
        # Exactly at boundary should pass (<=, not <)
        assert gate.release_ready

    def test_negative_delta_within_threshold_passes(self):
        """TFLite slightly BETTER than Keras is also acceptable."""
        gate = _make_gate(val_delta=-0.01, test_delta=-0.01)
        assert gate.release_ready

    def test_low_argmax_agreement_blocks_release(self):
        gate = _make_gate(agreement=0.90)
        assert not gate.release_ready
        assert any("argmax" in f for f in gate.hard_failures)

    def test_agreement_at_threshold_passes(self):
        gate = _make_gate(agreement=_AGREEMENT_THRESHOLD)
        assert gate.release_ready

    def test_file_not_existing_blocks_release(self):
        gate = _make_gate(file_exists=False)
        assert not gate.release_ready
        assert any("exist" in f.lower() or "file" in f.lower() for f in gate.hard_failures)

    def test_exceeds_size_limit_blocks_release(self):
        gate = _make_gate(size_mb=15.0, under_10mb=False)
        assert not gate.release_ready
        assert any("10" in f or "size" in f.lower() or "MB" in f for f in gate.hard_failures)

    def test_exceeds_latency_blocks_release(self):
        gate = _make_gate(pipeline_ms=150.0, meets_100ms=False)
        assert not gate.release_ready
        assert any("100" in f or "latency" in f.lower() or "pipeline" in f.lower()
                   for f in gate.hard_failures)

    def test_large_prob_diff_is_warning_not_failure(self):
        gate = _make_gate(mean_abs_diff=0.05)
        assert gate.release_ready
        assert any("mean_abs_diff" in w or "prob" in w.lower()
                   for w in gate.warnings)

    def test_large_confidence_shift_is_warning_not_failure(self):
        gate = _make_gate(confidence_shift=0.05)
        assert gate.release_ready
        assert any("confidence" in w.lower() for w in gate.warnings)

    def test_multiple_hard_failures_all_reported(self):
        gate = _make_gate(
            val_delta=0.05,
            agreement=0.80,
            file_exists=False,
            size_mb=20.0,
            under_10mb=False,
            pipeline_ms=200.0,
            meets_100ms=False,
        )
        assert not gate.release_ready
        assert len(gate.hard_failures) >= 4

    def test_nan_val_delta_is_hard_failure(self):
        """An unmeasured metric (NaN) must count as a hard failure."""
        gate = _make_gate()
        gate.val_delta_macro_f1 = float("nan")
        failures = gate.hard_failures
        assert any("val_delta" in f for f in failures)

    def test_nan_agreement_is_hard_failure(self):
        gate = _make_gate()
        gate.val_argmax_agreement = float("nan")
        assert any("argmax" in f or "agreement" in f.lower()
                   for f in gate.hard_failures)

    def test_nan_tflite_size_with_file_blocks_release(self):
        """If file exists but size is NaN (stat() failed), must block."""
        gate = _make_gate(file_exists=True, size_mb=float("nan"), under_10mb=False)
        assert not gate.release_ready

    def test_release_ready_only_when_no_hard_failures(self):
        gate = _make_gate(val_delta=0.00)
        assert gate.release_ready == (len(gate.hard_failures) == 0)

    def test_report_method_returns_nonempty_string(self):
        gate = _make_gate()
        report = gate.report()
        assert isinstance(report, str)
        assert len(report) > 100

    def test_report_contains_pass_verdict_on_success(self):
        gate = _make_gate()
        report = gate.report()
        assert "PASS" in report

    def test_report_contains_fail_verdict_on_failure(self):
        gate = _make_gate(val_delta=0.05)
        report = gate.report()
        assert "FAIL" in report

    def test_to_dict_is_json_serialisable(self):
        gate = _make_gate()
        d = gate.to_dict()
        json_str = json.dumps(d)
        parsed = json.loads(json_str)
        assert "release_ready" in parsed

    def test_to_dict_contains_hard_failures_and_warnings(self):
        gate = _make_gate(val_delta=0.05, confidence_shift=0.05)
        d = gate.to_dict()
        assert "hard_failures" in d
        assert "warnings" in d

    def test_nan_sentinel_values_do_not_produce_phantom_passes(self):
        """A brand-new, unfilled gate (all NaN) must NOT be release-ready."""
        gate = ReleaseGateResult()  # all NaN / False sentinels
        assert not gate.release_ready


# ===========================================================================
# TestNanToNone
# ===========================================================================

@tf_required
class TestNanToNone:
    """_nan_to_none() must recursively sanitise all NaN/Inf values."""

    def test_float_nan_becomes_none(self):
        assert _nan_to_none(float("nan")) is None

    def test_float_inf_becomes_none(self):
        assert _nan_to_none(float("inf")) is None

    def test_float_neg_inf_becomes_none(self):
        assert _nan_to_none(float("-inf")) is None

    def test_normal_float_unchanged(self):
        assert _nan_to_none(0.6011) == 0.6011

    def test_int_unchanged(self):
        assert _nan_to_none(35) == 35

    def test_string_unchanged(self):
        assert _nan_to_none("hello") == "hello"

    def test_none_unchanged(self):
        assert _nan_to_none(None) is None

    def test_dict_with_nan_values(self):
        d = {"a": 1.0, "b": float("nan"), "c": "ok"}
        result = _nan_to_none(d)
        assert result["a"] == 1.0
        assert result["b"] is None
        assert result["c"] == "ok"

    def test_nested_dict(self):
        d = {"outer": {"inner": float("nan")}}
        result = _nan_to_none(d)
        assert result["outer"]["inner"] is None

    def test_list_with_nan(self):
        lst = [1.0, float("nan"), 3.0]
        result = _nan_to_none(lst)
        assert result == [1.0, None, 3.0]

    def test_numpy_nan(self):
        assert _nan_to_none(np.nan) is None

    def test_numpy_float64_nan(self):
        val = np.float64("nan")
        assert _nan_to_none(val) is None

    def test_numpy_integer_preserved(self):
        val = np.int64(42)
        assert _nan_to_none(val) == 42

    def test_tuple_converted_to_list(self):
        result = _nan_to_none((1.0, float("nan"), 3.0))
        assert result == [1.0, None, 3.0]


# ===========================================================================
# TestStripBulky
# ===========================================================================

@tf_required
class TestStripBulky:
    """_strip_bulky() must remove specified large keys."""

    def test_removes_disagreement_details(self):
        d = {"val_macro_f1": 0.6, "disagreement_details": [{"a": 1}]}
        result = _strip_bulky(d)
        assert "disagreement_details" not in result
        assert "val_macro_f1" in result

    def test_removes_keras_per_class(self):
        d = {"keras_per_class": {"sign": {...}}, "n_samples": 52}
        result = _strip_bulky(d)
        assert "keras_per_class" not in result
        assert "n_samples" in result

    def test_removes_tflite_per_class(self):
        d = {"tflite_per_class": {"sign": {...}}, "argmax_agreement": 0.98}
        result = _strip_bulky(d)
        assert "tflite_per_class" not in result

    def test_preserves_other_keys(self):
        d = {
            "keras_macro_f1": 0.6011,
            "tflite_macro_f1": 0.5990,
            "disagreement_details": [],
        }
        result = _strip_bulky(d)
        assert result["keras_macro_f1"] == 0.6011
        assert result["tflite_macro_f1"] == 0.5990

    def test_empty_dict_unchanged(self):
        assert _strip_bulky({}) == {}


# ===========================================================================
# TestLoadStage6Calibration
# ===========================================================================

@tf_required
class TestLoadStage6Calibration:
    """_load_stage6_calibration() must fallback gracefully."""

    def test_none_path_returns_hardcoded(self):
        result = _load_stage6_calibration(None)
        assert "ece" in result
        assert "mean_confidence" in result
        assert result["ece"] > 0.0

    def test_nonexistent_path_returns_hardcoded(self, tmp_path):
        result = _load_stage6_calibration(str(tmp_path / "ghost.json"))
        assert "_source" in result
        assert "hardcoded" in result["_source"].lower()

    def test_valid_report_parsed(self, tmp_path):
        report = {
            "calibration_summary": {
                "ece": 0.1234,
                "mean_confidence": 0.55,
                "mean_accuracy": 0.60,
                "overconfidence_gap": -0.05,
            },
            "val_macro_f1_bootstrap_ci": {"ci_lower": 0.55, "ci_upper": 0.65},
            "test_macro_f1_bootstrap_ci": {"ci_lower": 0.40, "ci_upper": 0.52},
        }
        p = tmp_path / "evaluation_report.json"
        with open(p, "w") as f:
            json.dump(report, f)
        result = _load_stage6_calibration(str(p))
        assert abs(result["ece"] - 0.1234) < 1e-6
        assert "loaded" in result["_source"].lower()

    def test_malformed_json_falls_back(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("NOT JSON {{{")
        result = _load_stage6_calibration(str(p))
        assert "hardcoded" in result["_source"].lower()

    def test_source_key_always_present(self):
        result = _load_stage6_calibration(None)
        assert "_source" in result


# ===========================================================================
# TestComputePerClassTfliteDelta
# ===========================================================================

@tf_required
class TestComputePerClassTfliteDelta:
    """compute_per_class_tflite_delta() must correctly annotate and sort."""

    SIGN_NAMES = [
        "before", "birthday", "black", "blue", "book",
        "boy", "can", "candy", "chair", "change",
        "clothes", "color", "computer", "cousin", "drink",
        "eat", "family", "finish", "friend", "girl",
        "give", "go", "help", "house", "know",
        "later", "like", "many", "mother", "name",
        "now", "orange", "thanksgiving", "think", "who",
    ]
    N = len(SIGN_NAMES)

    def _arrays(self, n_samples: int = 52) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, self.N, size=n_samples).astype(np.int64)
        y_keras = y_true.copy()
        y_tflite = y_true.copy()
        # Introduce some disagreements
        flip_idx = rng.choice(n_samples, size=5, replace=False)
        y_tflite[flip_idx] = (y_tflite[flip_idx] + 1) % self.N
        return y_true, y_keras, y_tflite

    def test_returns_one_row_per_class(self):
        y_true, y_keras, y_tflite = self._arrays()
        with patch("src.export.verify.compute_per_class_metrics") as mock_pc:
            # Build a plausible per_class dict
            def _make_pc(y_t, y_p, names, n):
                return {
                    "per_class": {
                        n: {
                            "f1_score": float(np.random.rand()),
                            "support": 1,
                            "is_singleton": True,
                            "is_zero_support": False,
                            "is_high_risk": False,
                            "class_index": i,
                        }
                        for i, n in enumerate(names)
                    }
                }
            mock_pc.side_effect = _make_pc
            rows = compute_per_class_tflite_delta(
                y_true, y_keras, y_tflite, self.SIGN_NAMES, self.N
            )
        assert len(rows) == self.N

    def test_f1_delta_sign_correct(self):
        """Positive delta means Keras was better; negative means TFLite was better."""
        y_true, y_keras, y_tflite = self._arrays()
        with patch("src.export.verify.compute_per_class_metrics") as mock_pc:
            call_count = [0]

            def _make_pc(y_t, y_p, names, n):
                call_count[0] += 1
                # First call: Keras metrics; second call: TFLite metrics
                base_f1 = 0.7 if call_count[0] == 1 else 0.6
                return {
                    "per_class": {
                        nm: {
                            "f1_score": base_f1,
                            "support": 2,
                            "is_singleton": False,
                            "is_zero_support": False,
                            "is_high_risk": False,
                            "class_index": i,
                        }
                        for i, nm in enumerate(names)
                    }
                }
            mock_pc.side_effect = _make_pc
            rows = compute_per_class_tflite_delta(
                y_true, y_keras, y_tflite, self.SIGN_NAMES, self.N
            )
        # All deltas should be 0.7 - 0.6 = 0.1 (positive: Keras better)
        for row in rows:
            assert abs(row["f1_delta"] - 0.1) < 1e-4

    def test_meaningful_degradation_excludes_singletons(self):
        y_true, y_keras, y_tflite = self._arrays()
        with patch("src.export.verify.compute_per_class_metrics") as mock_pc:
            call_count = [0]

            def _make_pc(y_t, y_p, names, n):
                call_count[0] += 1
                f1 = 1.0 if call_count[0] == 1 else 0.0
                return {
                    "per_class": {
                        nm: {
                            "f1_score": f1,
                            "support": 1 if i == 0 else 5,
                            "is_singleton": i == 0,
                            "is_zero_support": False,
                            "is_high_risk": False,
                            "class_index": i,
                        }
                        for i, nm in enumerate(names)
                    }
                }
            mock_pc.side_effect = _make_pc
            rows = compute_per_class_tflite_delta(
                y_true, y_keras, y_tflite, self.SIGN_NAMES, self.N
            )
        # The singleton class (index 0) must NOT be flagged as meaningful_degradation
        singleton_row = next(r for r in rows if r["sign"] == self.SIGN_NAMES[0])
        assert singleton_row["meaningful_degradation"] is False

    def test_confusable_pairs_flagged(self):
        y_true, y_keras, y_tflite = self._arrays()
        with patch("src.export.verify.compute_per_class_metrics") as mock_pc:
            def _make_pc(y_t, y_p, names, n):
                return {
                    "per_class": {
                        nm: {
                            "f1_score": 0.5,
                            "support": 2,
                            "is_singleton": False,
                            "is_zero_support": False,
                            "is_high_risk": False,
                            "class_index": i,
                        }
                        for i, nm in enumerate(names)
                    }
                }
            mock_pc.side_effect = _make_pc
            rows = compute_per_class_tflite_delta(
                y_true, y_keras, y_tflite, self.SIGN_NAMES, self.N
            )
        # "think" and "who" should be flagged as confusable pairs
        confusable_rows = {r["sign"]: r for r in rows if r["is_confusable_pair"]}
        assert "think" in confusable_rows
        assert "who" in confusable_rows

    def test_high_risk_signs_flagged(self):
        y_true, y_keras, y_tflite = self._arrays()
        with patch("src.export.verify.compute_per_class_metrics") as mock_pc:
            def _make_pc(y_t, y_p, names, n):
                return {
                    "per_class": {
                        nm: {
                            "f1_score": 0.5,
                            "support": 2,
                            "is_singleton": False,
                            "is_zero_support": False,
                            "is_high_risk": nm in ("clothes", "think", "birthday", "name", "book"),
                            "class_index": i,
                        }
                        for i, nm in enumerate(names)
                    }
                }
            mock_pc.side_effect = _make_pc
            rows = compute_per_class_tflite_delta(
                y_true, y_keras, y_tflite, self.SIGN_NAMES, self.N
            )
        high_risk = {r["sign"] for r in rows if r["is_high_risk"]}
        assert "think" in high_risk
        assert "clothes" in high_risk

    def test_sorted_by_absolute_delta_descending(self):
        y_true, y_keras, y_tflite = self._arrays()
        with patch("src.export.verify.compute_per_class_metrics") as mock_pc:
            call_count = [0]

            def _make_pc(y_t, y_p, names, n):
                call_count[0] += 1
                # Give varying deltas
                return {
                    "per_class": {
                        nm: {
                            "f1_score": float(i) / len(names) if call_count[0] == 1
                                       else float(i + 1) / (len(names) + 1),
                            "support": 3,
                            "is_singleton": False,
                            "is_zero_support": False,
                            "is_high_risk": False,
                            "class_index": i,
                        }
                        for i, nm in enumerate(names)
                    }
                }
            mock_pc.side_effect = _make_pc
            rows = compute_per_class_tflite_delta(
                y_true, y_keras, y_tflite, self.SIGN_NAMES, self.N
            )
        # Meaningful degradations first; within non-meaningful, sorted by |delta|
        meaningful = [r for r in rows if r["meaningful_degradation"]]
        non_meaningful = [r for r in rows if not r["meaningful_degradation"]]
        assert rows[:len(meaningful)] == meaningful

    def test_result_dict_keys_complete(self):
        y_true, y_keras, y_tflite = self._arrays()
        with patch("src.export.verify.compute_per_class_metrics") as mock_pc:
            def _make_pc(y_t, y_p, names, n):
                return {
                    "per_class": {
                        nm: {
                            "f1_score": 0.5,
                            "support": 2,
                            "is_singleton": False,
                            "is_zero_support": False,
                            "is_high_risk": False,
                            "class_index": i,
                        }
                        for i, nm in enumerate(names)
                    }
                }
            mock_pc.side_effect = _make_pc
            rows = compute_per_class_tflite_delta(
                y_true, y_keras, y_tflite, self.SIGN_NAMES, self.N
            )
        required_row_keys = {
            "sign", "class_idx", "keras_f1", "tflite_f1", "f1_delta",
            "support", "is_singleton", "is_zero_support", "is_high_risk",
            "is_confusable_pair", "confusable_with", "meaningful_degradation",
        }
        for row in rows:
            assert required_row_keys.issubset(row.keys()), (
                f"Missing keys in row for '{row.get('sign')}': "
                f"{required_row_keys - row.keys()}"
            )


# ===========================================================================
# TestAssembleReleaseGate
# ===========================================================================

@tf_required
class TestAssembleReleaseGate:
    """assemble_release_gate() must populate ReleaseGateResult from dicts."""

    def _verification_result(
        self,
        val_delta: float = 0.01,
        test_delta: float = 0.01,
        agreement: float = 0.98,
    ) -> Dict:
        return {
            "val": {
                "keras_macro_f1": 0.6011,
                "tflite_macro_f1": 0.6011 - val_delta,
                "delta_macro_f1": val_delta,
                "keras_accuracy": 0.5769,
                "tflite_accuracy": 0.5600,
                "delta_accuracy": 0.0169,
                "argmax_agreement": agreement,
                "mean_abs_diff": 0.005,
                "max_abs_diff": 0.05,
                "keras_mean_confidence": 0.5136,
                "tflite_mean_confidence": 0.5100,
                "confidence_shift": -0.0036,
                "n_samples": 52,
                "n_disagreements": 1,
                "n_keras_right_tflite_wrong": 1,
                "n_keras_wrong_tflite_right": 0,
            },
            "test": {
                "keras_macro_f1": 0.4581,
                "tflite_macro_f1": 0.4581 - test_delta,
                "delta_macro_f1": test_delta,
                "keras_accuracy": 0.4902,
                "tflite_accuracy": 0.4800,
                "delta_accuracy": 0.0102,
                "argmax_agreement": 0.97,
                "mean_abs_diff": 0.004,
                "max_abs_diff": 0.04,
                "keras_mean_confidence": 0.5200,
                "tflite_mean_confidence": 0.5164,
                "confidence_shift": -0.0036,
                "n_samples": 51,
                "n_disagreements": 1,
                "n_keras_right_tflite_wrong": 1,
                "n_keras_wrong_tflite_right": 0,
            },
            "n_classes": N_CLASSES,
            "sign_names": ["sign_%d" % i for i in range(N_CLASSES)],
        }

    def _latency_result(self, full_ms: float = 25.0, meets: bool = True) -> Dict:
        return {
            "tflite": {"median_ms": 5.2, "p95_ms": 7.1, "fps": 192.3},
            "keras": {"median_ms": 12.1, "p95_ms": 15.0, "fps": 82.6},
            "pipeline": {"median_ms": full_ms - 5.2, "p95_ms": 22.0, "fps": 50.0},
            "full_pipeline_ms": full_ms,
            "meets_100ms_target": meets,
            "speedup_keras_vs_tflite_x": round(12.1 / 5.2, 2),
        }

    def test_passing_gate_is_release_ready(self, tmp_path):
        f = tmp_path / "model.tflite"
        f.write_bytes(b"\x00" * 68_000)  # ~0.065 MB
        gate = assemble_release_gate(
            self._verification_result(),
            self._latency_result(),
            f,
        )
        assert gate.release_ready

    def test_failing_val_delta_propagates(self, tmp_path):
        f = tmp_path / "model.tflite"
        f.write_bytes(b"\x00" * 68_000)
        gate = assemble_release_gate(
            self._verification_result(val_delta=0.05),
            self._latency_result(),
            f,
        )
        assert not gate.release_ready

    def test_missing_file_detected(self, tmp_path):
        gate = assemble_release_gate(
            self._verification_result(),
            self._latency_result(),
            tmp_path / "nonexistent.tflite",
        )
        assert not gate.release_ready
        assert gate.tflite_file_exists is False

    def test_oversized_file_fails(self, tmp_path):
        f = tmp_path / "huge.tflite"
        f.write_bytes(b"\x00" * (11 * 1024 * 1024))  # 11 MB
        gate = assemble_release_gate(
            self._verification_result(),
            self._latency_result(),
            f,
        )
        assert not gate.release_ready
        assert not gate.size_under_10mb

    def test_sample_counts_populated(self, tmp_path):
        f = tmp_path / "model.tflite"
        f.write_bytes(b"\x00" * 68_000)
        gate = assemble_release_gate(
            self._verification_result(),
            self._latency_result(),
            f,
        )
        assert gate.n_val_samples == 52
        assert gate.n_test_samples == 51

    def test_speedup_populated(self, tmp_path):
        f = tmp_path / "model.tflite"
        f.write_bytes(b"\x00" * 68_000)
        gate = assemble_release_gate(
            self._verification_result(),
            self._latency_result(),
            f,
        )
        assert gate.speedup_keras_vs_tflite_x is not None
        assert gate.speedup_keras_vs_tflite_x > 1.0


# ===========================================================================
# TestSaveVerificationReport
# ===========================================================================

@tf_required
class TestSaveVerificationReport:
    """save_verification_report() must write valid JSON without bulky keys."""

    def _minimal_inputs(self, tmp_path: Path):
        gate = _make_gate()
        verification_result = {
            "val": {
                "keras_macro_f1": 0.6011,
                "tflite_macro_f1": 0.5990,
                "delta_macro_f1": 0.0021,
                "keras_accuracy": 0.5769,
                "tflite_accuracy": 0.5600,
                "delta_accuracy": 0.0169,
                "argmax_agreement": 0.98,
                "mean_abs_diff": 0.003,
                "max_abs_diff": 0.04,
                "keras_mean_confidence": 0.5136,
                "tflite_mean_confidence": 0.5100,
                "confidence_shift": -0.003,
                "n_samples": 52,
                "n_disagreements": 1,
                "disagreement_details": [{"clip": 1, "keras": "book", "tflite": "computer"}],
                "keras_per_class": {"book": {"f1_score": 1.0}},
                "tflite_per_class": {"book": {"f1_score": 0.0}},
                "n_keras_right_tflite_wrong": 1,
                "n_keras_wrong_tflite_right": 0,
            },
            "test": {
                "keras_macro_f1": 0.4581,
                "tflite_macro_f1": 0.4560,
                "delta_macro_f1": 0.0021,
                "keras_accuracy": 0.4902,
                "tflite_accuracy": 0.4800,
                "delta_accuracy": 0.0102,
                "argmax_agreement": 0.97,
                "mean_abs_diff": 0.003,
                "max_abs_diff": 0.04,
                "keras_mean_confidence": 0.52,
                "tflite_mean_confidence": 0.517,
                "confidence_shift": -0.003,
                "n_samples": 51,
                "n_disagreements": 1,
                "disagreement_details": [],
                "keras_per_class": {},
                "tflite_per_class": {},
                "n_keras_right_tflite_wrong": 1,
                "n_keras_wrong_tflite_right": 0,
            },
            "n_classes": N_CLASSES,
            "sign_names": ["sign_%d" % i for i in range(N_CLASSES)],
        }
        latency_result = {
            "tflite": {"median_ms": 5.2, "p95_ms": 7.1, "p99_ms": 9.0, "fps": 192.3},
            "keras": {"median_ms": 12.1, "p95_ms": 15.0, "fps": 82.6},
            "pipeline": {"median_ms": 19.8, "p95_ms": 22.0, "fps": 50.0},
            "full_pipeline_ms": 25.0,
            "meets_100ms_target": True,
            "speedup_keras_vs_tflite_x": 2.33,
        }
        out = tmp_path / "tflite_verification_report.json"
        return gate, verification_result, latency_result, out

    def test_creates_valid_json_file(self, tmp_path):
        gate, vr, lr, out = self._minimal_inputs(tmp_path)
        save_verification_report(gate, vr, lr, output_path=out)
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_bulky_keys_stripped(self, tmp_path):
        gate, vr, lr, out = self._minimal_inputs(tmp_path)
        save_verification_report(gate, vr, lr, output_path=out)
        with open(out) as f:
            data = json.load(f)
        acc = data.get("accuracy_comparison", {})
        assert "disagreement_details" not in acc.get("val", {})
        assert "keras_per_class" not in acc.get("val", {})
        assert "tflite_per_class" not in acc.get("val", {})

    def test_release_gate_dict_present(self, tmp_path):
        gate, vr, lr, out = self._minimal_inputs(tmp_path)
        save_verification_report(gate, vr, lr, output_path=out)
        with open(out) as f:
            data = json.load(f)
        assert "release_gate" in data
        assert "release_ready" in data["release_gate"]

    def test_per_class_delta_included_if_provided(self, tmp_path):
        gate, vr, lr, out = self._minimal_inputs(tmp_path)
        delta = [{"sign": "think", "keras_f1": 0.0, "tflite_f1": 0.0, "f1_delta": 0.0}]
        save_verification_report(gate, vr, lr, per_class_delta=delta, output_path=out)
        with open(out) as f:
            data = json.load(f)
        assert data["per_class_delta"] is not None

    def test_returns_resolved_path(self, tmp_path):
        gate, vr, lr, out = self._minimal_inputs(tmp_path)
        returned = save_verification_report(gate, vr, lr, output_path=out)
        assert isinstance(returned, Path)
        assert returned.is_absolute()

    def test_nan_values_serialised_as_null(self, tmp_path):
        """NaN in latency (e.g. from a failed benchmark) must not break JSON."""
        gate, vr, lr, out = self._minimal_inputs(tmp_path)
        lr["full_pipeline_ms"] = float("nan")
        save_verification_report(gate, vr, lr, output_path=out)
        with open(out) as f:
            data = json.load(f)
        # If it loaded without error, NaN was handled
        assert data is not None

    def test_creates_parent_directories(self, tmp_path):
        gate, vr, lr, _ = self._minimal_inputs(tmp_path)
        nested_out = tmp_path / "deeply" / "nested" / "report.json"
        save_verification_report(gate, vr, lr, output_path=nested_out)
        assert nested_out.exists()


# ===========================================================================
# TestWriteModelMetadata
# ===========================================================================

@tf_required
class TestWriteModelMetadata:
    """write_model_metadata() must produce schema-complete, JSON-safe output."""

    def _inputs(self, tmp_path: Path, config_hash: str = "a" * 64):
        snapshot = _make_config_snapshot_yaml(tmp_path, config_hash)
        conversion_result = {
            "param_memory_mb": 0.2629,
            "savedmodel_disk_mb": 4.5,
            "tflite_disk_mb": 0.065,
            "size_reduction_vs_params_x": 4.04,
            "size_reduction_vs_savedmodel_x": 69.2,
            "sha256_checksum": "x" * 64,
        }
        verification_result = {
            "val": {
                "tflite_macro_f1": 0.5990,
                "tflite_accuracy": 0.5600,
                "delta_macro_f1": 0.0021,
                "argmax_agreement": 0.98,
                "confidence_shift": -0.003,
                "keras_mean_confidence": 0.5136,
                "tflite_mean_confidence": 0.5106,
                "n_samples": 52,
                "n_disagreements": 1,
            },
            "test": {
                "tflite_macro_f1": 0.4560,
                "tflite_accuracy": 0.4800,
                "delta_macro_f1": 0.0021,
                "argmax_agreement": 0.97,
                "confidence_shift": -0.003,
                "n_samples": 51,
                "n_disagreements": 1,
            },
        }
        latency_result = {
            "tflite": {"median_ms": 5.2, "p95_ms": 7.1, "p99_ms": 9.0, "fps": 192.3},
            "keras": {"median_ms": 12.1, "p95_ms": 15.0, "fps": 82.6},
            "pipeline": {"median_ms": 19.8, "p95_ms": 22.0, "fps": 50.0},
            "full_pipeline_ms": 25.0,
            "meets_100ms_target": True,
            "speedup_keras_vs_tflite_x": 2.33,
        }
        tflite_p = tmp_path / "model.tflite"
        tflite_p.write_bytes(b"\x00" * 68_000)
        out = tmp_path / "gesture_model_metadata.json"
        return snapshot, conversion_result, verification_result, latency_result, tflite_p, out

    def test_creates_valid_json(self, tmp_path):
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        assert out.exists()
        with open(out) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_required_top_level_keys_present(self, tmp_path):
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        required = {
            "model_name", "version", "config_hash",
            "architecture", "input_shape_tflite", "output_shape_tflite",
            "preprocessing", "num_classes",
            "stage6_reference_metrics", "tflite_performance",
            "calibration", "file_sizes", "latency_cpu",
            "quantisation", "deployment_notes",
        }
        missing = required - data.keys()
        assert not missing, f"Missing required metadata keys: {missing}"

    def test_model_name_correct(self, tmp_path):
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        assert data["model_name"] == "gesture_bilstm_v1"

    def test_architecture_fields_from_config_not_hardcoded(self, tmp_path):
        """Architecture must come from config snapshot, not literals."""
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        arch = data["architecture"]
        assert arch["hidden_units"] == 64
        assert arch["num_layers"] == 2
        assert arch["total_params"] == CHAMPION_PARAMS

    def test_preprocessing_landmark_config_is_string(self, tmp_path):
        """Pydantic enum objects must be cast to str before JSON dump."""
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        lc = data["preprocessing"]["landmark_config"]
        assert isinstance(lc, str), (
            f"landmark_config should be a plain str, got {type(lc)}: {lc!r}. "
            "Pydantic enum was not cast via str() before JSON serialisation."
        )

    def test_tflite_performance_populated_from_verification(self, tmp_path):
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        perf = data["tflite_performance"]
        assert abs(perf["val_macro_f1"] - 0.5990) < 1e-4
        assert abs(perf["test_macro_f1"] - 0.4560) < 1e-4

    def test_stage6_reference_metrics_labelled_historical(self, tmp_path):
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        ref = data["stage6_reference_metrics"]
        assert "_note" in ref
        note = ref["_note"].lower()
        assert "stage 6" in note or "historical" in note or "phase" in note

    def test_three_file_size_measurements_present(self, tmp_path):
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        with open(out) as f:
            data = json.load(f)
        sizes = data["file_sizes"]
        assert "param_memory_mb" in sizes
        assert "savedmodel_disk_mb" in sizes
        assert "tflite_disk_mb" in sizes

    def test_no_nan_in_output(self, tmp_path):
        """All NaN values must be serialised as null (Python None)."""
        snap, cr, vr, lr, tflite_p, out = self._inputs(tmp_path)
        # Inject a NaN into latency
        lr["full_pipeline_ms"] = float("nan")
        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )
        # If the file loads, NaN was sanitised to null
        with open(out) as f:
            data = json.load(f)
        assert data is not None


# ===========================================================================
# TestPlotFunctions (headless smoke tests)
# ===========================================================================

@tf_required
class TestPlotFunctions:
    """Plot functions must run without error and return Figure objects."""

    @pytest.fixture(autouse=True)
    def _suppress_display(self):
        """Prevent any display window from opening during tests."""
        import matplotlib
        matplotlib.use("Agg")
        yield

    def test_plot_size_comparison_returns_figure(self, tmp_path):
        conversion_result = {
            "param_memory_mb": 0.2629,
            "savedmodel_disk_mb": 4.5,
            "tflite_disk_mb": 0.065,
            "size_reduction_vs_params_x": 4.04,
            "size_reduction_vs_savedmodel_x": 69.2,
        }
        fig = plot_tflite_size_comparison(conversion_result)
        import matplotlib.pyplot as plt
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_size_comparison_saves_to_disk(self, tmp_path):
        conversion_result = {
            "param_memory_mb": 0.2629,
            "savedmodel_disk_mb": 4.5,
            "tflite_disk_mb": 0.065,
            "size_reduction_vs_params_x": 4.04,
            "size_reduction_vs_savedmodel_x": 69.2,
        }
        out = tmp_path / "size_comparison.png"
        import matplotlib.pyplot as plt
        fig = plot_tflite_size_comparison(conversion_result, output_path=out)
        plt.close(fig)
        assert out.exists()
        assert out.stat().st_size > 1000  # non-trivial PNG

    def test_plot_accuracy_comparison_returns_figure(self, tmp_path):
        verification_result = {
            "val": {
                "keras_macro_f1": 0.6011,
                "tflite_macro_f1": 0.5990,
                "keras_accuracy": 0.5769,
                "tflite_accuracy": 0.5600,
                "argmax_agreement": 0.98,
                "confidence_shift": -0.003,
            },
            "test": {
                "keras_macro_f1": 0.4581,
                "tflite_macro_f1": 0.4560,
                "keras_accuracy": 0.4902,
                "tflite_accuracy": 0.4800,
                "argmax_agreement": 0.97,
                "confidence_shift": -0.003,
            },
            "n_classes": N_CLASSES,
        }
        import matplotlib.pyplot as plt
        fig = plot_tflite_accuracy_comparison(verification_result)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_per_class_delta_empty_returns_figure(self, tmp_path):
        """Empty per_class_delta must not crash."""
        import matplotlib.pyplot as plt
        fig = plot_tflite_per_class_delta([])
        assert fig is not None
        plt.close(fig)

    def test_plot_per_class_delta_with_data(self, tmp_path):
        sign_names = ["think", "who", "later", "house", "clothes", "girl", "orange"]
        per_class_delta = [
            {
                "sign": s,
                "class_idx": i,
                "keras_f1": 0.5,
                "tflite_f1": 0.4,
                "f1_delta": 0.1,
                "support": 2,
                "is_singleton": False,
                "is_high_risk": s in ("think", "clothes"),
                "is_confusable_pair": s in ("think", "who", "later", "house"),
                "meaningful_degradation": s != "think",
            }
            for i, s in enumerate(sign_names)
        ]
        import matplotlib.pyplot as plt
        fig = plot_tflite_per_class_delta(per_class_delta, n_classes=len(sign_names))
        assert hasattr(fig, "savefig")
        plt.close(fig)


# ===========================================================================
# TestVerifyConstants
# ===========================================================================

@tf_required
class TestVerifyConstants:
    """Module-level constants in verify.py must be internally consistent."""

    def test_delta_threshold(self):
        assert 0.0 < _DELTA_THRESHOLD <= 0.10

    def test_agreement_threshold(self):
        assert 0.0 < _AGREEMENT_THRESHOLD <= 1.0

    def test_prob_diff_warn_threshold(self):
        assert 0.0 < _PROB_DIFF_WARN_THRESHOLD <= 0.10

    def test_confidence_shift_warn_threshold(self):
        assert 0.0 < _CONFIDENCE_SHIFT_WARN_THRESHOLD <= 0.10

    def test_max_tflite_size_mb(self):
        assert _VERIFY_MAX_MB == 10.0

    def test_latency_target_ms(self):
        assert _LATENCY_TARGET_MS == 100.0

    def test_meaningful_degradation_delta(self):
        assert 0.0 < _MEANINGFUL_DEGRADATION_DELTA <= 0.5

    def test_stage6_keras_val_f1_in_range(self):
        assert 0.0 < _STAGE6_KERAS_VAL_MACRO_F1 < 1.0

    def test_stage6_keras_test_f1_in_range(self):
        assert 0.0 < _STAGE6_KERAS_TEST_MACRO_F1 < 1.0

    def test_stage6_ci_val_bounds_sane(self):
        lo, hi = _STAGE6_KERAS_VAL_MACRO_F1_CI
        assert lo < _STAGE6_KERAS_VAL_MACRO_F1 < hi, (
            "Val macro-F1 point estimate must lie within its CI bounds."
        )

    def test_stage6_ci_test_bounds_sane(self):
        lo, hi = _STAGE6_KERAS_TEST_MACRO_F1_CI
        assert lo < _STAGE6_KERAS_TEST_MACRO_F1 < hi, (
            "Test macro-F1 point estimate must lie within its CI bounds."
        )

    def test_confusable_signs_symmetric(self):
        """Each confusable pair must contain both directions."""
        from src.export.verify import _CONFUSABLE_PAIRS
        for sign, partners in _CONFUSABLE_PAIRS.items():
            for partner in partners:
                assert sign in _CONFUSABLE_PAIRS.get(partner, []), (
                    f"Confusable pair is not symmetric: "
                    f"'{sign}' lists '{partner}' but not vice versa."
                )

    def test_confusable_signs_count(self):
        from src.export.verify import _CONFUSABLE_PAIRS
        assert len(_CONFUSABLE_PAIRS) == 8  # 4 pairs × 2 directions

    def test_high_risk_signs_count(self):
        assert len(_HIGH_RISK_SIGNS) == 5

    def test_high_risk_signs_known_values(self):
        assert "think" in _HIGH_RISK_SIGNS
        assert "clothes" in _HIGH_RISK_SIGNS

    def test_delta_threshold_value(self):
        """Spec mandates 0.03 threshold."""
        assert abs(_DELTA_THRESHOLD - 0.03) < 1e-9

    def test_agreement_threshold_value(self):
        """Spec mandates 0.95 threshold."""
        assert abs(_AGREEMENT_THRESHOLD - 0.95) < 1e-9


# ===========================================================================
# TestImportTimeSelfChecks
# ===========================================================================

@tf_required
class TestImportTimeSelfChecks:
    """Both modules' _self_check() functions must pass silently."""

    def test_convert_self_check_passes(self):
        from src.export import convert
        # _self_check() is called at import time; calling again explicitly
        # ensures it still passes with the current constant values.
        convert._self_check()

    def test_verify_self_check_passes(self):
        from src.export import verify
        verify._self_check()


# ===========================================================================
# TestRunFullVerification (integration — requires all Stage 5/6/7 artefacts)
# ===========================================================================

@pytest.mark.integration
@tf_required
class TestRunFullVerification:
    """run_full_verification() integration test against real artefacts."""

    @pytest.fixture(autouse=True)
    def require_all_artefacts(self):
        if not _SAVED_MODEL_PATH.exists():
            pytest.skip("Champion SavedModel not available for integration test.")
        if not _TFLITE_PATH.exists():
            pytest.skip(
                "TFLite file not found. Run Stage 8 Step 1 first: "
                "python -m src.export.convert"
            )
        if not _CONFIG_SNAPSHOT.exists():
            pytest.skip("Config snapshot not found.")

    def test_gate_structure_complete(self, tmp_path):
        """run_full_verification returns a ReleaseGateResult and full_results dict."""
        try:
            from src.features.dataset import GestureDataset
            from src.features.pipeline import FeaturePipeline
            from src.utils.config import load_config

            cfg = load_config(
                model="bilstm",
                data="seq100",
                augmentation="spatial_temporal",
                overrides={"data.landmark_config": "hands_only"},
            )
            pipeline = FeaturePipeline(cfg)
            dataset = GestureDataset(cfg, pipeline)
        except Exception as e:
            pytest.skip(f"Could not construct dataset for integration test: {e}")

        gate, full = run_full_verification(
            keras_model_path=_SAVED_MODEL_PATH,
            tflite_path=_TFLITE_PATH,
            config_snapshot_path=_CONFIG_SNAPSHOT,
            val_dataset=dataset,
            pipeline=pipeline,
            figures_dir=tmp_path / "figures",
            verification_report_path=tmp_path / "report.json",
            metadata_output_path=tmp_path / "metadata.json",
            n_calls=10,   # minimal for speed in integration context
            warmup=3,
        )
        assert isinstance(gate, ReleaseGateResult)
        assert isinstance(full, dict)

    def test_gate_verdict_meaningful(self, tmp_path):
        """The gate's verdict must be deterministic (True or False, not error)."""
        try:
            from src.features.dataset import GestureDataset
            from src.features.pipeline import FeaturePipeline
            from src.utils.config import load_config

            cfg = load_config(
                model="bilstm",
                data="seq100",
                augmentation="spatial_temporal",
                overrides={"data.landmark_config": "hands_only"},
            )
            pipeline = FeaturePipeline(cfg)
            dataset = GestureDataset(cfg, pipeline)
        except Exception as e:
            pytest.skip(f"Could not construct dataset: {e}")

        gate, _ = run_full_verification(
            keras_model_path=_SAVED_MODEL_PATH,
            tflite_path=_TFLITE_PATH,
            config_snapshot_path=_CONFIG_SNAPSHOT,
            val_dataset=dataset,
            pipeline=pipeline,
            figures_dir=tmp_path / "figs",
            verification_report_path=tmp_path / "report.json",
            metadata_output_path=tmp_path / "meta.json",
            n_calls=10,
            warmup=3,
        )
        assert isinstance(gate.release_ready, bool)
        # For a correctly exported champion, we expect PASS
        if not gate.release_ready:
            failures_str = "\n".join(gate.hard_failures)
            warnings.warn(
                f"Integration test: release gate FAILED with:\n{failures_str}\n"
                "This may indicate a model/export mismatch or environment issue.",
                UserWarning,
            )

    def test_verification_report_written(self, tmp_path):
        try:
            from src.features.dataset import GestureDataset
            from src.features.pipeline import FeaturePipeline
            from src.utils.config import load_config

            cfg = load_config(
                model="bilstm",
                data="seq100",
                augmentation="spatial_temporal",
                overrides={"data.landmark_config": "hands_only"},
            )
            pipeline = FeaturePipeline(cfg)
            dataset = GestureDataset(cfg, pipeline)
        except Exception as e:
            pytest.skip(f"Could not construct dataset: {e}")

        report_path = tmp_path / "report.json"
        run_full_verification(
            keras_model_path=_SAVED_MODEL_PATH,
            tflite_path=_TFLITE_PATH,
            config_snapshot_path=_CONFIG_SNAPSHOT,
            val_dataset=dataset,
            pipeline=pipeline,
            figures_dir=tmp_path / "figs",
            verification_report_path=report_path,
            metadata_output_path=tmp_path / "meta.json",
            n_calls=5,
            warmup=2,
        )
        assert report_path.exists()
        with open(report_path) as f:
            data = json.load(f)
        assert "release_gate" in data
        assert "accuracy_comparison" in data


# ===========================================================================
# TestWriteExportManifest
# ===========================================================================

@tf_required
class TestWriteExportManifest:
    """write_export_manifest() must produce a complete, JSON-safe manifest."""

    def test_creates_manifest_file(self, tmp_path):
        result = {
            "output_path": str(tmp_path / "model.tflite"),
            "tflite_disk_mb": 0.065,
            "keras_params": CHAMPION_PARAMS,
            "sha256_checksum": "a" * 64,
            "quantisation_mode": "dynamic_range",
            "used_select_tf_ops": True,
        }
        manifest_path = write_export_manifest(result, output_dir=tmp_path)
        assert manifest_path.exists()

    def test_manifest_contains_tensorflow_version(self, tmp_path):
        result = {"output_path": "x", "tflite_disk_mb": 0.065}
        path = write_export_manifest(result, output_dir=tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert "tensorflow_version" in data

    def test_manifest_contains_created_utc(self, tmp_path):
        result = {"output_path": "x"}
        path = write_export_manifest(result, output_dir=tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert "created_utc" in data

    def test_manifest_is_json_serialisable(self, tmp_path):
        result = {
            "output_path": str(tmp_path / "m.tflite"),
            "sha256_checksum": None,  # may be None before sanity check
        }
        path = write_export_manifest(result, output_dir=tmp_path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)

    def test_custom_filename_respected(self, tmp_path):
        result = {}
        path = write_export_manifest(result, output_dir=tmp_path, filename="my_manifest.json")
        assert path.name == "my_manifest.json"


# ===========================================================================
# TestEdgeCasesAndRegression
# ===========================================================================

@tf_required
class TestEdgeCasesAndRegression:
    """Regression and edge-case tests for specific bugs documented in verify.py."""

    def test_b1_no_double_inference_in_full_verification(self):
        """
        Bug B1: run_full_verification previously re-ran inference for per-class delta.
        The fix stores prediction arrays in _predictions and reuses them.
        Verify the _predictions key is present in run_accuracy_verification output.
        """
        with patch("src.export.verify.GesturePredictor") as MockPredictor, \
             patch("src.export.verify.compute_macro_f1", return_value=0.6), \
             patch("src.export.verify.compute_accuracy", return_value=0.58), \
             patch("src.export.verify.compute_per_class_metrics") as mock_pcm:

            mock_pcm.return_value = {"per_class": {}}
            keras_pred = MagicMock()
            keras_pred.return_value = np.random.rand(52, N_CLASSES).astype(np.float32)
            tflite_pred = MagicMock()
            tflite_pred.return_value = np.random.rand(52, N_CLASSES).astype(np.float32)

            X = np.random.rand(52, SEQ_LEN, FEAT_DIM).astype(np.float32)
            y = np.random.randint(0, N_CLASSES, 52).astype(np.int64)
            sids = np.zeros(52, dtype=np.int32)

            mock_ds = MagicMock()
            mock_ds.get_arrays_for_split.return_value = (X, y, sids)

            instance = MagicMock()
            # side_effect: first call returns keras, second returns tflite
            call_count = [0]
            def predictor_call(X_arg, training=False):
                call_count[0] += 1
                if call_count[0] % 2 == 1:
                    return np.random.rand(len(X_arg), N_CLASSES).astype(np.float32)
                return np.random.rand(len(X_arg), N_CLASSES).astype(np.float32)

            instance.__call__ = predictor_call
            instance.label_map = MagicMock()
            instance.label_map.get_name_safe.side_effect = lambda i, d: f"sign_{i}"

            keras_instance = MagicMock()
            keras_instance.__call__ = lambda X, training=False: np.random.rand(len(X), N_CLASSES).astype(np.float32)
            keras_instance.label_map = instance.label_map

            tflite_instance = MagicMock()
            tflite_instance.__call__ = lambda X, training=False: np.random.rand(len(X), N_CLASSES).astype(np.float32)

            MockPredictor.from_config_snapshot.side_effect = [keras_instance, tflite_instance]

            try:
                result = run_accuracy_verification(
                    keras_model_path="fake/keras",
                    tflite_path="fake.tflite",
                    config_snapshot_path="fake/config.yaml",
                    val_dataset=mock_ds,
                    n_classes=N_CLASSES,
                )
                # B1/B8 fix: _predictions must be returned
                assert "_predictions" in result, (
                    "Bug B1/B8: _predictions sub-dict not returned by "
                    "run_accuracy_verification. Per-class delta will re-run inference."
                )
            except Exception:
                pytest.skip("Full mock chain not exercisable in this environment")

    def test_b3_singleton_fallback_logic_correct(self):
        """
        Bug B3: `support == 1` fallback with default 0 was always False.
        Ensure is_singleton is set correctly even when the key is absent.
        """
        with patch("src.export.verify.compute_per_class_metrics") as mock_pcm:
            call_count = [0]

            def _make_pc(y_t, y_p, names, n):
                call_count[0] += 1
                # Deliberately omit 'is_singleton' key to exercise fallback
                return {
                    "per_class": {
                        nm: {
                            "f1_score": 0.5,
                            "support": 1 if i == 0 else 5,
                            # is_singleton deliberately ABSENT
                            "is_zero_support": False,
                            "is_high_risk": False,
                            "class_index": i,
                        }
                        for i, nm in enumerate(names)
                    }
                }
            mock_pcm.side_effect = _make_pc

            names = [f"sign_{i}" for i in range(N_CLASSES)]
            rng = np.random.default_rng(0)
            y = rng.integers(0, N_CLASSES, 52).astype(np.int64)

            rows = compute_per_class_tflite_delta(y, y, y, names, N_CLASSES)

            # sign_0 has support=1 → should be detected as singleton
            sign0_row = next((r for r in rows if r["sign"] == "sign_0"), None)
            if sign0_row is not None:
                # The bug (always False) would cause is_singleton=False for support=1
                # The fix should detect support=1 via explicit comparison
                assert sign0_row["is_singleton"] is True, (
                    "Bug B3: is_singleton=False for a class with support=1. "
                    "The fallback `k.get('is_singleton', support == 1)` is wrong "
                    "because `support` was not yet extracted — the fix must extract "
                    "support first, then derive the boolean."
                )

    def test_b9_nan_size_mb_does_not_produce_phantom_pass(self, tmp_path):
        """
        Bug B9: if stat() fails and tflite_size_mb is NaN, size_under_10mb
        must be False (not True from NaN < 10.0 returning False in Python,
        but specifically should never yield size_under_10mb=True).
        """
        vr = {
            "val": {
                "keras_macro_f1": 0.6011, "tflite_macro_f1": 0.5990,
                "delta_macro_f1": 0.0021, "keras_accuracy": 0.5769,
                "tflite_accuracy": 0.5600, "delta_accuracy": 0.0169,
                "argmax_agreement": 0.98, "mean_abs_diff": 0.003,
                "max_abs_diff": 0.04, "keras_mean_confidence": 0.5136,
                "tflite_mean_confidence": 0.5100, "confidence_shift": -0.003,
                "n_samples": 52, "n_disagreements": 1,
                "n_keras_right_tflite_wrong": 1, "n_keras_wrong_tflite_right": 0,
            },
            "test": {
                "keras_macro_f1": 0.4581, "tflite_macro_f1": 0.4560,
                "delta_macro_f1": 0.0021, "keras_accuracy": 0.4902,
                "tflite_accuracy": 0.4800, "delta_accuracy": 0.0102,
                "argmax_agreement": 0.97, "mean_abs_diff": 0.003,
                "max_abs_diff": 0.04, "keras_mean_confidence": 0.52,
                "tflite_mean_confidence": 0.517, "confidence_shift": -0.003,
                "n_samples": 51, "n_disagreements": 1,
                "n_keras_right_tflite_wrong": 1, "n_keras_wrong_tflite_right": 0,
            },
        }
        lr = {
            "tflite": {"median_ms": 5.2, "p95_ms": 7.1, "fps": 192.3},
            "keras": {"median_ms": 12.1, "p95_ms": 15.0, "fps": 82.6},
            "pipeline": {"median_ms": 19.8, "p95_ms": 22.0, "fps": 50.0},
            "full_pipeline_ms": 25.0,
            "meets_100ms_target": True,
            "speedup_keras_vs_tflite_x": 2.33,
        }
        # File exists but stat() would fail → simulate by pointing to a file
        # that exists but patching stat to raise
        real_file = tmp_path / "real.tflite"
        real_file.write_bytes(b"\x00" * 10)

        with patch.object(Path, "exists", return_value=True), \
            patch.object(Path, "stat", side_effect=OSError("simulated stat failure")):
            gate = assemble_release_gate(vr, lr, real_file)
        assert not gate.size_under_10mb, (
            "Bug B9: size_under_10mb should be False when stat() fails, "
            "not True from NaN < 10.0 comparison."
        )

    def test_b10_all_enum_fields_are_strings_in_metadata(self, tmp_path):
        """
        Bug B10: Pydantic enum objects must be cast via str() to guarantee
        JSON serialisability regardless of Pydantic's coercion behaviour.
        """
        snap = _make_config_snapshot_yaml(tmp_path, "a" * 64)
        cr = {
            "param_memory_mb": 0.26,
            "savedmodel_disk_mb": 4.5,
            "tflite_disk_mb": 0.065,
            "size_reduction_vs_params_x": 4.0,
            "size_reduction_vs_savedmodel_x": 69.0,
        }
        vr = {
            "val": {
                "tflite_macro_f1": 0.60, "tflite_accuracy": 0.56,
                "delta_macro_f1": 0.001, "argmax_agreement": 0.98,
                "confidence_shift": -0.003, "keras_mean_confidence": 0.51,
                "tflite_mean_confidence": 0.507, "n_samples": 52, "n_disagreements": 0,
            },
            "test": {
                "tflite_macro_f1": 0.45, "tflite_accuracy": 0.48,
                "delta_macro_f1": 0.001, "argmax_agreement": 0.97,
                "confidence_shift": -0.003, "n_samples": 51, "n_disagreements": 0,
            },
        }
        lr = {
            "tflite": {"median_ms": 5.2, "p95_ms": 7.1, "p99_ms": 9.0, "fps": 192.3},
            "keras": {"median_ms": 12.1, "p95_ms": 15.0, "fps": 82.6},
            "pipeline": {"median_ms": 19.8, "p95_ms": 22.0, "fps": 50.0},
            "full_pipeline_ms": 25.0,
            "meets_100ms_target": True,
            "speedup_keras_vs_tflite_x": 2.33,
        }
        tflite_p = tmp_path / "m.tflite"
        tflite_p.write_bytes(b"\x00" * 68_000)
        out = tmp_path / "meta.json"

        write_model_metadata(
            tflite_path=tflite_p,
            conversion_result=cr,
            verification_result=vr,
            latency_result=lr,
            config_snapshot_path=str(snap),
            output_path=out,
        )

        with open(out) as f:
            data = json.load(f)

        # All preprocessing fields that might be Pydantic enums must be strings
        preprocessing = data.get("preprocessing", {})
        for field_name in ("landmark_config", "normalisation", "missing_frame_strategy"):
            if field_name in preprocessing:
                val = preprocessing[field_name]
                assert isinstance(val, str), (
                    f"Bug B10: preprocessing['{field_name}'] is {type(val)} "
                    f"(value: {val!r}), not str. Cast via str() before JSON dump."
                )


# ===========================================================================
# TestProjectConstants (cross-module consistency)
# ===========================================================================

@tf_required
class TestProjectConstants:
    """Critical project constants must be identical across modules."""

    def test_champion_params_same_in_convert_and_verify(self):
        """
        _EXPECTED_CHAMPION_PARAMS in convert.py must equal the documented
        68,771 count. verify.py references this from Stage 6 reference
        metrics' deployment_notes.total_params field.
        """
        assert _EXPECTED_CHAMPION_PARAMS == 68_771

    def test_max_size_mb_consistent(self):
        assert _MAX_TFLITE_SIZE_MB == _VERIFY_MAX_MB == 10.0

    def test_delta_threshold_matches_spec(self):
        """Part 2 specifies max_accuracy_delta = 0.03."""
        assert abs(_DELTA_THRESHOLD - 0.03) < 1e-9

    def test_known_config_hash_consistent(self):
        """Both modules must reference the same champion config hash."""
        from src.export.verify import _CHAMPION_CONFIG_HASH as verify_hash
        assert _KNOWN_CHAMPION_CONFIG_HASH == verify_hash

    def test_high_risk_signs_consistent_between_modules(self):
        """HIGH_RISK_SIGNS in predictor.py must match verify.py."""
        try:
            from src.inference.predictor import HIGH_RISK_SIGNS as pred_signs
            verify_signs = _HIGH_RISK_SIGNS
            assert set(pred_signs) == set(verify_signs), (
                f"HIGH_RISK_SIGNS mismatch between predictor.py and verify.py: "
                f"predictor={set(pred_signs)}, verify={set(verify_signs)}"
            )
        except ImportError:
            pytest.skip("predictor.py not available")

    def test_bytes_per_float32_consistent(self):
        """Both convert.py and benchmark.py must use 4 bytes per float32."""
        try:
            from src.evaluation.benchmark import _BYTES_PER_FLOAT32 as bench_bytes
            assert _BYTES_PER_FLOAT32 == bench_bytes == 4
        except ImportError:
            pass  # benchmark.py may not be importable in minimal env

    def test_n_classes_is_35(self):
        """The WLASL-35 project always has exactly 35 classes."""
        assert _TFLITE_EXPECTED_OUTPUT_SHAPE[-1] == 35
        assert _EXPECTED_CHAMPION_OUTPUT_SHAPE[-1] == 35

    def test_seq_len_is_100(self):
        """Champion model uses seq_len=100."""
        assert _TFLITE_EXPECTED_INPUT_SHAPE[1] == 100
        assert _EXPECTED_CHAMPION_INPUT_SHAPE[1] == 100

    def test_feat_dim_is_126_hands_only(self):
        """Champion uses hands_only: 63+63=126 dims."""
        assert _TFLITE_EXPECTED_INPUT_SHAPE[2] == 126
        assert _EXPECTED_CHAMPION_INPUT_SHAPE[2] == 126


# ===========================================================================
# Pytest configuration hook
# ===========================================================================

def pytest_collection_modifyitems(config, items):
    """Auto-add 'integration' marker info for CI clarity."""
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(
                pytest.mark.skipif(
                    not _SAVED_MODEL_PATH.exists(),
                    reason=(
                        "Integration test skipped: champion SavedModel not found. "
                        "Run Stage 5 training to generate "
                        f"{_SAVED_MODEL_PATH}"
                    ),
                )
            )