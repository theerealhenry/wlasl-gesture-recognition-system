# Makefile — WLASL Gesture Recognition Pipeline
# Author: Henry Otsyula — Senior Data Scientist & ML Engineer
#
# Usage: make <target>
# Run    make help    for a full description of each target.
#
# All targets assume a virtual environment is already active.
# Run `make setup` on first install to install all dependencies.

.PHONY: help setup preprocess extract train evaluate export register \
        demo lint test test-fast docker-train docker-inference \
        docker-test mlflow clean clean-models clean-logs

# Default target
.DEFAULT_GOAL := help

# ── Self-documenting help ─────────────────────────────────────────────────
# Prints all targets with their ## comments, sorted by category.
help:
	@echo ""
	@echo "WLASL Gesture Recognition — Make Targets"
	@echo "=========================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────
setup: ## Install all training + dev dependencies (run once)
	pip install protobuf==3.20.3
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	python -c "from src.utils.config import write_default_configs; write_default_configs()"
	@echo "Setup complete. Run 'make preprocess' to begin the pipeline."

# ── Data pipeline ─────────────────────────────────────────────────────────
preprocess: ## Stage 1: Download WLASL videos and create signer-aware splits
	python pipelines/run_preprocessing.py

extract: ## Stage 3: Extract MediaPipe Holistic landmarks from all videos
	python pipelines/run_landmark_extraction.py --split all

# ── Training ──────────────────────────────────────────────────────────────
train: ## Stage 5: Run full 23-run ablation experiment matrix
	python pipelines/run_all_experiments.py

train-champion: ## Stage 5: Run champion config only (bilstm/seq100/spatial_temporal/hands_only)
	python pipelines/run_training.py \
		--model bilstm \
		--data seq100 \
		--augmentation spatial_temporal \
		--run-name bilstm_hands_only_v4_aug \
		--overrides data.landmark_config=hands_only training.learning_rate=0.0005

# ── Evaluation ────────────────────────────────────────────────────────────
evaluate: ## Stage 6: Run full evaluation suite (confusion matrix, SHAP, calibration)
	python pipelines/run_evaluation.py \
		--champion-run bilstm_hands_only_v4_aug \
		--splits val test \
		--output-dir reports/evaluation/

# ── Export ────────────────────────────────────────────────────────────────
export: ## Stage 8: TFLite export + verification gate (requires Stage 5 SavedModel)
	python pipelines/run_export_verification.py

export-dry-run: ## Stage 8: Fast end-to-end sanity check (n_calls=5, skips latency)
	python pipelines/run_export_verification.py --dry-run

# ── Model registration ────────────────────────────────────────────────────
register: ## Stage 10: Register champion in MLflow Model Registry
	python pipelines/run_model_registration.py

# ── Demo ─────────────────────────────────────────────────────────────────
demo: ## Stage 9: Launch real-time webcam demo (TFLite)
	python src/demo/webcam_demo.py

demo-minimal: ## Stage 9: Launch demo with minimal HUD (better FPS on slower hardware)
	python src/demo/webcam_demo.py --minimal-hud

demo-record: ## Stage 9: Launch demo and record to reports/demo_recording.mp4
	python src/demo/webcam_demo.py --record reports/demo_recording.mp4

gif: ## Stage 9: Convert demo recording to GIF for README embedding
	@if [ ! -f reports/demo_recording.mp4 ]; then \
		echo "Run 'make demo-record' first to create the recording."; exit 1; \
	fi
	ffmpeg -i reports/demo_recording.mp4 \
		-vf "fps=10,scale=720:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" \
		-loop 0 \
		reports/demo.gif
	@echo "GIF written to reports/demo.gif"

# ── Quality ───────────────────────────────────────────────────────────────
lint: ## Run flake8 over all source, pipeline, and test files
	flake8 src/ pipelines/ tests/ --max-line-length 100 --extend-ignore E203,W503

format: ## Auto-format code with black (modifies files in place)
	black src/ pipelines/ tests/ --line-length 100

test: ## Run full test suite with coverage report
	pytest tests/ -v \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=xml:coverage.xml \
		--cov-fail-under=60

test-fast: ## Run tests excluding slow integration tests (unit tests only)
	pytest tests/ -v -m "not integration" --tb=short

test-integration: ## Run only integration tests (requires SavedModel and TFLite on disk)
	pytest tests/ -v -m integration --tb=long

# ── MLflow ────────────────────────────────────────────────────────────────
mlflow: ## Launch MLflow UI (http://localhost:5000)
	mlflow ui --host 0.0.0.0 --port 5000

# ── Docker ────────────────────────────────────────────────────────────────
docker-build: ## Build training image (wlasl-train:latest)
	docker build -t wlasl-train .

docker-build-inference: ## Build inference image (wlasl-inference:latest)
	docker build -f Dockerfile.inference -t wlasl-inference .

docker-train: ## Run full training pipeline inside Docker
	docker-compose run --rm train

docker-evaluate: ## Run evaluation suite inside Docker
	docker-compose run --rm evaluate

docker-export: ## Run TFLite export + verification inside Docker
	docker-compose run --rm export

docker-test: ## Run test suite inside Docker (mirrors CI environment exactly)
	docker-compose run --rm test

docker-lint: ## Run linting inside Docker
	docker-compose run --rm lint

# ── Cleanup ───────────────────────────────────────────────────────────────
clean: ## Remove __pycache__, .pyc files, and pytest/coverage artefacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage coverage.xml htmlcov/

clean-logs: ## Remove all log files (logs/ directory)
	rm -rf logs/*
	@echo "Logs cleared."

clean-models: ## DANGER: Remove all SavedModels (keeps TFLite and metadata)
	@echo "WARNING: This will delete all Keras SavedModels in models/."
	@read -p "Are you sure? [y/N] " confirm; \
		if [ "$$confirm" = "y" ]; then \
			find models/ -name "*_saved_model" -type d -exec rm -rf {} + 2>/dev/null || true; \
			find models/ -name "*_best_weights" -type d -exec rm -rf {} + 2>/dev/null || true; \
			echo "SavedModels removed. TFLite artefacts preserved."; \
		else \
			echo "Cancelled."; \
		fi