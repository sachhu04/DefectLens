.PHONY: help all install run run-backend run-frontend run-interface ui cli predict build-mb build-mb-baseline build-mb-mctf build-mb-all test test-unit test-pipeline evaluate eval download-data build-frontend clean

# Default configurable variables (override via command line: make cli CATEGORY=bottle MODEL=mctf)
CATEGORY ?= bottle
MODEL ?= mctf
IMAGE ?= data/mvtec/bottle/test/broken_small/000.png
PORT ?= 8000
FRONTEND_PORT ?= 5173

# Default target
all: help

## ── Help ─────────────────────────────────────────────────────────────
help:
	@echo "========================================================================"
	@echo " DefectLens: Vision Transformer Anomaly Detection (MCTF + PatchCore)    "
	@echo "========================================================================"
	@echo "Usage: make <target> [VARIABLE=value]"
	@echo ""
	@echo "Environment & Installation:"
	@echo "  make install           Install Python dependencies, editable pkg, & frontend"
	@echo "  make download-data     Download MVTec AD category data (CATEGORY=bottle)"
	@echo ""
	@echo "Memory Bank Generation:"
	@echo "  make build-mb          Build memory bank (both baseline & mctf) for CATEGORY"
	@echo "  make build-mb-baseline Build baseline memory bank for CATEGORY (default: bottle)"
	@echo "  make build-mb-mctf     Build MCTF memory bank for CATEGORY (default: bottle)"
	@echo "  make build-mb-all      Build memory banks for all categories (bottle, cable, metal_nut)"
	@echo ""
	@echo "Interfaces (Web UI, Backend & CLI):"
	@echo "  make run / make ui     Run full stack (FastAPI backend + Vite React frontend)"
	@echo "  make run-backend       Run FastAPI backend only (http://localhost:8000)"
	@echo "  make run-frontend      Run Vite frontend only (http://localhost:5173)"
	@echo "  make cli / predict     Run CLI inference on an image"
	@echo "                         (IMAGE=path/to/img.png CATEGORY=bottle MODEL=mctf)"
	@echo ""
	@echo "Testing & Evaluation:"
	@echo "  make test              Run unit tests and end-to-end integration test suite"
	@echo "  make test-unit         Run pytest suite (API & pipeline unit tests)"
	@echo "  make test-pipeline     Run full 5-module end-to-end integration test"
	@echo "  make evaluate          Run full AUROC & latency evaluation (MODEL=mctf CATEGORY=bottle)"
	@echo ""
	@echo "Build & Maintenance:"
	@echo "  make build-frontend    Build production bundle for frontend"
	@echo "  make clean             Clean build artifacts, caches, and temp files"
	@echo "========================================================================"

## ── Installation ─────────────────────────────────────────────────────
install:
	@echo "==> Installing Python dependencies..."
	pip install -r requirements.txt
	pip install -e .
	@echo "==> Installing Frontend dependencies..."
	cd frontend && npm install

download-data:
	@echo "==> Downloading dataset for category '$(CATEGORY)'..."
	python scripts/download_dataset.py --category $(CATEGORY)

## ── Memory Bank ──────────────────────────────────────────────────────
build-mb-baseline:
	@echo "==> Building Baseline Memory Bank for '$(CATEGORY)'..."
	python scripts/build_memory_bank.py --config configs/baseline.yaml --category $(CATEGORY)

build-mb-mctf:
	@echo "==> Building MCTF Memory Bank for '$(CATEGORY)'..."
	python scripts/build_memory_bank.py --config configs/mctf.yaml --category $(CATEGORY)

build-mb: build-mb-baseline build-mb-mctf
	@echo "==> Finished building Baseline and MCTF Memory Banks for '$(CATEGORY)'."

build-mb-all:
	@for cat in bottle cable metal_nut; do \
		echo "==> Building memory banks for $$cat..."; \
		python scripts/build_memory_bank.py --config configs/baseline.yaml --category $$cat; \
		python scripts/build_memory_bank.py --config configs/mctf.yaml --category $$cat; \
	done
	@echo "==> Finished building memory banks for all categories."

## ── CLI Inference ────────────────────────────────────────────────────
cli predict:
	@echo "==> Running CLI inference (Model: $(MODEL), Category: $(CATEGORY), Image: $(IMAGE))..."
	python scripts/predict.py --image $(IMAGE) --model $(MODEL) --category $(CATEGORY)

## ── Interface & Services ─────────────────────────────────────────────
# Run both backend and frontend concurrently
run run-interface ui:
	@echo "Starting DefectLens (FastAPI http://localhost:$(PORT) + Vite http://localhost:$(FRONTEND_PORT))..."
	@trap 'kill 0' SIGINT SIGTERM EXIT; PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port $(PORT) --reload & (cd frontend && npm run dev) & wait

# Run backend only
run-backend:
	PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port $(PORT) --reload

# Run frontend only
run-frontend:
	cd frontend && npm run dev

build-frontend:
	cd frontend && npm run build

## ── Testing & Evaluation ─────────────────────────────────────────────
test: test-unit test-pipeline

test-unit:
	@echo "==> Running pytest test suite..."
	pytest tests/ -v

test-pipeline:
	@echo "==> Running 5-module end-to-end integration test..."
	python scripts/test_pipeline.py

evaluate eval:
	@echo "==> Running quantitative evaluation (Model: $(MODEL), Category: $(CATEGORY))..."
	python scripts/evaluate.py --config configs/$(MODEL).yaml --category $(CATEGORY)

## ── Maintenance ──────────────────────────────────────────────────────
clean:
	@echo "==> Cleaning temporary cache and build artifacts..."
	rm -rf .pytest_cache frontend/dist frontend/node_modules/.vite
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	@echo "==> Clean complete."
