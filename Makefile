SHELL := /bin/bash
ROOT_DIR := $(shell pwd)
VENVS_DIR := $(ROOT_DIR)/.venvs

# ─────────────────────────────────────────────────────────────────────────────
# Model-specific test targets
# Most models have their own venv; AutoGluon-backed models share .venvs/autogluon.
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test-ttm test-sundial test-timesfm test-timesfm3 test-cgm_jepa test-autogluon test-chronos2 test-models test lint help

test-ttm:
	$(VENVS_DIR)/ttm/bin/python -m pytest tests/models/ -v -k ttm

test-sundial:
	$(VENVS_DIR)/sundial/bin/python -m pytest tests/models/ -v -k "not ttm and not timesfm and not chronos2"

test-timesfm:
	$(VENVS_DIR)/timesfm/bin/python -m pytest tests/models/ -v -k timesfm

# TimesFM 3.0 is a separate family in its own env (.venvs/timesfm3). The
# TORCH_DISABLE_NATIVE_JIT env var forces the covariate `bmm` path onto standard
# aten kernels on hosts without CUDA/Python build headers for Triton.
test-timesfm3:
	TORCH_DISABLE_NATIVE_JIT=1 $(VENVS_DIR)/timesfm3/bin/python -m pytest tests/models/ -v -k timesfm3

test-cgm_jepa:
	$(VENVS_DIR)/cgm_jepa/bin/python -m pytest tests/models/ -v -k cgm_jepa

test-autogluon:
	$(VENVS_DIR)/autogluon/bin/python -m pytest tests/models/ -v -k "chronos2 or autogluon_base or deepar or patchtst or tft or naive_baseline or statistical"

test-chronos2: test-autogluon

## Run all per-model tests sequentially with their correct venvs
test-models: test-ttm test-sundial test-timesfm test-autogluon

# ─────────────────────────────────────────────────────────────────────────────
# General tests (main venv — no model-specific deps required)
# ─────────────────────────────────────────────────────────────────────────────

test:
	.venv/bin/python -m pytest tests/ -v --ignore=tests/models

## Run everything: common tests + all model tests
test-all: test test-models

# ─────────────────────────────────────────────────────────────────────────────
# Linting
# ─────────────────────────────────────────────────────────────────────────────

lint:
	.venv/bin/python -m ruff check src/ tests/

help:
	@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'
	@echo ""
	@echo "Targets:"
	@echo "  test-ttm       Run TTM tests using .venvs/ttm"
	@echo "  test-sundial   Run Sundial tests using .venvs/sundial"
	@echo "  test-timesfm   Run TimesFM tests using .venvs/timesfm"
	@echo "  test-timesfm3  Run TimesFM 3.0 tests using .venvs/timesfm3"
	@echo "  test-autogluon Run AutoGluon-family tests using .venvs/autogluon"
	@echo "  test-chronos2  Alias for test-autogluon (backward compatible)"
	@echo "  test-models    Run all model tests (all venvs)"
	@echo "  test           Run non-model tests (main venv)"
	@echo "  test-all       Run all tests"
	@echo "  lint           Run ruff linter"
