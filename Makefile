# Makefile for Diversity Evaluation Pipeline
#
# Core targets for reproducing paper results:
#   make install     → Set up environment
#   make eval-div    → Run diversity evaluation on all models
#   make quality     → Compute quality metrics
#   make qfd         → Compute quality-filtered diversity

SHELL := /bin/bash
PYTHON ?= python
PIP ?= pip

.PHONY: help install quality qfd eval-div

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install core dependencies
	$(PIP) install -e .

eval-div: ## Run diversity evaluation on all models via lighteval
	$(PYTHON) src/evaluation/lighteval_runner.py \
		--all-models \
		--tasks tldr_diversity cnn_dailymail_diversity xsum_diversity \
		--output-dir ./runs/diversity \
		--lighteval-args --custom-tasks src/evaluation/lighteval_summarization_tasks.py

quality: ## Compute quality metrics for all tasks
	$(PYTHON) scripts/compute_math_quality.py
	$(PYTHON) scripts/compute_code_quality.py
	$(PYTHON) scripts/compute_ifeval_quality.py
	$(PYTHON) scripts/compute_summarization_quality.py
	$(PYTHON) scripts/compute_wildbench_quality.py
	$(PYTHON) scripts/compute_writingprompts_quality.py

qfd: ## Compute quality-filtered diversity decomposition
	$(PYTHON) scripts/quality_filtered_diversity.py
	$(PYTHON) scripts/quality_filtered_diversity_gsm8k.py
	$(PYTHON) scripts/compute_code_diversity.py
	$(PYTHON) scripts/compute_vendi_from_generations.py
