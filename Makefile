PY = $(if $(wildcard .venv/bin/python),.venv/bin/python,$(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,python3))
BOOTSTRAP_PY ?= python3

GROUP ?= Gustavo_Lucas_Pedro
ZIP_NAME = IA_Trabalho_$(GROUP).zip
PROJECT_DIR = $(notdir $(CURDIR))

.DEFAULT_GOAL := help
.PHONY: help venv install check-data smoke quick run dota 2 lol significance \
		summary freeze clean clean-results zip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

venv:  ## Create the virtual environment (.venv)
	$(BOOTSTRAP_PY) -m venv .venv

install: venv  ## Create .venv and install requirements.txt
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt


check-data:  ## Check that the datasets are in data/raw/
	@test -f data/raw/dota2/dota2Train.csv || { echo "Missing data/raw/dota2/dota2Train.csv (see README.md)"; exit 1; }
	@test -f data/raw/dota2/dota2Test.csv  || { echo "Missing data/raw/dota2/dota2Test.csv (see README.md)"; exit 1; }
	@test -f data/raw/lol/games.csv        || { echo "Missing data/raw/lol/games.csv (see README.md)"; exit 1; }
	@echo "All datasets found."

smoke: check-data  ## Load both datasets and print their summaries
	$(PY) -m src.data_loading

quick: check-data  ## Fast end-to-end check (reduced grids; overwrites results/)
	$(PY) -m src.main --quick

run: check-data  ## Full experiment, 3 scenarios (about 15 minutes)
	$(PY) -m src.main

dota2: check-data  ## Full experiment, Dota 2 only
	$(PY) -m src.main --scenarios dota2

lol: check-data  ## Full experiment, both LoL scenarios
	$(PY) -m src.main --scenarios lol_draft_only lol_draft_ingame

significance: check-data  ## McNemar and bootstrap tests (needs `make run` first)
	$(PY) -m src.significance

summary:  ## Rebuild summary.csv and model_comparison.png from existing tables
	$(PY) -c "from src import main; main.combine_summaries()"

lint:  ## Fail if a forbidden library is imported in src/
	@if grep -rEin "^[[:space:]]*(import|from)[[:space:]]+(tensorflow|keras|torch|pytorch|xgboost|lightgbm|catboost|jax|theano|fastai|transformers)\b" --include="*.py" src/ ; then \
		echo "ERROR: forbidden library found."; exit 1; \
	fi
	$(PY) -m compileall -q src/
	@echo "OK: only allowed libraries, syntax valid."

clean:  ## Remove Python caches
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +

clean-results:  ## Remove generated tables and figures
	rm -f results/tables/*.csv results/figures/*.png

zip: lint  ## Build ../IA_Trabalho_<GROUP>.zip (report PDF must be in report/)
	@ls report/*.pdf >/dev/null 2>&1 || { echo "ERROR: put the report PDF in report/ first."; exit 1; }
	cd .. && rm -f $(ZIP_NAME) && zip -r $(ZIP_NAME) $(PROJECT_DIR) \
		-x "$(PROJECT_DIR)/.git/*" "$(PROJECT_DIR)/.venv/*" "$(PROJECT_DIR)/.github/*" \
		   "$(PROJECT_DIR)/data/raw/*" "$(PROJECT_DIR)/data/processed/*" \
		   "*/__pycache__/*" "*.log" "*/.vscode/*" "*/notebooks/.ipynb_checkpoints/*"
	@echo "Created ../$(ZIP_NAME)"