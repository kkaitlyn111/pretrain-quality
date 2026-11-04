PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help install quickstart extract fit test clean

help:
	@echo "make install      create $(VENV) and install the package (+ pytest)"
	@echo "make quickstart   feature extraction -> EBM fit on the bundled 10k-doc sample (CPU)"
	@echo "make test         unit tests for the heuristic features"
	@echo "Full cluster pipeline: see pipeline/README.md"

$(BIN)/pq:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip
	$(BIN)/pip install -q -e ".[dev]"

install: $(BIN)/pq

extract: install
	$(BIN)/pq extract

fit: install
	$(BIN)/pq fit

quickstart: extract fit

test: install
	$(BIN)/pytest -q

clean:
	rm -rf runs
