PYTHON ?= .venv/bin/python

.PHONY: test-cpu test-gpu lint install

install:
	$(PYTHON) -m pip install --no-deps .

lint:
	$(PYTHON) -m ruff format --check src tests scripts
	$(PYTHON) -m ruff check src tests scripts

test-cpu: install lint
	$(PYTHON) -m pytest -q -m 'not gpu' -W error

test-gpu: install lint
	$(PYTHON) -m pytest -q -m gpu -W error
