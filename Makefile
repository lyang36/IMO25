# IMO25 Solution Testing Makefile

# Python to use
PYTHON := python3

# Actual file targets
.env: .env.example
	cp $< $@
	@echo "Created .env from .env.example - please update with your API credentials"

# Phony targets
.PHONY: help setup test clean env run-agent run-imo01

help:
	@echo "Available targets:"
	@echo "  make env         - Create .env from .env.example"
	@echo "  make setup       - Set up Python environment"
	@echo "  make test        - Run tests"
	@echo "  make run-imo01   - Run agent on problem 1"
	@echo "  make run-agent   - Run agent with custom problem file"
	@echo "  make clean       - Clean generated files"

env: .env

setup: env
	$(PYTHON) -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

test:
	. venv/bin/activate && $(PYTHON) code/run_agent.py

clean:
	rm -rf venv __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Run agent on the first IMO problem
run-imo01: .env
	@echo "Running agent on IMO problem 1..."
	$(PYTHON) code/agent.py problems/imo01.txt --max_runs 1

# Run agent with custom problem file (use PROBLEM=path/to/problem.txt)
run-agent: .env
	@if [ -z "$(PROBLEM)" ]; then \
		echo "Usage: make run-agent PROBLEM=path/to/problem.txt"; \
		exit 1; \
	fi
	$(PYTHON) code/agent.py $(PROBLEM)