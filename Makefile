# Windows: use .venv/Scripts/python; Linux/macOS: .venv/bin/python
PY ?= .venv/Scripts/python
export PYTHONPATH := src$(if $(filter Windows_NT,$(OS)),;,:).

.PHONY: install lint test api demo stub-stats docker

install:
	$(PY) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[ml,clip,llm,demo,dev]"

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest

api:
	$(PY) -m uvicorn app.api:app --reload --port 8000

demo:
	$(PY) -m streamlit run app/demo.py

stub-stats:
	$(PY) -m app.predictors

docker:
	docker build -t sf311-triage .
