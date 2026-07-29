.PHONY: install dev test lint selftest data train robustness explain clean

DATA ?= data/sleep-edf
RUN  ?= runs/main
CKPT ?= $(RUN)/cv/fold0.pt

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

selftest:            ## full pipeline on synthetic data, no download needed
	somnoscope selftest

data:                ## N=20 records by default; `make data N=0` for the full set
	bash scripts/download_sleepedf.sh $(DATA) $(or $(N),20)

train:
	somnoscope train --data $(DATA) --out $(RUN)

train-vulnerable:    ## the amplitude-dependent baseline, for the shortcut comparison
	somnoscope train --data $(DATA) --out $(RUN)-global --norm global

robustness:
	somnoscope robustness --checkpoint $(CKPT) --data $(DATA) --out $(RUN)/robustness.json

explain:
	somnoscope explain --checkpoint $(CKPT) --data $(DATA) --out $(RUN)/audit.json

audit: robustness explain

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
