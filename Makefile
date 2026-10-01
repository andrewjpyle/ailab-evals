.PHONY: demo test lint hooks fixtures record

demo:  ## replay every suite from the committed cassettes (no network, no key)
	uv run ailab-evals run suites/substance.json --out /tmp/ailab-demo/substance
	uv run ailab-evals judge suites/judge.json --out /tmp/ailab-demo/judge
	uv run ailab-evals pairwise suites/pairwise.json --out /tmp/ailab-demo/pairwise
	uv run ailab-evals retrieval suites/retrieval.json --out /tmp/ailab-demo/retrieval

test:
	uv run --group dev pytest -q

lint:
	uv run --group dev ruff check .

hooks:  ## enable the mandatory pre-push secret wall
	git config core.hooksPath .githooks

fixtures:
	python3 scripts/build_fixtures.py

record:  ## re-record cassettes against live models (needs ANTHROPIC_API_KEY + a local Ollama)
	uv run ailab-evals run suites/substance.json --mode record
	uv run ailab-evals judge suites/judge.json --mode record
	uv run ailab-evals pairwise suites/pairwise.json --mode record
