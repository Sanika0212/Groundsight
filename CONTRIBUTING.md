# Contributing

Thanks for your interest in Groundsight.

## Development setup

```bash
git clone https://github.com/Sanika0212/Groundsight.git
cd Groundsight
uv sync --extra models --extra bench --extra viz
uv run pre-commit install
```

## Checks

Every pull request must pass the same checks CI runs:

```bash
uv run ruff check src tests scripts examples
uv run ruff format --check src tests scripts examples
uv run mypy
uv run coverage run -m pytest && uv run coverage report --fail-under=85
uv run groundsight demo --checker lexical
```

Coverage is measured with `coverage run`, not `pytest --cov`. The pytest plugin is
loaded before pytest-cov starts, so `--cov` would miss the module-level lines.

## Ground rules

1. **Tests never download weights or call an API.** Use the `lexical` checker, scripted
   checkers and `httpx.MockTransport`. Tests that need real weights are marked `models`
   and run only with `GROUNDSIGHT_TEST_MODELS=1`.
2. **Numbers come from runs.** Every number in the README must be reproducible with
   `groundsight bench` and `scripts/make_figures.py`, and docs/benchmarks.md must list
   the exact commands.
3. **Verdicts carry evidence.** A new checker or rule must still point each verdict at
   the passage span it rests on.
4. Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
   `docs:`, `test:`, `chore:`, `ci:`).
