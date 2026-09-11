# Contributing

Thanks for your interest in contributing to `dg-solvax`! Bug reports, bug fixes, documentation improvements, and new features (including benchmarks and example problems) are all welcome.


## Development setup

Clone the repository and install the development dependencies with [uv](https://docs.astral.sh/uv/):

```sh
uv sync
uv run pre-commit install
```

This installs a pre-commit hook that runs formatting, linting, the unit tests, and secret scanning on every commit. The secret scanning hook requires the `gitleaks`, e.g. installed via Homebrew:

```sh
brew install gitleaks
```

To run all checks on demand, use:
```sh
uv run pre-commit run --all-files
```


## Making changes

- Format and lint: `uv run --group lint ruff format .` and `uv run --group lint ruff check --fix .`.
- Unit tests: `uv run pytest`. 
- Benchmarks: `uv run pytest benchmarks`.
- Examples (need plotly from the `utils` group): `uv run --group utils python -m examples.linear_advection_eq`.
- Docs preview: `uv run --group docs mkdocs serve`.

Please add or adjust unit tests for any changed public behavior. Bug fixes should come with a test that fails without the fix.


## Submitting changes

Open a pull request against `main` with a short description of the change and how you verified it. CI must pass; it runs formatting and linting, the unit tests on all supported Python versions, the package build, and secret scanning.
