# Default recipe to display available commands.
_:
  @just --list

# Format and lint the package using ruff, and lint the examples using marimo.
lint:
  ruff format
  ruff check

# Run the full test suite.
test:
  pytest --log-cli-level=INFO

# Run tests with coverage report (requires pytest-cov).
coverage:
  pytest --cov=sgam --cov-report=term-missing --cov-fail-under=90

# Run static type checker (requires mypy in dev group).
typecheck:
  mypy src/sgam

# Regenerate PFT table and figures referenced in docs/science.md.
_gen-science-assets:
  python scripts/gen_science_assets.py

# Build the documentation using Zensical.
docs: _gen-science-assets
  zensical build
