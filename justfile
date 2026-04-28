# Default recipe to display available commands.
_:
  @just --list

# Format and lint the package using ruff, and lint the examples using marimo.
lint:
  ruff format
  ruff check

# Run the full test suite.
test:
  pytest

# Build the documentation using Zensical.
docs:
  zensical build
