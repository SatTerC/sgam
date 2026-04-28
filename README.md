# SGAM: Simplified Growth and Allocation Model

A Simplified Growth and Allocation Model (SGAM) for the allocation of gross primary productivity (GPP) to plant carbon pools across different plant types.

**Full documentation**: [satterc.github.io/sgam](https://satterc.github.io/sgam)

> [!WARNING]
> This project is in the early stages of development and should be used with caution.

## Quick start

Requires Python >=3.13.

```sh
# Clone the repository
git clone https://github.com/SatTerC/sgam.git
cd sgam

# Install the project and dependencies with uv
uv sync
```

## Development workflow

This project uses [`uv`](https://docs.astral.sh/uv/) for package management and [`just`](https://just.systems/) as a command runner.

```sh
# See all available commands
just

# Format and lint the code
just lint

# Run the test suite
just test

# Build the documentation
just docs
```

## Contributing

Well that would be nice.
Please feel free to leave an issue or open a PR.
