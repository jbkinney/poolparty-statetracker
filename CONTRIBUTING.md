# Contributing to PoolParty and StateTracker

Thank you for your interest in contributing! This document provides guidelines
for contributing to this project.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you are expected to uphold this code.

## How to Contribute

### Reporting Bugs

Before submitting a bug report:
1. Check the [existing issues](https://github.com/jbkinney/poolparty-statetracker/issues)
2. Ensure the bug is reproducible with the latest version

When submitting a bug report, include:
- Python version (`python --version`)
- Package versions (`pip show poolparty statetracker`)
- Operating system
- Minimal code example that reproduces the issue
- Full error traceback

### Suggesting Features

Feature requests are welcome! Please open an issue describing:
- The problem you're trying to solve
- Your proposed solution
- Any alternatives you've considered

### Submitting Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`uv run pytest`)
5. Run `pre-commit run --all-files` to check formatting
6. Commit with a clear message
7. Push to your fork
8. Open a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/jbkinney/poolparty-statetracker.git
cd poolparty-statetracker

# Install with uv (recommended)
uv sync --group dev

# Or install with pip
pip install -e ./statetracker[dev]
pip install -e ./poolparty[dev]

# Set up pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific package
uv run pytest statetracker/tests/
uv run pytest poolparty/tests/

# Run with coverage
uv run pytest --cov=statetracker --cov=poolparty
```

### Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Check for issues
ruff check .

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

## Project Structure

```
poolparty-statetracker/
├── poolparty/           # DNA library design package
│   ├── src/poolparty/   # Source code
│   ├── tests/           # Test suite
│   └── pyproject.toml
├── statetracker/        # Combinatorial state management
│   ├── src/statetracker/
│   ├── tests/
│   ├── docs/            # Sphinx documentation
│   └── pyproject.toml
└── pyproject.toml       # Workspace configuration
```

## Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Prefix with package name for package-specific changes:
  - `poolparty: Add new operation`
  - `statetracker: Fix value propagation bug`

## Questions?

Feel free to open an issue for any questions not covered here.
