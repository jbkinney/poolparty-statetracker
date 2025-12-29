# poolparty-repo

This monorepo contains two related Python packages:

- **[poolparty](poolparty/)** - A Python package for designing oligonucleotide sequence libraries
- **[statecounter](statecounter/)** - Composable counters with unidirectional state propagation for enumerating combinatorial spaces

## Repository Structure

```
poolparty-repo/
├── poolparty/          # poolparty package
│   ├── src/poolparty/  # source code
│   ├── tests/          # tests
│   └── pyproject.toml
├── statecounter/       # statecounter package
│   ├── src/statecounter/
│   ├── tests/
│   ├── docs/
│   └── pyproject.toml
└── notebooks/          # shared notebooks
```

## Installation

Each package can be installed independently. For development, install in editable mode:

```bash
# Install statecounter
cd statecounter
pip install -e ".[dev]"

# Install poolparty
cd ../poolparty
pip install -e ".[dev]"
```

## Running Tests

```bash
# Test statecounter
cd statecounter
pytest

# Test poolparty
cd ../poolparty
pytest
```

## License

Both packages are released under the MIT License.
