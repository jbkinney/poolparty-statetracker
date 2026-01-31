# poolparty-repo

This monorepo contains two related Python packages:

- **[poolparty](poolparty/)** - A Python package for designing oligonucleotide sequence libraries
- **[statetracker](statetracker/)** - Composable states with unidirectional value propagation for enumerating combinatorial spaces

## Repository Structure

```
poolparty-repo/
├── poolparty/          # poolparty package
│   ├── src/poolparty/  # source code
│   ├── tests/          # tests
│   └── pyproject.toml
├── statetracker/       # statetracker package
│   ├── src/statetracker/
│   ├── tests/
│   ├── docs/
│   └── pyproject.toml
└── notebooks/          # shared notebooks
```

## Installation

Each package can be installed independently. For development, install in editable mode:

```bash
# Install statetracker
cd statetracker
pip install -e ".[dev]"

# Install poolparty
cd ../poolparty
pip install -e ".[dev]"
```

## Running Tests

```bash
# Test statetracker
cd statetracker
pytest

# Test poolparty
cd ../poolparty
pytest
```

## License

Both packages are released under the MIT License.
