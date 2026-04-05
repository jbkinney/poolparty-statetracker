# poolparty-statetracker

This monorepo contains two related Python packages:

- **[poolparty](poolparty/)** - A Python package for designing complex oligonucleotide sequence libraries
- **[statetracker](statetracker/)** - Composable states with unidirectional value propagation for enumerating combinatorial spaces

## Repository Structure

```
poolparty-statetracker/
├── poolparty/          # poolparty package
│   ├── src/poolparty/  # source code
│   ├── tests/          # tests
│   ├── docs/           # Sphinx documentation
│   └── pyproject.toml
├── statetracker/       # statetracker package
│   ├── src/statetracker/
│   ├── tests/
│   ├── docs/
│   └── pyproject.toml
```

## Installation

Each package can be installed independently from PyPI:

```bash
pip install poolparty      # includes statetracker as a dependency
pip install statetracker   # standalone
```

For development, clone the repo and install both in editable mode:

```bash
git clone https://github.com/jbkinney/poolparty-statetracker.git
cd poolparty-statetracker
pip install -e ./statetracker[dev]
pip install -e ./poolparty[dev]
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

## Citation

If you use this software, please cite it using the metadata in [CITATION.cff](CITATION.cff).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Both packages are released under the MIT License.
