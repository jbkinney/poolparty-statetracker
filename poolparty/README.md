# PoolParty

[![PyPI version](https://badge.fury.io/py/poolparty.svg)](https://badge.fury.io/py/poolparty)
[![Documentation Status](https://readthedocs.org/projects/poolparty/badge/?version=latest)](https://poolparty.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PoolParty** is a Python package for designing complex oligonucleotide sequence libraries. It provides a declarative, composable interface for generating DNA libraries used in MPRA (massively parallel reporter assays), deep mutational scanning, and other high-throughput experiments.

## Why PoolParty?

Designing DNA libraries often involves combining multiple types of sequence variation—point mutations, deletions, insertions, shuffled regions—each with different parameters and positions. PoolParty lets you:

- **Compose operations**: Chain operations like `.mutagenize()`, `.deletion_scan()`, and `.insertion_scan()` to build complex libraries
- **Use lazy evaluation**: Sequences are generated on-demand, enabling libraries with billions of potential variants
- **Track provenance**: Every sequence includes metadata tracing its combinatorial origin
- **Tag regions**: Use XML-like syntax to mark and manipulate specific regions of sequences

## Installation

```bash
pip install poolparty
```

For development:
```bash
git clone https://github.com/jbkinney/poolparty-statecounter.git
cd poolparty-statecounter/poolparty
pip install -e ".[dev]"
```

## Quick Start

```python
import poolparty as pp

# Initialize poolparty
pp.init()

# Create a template sequence with tagged regions
template = pp.from_seq("ACGT<cre>GGAAAGCGGGCAGTGAGC</cre>TTTT<bc/>GGGG")

# Generate single-nucleotide mutations in the CRE region
mutant_library = template.mutagenize(
    region="cre",
    num_mutations=1,
    mode="sequential"  # Generate all possible single mutants
)

# Generate the library as a DataFrame
df = mutant_library.generate_library()
print(df)
```

## Key Features

### Region Tagging

Mark regions of interest with XML-like tags:

```python
# Self-closing tag for insertion points
seq = pp.from_seq("ACGT<barcode/>TTTT")

# Paired tags for regions
seq = pp.from_seq("ACGT<promoter>GGAAAGCGGG</promoter>TTTT")
```

### Scanning Operations

Apply systematic mutations across a region:

```python
# Tiled deletions
deletions = template.deletion_scan(
    region="cre",
    deletion_length=5,
    positions=slice(None, None, 3)  # Every 3rd position
)

# Tiled insertions
inserts = pp.from_seqs(["AAAAAA", "TTTTTT"])
insertions = template.insertion_scan(
    region="cre",
    ins_pool=inserts,
    positions=slice(0, 10, 2)
)

# Replacement scanning
replacements = template.replacement_scan(
    region="cre",
    ins_pool=pp.get_kmers(length=5),
)
```

### Combining Libraries

Stack different variant types into a single library:

```python
# Create different variant pools
mutations = template.mutagenize(region="cre", num_mutations=1)
deletions = template.deletion_scan(region="cre", deletion_length=5)

# Combine into one library
combined = pp.stack([mutations, deletions])

# Add barcodes to all variants
barcoded = combined.insert_kmers(region="bc", length=10)

# Generate final library
df = barcoded.generate_library()
```

### Random vs Sequential Mode

Control how variants are generated:

```python
# Sequential: enumerate all possible variants
all_mutants = template.mutagenize(
    num_mutations=1,
    mode="sequential"
)

# Random: sample from variant space
random_mutants = template.mutagenize(
    num_mutations=2,
    mode="random",
    num_states=100  # Generate 100 random double mutants
)
```

### Codon-Aware Mutagenesis

Preserve reading frames during mutagenesis:

```python
orf_mutants = template.mutagenize_orf(
    region="cre",
    num_mutations=1,
    mutation_type="synonymous"  # or "nonsynonymous", "nonsense"
)
```

## Operations

PoolParty provides 50+ composable operations for DNA library design.
See the [full API reference](https://poolparty.readthedocs.io) for details.

| Goal | Key operations |
|------|----------------|
| Create pools | `from_seq`, `from_seqs`, `from_fasta`, `from_iupac`, `get_kmers`, `get_barcodes` |
| Mutate | `mutagenize`, `mutagenize_orf`, `shuffle_seq`, `recombine`, `flip` |
| Scan across positions | `deletion_scan`, `insertion_scan`, `replacement_scan`, `mutagenize_scan`, `subseq_scan` |
| Work with regions | `annotate_region`, `extract_region`, `replace_region`, `insert_tags` |
| Combine & control | `stack`, `sample`, `repeat`, `sync`, `filter`, `score` |
| Export | `generate_library`, `to_df`, `to_file` |

## Documentation

Full documentation is available at [poolparty.readthedocs.io](https://poolparty.readthedocs.io).

## Related Projects

PoolParty is built on [StateTracker](https://github.com/jbkinney/poolparty-statecounter/tree/main/statetracker), a library for composable state management that enables efficient random access to combinatorial spaces.

## License

MIT License - see [LICENSE](LICENSE) for details.
