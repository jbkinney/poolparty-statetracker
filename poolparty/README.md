# PoolParty

[![PyPI version](https://badge.fury.io/py/poolparty.svg)](https://badge.fury.io/py/poolparty)
[![Documentation Status](https://readthedocs.org/projects/poolparty/badge/?version=latest)](https://poolparty.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**PoolParty** is a Python package that streamlines the design of complex
DNA sequence libraries. Each library is specified as a computational graph
in a few lines of code, with sequences generated on demand. Over 50
built-in operations cover nucleotide- and codon-level mutagenesis,
scanning, barcode generation, and more. Applications include massively
parallel reporter assays, deep mutational scanning, and in silico analysis
of genomic AI models.

<p align="center">
  <img src="images/poolparty_schematic.png" alt="PoolParty overview: Pools represent sequence collections; Operations transform them into a DAG that generates libraries on demand" width="700">
</p>

## Why PoolParty?

Designing DNA libraries often involves combining multiple types of
sequence modifications (mutations, insertions, deletions, replacements)
across multiple regions. PoolParty lets you:

- **Chain operations**: Build pipelines from operations like `mutagenize`,
  `deletion_scan`, and `insertion_scan` to produce complex variant libraries
- **Tag regions**: Mark segments of a sequence with XML-style tags so
  operations can target them by name
- **Track construction history**: Each sequence carries a design card
  recording how it was built, ready for filtering and analysis
- **Style output**: Visual annotations highlight mutations, deletions,
  and regions for quick auditing

## Installation

```bash
pip install poolparty
```

Requires Python >= 3.10.

## Quick example

Create a template with a tagged region, branch into mutagenesis and
deletion scanning, and stack them into a single library:

```python
import poolparty as pp
pp.init()

template = pp.from_seq("TCCGACT<tag>GCA</tag>ATTCGGA")

mut_pool = template.mutagenize(
    num_mutations=1, region="tag", prefix="mut", mode="sequential",
    cards={"positions": "mut_pos", "wt_chars": "wt", "mut_chars": "mut"},
)

del_pool = template.deletion_scan(
    deletion_length=1, region="tag", prefix="del", mode="sequential"
).repeat(times=2, prefix="rep")

library = pp.stack([mut_pool, del_pool]).named("library")
library.print_library(show_name=True)
```

```
library: seq_length=17, num_states=15
name         seq
mut_0        TCCGACT<tag>ACA</tag>ATTCGGA
mut_1        TCCGACT<tag>CCA</tag>ATTCGGA
mut_2        TCCGACT<tag>TCA</tag>ATTCGGA
mut_3        TCCGACT<tag>GAA</tag>ATTCGGA
mut_4        TCCGACT<tag>GGA</tag>ATTCGGA
mut_5        TCCGACT<tag>GTA</tag>ATTCGGA
mut_6        TCCGACT<tag>GCC</tag>ATTCGGA
mut_7        TCCGACT<tag>GCG</tag>ATTCGGA
mut_8        TCCGACT<tag>GCT</tag>ATTCGGA
del_0.rep_0  TCCGACT<tag>-CA</tag>ATTCGGA
del_0.rep_1  TCCGACT<tag>-CA</tag>ATTCGGA
del_1.rep_0  TCCGACT<tag>G-A</tag>ATTCGGA
del_1.rep_1  TCCGACT<tag>G-A</tag>ATTCGGA
del_2.rep_0  TCCGACT<tag>GC-</tag>ATTCGGA
del_2.rep_1  TCCGACT<tag>GC-</tag>ATTCGGA
```

`mutagenize` in sequential mode generates all 9 single-nucleotide
substitutions within the 3 bp `<tag>` region. `deletion_scan` produces
3 single-base deletions, and `repeat` duplicates each for replication.
`stack` merges the two branches into a 15-variant library. The `prefix`
parameter on each operation labels variants so they can be traced back
to their source.

The `cards` parameter records design choices as DataFrame columns,
so each sequence carries a structured record of how it was built:

```python
df = library.generate_library()
print(df[["name", "mut_pos", "wt", "mut"]].head(5))
```

```
   name mut_pos   wt  mut
  mut_0    (0,) (G,) (A,)
  mut_1    (0,) (G,) (C,)
  mut_2    (0,) (G,) (T,)
  mut_3    (1,) (C,) (A,)
  mut_4    (1,) (C,) (G,)
```

## Operations

PoolParty provides over 50 composable operations for DNA library design.
See the [full documentation](https://poolparty.readthedocs.io) for details.

| Goal | Key operations |
|------|----------------|
| Create pools | `from_seq`, `from_seqs`, `from_fasta`, `from_iupac`, `get_kmers`, `get_barcodes` |
| Mutate | `mutagenize`, `mutagenize_orf`, `shuffle_seq`, `recombine`, `flip` |
| Scan across positions | `deletion_scan`, `insertion_scan`, `replacement_scan`, `mutagenize_scan`, `subseq_scan` |
| Work with regions | `annotate_region`, `extract_region`, `replace_region`, `insert_tags` |
| Combine and control | `stack`, `join`, `sample`, `repeat`, `sync`, `filter`, `score` |
| Export | `generate_library`, `to_df`, `to_file` |

## Documentation

Full documentation is available at
[poolparty.readthedocs.io](https://poolparty.readthedocs.io), including a
[quickstart guide](https://poolparty.readthedocs.io/en/latest/quickstart.html)
and tutorials for
[deep mutational scanning](https://poolparty.readthedocs.io/en/latest/tutorials/dms_gb1.html)
and
[MPRA library design](https://poolparty.readthedocs.io/en/latest/tutorials/mpra_regulatory_grammar.html).

## Development

```bash
git clone https://github.com/jbkinney/poolparty-statetracker.git
cd poolparty-statetracker
pip install -e ./statetracker[dev]
pip install -e ./poolparty[dev]
pytest poolparty/
```

## Citation

If you use PoolParty, please cite it using the metadata in
[CITATION.cff](https://github.com/jbkinney/poolparty-statetracker/blob/main/CITATION.cff).

## See Also

[StateTracker](https://statetracker.readthedocs.io): Composable states
for combinatorial enumeration (used internally by PoolParty).

## License

MIT License. See [LICENSE](LICENSE) for details.
