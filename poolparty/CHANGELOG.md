# Changelog

All notable changes to PoolParty will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-03

Initial release. See [Liu, Cordero, and Kinney (2026)](https://doi.org/XXXX) for a full description.

### Added
- Declarative, DAG-based library design with lazy sequence generation
- Three operation modes: sequential (exhaustive enumeration), random (stochastic sampling), and fixed (deterministic)
- 50+ composable operations across four functional categories:
  - **Source**: create pools from sequences, IUPAC codes, motifs, k-mers, and constrained barcodes
  - **Transformation**: nucleotide and codon-level mutagenesis, shuffling, recombination, and positional scanning (deletion, insertion, replacement)
  - **Composition**: concatenation (join) and pool merging (stack)
  - **State**: select, reorder, filter, replicate, synchronize, and score sequences
- Region tagging with XML-style syntax for targeting operations to specific sequence regions
- Design cards for structured sequence provenance tracking
- Sequence naming via configurable prefix system
- Composable sequence styling with ANSI escape codes
- Export to DataFrame, CSV, and FASTA
- DAG visualization with `print_dag`
