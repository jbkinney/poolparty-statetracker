# Changelog

All notable changes to PoolParty will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD workflow for automated testing
- Ruff linter configuration
- Dependabot for automated dependency updates

## [0.1.0] - 2026-01-30

Initial release of PoolParty for declarative DNA library design.

### Added
- Core Pool class for composable sequence generation
- DAG-based library design with automatic state tracking via StateTracker
- Logging support with configurable verbosity levels

#### Base Operations
- `from_seq()` - Single sequence pool
- `from_seqs()` - Multiple sequences pool
- `from_fasta()` - Load sequences from FASTA files
- `from_iupac()` - Generate sequences from IUPAC ambiguity codes
- `from_motif()` - Generate sequences from motif patterns
- `get_kmers()` - Generate all k-mers of specified length
- `mutagenize()` - Apply random mutations to sequences
- `shuffle_seq()` - Shuffle sequence positions
- `recombine()` - Evolutionary recombination simulation with style cycling

#### Scan Operations
- `insertion_scan()` - Tiled insertion mutagenesis
- `deletion_scan()` - Tiled deletion mutagenesis
- `replacement_scan()` - Tiled replacement mutagenesis
- `shuffle_scan()` - Positional shuffle scanning
- `mutagenize_scan()` - Positional mutagenesis scanning
- `subseq_scan()` - Subsequence extraction scanning

#### Region Operations
- Region tagging with XML-like syntax
- `tag_region()` - Add region annotations
- `extract_region()` - Extract tagged regions
- `replace_region()` - Replace region content
- `replacement_multiscan()` - Multi-region scanning

#### ORF Operations
- `mutagenize_orf()` - Codon-aware mutagenesis preserving reading frames

#### State Operations (via StateTracker)
- `state_slice()` - Slice state space
- `state_shuffle()` - Shuffle state ordering
- `state_sample()` - Sample from state space
- `state_repeat()` - Repeat state patterns

#### Utilities
- `generate_library()` - Generate complete library with metadata
- TOML-based configuration system for column filtering
- `toggle_styles()` - Enable/disable inline styling overhead
- `toggle_cards()` - Control design card display
- DataFrame output with sequence provenance tracking
- Text visualization for DAG structure

### Changed
- `remove_tags` parameter defaults to `False` (preserve region tags in output)

### Performance
- Optimized `RegionContext` with cached parsed regions (~25% speedup)
- Optimized `recombine()` performance (31% improvement)
- Optimized `shuffle_seq()` with numpy array caching (40% speedup)
- Optimized sequence parsing with fast paths for tag-free strings
- Vectorized `_random_mutation()` for improved performance
- Removed beartype from hot paths for better runtime performance

### Fixed
- Stack naming bug where inactive branch prefixes leaked into names
- Integer overflow in `from_iupac()` random mode
- Benchmark errors and beartype usage cleanup
