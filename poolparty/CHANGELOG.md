# Changelog

All notable changes to PoolParty will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING:** ORF operations now share one documented frame-offset
  convention. `mutagenize_orf` and `stylize_orf` previously placed
  the first complete codon `(4 - |frame|) % 3` bases into the region, while
  `translate` used `|frame| - 1`. The two agree only at `|frame| == 1`; for
  `|frame|` of 2 or 3 they were exactly swapped, so one `OrfRegion.frame`
  selected different codons depending on which operation read it.

  The unification covers the frame offset for unambiguous sequences and
  consistently resolved named regions. Two pre-existing sources of divergence
  are **not** addressed here and remain: `stylize_orf` excludes IUPAC ambiguity
  codes from its molecular positions where `translate` and `mutagenize_orf`
  include them, and the three operations interpret an interval `region=[a, b]`
  in different coordinate systems (molecular, nontag and literal respectively),
  which differ once a sequence contains gaps.

  All operations now follow `translate`'s convention, which matches the
  `OrfRegion` docstring and standard six-frame usage: **the first complete
  codon begins at base `|frame|` of the region**, counted from the 5' end for
  positive frames and from the 3' end for negative frames.

  Consequences at `|frame|` of 2 or 3:

  - For `mutagenize_orf`, old `±2` codon geometry is now obtained with `±3`
    and vice versa. `stylize_orf` codon boundaries undergo the same swap, but
    its complete rendered output is *not* equivalent under that swap, because
    orphan styling and codon-style numbering also change (below).
  - The number of complete mutable codons changes only when the region length
    is congruent to 1 modulo 3. Depending on mutation mode, eligible
    positions, and `num_states`, this may change the enumerated state count or
    variant count. For other lengths the count may be unchanged even though
    different nucleotides are mutated.
  - Most visibly, `mutagenize_orf(mutation_type="nonsense")` previously failed
    to introduce a premature stop into the protein produced by `translate()` at
    `|frame|` of 2 or 3, because the stop was written one nucleotide away from
    where `translate` reads. On the sequences tested this was every variant; an
    off-grid substitution could in principle create an in-frame stop by
    coincidence.

  Libraries designed at `frame=±1` are unaffected.

- **BREAKING:** `stylize_orf` no longer styles orphan bases — the frame offset
  at the start of the reading direction, and any trailing remainder shorter
  than a complete codon. This applies at **every** frame, including `±1`,
  where a trailing partial codon was previously styled.
- `stylize_orf` codon numbering: a leading partial group no longer consumes
  `style_codons[0]`, so `style_codons[i]` now indexes the same codon as
  `mutagenize_orf`'s `codon_positions=i`.
- **BREAKING:** `replace_region` now defaults to `sync=True` and
  `keep_tags=True` (previously both `False`). Code relying on the old
  Cartesian-product or tag-stripping behavior must pass `sync=False`
  and/or `keep_tags=False` explicitly.
- Vectorized `get_barcodes` internals with NumPy for significantly
  faster barcode generation at scale.

### Added
- `orf_ops/_frame.py`, holding the single `frame_offset()` and `resolve_frame()`
  used by the three frame-aware operations (`translate`, `mutagenize_orf`,
  `stylize_orf`). `resolve_frame` was previously defined once per operation
  module; `annotate_orf` and `reverse_translate` use neither helper.
- `tests/test_orf_frame_consistency.py`: hand-derived codon anchors and
  cross-operation agreement tests over all six frames, plus orphan-base and
  end-to-end nonsense tests.
- DMS (protein GB1) and MPRA (regulatory grammar) tutorial pages.

### Fixed
- `stylize_orf(style_frames=...)` took the style group from an unshifted index
  and the position-within-codon from a shifted one, so at `|frame|` of 2 or 3 a
  single codon could straddle two style groups. Both are now derived from the
  same index.
- Fixed CSV/TSV export writing `\r\n` line endings on Windows.

### Removed
- `StylizeOrfOp.region_frame`. The attribute held an internal grid shift that no
  longer exists; codon geometry now comes from `frame_offset()`. `StylizeOrfOp`
  is exported at package level, so this is a public attribute removal.
- Stale scalar helpers `_check_gc_content` and `_check_homopolymer`
  from `get_barcodes` (replaced by vectorized batch equivalents).

## [0.1.1] - 2026-04-06

### Fixed
- Added `typing_extensions` to dependencies.
- Bumped statetracker dependency to v0.1.1.

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
