# Pre-Publish Package Audit

## Phase 1: Dead / Deprecated Code Cleanup

### A1. Remove dead code

- ~~Remove unreachable `"hybrid"` mode branch in `region_scan.py` `_compute_core` (line ~233)~~ **DONE**
- ~~Remove `"hybrid"` from `shuffle_scan.py` docstring (line 47)~~ **DONE**
- ~~Remove unreachable `"hybrid"` mode branch in `mutagenize_orf.py` `_compute_core`~~ **DONE**
- ~~Rename `hybrid` test names to use `random_num_states` / `random_mode_with_num_states`~~ **DONE**
- ~~Audit `_defaults` / `get_default` / `set_default` / `load_defaults` in `party.py`~~ **DONE** — removed (dead code; `_defaults` was never read by any operation)

### A2. ~~Clean up deprecated shims~~ **DONE**

- ~~`party.py`: `counter_manager` property deleted, `_counter_manager` renamed to `_state_manager`~~ **DONE**
- ~~`party.py`: `set_default`/`get_default`/`load_defaults` and `_defaults` dict removed (dead code)~~ **DONE**
- ~~`pool.py`: `report_seq`→`show_seq`, `report_pool_seqs`→`show_pool_seqs`, `report_pool_states`→`show_pool_states`, `report_op_states`→`show_op_states`, `report_op_keys`→`show_op_keys`~~ **DONE**
- ~~`pool_mixins/__init__.py`: `BaseOpsMixin`, `FixedOpsMixin`, `OrfOpsMixin` files deleted and exports removed~~ **DONE**
- ~~`base_ops/filter_seq.py`: `filter_seq` alias removed, tests updated to use `filter`~~ **DONE**

### A3. Clean up placeholder tests

- 11 test methods with `pass` body referencing removed `breakpoint_scan` across `test_replacement_multiscan.py`, `test_deletion_scan.py`, `test_subseq_scan.py`, `test_replacement_scan.py`, `test_insertion_scan.py`, `test_join.py`, `test_mutagenize.py`, `test_deletion_multiscan.py`
- Either remove or replace with `@pytest.mark.skip(reason="...")`

---

## Phase 1.5: Packaging Fixes

### A4. Ensure non-Python files are included in built package

- Add `[tool.setuptools.package-data]` to `pyproject.toml` so `default_config.toml` is included:
  ```toml
  [tool.setuptools.package-data]
  poolparty = ["*.toml"]
  ```

### A5. Pin statetracker dependency

- Change `"statetracker"` to `"statetracker>=0.1.0,<1.0"` in `pyproject.toml`

### A6. Add `py.typed` marker

- Create empty `poolparty/src/poolparty/py.typed` for PEP 561 compliance

---

## Phase 2: New Features

Implement new features early so they are included in the subsequent group audits and consistency checks. Follow the new operation checklist in `.cursor/rules/new_operation.mdc`.

### ~~B1. Dinucleotide shuffle~~ **DONE**

- ~~Add `shuffle_type: Literal["mono", "dinuc"] = "mono"` to `shuffle_seq` and `shuffle_scan`~~ **DONE**
- ~~Implement Altschul-Erikson / Euler path algorithm for dinucleotide-preserving shuffle~~ **DONE** — `utils/shuffle_utils.py` with Wilson's loop-erased random walk for spanning arborescence
- ~~Wire through: mixin, submodule `__init__`, top-level `__init__`/`__all__`, tests~~ **DONE** — `shuffle_seq.py`, `shuffle_scan.py`, `common_ops_mixin.py`, `scan_ops_mixin.py`; 23 new tests in `test_dinuc_shuffle.py`

### B2. Orientation operation

- New operation: sequential mode produces 2 states (forward + rc), random mode randomly applies rc
- Design card: `{"orientation": "forward" | "rc"}`
- Supports `region`, `style`, `prefix`
- Wire through: mixin, submodule `__init__`, top-level `__init__`/`__all__`, `default_config.toml`, tests

### B3. Score operation

- Design discussion needed before implementation:
  - Storing user callables has implications for serialization (pickle) and reproducibility
  - Need error handling strategy: what if `fn` raises or returns non-numeric?
  - Consider whether `fn` should receive raw sequence or stripped (no tags) sequence
- Passthrough operation that applies a user-supplied `fn(seq) -> value` and records result in design card
- Design card key configurable (default `"score"`)
- Wire through: mixin, submodule `__init__`, top-level `__init__`/`__all__`, `default_config.toml`, tests

---

## Phase 3: Operation Group Audit

Systematic review of each group for: bugs, edge cases, missing validation, interface consistency within the group. New features from Phase 2 are included in the relevant group audits.

### C1. Source operations (`from_seq`, `from_seqs`, `from_fasta`, `from_iupac`, `from_motif`, `get_kmers`)

- Consistent parameter naming and types
- Edge cases: empty inputs, single-char sequences, invalid alphabets
- Error messages consistency

### C2. Mutation operations (`mutagenize`, `mutagenize_orf`, `recombine`)

- Parameter consistency (mode, num_states, style, region)
- Edge cases: num_mutations > seq_length, zero-length regions
- `recombine` style cycling correctness

### C3. Fixed operations (`rc`, `upper`, `lower`, `swapcase`, `join`, `slice_seq`, `stylize`, `clear_gaps`, `clear_annotation`, `remove_tags`, `add_prefix`, `orientation`, `score`)

- All should accept `region`, `style`, `prefix` consistently (where applicable)
- Edge cases: empty sequences, missing regions
- Includes new `orientation` and `score` operations

### C4. Region operations (`region_scan`, `region_multiscan`, `annotate_region`, `insert_tags`, `extract_region`, `replace_region`, `apply_at_region`)

- Add `@beartype` and type hints to `region_scan`, `region_multiscan`, `apply_at_region` (currently missing, unlike all other factory functions)
- Consistent parameter naming (verify `tag_name`/`tag_names` rename landed everywhere)
- Edge cases: overlapping regions, zero-length regions
- ~~`remove_tags`, `replace_region`, `_replace_keeping_tags`, `clear_tags` hardcode `seq_length=None`~~ **DONE** — all now compute deterministic output lengths (see bug log #22–#25)
- ~~`replacement_multiscan`/`insertion_multiscan` region name collision gives cryptic error~~ **DONE** — added descriptive hint to `ValueError` (see bug log #26)

### C5. Scan operations (`deletion_scan`, `insertion_scan`, `replacement_scan`, `shuffle_scan`, `mutagenize_scan`, `subseq_scan`)

- Consistent parameter order: `pool, length_param, positions, region, prefix, mode, num_states, style, iter_order`
- Consistent `_factory_name` support (missing from `deletion_scan`, `subseq_scan`)
- Edge cases: window larger than sequence, positions out of range
- Includes new dinucleotide shuffle mode in `shuffle_scan`
- ~~`mutagenize_scan` type hint rejects `None` inside `num_states` tuple~~ **DONE** — widened to `Sequence[Optional[Integral]]`, updated docstring (see bug log #27)

### C6. Multiscan operations (`deletion_multiscan`, `insertion_multiscan`, `replacement_multiscan`)

- Consistent parameter order and types with scan ops
- Consistent `_factory_name` support (currently missing from all multiscan ops)
- Edge cases: num_insertions > available positions, overlapping regions

### C7. State operations (`stack`, `sample`, `state_slice`, `state_shuffle`, `repeat`, `sync`)

- Consistent interface
- Edge cases: empty pools, zero states, slicing beyond range
- ~~`pp.sync` passed list to `st.sync` instead of pairwise calls; return type was `-> Pool` instead of `-> None`~~ **DONE** (see bug log #20–#21)

### C8. ORF operations (`translate`, `reverse_translate`, `mutagenize_orf`, `stylize_orf`, `annotate_orf`)

- Frame handling consistency
- Edge cases: non-divisible-by-3 sequences, stop codons

---

## Phase 4: Cross-Package Interface Consistency Check

### D0. Migrate remaining operations to `cards` parameter — **DONE** (scan/region ops)

- `cards: CardsType = None` added to factory + Op.__init__ + super().__init__() for: `from_seqs`, `from_iupac`, `from_motif`, `get_kmers`, `mutagenize`, `mutagenize_orf`, `shuffle_seq`, `recombine`, `filter`, `materialize`, `repeat`, `stack`
- ~~`region_scan`, `region_multiscan`~~ **DONE** — migrated to new cards system, replaced `cards_suppressed()` with `self._party.suppress_cards`
- ~~Scan ops: `mutagenize_scan`, `shuffle_scan` (tuple cards), `deletion_scan`, `insertion_scan`, `replacement_scan`, `subseq_scan` (simple cards)~~ **DONE** — `cards` param threaded through to inner `region_scan` calls
- ~~Multiscan consumers: `deletion_multiscan`, `insertion_multiscan`, `replacement_multiscan` (simple cards)~~ **DONE** — `cards` param threaded through to inner `region_multiscan` calls
- ~~`scan_ops_mixin.py`~~ **DONE** — all 9 mixin methods updated
- Remaining (no meaningful `design_card_keys`; migration deferred): `from_seq`, `from_fasta`, `rc`, `upper`, `lower`, `swapcase`, `clear_gaps`, `clear_annotation`, `stylize`, `stylize_orf`, `slice_seq`, `translate`, `reverse_translate`, `annotate_orf`, `insert_tags`, `remove_tags`, `replace_region`, `apply_at_region`, `state_slice`, `state_shuffle`, `sample`, `sync`

### D1. Parameter naming audit

- `mode` type: should be `ModeType` everywhere (already fixed for multiscan, verify all)
- `region` type: should be `RegionType` everywhere
- `positions` type: should be `PositionsType` / `MultiPositionsType` consistently
- `style` parameter: verify availability across all ops that modify sequences
- `prefix` parameter: verify availability across all ops

### D2. Return type consistency

- All factory functions should return `Pool` (not `DnaPool` or `ProteinPool` directly)
- Pool type should be preserved from input

### D3. Error message consistency

- Standardize error message patterns (e.g., "X must be Y, got Z")
- Ensure all user-facing errors use `ValueError` / `TypeError` appropriately

### D4. Version alignment

- `pyproject.toml` says `0.1.0`, `__init__.py` says `0.3.0` -- align

---

## Phase 5: Test Coverage

### E1. Add `test_mutagenize_scan.py`

- Random/sequential mode, region constraint, style, scalar broadcast warnings, Cartesian product state count

### E2. Add tests for `clear_gaps`, `clear_annotation`

### E3. Add basic `text_viz` smoke tests

---

## Phase 6: Documentation

Fix docs last since interfaces may change during earlier phases.

### F1. Fix README.md

- `replacement_scan` example uses wrong params (`replacement_pool`, `replacement_length` -> `ins_pool`)

### F2. Fix `docs/quickstart.rst` and `docs/index.rst`

- Remove non-existent `alphabet="ACGT"` parameter from `get_kmers` and `mutagenize` examples
- Fix `mutagenize` examples (`k=1` -> `num_mutations=1`)
- Fix `deletion_scan` examples (remove `step`)
- Fix `replacement_scan` examples

### F0. Add docstrings to mixin wrapper methods

- Most mixin methods in `common_ops_mixin.py`, `scan_ops_mixin.py`, `generic_fixed_ops_mixin.py`, `dna_mixin.py`, `region_ops_mixin.py`, `state_ops_mixin.py` lack docstrings
- IDE hover shows nothing for fluent API calls like `pool.mutagenize(...)`
- Either add short docstrings or inherit from the factory functions

### F3. Add missing docs pages

- Add `docs/operations/subseq_scan.rst`
- Add docs pages for new operations (orientation, score)

---

## Phase 7: Pre-Publish Checklist

### G1. CHANGELOG update

- Update `[Unreleased]` section with all changes from this audit

### G2. Final verification

- `uv run pytest` -- all tests pass
- `uv run ruff check .` -- no lint errors
- `python -m build` -- package builds cleanly
- `twine check dist/*` -- metadata valid

---

## Known Accepted Issues

- `poolparty.filter` shadows the Python builtin `filter()`. This only affects `from poolparty import *` (discouraged); fluent API (`pool.filter(...)`) is unaffected. Not worth a breaking rename.
