# Poolparty Benchmarks

Performance profiling suite for poolparty runtime and memory analysis.

## Quick Start

```bash
# Install benchmark dependencies
uv sync --group benchmark

# Run all timing benchmarks
uv run pytest benchmarks/timing/all.py -v

# Run specific category
uv run pytest benchmarks/timing/base_ops.py -v
```

## Benchmark Structure

```
benchmarks/
├── timing/                    # Timing benchmarks (self-runnable)
│   ├── __init__.py            # Auto-discovers modules, exports ALL_WORKLOADS
│   ├── _utils.py              # Shared utilities
│   ├── base_ops.py            # mutagenize, shuffle_seq, get_kmers, from_iupac, recombine
│   ├── scan_ops.py            # deletion_scan, insertion_scan
│   ├── dag_ops.py             # chain_of_joins, tree_of_joins
│   ├── examples.py            # mpra_example
│   └── all.py                 # Aggregates all benchmarks
├── benchmark_utils.py         # Utilities for generating benchmark tests
├── run_benchmarks.py          # Run benchmarks and export to CSV
├── run_profile.py             # CLI for ad-hoc profiling
├── test_memory.py             # Memory profiling using tracemalloc
└── test_scalability.py        # Scalability tests for various parameters
```

## Timing Benchmarks

Each file in `timing/` is self-runnable with pytest:

```bash
# Run specific category
uv run pytest benchmarks/timing/base_ops.py -v      # Base operations
uv run pytest benchmarks/timing/scan_ops.py -v      # Scan operations
uv run pytest benchmarks/timing/dag_ops.py -v       # DAG operations
uv run pytest benchmarks/timing/examples.py -v      # Complex examples

# Run ALL timing benchmarks
uv run pytest benchmarks/timing/all.py -v

# Run specific test class
uv run pytest benchmarks/timing/base_ops.py::TestMutagenize -v

# Disable actual benchmarking (just verify tests work)
uv run pytest benchmarks/timing/all.py --benchmark-disable -v
```

### Available Workloads

| File | Workloads | Test Classes |
|------|-----------|--------------|
| `base_ops.py` | mutagenize, shuffle_seq, get_kmers, from_iupac, recombine | TestMutagenize, TestShuffleSeq, TestGetKmers, TestFromIupac, TestRecombine |
| `scan_ops.py` | deletion_scan, insertion_scan | TestDeletionScan, TestInsertionScan |
| `dag_ops.py` | chain_of_joins, tree_of_joins | TestDAGSize |
| `examples.py` | mpra_example | TestMPRAExample |

### Adding New Benchmarks

1. Add a workload function with `workload_` prefix
2. Attach `.benchmark_specs` attribute with test specifications:

```python
def workload_my_operation(seq_len: int = 100, num_seqs: int = 100):
    # ... implementation ...
    pass

workload_my_operation.benchmark_specs = [
    # (TestClassName, param_name, param_values)
    ("TestMyOperation", "seq_len", [10, 30, 100, 300]),
    # With constants: (TestClassName, param_name, values, {constants})
    ("TestMyOperation", "num_seqs", [10, 100, 1000], {"seq_len": 50}),
]
```

The test classes are auto-generated when the module is imported.

## Export Benchmarks to CSV

Use `run_benchmarks.py` to run benchmarks and export results:

```bash
# Run and save to CSV
uv run python benchmarks/run_benchmarks.py timing/base_ops.py

# Print formatted table to stdout
uv run python benchmarks/run_benchmarks.py timing/base_ops.py --table

# Run specific test class
uv run python benchmarks/run_benchmarks.py timing/base_ops.py -c TestMutagenize --table

# Specify custom output path
uv run python benchmarks/run_benchmarks.py timing/all.py -o results.csv
```

The CSV contains: `test_name`, `mean`, `stddev`, `min`, `max`, `rounds`.

## Ad-hoc Profiling

Use `run_profile.py` for interactive profiling:

```bash
# Profile with pyinstrument (call tree)
uv run python benchmarks/run_profile.py mutagenize

# Profile with cProfile (detailed stats)
uv run python benchmarks/run_profile.py mutagenize --cprofile

# Profile with memray (memory)
uv run python benchmarks/run_profile.py mutagenize --memray

# Custom parameters
uv run python benchmarks/run_profile.py mutagenize --seq-len 200 --num-seqs 5000

# List available workloads
uv run python benchmarks/run_profile.py --list
```

## Memory Benchmarks

```bash
# Run memory tests (uses tracemalloc)
uv run pytest benchmarks/test_memory.py -v -s

# Profile with memray (more detailed)
uv run python benchmarks/run_profile.py mutagenize --memray

# Generate flamegraph
uv run memray flamegraph benchmarks/profiles/mutagenize.bin
```

## Scalability Tests

```bash
# Run scalability tests
uv run pytest benchmarks/test_scalability.py -v -s

# Test specific scaling dimension
uv run pytest benchmarks/test_scalability.py::TestScaleNumSeqs -v -s
uv run pytest benchmarks/test_scalability.py::TestScaleSeqLength -v -s
uv run pytest benchmarks/test_scalability.py::TestScaleNumMutations -v -s
```

## Interpreting Results

### Runtime

pytest-benchmark provides:
- **min/max/mean/stddev** - Statistical timing data
- **rounds** - Number of iterations
- **ops/sec** - Operations per second

### Memory

tracemalloc reports:
- **current** - Memory currently allocated
- **peak** - Maximum memory used during execution

memray provides:
- Flamegraphs showing allocation call stacks
- Summary of top allocators
- Leak detection

## Performance Tips

Key areas that affect performance:

1. **Sequential mode cache size** - Grows as C(n,k) × 3^k for mutagenize
2. **DAG complexity** - More operations = more per-sequence overhead
3. **Region parsing** - Tagged sequences have coordinate conversion overhead
4. **Style tracking** - Can be disabled with `pp.toggle_styles(on=False)`
5. **Design cards** - Can be disabled with `pp.toggle_cards(on=False)`
