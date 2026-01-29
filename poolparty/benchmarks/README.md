# Poolparty Benchmarks

Performance profiling suite for poolparty runtime and memory analysis.

## Quick Start

```bash
# Install benchmark dependencies
uv sync --group benchmark

# Run small benchmarks (fast)
uv run pytest poolparty/benchmarks/ -v

# Run all benchmarks including slow ones
uv run pytest poolparty/benchmarks/ -v --run-slow
```

## Benchmark Structure

- `workloads.py` - Parameterized benchmark workloads
- `benchmark_base_ops.py` - Runtime benchmarks for base operations
- `benchmark_scan_ops.py` - Runtime benchmarks for scan operations
- `benchmark_utils.py` - Utilities for generating benchmark tests
- `run_benchmarks.py` - Run benchmarks and export to CSV
- `test_memory.py` - Memory profiling using tracemalloc
- `test_scalability.py` - Scalability tests for various parameters
- `run_profile.py` - CLI for ad-hoc profiling

## Workloads

Available workloads (defined in `workloads.py`):

| Workload | Description |
|----------|-------------|
| `mutagenize` | Random mutagenesis |
| `mutagenize_sequential` | Sequential enumeration of all mutations |
| `recombine` | Recombination with breakpoints |
| `deletion_scan` | Deletion scanning |
| `insertion_scan` | Insertion scanning |
| `complex_dag` | Multi-operation DAG (mutagenize + join + barcode) |
| `region_operations` | Operations on tagged regions |
| `stack` | Stacking multiple pools |
| `get_kmers` | K-mer generation |

## Runtime Benchmarks

```bash
# Run with pytest-benchmark
uv run pytest poolparty/benchmarks/test_runtime.py -v

# Compare against baseline
uv run pytest poolparty/benchmarks/test_runtime.py --benchmark-compare

# Save results to JSON
uv run pytest poolparty/benchmarks/test_runtime.py --benchmark-json=results.json
```

## Export Benchmarks to CSV

Use `run_benchmarks.py` to run benchmarks and export results to CSV:

```bash
# Run benchmarks and save to benchmark_base_ops.results.csv
uv run python poolparty/benchmarks/run_benchmarks.py benchmark_base_ops.py

# Also print a formatted table to stdout
uv run python poolparty/benchmarks/run_benchmarks.py benchmark_base_ops.py --table

# Specify custom output path
uv run python poolparty/benchmarks/run_benchmarks.py benchmark_base_ops.py -o results.csv
```

The CSV contains: `test_name`, `mean`, `stddev`, `min`, `max`, `rounds`.

## Memory Benchmarks

```bash
# Run memory tests (uses tracemalloc)
uv run pytest poolparty/benchmarks/test_memory.py -v -s

# Profile with memray (more detailed)
uv run python -m poolparty.benchmarks.run_profile mutagenize --memray

# Generate flamegraph
uv run memray flamegraph poolparty/benchmarks/profiles/mutagenize.bin
```

## Ad-hoc Profiling

Use `run_profile.py` for interactive profiling:

```bash
# Profile with pyinstrument (call tree)
uv run python poolparty/benchmarks/run_profile.py mutagenize

# Profile with cProfile (detailed stats)
uv run python poolparty/benchmarks/run_profile.py mutagenize --cprofile

# Profile with memray (memory)
uv run python poolparty/benchmarks/run_profile.py mutagenize --memray

# Custom parameters
uv run python poolparty/benchmarks/run_profile.py mutagenize --seq-len 200 --num-seqs 5000

# List available workloads
uv run python poolparty/benchmarks/run_profile.py --list
```

## Scalability Tests

```bash
# Run scalability tests
uv run pytest poolparty/benchmarks/test_scalability.py -v -s

# Test specific scaling dimension
uv run pytest poolparty/benchmarks/test_scalability.py::TestScaleNumSeqs -v -s
uv run pytest poolparty/benchmarks/test_scalability.py::TestScaleSeqLength -v -s
uv run pytest poolparty/benchmarks/test_scalability.py::TestScaleNumMutations -v -s
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
