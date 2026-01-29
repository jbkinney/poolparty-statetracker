"""Runtime benchmarks for base operations using pytest-benchmark."""
from .workloads import (
    workload_mutagenize,
    workload_shuffle_seq,
    workload_get_kmers,
    workload_from_iupac,
    workload_recombine,
)
from .benchmark_utils import generate_benchmark_tests

BENCHMARK_SPECS = {
    "TestMutagenize": [
        # (workload, param_name, values, constants, enabled)
        (workload_mutagenize, "num_mut", [1, 3, 10, 30, 100], {}, True), 
        (workload_mutagenize, "mut_rate", [0.01, 0.03, 0.10, 0.30, 1.00], {}, True),
        (workload_mutagenize, "seq_len", [10, 30, 100, 300, 1_000, 3_000], dict(mut_rate=0.1), True),
    ],
    "TestShuffleSeq": [
        (workload_shuffle_seq, "seq_len", [10, 30, 100, 300, 1000], {}, True),
    ],
    "TestGetKmers": [
        (workload_get_kmers, "kmer_len", [1, 3, 10, 30, 100], {}, True),
    ],
    "TestFromIupac": [
        (workload_from_iupac, "seq_len", [1, 3, 10, 30, 100], {}, True),
    ],
    "TestRecombine": [
        (workload_recombine, "seq_len", [10, 30, 100, 300, 1000], {}, True),
        (workload_recombine, "num_breakpoints", [1, 3, 10, 30, 99], {}, True),
        (workload_recombine, "num_sources", [2, 4, 10, 30, 100], {}, True),
    ],
}

# Generate and inject test classes into module namespace
globals().update(generate_benchmark_tests(BENCHMARK_SPECS))
