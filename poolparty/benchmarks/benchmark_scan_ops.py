"""Runtime benchmarks for scan operations using pytest-benchmark."""
from .workloads import workload_deletion_scan, workload_insertion_scan
from .benchmark_utils import generate_benchmark_tests

BENCHMARK_SPECS = {
    "DeletionScan": [
        # (workload, param_name, values, constants, enabled)
        (workload_deletion_scan, "seq_len", [10, 30, 100, 300, 1000], {}, True),
    ],
    "InsertionScan": [
        (workload_insertion_scan, "seq_len", [10, 30, 100, 300, 1000], {}, False),
        (workload_insertion_scan, "ins_len", [1, 3, 10, 30, 100], {}, True),
    ],
}

globals().update(generate_benchmark_tests(BENCHMARK_SPECS))
