"""Run all timing benchmarks."""
from . import collect_benchmark_specs
from ..benchmark_utils import generate_benchmark_tests

globals().update(generate_benchmark_tests(collect_benchmark_specs()))
