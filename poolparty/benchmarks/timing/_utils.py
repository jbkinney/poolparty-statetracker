"""Shared utilities for timing benchmarks."""


def make_sequence(length: int) -> str:
    """Generate a DNA sequence of specified length."""
    bases = 'ACGT'
    return (bases * (length // 4 + 1))[:length]


def collect_local_specs(module_globals: dict) -> dict:
    """Collect benchmark specs from workload functions in the given module."""
    specs = {}
    for name, obj in module_globals.items():
        if name.startswith('workload_') and hasattr(obj, 'benchmark_specs'):
            for spec in obj.benchmark_specs:
                test_class, param, values = spec[:3]
                constants = spec[3] if len(spec) > 3 else {}
                if test_class not in specs:
                    specs[test_class] = []
                specs[test_class].append((obj, param, values, constants, True))
    return specs
