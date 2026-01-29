"""Timing benchmarks for poolparty profiling.

This module auto-discovers all workload_* functions from public modules
and builds ALL_WORKLOADS and collect_benchmark_specs() automatically.
"""
import importlib
import pkgutil
from pathlib import Path

# Auto-discover all public modules (exclude _utils.py, __init__.py, all.py)
_workload_modules = []
for module_info in pkgutil.iter_modules([str(Path(__file__).parent)]):
    if not module_info.name.startswith('_') and module_info.name != 'all':
        mod = importlib.import_module(f'.{module_info.name}', __package__)
        _workload_modules.append(mod)

# Build ALL_WORKLOADS from discovered modules
ALL_WORKLOADS = {}
for mod in _workload_modules:
    for name in dir(mod):
        if name.startswith('workload_'):
            fn = getattr(mod, name)
            key = name.replace('workload_', '')
            ALL_WORKLOADS[key] = fn


def collect_benchmark_specs() -> dict:
    """Collect benchmark specs from all workload functions."""
    specs = {}
    for name, workload_fn in ALL_WORKLOADS.items():
        if hasattr(workload_fn, 'benchmark_specs'):
            for spec in workload_fn.benchmark_specs:
                test_class, param, values = spec[:3]
                constants = spec[3] if len(spec) > 3 else {}
                if test_class not in specs:
                    specs[test_class] = []
                specs[test_class].append((workload_fn, param, values, constants, True))
    return specs
