#!/usr/bin/env python
"""CLI script for ad-hoc profiling of poolparty workloads.

Usage:
    # Profile with pyinstrument (call tree)
    uv run python poolparty/benchmarks/run_profile.py mutagenize
    
    # Profile with cProfile (detailed stats)
    uv run python poolparty/benchmarks/run_profile.py mutagenize --cprofile
    
    # Profile with memray (memory)
    uv run python poolparty/benchmarks/run_profile.py mutagenize --memray
    
    # List available workloads
    uv run python poolparty/benchmarks/run_profile.py --list
    
    # Custom parameters
     
"""
import argparse
import sys
from pathlib import Path

# Add benchmarks directory to path for imports
BENCHMARKS_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCHMARKS_DIR))

PROFILES_DIR = BENCHMARKS_DIR / "profiles"


def list_workloads():
    """Print available workloads."""
    from timing import ALL_WORKLOADS
    print("Available workloads:")
    for name in ALL_WORKLOADS:
        print(f"  - {name}")


def profile_pyinstrument(workload_func, **kwargs):
    """Profile with pyinstrument (call tree visualization)."""
    from pyinstrument import Profiler
    
    profiler = Profiler()
    profiler.start()
    
    try:
        result = workload_func(**kwargs)
    finally:
        profiler.stop()
    
    print(profiler.output_text(unicode=True, color=True))
    return result


def profile_cprofile(workload_func, output_file=None, **kwargs):
    """Profile with cProfile (detailed stats)."""
    import cProfile
    import pstats
    from io import StringIO
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        result = workload_func(**kwargs)
    finally:
        profiler.disable()
    
    # Print top 30 functions by cumulative time
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(30)
    print(stream.getvalue())
    
    # Save to file if requested
    if output_file:
        stats.dump_stats(str(output_file))
        print(f"\nProfile saved to: {output_file}")
    
    return result


def profile_memray(workload_func, workload_name, **kwargs):
    """Profile with memray (memory tracking)."""
    try:
        import memray
    except ImportError:
        print("memray not installed. Install with: uv add --group benchmark memray")
        sys.exit(1)
    
    output_file = PROFILES_DIR / f"{workload_name}.bin"
    
    print(f"Running with memray tracking...")
    print(f"Output: {output_file}")
    
    with memray.Tracker(str(output_file)):
        result = workload_func(**kwargs)
    
    print(f"\nDone. Generate reports with:")
    print(f"  memray flamegraph {output_file}")
    print(f"  memray summary {output_file}")
    print(f"  memray tree {output_file}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Profile poolparty workloads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "workload",
        nargs="?",
        help="Workload name to profile",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available workloads",
    )
    parser.add_argument(
        "--cprofile",
        action="store_true",
        help="Use cProfile instead of pyinstrument",
    )
    parser.add_argument(
        "--memray",
        action="store_true",
        help="Use memray for memory profiling",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=100,
        help="Sequence length (default: 100)",
    )
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=1000,
        help="Number of sequences (default: 1000)",
    )
    parser.add_argument(
        "--num_mut",
        type=str,
        default='None',
        help="Number of mutations (default: None)",
    )
    parser.add_argument(
        "--mut_rate",
        type=str,
        default='None',
        help="Mutations rate (default: None)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for profile data",
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_workloads()
        return
    
    if not args.workload:
        parser.print_help()
        return
    
    from timing import ALL_WORKLOADS
    
    if args.workload not in ALL_WORKLOADS:
        print(f"Unknown workload: {args.workload}")
        list_workloads()
        sys.exit(1)
    
    workload_func = ALL_WORKLOADS[args.workload]
    
    # Build kwargs based on workload
    kwargs = {}
    if 'seq_len' in workload_func.__code__.co_varnames:
        kwargs['seq_len'] = args.seq_len
    if 'num_seqs' in workload_func.__code__.co_varnames:
        kwargs['num_seqs'] = args.num_seqs
    if 'num_mut' in workload_func.__code__.co_varnames:
        kwargs['num_mut'] = None if args.num_mut=='None' else int(args.num_mut)
    if 'mut_rate' in workload_func.__code__.co_varnames:
        kwargs['mut_rate'] = None if args.mut_rate=='None' else float(args.mut_rate)
    
    print(f"Profiling: {args.workload}")
    print(f"Parameters: {kwargs}")
    print("-" * 50)
    
    # Run with selected profiler
    if args.memray:
        result = profile_memray(workload_func, args.workload, **kwargs)
    elif args.cprofile:
        output_file = Path(args.output) if args.output else PROFILES_DIR / f"{args.workload}.prof"
        result = profile_cprofile(workload_func, output_file, **kwargs)
    else:
        result = profile_pyinstrument(workload_func, **kwargs)
    
    print(f"\nGenerated {len(result)} sequences")


if __name__ == "__main__":
    main()
