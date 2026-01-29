#!/usr/bin/env python
"""Run benchmarks and export results to CSV."""
import argparse
import json
import csv
import subprocess
import tempfile
from pathlib import Path

FIELDS = ['test_name', 'mean', 'stddev', 'min', 'max', 'rounds']

def run_pytest_benchmark(benchmark_target: str, json_output: str) -> bool:
    """Run pytest-benchmark and save JSON output."""
    cmd = [
        'uv', 'run', 'pytest', benchmark_target,
        f'--benchmark-json={json_output}',
        '-v'
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0

def load_benchmarks(json_path: str) -> list[dict]:
    """Load benchmark data from JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    
    rows = []
    for benchmark in data['benchmarks']:
        rows.append({
            'test_name': benchmark['name'],
            'mean': benchmark['stats']['mean'],
            'stddev': benchmark['stats']['stddev'],
            'min': benchmark['stats']['min'],
            'max': benchmark['stats']['max'],
            'rounds': benchmark['stats']['rounds'],
        })
    return rows

def format_time_ms(seconds: float) -> str:
    """Format time value in milliseconds with consistent decimal places."""
    ms = seconds * 1000
    return f"{ms:.2f}"

def print_table(rows: list[dict]):
    """Print benchmarks as a formatted table to stdout."""
    # Headers with units
    headers = ['Test Name', 'Mean (ms)', 'StdDev (ms)', 'Min (ms)', 'Max (ms)', 'Rounds']
    # Which columns are right-aligned (numeric)
    right_align = [False, True, True, True, True, True]
    
    formatted = []
    for row in rows:
        formatted.append([
            row['test_name'],
            format_time_ms(row['mean']),
            format_time_ms(row['stddev']),
            format_time_ms(row['min']),
            format_time_ms(row['max']),
            str(row['rounds']),
        ])
    
    widths = [len(h) for h in headers]
    for row in formatted:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))
    
    print()
    header_line = "  ".join(
        h.rjust(widths[i]) if right_align[i] else h.ljust(widths[i])
        for i, h in enumerate(headers)
    )
    print(header_line)
    print("-" * len(header_line))
    for row in formatted:
        print("  ".join(
            val.rjust(widths[i]) if right_align[i] else val.ljust(widths[i])
            for i, val in enumerate(row)
        ))

def write_csv(rows: list[dict], csv_path: str):
    """Write benchmarks to CSV file."""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} benchmarks to {csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks and export to CSV')
    parser.add_argument('benchmark_file', help='Benchmark file to run (e.g., benchmark_base_ops.py)')
    parser.add_argument('-c', '--class-name', help='Test class to run (e.g., TestMutagenize)')
    parser.add_argument('-t', '--table', action='store_true', help='Print results as formatted table')
    parser.add_argument('-o', '--output', help='Output CSV file (default: <benchmark_file>.results.csv)')
    args = parser.parse_args()
    
    benchmark_path = Path(args.benchmark_file)
    if not benchmark_path.exists():
        # Try relative to benchmarks directory
        benchmark_path = Path(__file__).parent / args.benchmark_file
    
    if not benchmark_path.exists():
        print(f"Error: Benchmark file not found: {args.benchmark_file}")
        return 1
    
    # Build pytest target (file or file::ClassName)
    if args.class_name:
        benchmark_target = f"{benchmark_path}::{args.class_name}"
        # Include class name in default output path
        default_csv = benchmark_path.with_suffix(f'.{args.class_name}.results.csv')
    else:
        benchmark_target = str(benchmark_path)
        default_csv = benchmark_path.with_suffix('.results.csv')
    
    csv_path = args.output or str(default_csv)
    
    # Run benchmarks with temp JSON file
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        json_path = tmp.name
    
    try:
        print(f"Running benchmarks: {benchmark_target}")
        if not run_pytest_benchmark(benchmark_target, json_path):
            print("Warning: Some benchmarks may have failed")
        
        rows = load_benchmarks(json_path)
        
        if args.table:
            print_table(rows)
        
        write_csv(rows, csv_path)
        
    finally:
        Path(json_path).unlink(missing_ok=True)
    
    return 0

if __name__ == '__main__':
    exit(main())
