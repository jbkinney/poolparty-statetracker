"""Memory profiling tests using tracemalloc and memray."""
import tracemalloc
import pytest
from pathlib import Path
from .timing import (
    workload_mutagenize,
    workload_mutagenize_sequential,
    workload_recombine,
    workload_complex_dag,
    WORKLOAD_SIZES,
)

PROFILES_DIR = Path(__file__).parent / "profiles"


def measure_peak_memory(func, *args, **kwargs):
    """Measure peak memory usage of a function call."""
    tracemalloc.start()
    try:
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        return result, current, peak
    finally:
        tracemalloc.stop()


class TestMemorySmall:
    """Memory profiling for small workloads."""
    
    def test_mutagenize_memory(self):
        result, current, peak = measure_peak_memory(
            workload_mutagenize,
            seq_len=WORKLOAD_SIZES['small']['seq_len'],
            num_seqs=WORKLOAD_SIZES['small']['num_seqs'],
        )
        assert len(result) == WORKLOAD_SIZES['small']['num_seqs']
        # Log memory usage (in MB)
        print(f"\nMutagenize small: current={current/1e6:.2f}MB, peak={peak/1e6:.2f}MB")
    
    def test_mutagenize_sequential_memory(self):
        result, current, peak = measure_peak_memory(
            workload_mutagenize_sequential, seq_len=20, num_mut=1
        )
        assert len(result) == 60
        print(f"\nMutagenize sequential (20bp, 1mut): current={current/1e6:.2f}MB, peak={peak/1e6:.2f}MB")
    
    def test_recombine_memory(self):
        result, current, peak = measure_peak_memory(
            workload_recombine,
            seq_len=WORKLOAD_SIZES['small']['seq_len'],
            num_seqs=WORKLOAD_SIZES['small']['num_seqs'],
        )
        assert len(result) == WORKLOAD_SIZES['small']['num_seqs']
        print(f"\nRecombine small: current={current/1e6:.2f}MB, peak={peak/1e6:.2f}MB")
    
    def test_complex_dag_memory(self):
        result, current, peak = measure_peak_memory(
            workload_complex_dag,
            seq_len=WORKLOAD_SIZES['small']['seq_len'],
            num_seqs=WORKLOAD_SIZES['small']['num_seqs'],
        )
        assert len(result) == WORKLOAD_SIZES['small']['num_seqs']
        print(f"\nComplex DAG small: current={current/1e6:.2f}MB, peak={peak/1e6:.2f}MB")


class TestMemoryMedium:
    """Memory profiling for medium workloads."""
    
    def test_mutagenize_memory(self):
        result, current, peak = measure_peak_memory(
            workload_mutagenize,
            seq_len=WORKLOAD_SIZES['medium']['seq_len'],
            num_seqs=WORKLOAD_SIZES['medium']['num_seqs'],
        )
        assert len(result) == WORKLOAD_SIZES['medium']['num_seqs']
        print(f"\nMutagenize medium: current={current/1e6:.2f}MB, peak={peak/1e6:.2f}MB")
    
    def test_complex_dag_memory(self):
        result, current, peak = measure_peak_memory(
            workload_complex_dag,
            seq_len=WORKLOAD_SIZES['medium']['seq_len'],
            num_seqs=WORKLOAD_SIZES['medium']['num_seqs'],
        )
        assert len(result) == WORKLOAD_SIZES['medium']['num_seqs']
        print(f"\nComplex DAG medium: current={current/1e6:.2f}MB, peak={peak/1e6:.2f}MB")


@pytest.mark.slow
class TestMemoryLarge:
    """Memory profiling for large workloads (requires --run-slow)."""
    
    def test_mutagenize_memory(self):
        result, current, peak = measure_peak_memory(
            workload_mutagenize,
            seq_len=WORKLOAD_SIZES['large']['seq_len'],
            num_seqs=WORKLOAD_SIZES['large']['num_seqs'],
        )
        assert len(result) == WORKLOAD_SIZES['large']['num_seqs']
        print(f"\nMutagenize large: current={current/1e6:.2f}MB, peak={peak/1e6:.2f}MB")
    
    def test_sequential_cache_memory(self):
        """Test memory usage of sequential cache building (combinatorial)."""
        # 50bp with 2 mutations creates large cache
        result, current, peak = measure_peak_memory(
            workload_mutagenize_sequential, seq_len=50, num_mut=2
        )
        assert len(result) == 11025
        print(f"\nMutagenize sequential (50bp, 2mut): current={current/1e6:.2f}MB, peak={peak/1e6:.2f}MB")


# --- Memray integration for detailed profiling ---

def run_with_memray(workload_name: str, output_file: str = None):
    """Run a workload with memray tracking. Call from CLI, not pytest."""
    try:
        import memray
    except ImportError:
        print("memray not installed. Install with: uv pip install memray")
        return
    
    from .timing import ALL_WORKLOADS
    
    if workload_name not in ALL_WORKLOADS:
        print(f"Unknown workload: {workload_name}")
        print(f"Available: {list(ALL_WORKLOADS.keys())}")
        return
    
    if output_file is None:
        output_file = PROFILES_DIR / f"{workload_name}.bin"
    
    workload_func = ALL_WORKLOADS[workload_name]
    
    print(f"Running {workload_name} with memray tracking...")
    print(f"Output: {output_file}")
    
    with memray.Tracker(str(output_file)):
        workload_func()
    
    print(f"Done. Generate flamegraph with: memray flamegraph {output_file}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        run_with_memray(sys.argv[1])
    else:
        print("Usage: python -m poolparty.benchmarks.test_memory <workload_name>")
        print("Available workloads: mutagenize, recombine, complex_dag, etc.")
