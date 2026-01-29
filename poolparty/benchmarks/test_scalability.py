"""Scalability tests to measure how performance scales with parameters."""
import pytest
import time
from .timing import workload_mutagenize, workload_recombine, workload_complex_dag


def time_function(func, *args, **kwargs):
    """Time a function call and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


class TestScaleNumSeqs:
    """Test scaling with increasing number of sequences."""
    
    @pytest.mark.parametrize("num_seqs", [10, 100, 500, 1000])
    def test_mutagenize_num_seqs(self, num_seqs):
        result, elapsed = time_function(
            workload_mutagenize, seq_len=50, num_seqs=num_seqs
        )
        assert len(result) == num_seqs
        # Log time per sequence
        per_seq = elapsed / num_seqs * 1000  # ms
        print(f"\nMutagenize {num_seqs} seqs: {elapsed:.3f}s total, {per_seq:.3f}ms/seq")
    
    @pytest.mark.parametrize("num_seqs", [10, 100, 500, 1000])
    def test_complex_dag_num_seqs(self, num_seqs):
        result, elapsed = time_function(
            workload_complex_dag, seq_len=50, num_seqs=num_seqs
        )
        assert len(result) == num_seqs
        per_seq = elapsed / num_seqs * 1000
        print(f"\nComplex DAG {num_seqs} seqs: {elapsed:.3f}s total, {per_seq:.3f}ms/seq")


class TestScaleSeqLength:
    """Test scaling with increasing sequence length."""
    
    @pytest.mark.parametrize("seq_len", [20, 50, 100, 200, 500])
    def test_mutagenize_seq_length(self, seq_len):
        result, elapsed = time_function(
            workload_mutagenize, seq_len=seq_len, num_seqs=100
        )
        assert len(result) == 100
        print(f"\nMutagenize {seq_len}bp: {elapsed:.3f}s")
    
    @pytest.mark.parametrize("seq_len", [20, 50, 100, 200])
    def test_recombine_seq_length(self, seq_len):
        result, elapsed = time_function(
            workload_recombine, seq_len=seq_len, num_seqs=100
        )
        assert len(result) == 100
        print(f"\nRecombine {seq_len}bp: {elapsed:.3f}s")


class TestScaleNumMutations:
    """Test scaling with increasing mutation count (sequential mode)."""
    
    @pytest.mark.parametrize("num_mut", [1, 2, 3])
    def test_mutagenize_sequential_mutations(self, num_mut):
        import poolparty as pp
        pp.init()
        
        seq_len = 20
        seq = 'ACGT' * (seq_len // 4)
        
        start = time.perf_counter()
        pool = pp.mutagenize(seq, num_mutations=num_mut, mode='sequential')
        df = pool.generate_library(num_cycles=1)
        elapsed = time.perf_counter() - start
        
        # Expected states: C(seq_len, num_mut) × 3^num_mut
        from math import comb
        expected = comb(seq_len, num_mut) * (3 ** num_mut)
        assert len(df) == expected
        print(f"\nSequential mutagenize {num_mut} mutations: {len(df)} states, {elapsed:.3f}s")


class TestScaleNumBreakpoints:
    """Test scaling with increasing breakpoints (recombine)."""
    
    @pytest.mark.parametrize("num_bp", [1, 2, 3])
    def test_recombine_breakpoints(self, num_bp):
        result, elapsed = time_function(
            workload_recombine,
            seq_len=50,
            num_sources=3,
            num_breakpoints=num_bp,
            num_seqs=100,
        )
        assert len(result) == 100
        print(f"\nRecombine {num_bp} breakpoints: {elapsed:.3f}s")


@pytest.mark.slow
class TestScaleLarge:
    """Large-scale tests (require --run-slow)."""
    
    @pytest.mark.parametrize("num_seqs", [1000, 5000, 10000])
    def test_mutagenize_large_scale(self, num_seqs):
        result, elapsed = time_function(
            workload_mutagenize, seq_len=200, num_seqs=num_seqs
        )
        assert len(result) == num_seqs
        per_seq = elapsed / num_seqs * 1000
        print(f"\nMutagenize {num_seqs} seqs (200bp): {elapsed:.3f}s total, {per_seq:.3f}ms/seq")
    
    @pytest.mark.parametrize("seq_len", [500, 1000, 2000])
    def test_mutagenize_long_sequences(self, seq_len):
        result, elapsed = time_function(
            workload_mutagenize, seq_len=seq_len, num_seqs=100
        )
        assert len(result) == 100
        print(f"\nMutagenize {seq_len}bp: {elapsed:.3f}s")
    
    def test_sequential_large_cache(self):
        """Test large sequential cache (50bp, 3 mutations)."""
        import poolparty as pp
        pp.init()
        
        seq_len = 30
        num_mut = 3
        seq = 'ACGT' * (seq_len // 4)
        
        start = time.perf_counter()
        pool = pp.mutagenize(seq, num_mutations=num_mut, mode='sequential')
        df = pool.generate_library(num_cycles=1)
        elapsed = time.perf_counter() - start
        
        # C(30, 3) × 3^3 = 4060 × 27 = 109620 states
        from math import comb
        expected = comb(seq_len, num_mut) * (3 ** num_mut)
        assert len(df) == expected
        print(f"\nSequential mutagenize {num_mut} mutations on {seq_len}bp: {len(df)} states, {elapsed:.3f}s")
