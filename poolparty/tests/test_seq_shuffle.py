"""Tests for the SeqShuffle operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.base_ops.seq_shuffle import SeqShuffleOp, seq_shuffle


class TestSeqShuffleFactory:
    """Factory function behavior."""
    
    def test_returns_pool(self):
        with pp.Party():
            pool = seq_shuffle('ACGT')
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_seqshuffle_op(self):
        with pp.Party():
            pool = seq_shuffle('ACGT')
            assert isinstance(pool.operation, SeqShuffleOp)


class TestSeqShuffleBehavior:
    """Core behavior tests."""
    
    def test_preserves_length(self):
        with pp.Party():
            pool = seq_shuffle('ACGTAC', start=1, end=5).named('shuf')
        df = pool.generate_seqs(num_seqs=10, seed=123)
        for seq in df['seq']:
            assert len(seq) == 6
    
    def test_random_variability(self):
        with pp.Party():
            pool = seq_shuffle('ACGTACGT', start=0, end=8).named('shuf')
        df = pool.generate_seqs(num_seqs=50, seed=42)
        assert df['seq'].nunique() > 5
    
    def test_hybrid_num_states(self):
        with pp.Party():
            pool = seq_shuffle('ACGT', mode='hybrid', num_hybrid_states=10).named('shuf')
        assert pool.operation.num_states == 10
        df = pool.generate_seqs(num_cycles=1, seed=99)
        assert len(df) == 10
    
    def test_region_only_shuffled(self):
        with pp.Party():
            pool = seq_shuffle('ABCD', start=1, end=3).named('shuf')
        df = pool.generate_seqs(num_seqs=5, seed=7)
        for seq in df['seq']:
            assert seq[0] == 'A'
            assert seq[3] == 'D'
            middle = seq[1:3]
            assert sorted(middle) == sorted('BC')
    
    def test_zero_length_region_noop(self):
        with pp.Party():
            pool = seq_shuffle('ABCDE', start=2, end=2).named('shuf')
        df = pool.generate_seqs(num_seqs=3, seed=1)
        assert set(df['seq']) == {'ABCDE'}


class TestSeqShuffleDesignCard:
    """Design card correctness."""
    
    def test_compute_and_apply_permutation(self):
        with pp.Party():
            pool = seq_shuffle('WXYZ', start=0, end=4)
            rng = np.random.default_rng(42)
            card = pool.operation.compute_design_card(['WXYZ'], rng)
            result = pool.operation.compute_seq_from_card(['WXYZ'], card)
        shuffled = result['seq_0']
        assert set(shuffled) == set('WXYZ')
        assert len(shuffled) == 4
        # applying again with same card should reproduce
        result2 = pool.operation.compute_seq_from_card(['WXYZ'], card)
        assert result2['seq_0'] == shuffled
