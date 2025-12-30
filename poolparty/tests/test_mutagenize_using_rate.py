"""Tests for the MutagenizeUsingRate operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.operations.mutagenize_using_rate import MutagenizeUsingRateOp, mutagenize_using_rate


class TestMutagenizeUsingRateFactory:
    """Test mutagenize_using_rate factory function."""
    
    def test_returns_pool(self):
        """mutagenize_using_rate returns a Pool object."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.1)
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_mutagenize_using_rate_op(self):
        """Pool's operation is MutagenizeUsingRateOp."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.1)
            assert isinstance(pool.operation, MutagenizeUsingRateOp)
    
    def test_accepts_string_input(self):
        """Factory accepts a string and converts to Pool internally."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.1).named('mutant')
        
        df = pool.generate_seqs(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 4
    
    def test_accepts_pool_input(self):
        """Factory accepts an existing Pool as input."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            pool = mutagenize_using_rate(seq, mutation_rate=0.1).named('mutant')
        
        df = pool.generate_seqs(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 4


class TestMutagenizeUsingRateMutations:
    """Test mutation correctness."""
    
    def test_mutations_preserve_length(self):
        """Output sequences have same length as input."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGTACGT', mutation_rate=0.2).named('mutant')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for mutant in df['seq']:
            assert len(mutant) == 8
    
    def test_mutations_stay_in_alphabet(self):
        """All characters in output are from the alphabet."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.3).named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert all(c in 'ACGT' for c in mutant)
    
    def test_mutations_differ_from_wildtype(self):
        """Mutated positions have different characters than original."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.5, op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=50, seed=42)
        
        for _, row in df.iterrows():
            mutant = row['seq']
            positions = row['mutant.op.key.positions']
            wt_chars = row['mutant.op.key.wt_chars']
            mut_chars = row['mutant.op.key.mut_chars']
            
            # Check that mutations differ from wild-type
            for pos, wt, mut in zip(positions, wt_chars, mut_chars):
                assert 'ACGT'[pos] == wt  # Original has wt at position
                assert mutant[pos] == mut  # Mutant has mut at position
                assert wt != mut  # Mutation differs from wild-type


class TestMutagenizeUsingRateRandomMode:
    """Test random mode behavior."""
    
    def test_random_mode_num_states_is_one(self):
        """Random mode has num_states=1."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.1, mode='random')
            assert pool.operation.num_states == 1
    
    def test_random_mode_produces_valid_output(self):
        """Random mode generates valid mutated sequences."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGTACGT', mutation_rate=0.2, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        assert len(df) == 100
        
        # All should be valid sequences
        for mutant in df['seq']:
            assert len(mutant) == 8
            assert all(c in 'ACGT' for c in mutant)
    
    def test_random_mode_variability(self):
        """Random mode produces varied outputs across many samples."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGTACGT', mutation_rate=0.3, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        unique_mutants = df['seq'].nunique()
        assert unique_mutants > 10  # Should have variety


class TestMutagenizeUsingRateHybridMode:
    """Test hybrid mode behavior."""
    
    def test_hybrid_requires_num_hybrid_states(self):
        """Hybrid mode raises ValueError if num_hybrid_states not provided."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                mutagenize_using_rate('ACGT', mutation_rate=0.1, mode='hybrid')
    
    def test_hybrid_uses_num_hybrid_states(self):
        """Hybrid mode sets num_states to num_hybrid_states."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.1, mode='hybrid', num_hybrid_states=100)
            assert pool.operation.num_states == 100
    
    def test_hybrid_generates_correct_count(self):
        """Hybrid mode generates num_hybrid_states sequences per iteration."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=50).named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1, seed=42)
        assert len(df) == 50
    
    def test_hybrid_reproducible_with_seed(self):
        """Same seed produces identical results in hybrid mode."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGTACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_seqs(num_complete_iterations=1, seed=42)
            results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGTACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_seqs(num_complete_iterations=1, seed=42)
            results2 = list(df['seq'])
        
        assert results1 == results2
    
    def test_hybrid_different_seeds_different_results(self):
        """Different seeds produce different results."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGTACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_seqs(num_complete_iterations=1, seed=42)
            results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGTACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_seqs(num_complete_iterations=1, seed=123)
            results2 = list(df['seq'])
        
        assert results1 != results2


class TestMutagenizeUsingRateAlphabets:
    """Test different alphabet support."""
    
    def test_dna_alphabet(self):
        """DNA alphabet mutations use only ACGT."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.3, alphabet='dna', mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert all(c in 'ACGT' for c in mutant)
    
    def test_rna_alphabet(self):
        """RNA alphabet mutations use only ACGU."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGU', mutation_rate=0.3, alphabet='rna', mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert all(c in 'ACGU' for c in mutant)
    
    def test_custom_alphabet(self):
        """Custom alphabet mutations use only specified characters."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('AB', mutation_rate=0.5, alphabet='AB', mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert all(c in 'AB' for c in mutant)


class TestMutagenizeUsingRateDesignCards:
    """Test design card output."""
    
    def test_positions_in_output(self):
        """Design card contains 'positions' column."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.3, mode='random', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4, seed=42)
        assert 'mutant.op.key.positions' in df.columns
    
    def test_wt_chars_in_output(self):
        """Design card contains 'wt_chars' column."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.3, mode='random', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4, seed=42)
        assert 'mutant.op.key.wt_chars' in df.columns
    
    def test_mut_chars_in_output(self):
        """Design card contains 'mut_chars' column."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.3, mode='random', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4, seed=42)
        assert 'mutant.op.key.mut_chars' in df.columns
    
    def test_design_card_consistency(self):
        """Design card values match actual mutations in sequence."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGT', mutation_rate=0.5, mode='random', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        
        for _, row in df.iterrows():
            mutant = row['seq']
            positions = row['mutant.op.key.positions']
            wt_chars = row['mutant.op.key.wt_chars']
            mut_chars = row['mutant.op.key.mut_chars']
            
            # Verify the mutation is at the stated position
            for pos, wt, mut in zip(positions, wt_chars, mut_chars):
                assert 'ACGT'[pos] == wt  # Original has wt at position
                assert mutant[pos] == mut  # Mutant has mut at position


class TestMutagenizeUsingRateErrors:
    """Test error handling."""
    
    def test_negative_rate_error(self):
        """mutation_rate < 0 raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize_using_rate('ACGT', mutation_rate=-0.1)
    
    def test_rate_greater_than_one_error(self):
        """mutation_rate > 1 raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize_using_rate('ACGT', mutation_rate=1.5)
    
    def test_sequential_mode_error(self):
        """mode='sequential' raises ValueError (not supported)."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mode must be 'random' or 'hybrid'"):
                mutagenize_using_rate('ACGT', mutation_rate=0.1, mode='sequential')


class TestMutagenizeUsingRateStatistics:
    """Test statistical properties."""
    
    def test_zero_mutations_possible(self):
        """Low mutation_rate can produce zero mutations."""
        with pp.Party() as party:
            pool = mutagenize_using_rate('ACGTACGTACGT', mutation_rate=0.001, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        
        # With very low rate, some sequences should have zero mutations
        zero_mut_count = 0
        for _, row in df.iterrows():
            positions = row['mutant.op.key.positions']
            if len(positions) == 0:
                zero_mut_count += 1
        
        # Should have at least some zero-mutation sequences
        assert zero_mut_count > 0
    
    def test_average_mutations_matches_rate(self):
        """Average number of mutations ≈ mutation_rate * seq_len over many samples."""
        mutation_rate = 0.1
        seq_len = 100
        num_samples = 1000
        
        with pp.Party() as party:
            pool = mutagenize_using_rate('A' * seq_len, mutation_rate=mutation_rate, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=num_samples, seed=42)
        
        total_mutations = 0
        for _, row in df.iterrows():
            positions = row['mutant.op.key.positions']
            total_mutations += len(positions)
        
        avg_mutations = total_mutations / num_samples
        expected_avg = mutation_rate * seq_len
        
        # Allow 20% tolerance for statistical variation
        assert abs(avg_mutations - expected_avg) < 0.2 * expected_avg
    
    def test_high_rate_many_mutations(self):
        """High mutation_rate produces many mutations per sequence."""
        mutation_rate = 0.8
        seq_len = 50
        
        with pp.Party() as party:
            pool = mutagenize_using_rate('A' * seq_len, mutation_rate=mutation_rate, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        
        total_mutations = 0
        for _, row in df.iterrows():
            positions = row['mutant.op.key.positions']
            total_mutations += len(positions)
        
        avg_mutations = total_mutations / 100
        expected_avg = mutation_rate * seq_len
        
        # With high rate, average should be close to expected
        assert avg_mutations > expected_avg * 0.7  # At least 70% of expected
