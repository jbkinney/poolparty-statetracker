"""Tests for the Mutagenize operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.operations.mutagenize import MutagenizeOp, mutagenize


class TestMutagenizeFactory:
    """Test mutagenize factory function."""
    
    def test_returns_pool(self):
        """mutagenize returns a Pool object."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1)
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_mutagenize_op(self):
        """Pool's operation is MutagenizeOp."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1)
            assert isinstance(pool.operation, MutagenizeOp)
    
    def test_accepts_string_input_with_num(self):
        """Factory accepts a string with num_mutations."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1).named('mutant')
        
        df = pool.generate_seqs(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 4
    
    def test_accepts_string_input_with_rate(self):
        """Factory accepts a string with mutation_rate."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', mutation_rate=0.1).named('mutant')
        
        df = pool.generate_seqs(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 4
    
    def test_accepts_pool_input(self):
        """Factory accepts an existing Pool as input."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            pool = mutagenize(seq, num_mutations=1).named('mutant')
        
        df = pool.generate_seqs(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 4


class TestMutagenizeParameterValidation:
    """Test parameter validation."""
    
    def test_requires_num_or_rate(self):
        """Must provide either num_mutations or mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Either num_mutations or mutation_rate must be provided"):
                mutagenize('ACGT')
    
    def test_exclusive_num_and_rate(self):
        """Cannot provide both num_mutations and mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Only one of num_mutations or mutation_rate"):
                mutagenize('ACGT', num_mutations=1, mutation_rate=0.1)
    
    def test_num_mutations_minimum(self):
        """num_mutations must be >= 1."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations must be >= 1"):
                mutagenize('ACGT', num_mutations=0)
    
    def test_negative_num_mutations(self):
        """Negative num_mutations raises error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations must be >= 1"):
                mutagenize('ACGT', num_mutations=-1)
    
    def test_negative_rate_error(self):
        """mutation_rate < 0 raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize('ACGT', mutation_rate=-0.1)
    
    def test_rate_greater_than_one_error(self):
        """mutation_rate > 1 raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize('ACGT', mutation_rate=1.5)
    
    def test_sequential_mode_requires_num_mutations(self):
        """mode='sequential' not allowed with mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mode='sequential' is not supported with mutation_rate"):
                mutagenize('ACGT', mutation_rate=0.1, mode='sequential')
    
    def test_k_greater_than_length_error(self):
        """Error when num_mutations > sequence length."""
        with pytest.raises(ValueError, match="Cannot apply 3 mutations"):
            with pp.Party() as party:
                pool = mutagenize('AC', num_mutations=3, mode='sequential')
                party.output(pool, name='mutant')


# =============================================================================
# Tests for num_mutations mode
# =============================================================================

class TestMutagenizeSingleMutation:
    """Test single mutation (num_mutations=1) behavior."""
    
    def test_single_mutation_count(self):
        """Test correct number of single mutants."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # 4 positions * 3 mutations each = 12
        assert len(df) == 12
    
    def test_single_mutation_correctness(self):
        """Test that all single mutants have exactly 1 difference."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        for mutant in df['seq']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_single_mutation_preserves_length(self):
        """Test that mutations preserve sequence length."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_seqs=10)
        for mutant in df['seq']:
            assert len(mutant) == 8


class TestMutagenizeMultipleMutations:
    """Test multiple mutations (num_mutations > 1) behavior."""
    
    def test_double_mutation_count(self):
        """Test correct number of double mutants."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=2, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # C(4,2) * 3^2 = 6 * 9 = 54
        assert len(df) == 54
    
    def test_double_mutation_correctness(self):
        """Test that all double mutants have exactly 2 differences."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=2, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        for mutant in df['seq']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 2
    
    def test_triple_mutation(self):
        """Test triple mutations."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=3, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # C(4,3) * 3^3 = 4 * 27 = 108
        assert len(df) == 108
        
        for mutant in df['seq']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 3


class TestMutagenizeSequentialMode:
    """Test Mutagenize in sequential mode (num_mutations only)."""
    
    def test_sequential_enumeration(self):
        """Test sequential enumeration of mutations."""
        with pp.Party() as party:
            pool = mutagenize('AC', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # 2 positions * 3 mutations = 6 mutants
        assert len(df) == 6
    
    def test_sequential_cycling(self):
        """Test that sequential mode cycles."""
        with pp.Party() as party:
            pool = mutagenize('AC', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_seqs=12)  # 2 complete cycles
        first_half = list(df['seq'][:6])
        second_half = list(df['seq'][6:])
        assert first_half == second_half
    
    def test_num_states_k1(self):
        """Test num_states for num_mutations=1."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential')
            # 4 positions * 3 mutations = 12
            assert pool.operation.num_states == 12
    
    def test_num_states_k2(self):
        """Test num_states for num_mutations=2."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=2, mode='sequential')
            # C(4,2) * 3^2 = 6 * 9 = 54
            assert pool.operation.num_states == 54


class TestMutagenizeRandomModeWithNum:
    """Test Mutagenize in random mode with num_mutations."""
    
    def test_random_sampling(self):
        """Test random sampling of mutations."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        assert len(df) == 100
        
        # All should be valid single mutants
        for mutant in df['seq']:
            assert len(mutant) == 8
            diffs = sum(1 for a, b in zip('ACGTACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_random_variability(self):
        """Test that random mode produces varied outputs."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        unique_mutants = df['seq'].nunique()
        assert unique_mutants > 10  # Should have variety
    
    def test_random_num_states_is_one(self):
        """Test that random mode has num_states=1."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='random')
            assert pool.operation.num_states == 1


# =============================================================================
# Tests for mutation_rate mode
# =============================================================================

class TestMutagenizeRandomModeWithRate:
    """Test random mode with mutation_rate."""
    
    def test_random_mode_num_states_is_one(self):
        """Random mode has num_states=1."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', mutation_rate=0.1, mode='random')
            assert pool.operation.num_states == 1
    
    def test_random_mode_produces_valid_output(self):
        """Random mode generates valid mutated sequences."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', mutation_rate=0.2, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        assert len(df) == 100
        
        # All should be valid sequences
        for mutant in df['seq']:
            assert len(mutant) == 8
            assert all(c in 'ACGT' for c in mutant)
    
    def test_random_mode_variability(self):
        """Random mode produces varied outputs across many samples."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', mutation_rate=0.3, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        unique_mutants = df['seq'].nunique()
        assert unique_mutants > 10  # Should have variety
    
    def test_mutations_preserve_length(self):
        """Output sequences have same length as input."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', mutation_rate=0.2).named('mutant')
        
        df = pool.generate_seqs(num_seqs=10, seed=42)
        for mutant in df['seq']:
            assert len(mutant) == 8
    
    def test_mutations_stay_in_alphabet(self):
        """All characters in output are from the alphabet."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', mutation_rate=0.3).named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert all(c in 'ACGT' for c in mutant)
    
    def test_mutations_differ_from_wildtype(self):
        """Mutated positions have different characters than original."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', mutation_rate=0.5, op_name='mutate').named('mutant')
        
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


class TestMutagenizeHybridModeWithRate:
    """Test hybrid mode with mutation_rate."""
    
    def test_hybrid_requires_num_hybrid_states(self):
        """Hybrid mode raises ValueError if num_hybrid_states not provided."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                mutagenize('ACGT', mutation_rate=0.1, mode='hybrid')
    
    def test_hybrid_uses_num_hybrid_states(self):
        """Hybrid mode sets num_states to num_hybrid_states."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', mutation_rate=0.1, mode='hybrid', num_hybrid_states=100)
            assert pool.operation.num_states == 100
    
    def test_hybrid_generates_correct_count(self):
        """Hybrid mode generates num_hybrid_states sequences per iteration."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=50).named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1, seed=42)
        assert len(df) == 50
    
    def test_hybrid_reproducible_with_seed(self):
        """Same seed produces identical results in hybrid mode."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_seqs(num_complete_iterations=1, seed=42)
            results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_seqs(num_complete_iterations=1, seed=42)
            results2 = list(df['seq'])
        
        assert results1 == results2
    
    def test_hybrid_different_seeds_different_results(self):
        """Different seeds produce different results."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_seqs(num_complete_iterations=1, seed=42)
            results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', mutation_rate=0.2, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_seqs(num_complete_iterations=1, seed=123)
            results2 = list(df['seq'])
        
        assert results1 != results2


class TestMutagenizeRateStatistics:
    """Test statistical properties of mutation_rate mode."""
    
    def test_zero_mutations_possible(self):
        """Low mutation_rate can produce zero mutations."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGTACGT', mutation_rate=0.001, mode='random').named('mutant')
        
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
            pool = mutagenize('A' * seq_len, mutation_rate=mutation_rate, mode='random').named('mutant')
        
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
            pool = mutagenize('A' * seq_len, mutation_rate=mutation_rate, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        
        total_mutations = 0
        for _, row in df.iterrows():
            positions = row['mutant.op.key.positions']
            total_mutations += len(positions)
        
        avg_mutations = total_mutations / 100
        expected_avg = mutation_rate * seq_len
        
        # With high rate, average should be close to expected
        assert avg_mutations > expected_avg * 0.7  # At least 70% of expected


# =============================================================================
# Common tests (both modes)
# =============================================================================

class TestMutagenizeAlphabets:
    """Test different alphabet support via Party."""
    
    def test_dna_alphabet_with_num(self):
        """DNA alphabet mutations use only ACGT (num_mutations)."""
        with pp.Party(alphabet='dna') as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        for mutant in df['seq']:
            assert all(c in 'ACGT' for c in mutant)
    
    def test_dna_alphabet_with_rate(self):
        """DNA alphabet mutations use only ACGT (mutation_rate)."""
        with pp.Party(alphabet='dna') as party:
            pool = mutagenize('ACGT', mutation_rate=0.3, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert all(c in 'ACGT' for c in mutant)
    
    def test_rna_alphabet(self):
        """RNA alphabet mutations use only ACGU."""
        with pp.Party(alphabet='rna') as party:
            pool = mutagenize('ACGU', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        for mutant in df['seq']:
            assert all(c in 'ACGU' for c in mutant)
    
    def test_rna_alphabet_with_rate(self):
        """RNA alphabet mutations use only ACGU (mutation_rate)."""
        with pp.Party(alphabet='rna') as party:
            pool = mutagenize('ACGU', mutation_rate=0.3, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert all(c in 'ACGU' for c in mutant)
    
    def test_custom_alphabet_with_num(self):
        """Custom alphabet mutations use only specified characters (num_mutations)."""
        from poolparty.alphabet import Alphabet
        custom_alph = Alphabet(chars=['A', 'B'])
        with pp.Party(alphabet=custom_alph) as party:
            pool = mutagenize('AB', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # 2 positions * 1 mutation each = 2 mutants
        assert len(df) == 2
        assert set(df['seq']) == {'BB', 'AA'}
    
    def test_custom_alphabet_with_rate(self):
        """Custom alphabet mutations use only specified characters (mutation_rate)."""
        from poolparty.alphabet import Alphabet
        custom_alph = Alphabet(chars=['A', 'B'])
        with pp.Party(alphabet=custom_alph) as party:
            pool = mutagenize('AB', mutation_rate=0.5, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=20, seed=42)
        for mutant in df['seq']:
            assert all(c in 'AB' for c in mutant)


class TestMutagenizeDesignCards:
    """Test design card output."""
    
    def test_positions_in_output(self):
        """Design card contains 'positions' column."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4)
        assert 'mutant.op.key.positions' in df.columns
    
    def test_wt_chars_in_output(self):
        """Design card contains 'wt_chars' column."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4)
        assert 'mutant.op.key.wt_chars' in df.columns
    
    def test_mut_chars_in_output(self):
        """Design card contains 'mut_chars' column."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4)
        assert 'mutant.op.key.mut_chars' in df.columns
    
    def test_design_card_consistency_with_num(self):
        """Design card values match actual mutations in sequence (num_mutations)."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=12)
        
        for _, row in df.iterrows():
            mutant = row['seq']
            positions = row['mutant.op.key.positions']
            wt_chars = row['mutant.op.key.wt_chars']
            mut_chars = row['mutant.op.key.mut_chars']
            
            # Verify the mutation is at the stated position
            for pos, wt, mut in zip(positions, wt_chars, mut_chars):
                assert 'ACGT'[pos] == wt  # Original has wt at position
                assert mutant[pos] == mut  # Mutant has mut at position
    
    def test_design_card_consistency_with_rate(self):
        """Design card values match actual mutations in sequence (mutation_rate)."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', mutation_rate=0.5, mode='random', op_name='mutate').named('mutant')
        
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


class TestMutagenizeMutationMap:
    """Test the mutation map correctness."""
    
    def test_mutations_are_different_from_wt(self):
        """Test that mutations are always different from wild-type."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        
        for _, row in df.iterrows():
            positions = row['mutant.op.key.positions']
            wt_chars = row['mutant.op.key.wt_chars']
            mut_chars = row['mutant.op.key.mut_chars']
            
            for wt, mut in zip(wt_chars, mut_chars):
                assert wt != mut
    
    def test_all_mutations_covered(self):
        """Test that all possible mutations are covered."""
        with pp.Party(alphabet='dna') as party:
            pool = mutagenize('A', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # A can mutate to C, G, T
        mutants = set(df['seq'])
        assert mutants == {'C', 'G', 'T'}


class TestMutagenizeCompute:
    """Test compute methods directly."""
    
    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='sequential')
        
        pool.operation.counter._state = 0
        card = pool.operation.compute_design_card(['ACGT'])
        result = pool.operation.compute_seq_from_card(['ACGT'], card)
        assert len(result['seq_0']) == 4
        assert card['positions'] is not None
    
    def test_compute_random_with_num(self):
        """Test compute in random mode with num_mutations."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='random')
        
        rng = np.random.default_rng(42)
        card = pool.operation.compute_design_card(['ACGT'], rng)
        result = pool.operation.compute_seq_from_card(['ACGT'], card)
        assert len(result['seq_0']) == 4
        
        # Verify exactly one mutation
        diffs = sum(1 for a, b in zip('ACGT', result['seq_0']) if a != b)
        assert diffs == 1
    
    def test_compute_random_with_rate(self):
        """Test compute in random mode with mutation_rate."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', mutation_rate=0.5, mode='random')
        
        rng = np.random.default_rng(42)
        card = pool.operation.compute_design_card(['ACGT'], rng)
        result = pool.operation.compute_seq_from_card(['ACGT'], card)
        assert len(result['seq_0']) == 4


class TestMutagenizeWithParentPool:
    """Test Mutagenize with various parent pool configurations."""
    
    def test_with_sequential_parent(self):
        """Test mutation scan with sequential parent pool."""
        with pp.Party() as party:
            seqs = pp.from_seqs(['AAA', 'TTT'], mode='sequential')
            mutants = mutagenize(seqs, num_mutations=1, mode='sequential').named('mutant')
        
        df = mutants.generate_seqs(num_seqs=10)
        # Should see mutations of both AAA and TTT
        assert len(df) == 10
    
    def test_with_breakpoint_output(self):
        """Test mutation scan on breakpoint output."""
        with pp.Party() as party:
            # Use positions=[1, 2, 3] to avoid empty segments
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1, positions=[1, 2, 3])
            mutated_right = mutagenize(right, num_mutations=1, mode='sequential').named('mutant')
        
        df = mutated_right.generate_seqs(num_seqs=5)
        assert len(df) == 5


class TestMutagenizeCustomName:
    """Test name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1)
            assert pool.operation.name.startswith('op[')
            assert ':mutagenize' in pool.operation.name
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, op_name='my_mutations')
            assert pool.operation.name == 'my_mutations'
    
    def test_custom_name_in_design_card(self):
        """Test custom name appears in design card columns with .key."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, op_name='mutants').named('mypool')
        
        df = pool.generate_seqs(num_seqs=1)
        assert 'mypool.op.key.positions' in df.columns
        assert 'mypool.op.key.wt_chars' in df.columns
        assert 'mypool.op.key.mut_chars' in df.columns


class TestMutagenizeHybridModeWithNum:
    """Test hybrid mode with num_mutations."""
    
    def test_hybrid_requires_num_hybrid_states_with_num(self):
        """Hybrid mode raises ValueError if num_hybrid_states not provided."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                mutagenize('ACGT', num_mutations=1, mode='hybrid')
    
    def test_hybrid_uses_num_hybrid_states_with_num(self):
        """Hybrid mode sets num_states to num_hybrid_states."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='hybrid', num_hybrid_states=100)
            assert pool.operation.num_states == 100
