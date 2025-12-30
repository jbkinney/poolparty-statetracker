"""Tests for the MutationScan operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.operations.mutation_scan import MutationScanOp, mutation_scan


class TestMutationScanFactory:
    """Test mutation_scan factory function."""
    
    def test_returns_pool(self):
        """Test that mutation_scan returns a Pool."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1)
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_mutation_scan_op(self):
        """Test that mutation_scan creates a MutationScanOp."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1)
            assert isinstance(pool.operation, MutationScanOp)
    
    def test_accepts_string_input(self):
        """Test that mutation_scan accepts string input."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1).named('mutant')
        
        df = pool.generate_seqs(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 4
    
    def test_accepts_pool_input(self):
        """Test that mutation_scan accepts Pool input."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            pool = mutation_scan(seq, num_mutations=1).named('mutant')
        
        df = pool.generate_seqs(num_seqs=1)
        assert len(df['seq'].iloc[0]) == 4


class TestMutationScanSingleMutation:
    """Test single mutation (num_mutations=1) behavior."""
    
    def test_single_mutation_count(self):
        """Test correct number of single mutants."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # 4 positions * 3 mutations each = 12
        assert len(df) == 12
    
    def test_single_mutation_correctness(self):
        """Test that all single mutants have exactly 1 difference."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        for mutant in df['seq']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_single_mutation_preserves_length(self):
        """Test that mutations preserve sequence length."""
        with pp.Party() as party:
            pool = mutation_scan('ACGTACGT', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_seqs=10)
        for mutant in df['seq']:
            assert len(mutant) == 8


class TestMutationScanMultipleMutations:
    """Test multiple mutations (k>1) behavior."""
    
    def test_double_mutation_count(self):
        """Test correct number of double mutants."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=2, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # C(4,2) * 3^2 = 6 * 9 = 54
        assert len(df) == 54
    
    def test_double_mutation_correctness(self):
        """Test that all double mutants have exactly 2 differences."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=2, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        for mutant in df['seq']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 2
    
    def test_triple_mutation(self):
        """Test triple mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=3, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # C(4,3) * 3^3 = 4 * 27 = 108
        assert len(df) == 108
        
        for mutant in df['seq']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 3


class TestMutationScanSequentialMode:
    """Test MutationScan in sequential mode."""
    
    def test_sequential_enumeration(self):
        """Test sequential enumeration of mutations."""
        with pp.Party() as party:
            pool = mutation_scan('AC', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # 2 positions * 3 mutations = 6 mutants
        assert len(df) == 6
    
    def test_sequential_cycling(self):
        """Test that sequential mode cycles."""
        with pp.Party() as party:
            pool = mutation_scan('AC', num_mutations=1, mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_seqs=12)  # 2 complete cycles
        first_half = list(df['seq'][:6])
        second_half = list(df['seq'][6:])
        assert first_half == second_half


class TestMutationScanRandomMode:
    """Test MutationScan in random mode."""
    
    def test_random_sampling(self):
        """Test random sampling of mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGTACGT', num_mutations=1, mode='random').named('mutant')
        
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
            pool = mutation_scan('ACGTACGT', num_mutations=1, mode='random').named('mutant')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        unique_mutants = df['seq'].nunique()
        assert unique_mutants > 10  # Should have variety
    
    def test_random_num_states_is_one(self):
        """Test that random mode has num_states=1."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='random')
            assert pool.operation.num_states == 1


class TestMutationScanAlphabets:
    """Test MutationScan with different alphabets."""
    
    def test_dna_alphabet(self):
        """Test DNA alphabet mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, alphabet='dna', mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        for mutant in df['seq']:
            assert all(c in 'ACGT' for c in mutant)
    
    def test_rna_alphabet(self):
        """Test RNA alphabet mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGU', num_mutations=1, alphabet='rna', mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        for mutant in df['seq']:
            assert all(c in 'ACGU' for c in mutant)
    
    def test_custom_alphabet(self):
        """Test custom alphabet mutations."""
        with pp.Party() as party:
            pool = mutation_scan('AB', num_mutations=1, alphabet='AB', mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # 2 positions * 1 mutation each = 2 mutants
        assert len(df) == 2
        assert set(df['seq']) == {'BB', 'AA'}


class TestMutationScanDesignCards:
    """Test MutationScan design card output."""
    
    def test_positions_in_output(self):
        """Test positions are in output."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4)
        assert 'mutant.op.key.positions' in df.columns
    
    def test_wt_chars_in_output(self):
        """Test wild-type characters are in output."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4)
        assert 'mutant.op.key.wt_chars' in df.columns
    
    def test_mut_chars_in_output(self):
        """Test mutant characters are in output."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_seqs=4)
        assert 'mutant.op.key.mut_chars' in df.columns
    
    def test_design_card_consistency(self):
        """Test design card values are consistent with mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
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


class TestMutationScanErrors:
    """Test MutationScan error handling."""
    
    def test_k_zero_error(self):
        """Test error for num_mutations=0."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations must be >= 1"):
                mutation_scan('ACGT', num_mutations=0)
    
    def test_k_negative_error(self):
        """Test error for negative k."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations must be >= 1"):
                mutation_scan('ACGT', num_mutations=-1)
    
    def test_k_greater_than_length_error(self):
        """Test error when k > sequence length."""
        # Error happens at init time when k exceeds sequence length
        with pytest.raises(ValueError, match="Cannot apply 3 mutations"):
            with pp.Party() as party:
                pool = mutation_scan('AC', num_mutations=3, mode='sequential')
                party.output(pool, name='mutant')


class TestMutationScanMutationMap:
    """Test the mutation map correctness."""
    
    def test_mutations_are_different_from_wt(self):
        """Test that mutations are always different from wild-type."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential', op_name='mutate').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        
        for _, row in df.iterrows():
            positions = row['mutant.op.key.positions']
            wt_chars = row['mutant.op.key.wt_chars']
            mut_chars = row['mutant.op.key.mut_chars']
            
            for wt, mut in zip(wt_chars, mut_chars):
                assert wt != mut
    
    def test_all_mutations_covered(self):
        """Test that all possible mutations are covered."""
        with pp.Party() as party:
            pool = mutation_scan('A', num_mutations=1, alphabet='dna', mode='sequential').named('mutant')
        
        df = pool.generate_seqs(num_complete_iterations=1)
        # A can mutate to C, G, T
        mutants = set(df['seq'])
        assert mutants == {'C', 'G', 'T'}


class TestMutationScanCompute:
    """Test MutationScan compute methods directly."""
    
    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential')
        
        pool.operation.counter._state = 0
        card = pool.operation.compute_design_card(['ACGT'])
        result = pool.operation.compute_seq_from_card(['ACGT'], card)
        assert len(result['seq_0']) == 4
        assert card['positions'] is not None
    
    def test_compute_random(self):
        """Test compute in random mode."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='random')
        
        rng = np.random.default_rng(42)
        card = pool.operation.compute_design_card(['ACGT'], rng)
        result = pool.operation.compute_seq_from_card(['ACGT'], card)
        assert len(result['seq_0']) == 4
        
        # Verify exactly one mutation
        diffs = sum(1 for a, b in zip('ACGT', result['seq_0']) if a != b)
        assert diffs == 1


class TestMutationScanWithParentPool:
    """Test MutationScan with various parent pool configurations."""
    
    def test_with_sequential_parent(self):
        """Test mutation scan with sequential parent pool."""
        with pp.Party() as party:
            seqs = pp.from_seqs(['AAA', 'TTT'], mode='sequential')
            mutants = mutation_scan(seqs, num_mutations=1, mode='sequential').named('mutant')
        
        df = mutants.generate_seqs(num_seqs=10)
        # Should see mutations of both AAA and TTT
        assert len(df) == 10
    
    def test_with_breakpoint_output(self):
        """Test mutation scan on breakpoint output."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            mutated_right = mutation_scan(right, num_mutations=1, mode='sequential').named('mutant')
        
        df = mutated_right.generate_seqs(num_seqs=5)
        assert len(df) == 5


class TestMutationScanCustomName:
    """Test MutationScan name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1)
            assert pool.operation.name.startswith('op[')
            assert ':mutation_scan' in pool.operation.name
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, op_name='my_mutations')
            assert pool.operation.name == 'my_mutations'
    
    def test_custom_name_in_design_card(self):
        """Test custom name appears in design card columns with .key."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, op_name='mutants').named('mypool')
        
        df = pool.generate_seqs(num_seqs=1)
        assert 'mypool.op.key.positions' in df.columns
        assert 'mypool.op.key.wt_chars' in df.columns
        assert 'mypool.op.key.mut_chars' in df.columns


class TestMutationScanNumStates:
    """Test MutationScan num_states calculation."""
    
    def test_num_states_k1(self):
        """Test num_states for num_mutations=1."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=1, mode='sequential')
            # 4 positions * 3 mutations = 12
            assert pool.operation.num_states == 12
    
    def test_num_states_k2(self):
        """Test num_states for num_mutations=2."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', num_mutations=2, mode='sequential')
            # C(4,2) * 3^2 = 6 * 9 = 54
            assert pool.operation.num_states == 54

