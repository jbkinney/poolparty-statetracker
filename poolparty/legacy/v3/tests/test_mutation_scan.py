"""Tests for the MutationScan operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty import reset_op_id_counter
from poolparty.operations.mutation_scan import MutationScanOp, mutation_scan


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


class TestMutationScanFactory:
    """Test mutation_scan factory function."""
    
    def test_returns_pool(self):
        """Test that mutation_scan returns a Pool."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1)
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_mutation_scan_op(self):
        """Test that mutation_scan creates a MutationScanOp."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1)
            assert isinstance(pool.operation, MutationScanOp)
    
    def test_accepts_string_input(self):
        """Test that mutation_scan accepts string input."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1)
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=1)
        assert len(df['mutant'].iloc[0]) == 4
    
    def test_accepts_pool_input(self):
        """Test that mutation_scan accepts Pool input."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            pool = mutation_scan(seq, k=1)
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=1)
        assert len(df['mutant'].iloc[0]) == 4


class TestMutationScanSingleMutation:
    """Test single mutation (k=1) behavior."""
    
    def test_single_mutation_count(self):
        """Test correct number of single mutants."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        # 4 positions * 3 mutations each = 12
        assert len(df) == 12
    
    def test_single_mutation_correctness(self):
        """Test that all single mutants have exactly 1 difference."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        for mutant in df['mutant']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_single_mutation_preserves_length(self):
        """Test that mutations preserve sequence length."""
        with pp.Party() as party:
            pool = mutation_scan('ACGTACGT', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=10)
        for mutant in df['mutant']:
            assert len(mutant) == 8


class TestMutationScanMultipleMutations:
    """Test multiple mutations (k>1) behavior."""
    
    def test_double_mutation_count(self):
        """Test correct number of double mutants."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=2, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        # C(4,2) * 3^2 = 6 * 9 = 54
        assert len(df) == 54
    
    def test_double_mutation_correctness(self):
        """Test that all double mutants have exactly 2 differences."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=2, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        for mutant in df['mutant']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 2
    
    def test_triple_mutation(self):
        """Test triple mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=3, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        # C(4,3) * 3^3 = 4 * 27 = 108
        assert len(df) == 108
        
        for mutant in df['mutant']:
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 3


class TestMutationScanSequentialMode:
    """Test MutationScan in sequential mode."""
    
    def test_sequential_enumeration(self):
        """Test sequential enumeration of mutations."""
        with pp.Party() as party:
            pool = mutation_scan('AC', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        # 2 positions * 3 mutations = 6 mutants
        assert len(df) == 6
    
    def test_sequential_cycling(self):
        """Test that sequential mode cycles."""
        with pp.Party() as party:
            pool = mutation_scan('AC', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=12)  # 2 complete cycles
        first_half = list(df['mutant'][:6])
        second_half = list(df['mutant'][6:])
        assert first_half == second_half


class TestMutationScanRandomMode:
    """Test MutationScan in random mode."""
    
    def test_random_sampling(self):
        """Test random sampling of mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGTACGT', k=1, mode='random')
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=100, seed=42)
        assert len(df) == 100
        
        # All should be valid single mutants
        for mutant in df['mutant']:
            assert len(mutant) == 8
            diffs = sum(1 for a, b in zip('ACGTACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_random_variability(self):
        """Test that random mode produces varied outputs."""
        with pp.Party() as party:
            pool = mutation_scan('ACGTACGT', k=1, mode='random')
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=100, seed=42)
        unique_mutants = df['mutant'].nunique()
        assert unique_mutants > 10  # Should have variety
    
    def test_random_requires_rng(self):
        """Test that random mode requires RNG."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='random')
        
        with pytest.raises(RuntimeError, match="Random mode requires RNG"):
            pool.operation.compute(['ACGT'], 0, None)


class TestMutationScanAlphabets:
    """Test MutationScan with different alphabets."""
    
    def test_dna_alphabet(self):
        """Test DNA alphabet mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, alphabet='dna', mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        for mutant in df['mutant']:
            assert all(c in 'ACGT' for c in mutant)
    
    def test_rna_alphabet(self):
        """Test RNA alphabet mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGU', k=1, alphabet='rna', mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        for mutant in df['mutant']:
            assert all(c in 'ACGU' for c in mutant)
    
    def test_custom_alphabet(self):
        """Test custom alphabet mutations."""
        with pp.Party() as party:
            pool = mutation_scan('AB', k=1, alphabet='AB', mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        # 2 positions * 1 mutation each = 2 mutants
        assert len(df) == 2
        assert set(df['mutant']) == {'BB', 'AA'}


class TestMutationScanDesignCards:
    """Test MutationScan design card output."""
    
    def test_positions_in_output(self):
        """Test positions are in output."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=4)
        assert 'mutation_scan.positions' in df.columns
    
    def test_wt_chars_in_output(self):
        """Test wild-type characters are in output."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=4)
        assert 'mutation_scan.wt_chars' in df.columns
    
    def test_mut_chars_in_output(self):
        """Test mutant characters are in output."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=4)
        assert 'mutation_scan.mut_chars' in df.columns
    
    def test_design_card_consistency(self):
        """Test design card values are consistent with mutations."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_seqs=12)
        
        for _, row in df.iterrows():
            mutant = row['mutant']
            positions = row['mutation_scan.positions']
            wt_chars = row['mutation_scan.wt_chars']
            mut_chars = row['mutation_scan.mut_chars']
            
            # Verify the mutation is at the stated position
            for pos, wt, mut in zip(positions, wt_chars, mut_chars):
                assert 'ACGT'[pos] == wt  # Original has wt at position
                assert mutant[pos] == mut  # Mutant has mut at position


class TestMutationScanErrors:
    """Test MutationScan error handling."""
    
    def test_k_zero_error(self):
        """Test error for k=0."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            mutation_scan('ACGT', k=0)
    
    def test_k_negative_error(self):
        """Test error for negative k."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            mutation_scan('ACGT', k=-1)
    
    def test_k_greater_than_length_error(self):
        """Test error when k > sequence length."""
        # When k > length, there are 0 possible mutation combinations
        # This raises an error at initialization time
        with pytest.raises(ValueError):
            mutation_scan('AC', k=3, mode='sequential')


class TestMutationScanMutationMap:
    """Test the mutation map correctness."""
    
    def test_mutations_are_different_from_wt(self):
        """Test that mutations are always different from wild-type."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        
        for _, row in df.iterrows():
            positions = row['mutation_scan.positions']
            wt_chars = row['mutation_scan.wt_chars']
            mut_chars = row['mutation_scan.mut_chars']
            
            for wt, mut in zip(wt_chars, mut_chars):
                assert wt != mut
    
    def test_all_mutations_covered(self):
        """Test that all possible mutations are covered."""
        with pp.Party() as party:
            pool = mutation_scan('A', k=1, alphabet='dna', mode='sequential')
            party.output(pool, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        # A can mutate to C, G, T
        mutants = set(df['mutant'])
        assert mutants == {'C', 'G', 'T'}


class TestMutationScanCompute:
    """Test MutationScan compute method directly."""
    
    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
        
        result = pool.operation.compute(['ACGT'], 0, None)
        assert len(result['seq_0']) == 4
        assert result['positions'] is not None
    
    def test_compute_random(self):
        """Test compute in random mode."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='random')
        
        rng = np.random.default_rng(42)
        result = pool.operation.compute(['ACGT'], 0, rng)
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
            mutants = mutation_scan(seqs, k=1, mode='sequential')
            party.output(mutants, name='mutant')
        
        df = party.generate(num_seqs=10)
        # Should see mutations of both AAA and TTT
        assert len(df) == 10
    
    def test_with_breakpoint_output(self):
        """Test mutation scan on breakpoint output."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            mutated_right = mutation_scan(right, k=1, mode='sequential')
            party.output(mutated_right, name='mutant')
        
        df = party.generate(num_seqs=5)
        assert len(df) == 5


class TestMutationScanCustomName:
    """Test MutationScan name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1)
            assert pool.operation.name == 'mutation_scan'
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, name='my_mutations')
            assert pool.operation.name == 'my_mutations'
    
    def test_custom_name_in_design_card(self):
        """Test custom name appears in design card columns."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, name='mutants')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=1)
        assert 'mutants.positions' in df.columns
        assert 'mutants.wt_chars' in df.columns
        assert 'mutants.mut_chars' in df.columns


class TestMutationScanNumStates:
    """Test MutationScan num_states calculation."""
    
    def test_num_states_k1(self):
        """Test num_states for k=1."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=1, mode='sequential')
            # 4 positions * 3 mutations = 12
            assert pool.operation.num_states == 12
    
    def test_num_states_k2(self):
        """Test num_states for k=2."""
        with pp.Party() as party:
            pool = mutation_scan('ACGT', k=2, mode='sequential')
            # C(4,2) * 3^2 = 6 * 9 = 54
            assert pool.operation.num_states == 54

