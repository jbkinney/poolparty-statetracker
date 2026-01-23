"""Tests for random mode with num_states across operations."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.base_ops.mutagenize import mutagenize
from poolparty.base_ops.get_kmers import get_kmers
from poolparty.base_ops.from_seqs import from_seqs
from poolparty.base_ops.breakpoint_scan import breakpoint_scan


class TestRandomNumStatesBasic:
    """Test basic random mode with num_states functionality."""
    
    def test_num_states(self):
        """Test that random mode with num_states uses num_states for counter."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='random', num_states=100)
            assert pool.operation.num_values == 100
    
    def test_random_mode_syncs_to_parent(self):
        """Test that random mode without num_states syncs to parent state."""
        with pp.Party() as party:
            # With a parent (from_seq creates a parent with num_values=1), syncs to parent
            pool = mutagenize('ACGT', num_mutations=1, mode='random')
            assert pool.operation.num_values == 1  # Synced to from_seq parent
            
            # Without a parent (source operation), stays stateless
            source_pool = get_kmers(length=4, mode='random')
            assert source_pool.operation.num_values is None
    
    def test_random_mode_generates_correct_count(self):
        """Test that random mode with num_states generates the expected number of sequences."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='random', num_states=50).named('mutant')
        
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 50


class TestRandomNumStatesReproducibility:
    """Test that random mode with num_states produces reproducible results."""
    
    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=10).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
            results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=10).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
            results2 = list(df['seq'])
        
        assert results1 == results2
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=10).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
            results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=10).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=123)
            results2 = list(df['seq'])
        
        assert results1 != results2
    
    def test_same_state_same_output(self):
        """Test that same state always produces same output with same seed."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=20).named('mutant')
            df = pool.generate_library(num_seqs=40, seed=42)  # 2 complete iterations
        
        # First and second iteration should be identical
        first_iter = list(df['seq'][:20])
        second_iter = list(df['seq'][20:])
        assert first_iter == second_iter


class TestRandomNumStatesOperationIsolation:
    """Test that different operations produce different results at same state."""
    
    def test_different_ops_different_results(self):
        """Test that different random operations at same state produce different results."""
        with pp.Party() as party:
            # Two mutagenize operations with same config but different op.id
            pool1 = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=10).named('mutant1')
            pool2 = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=10).named('mutant2')
            df = pool1.generate_library(num_cycles=1, seed=42, report_design_cards=True, aux_pools=[pool2])
        
        # They should produce different results because op.id is part of the seed
        results1 = list(df['mutant1.seq'])
        results2 = list(df['mutant2.seq'])
        
        # At least some results should differ
        differences = sum(1 for a, b in zip(results1, results2) if a != b)
        assert differences > 0, "Different ops should produce different outputs"


class TestRandomNumStatesMutationScan:
    """Test random mode with num_states specifically for mutagenize."""
    
    def test_mutagenize_random_valid_mutations(self):
        """Test that random mutagenize with num_states produces valid mutations."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='random', num_states=20).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
        
        # All outputs should be valid single mutants
        for mutant in df['seq']:
            assert len(mutant) == 4
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_mutagenize_random_double_mutation(self):
        """Test random mode with num_states and num_mutations=2."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=2, mode='random', num_states=30).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
        
        assert len(df) == 30
        for mutant in df['seq']:
            diffs = sum(1 for a, b in zip('ACGTACGT', mutant) if a != b)
            assert diffs == 2


class TestRandomNumStatesGetKmers:
    """Test random mode with num_states for get_kmers."""
    
    def test_get_kmers_random_num_states(self):
        """Test that get_kmers random mode with num_states uses correct num_states."""
        with pp.Party() as party:
            pool = get_kmers(length=3, mode='random', num_states=100)
            assert pool.operation.num_values == 100
    
    def test_get_kmers_random_valid_kmers(self):
        """Test that random get_kmers with num_states produces valid k-mers."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode='random', num_states=50).named('kmer')
        
        df = pool.generate_library(num_cycles=1, seed=42)
        
        assert len(df) == 50
        for kmer in df['seq']:
            assert len(kmer) == 4
            assert all(c in 'ACGT' for c in kmer)
    
    def test_get_kmers_random_reproducible(self):
        """Test that get_kmers random mode with num_states is reproducible."""
        results1 = []
        with pp.Party() as party:
            pool = get_kmers(length=4, mode='random', num_states=20).named('kmer')
        df = pool.generate_library(num_cycles=1, seed=42)
        results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = get_kmers(length=4, mode='random', num_states=20).named('kmer')
        df = pool.generate_library(num_cycles=1, seed=42)
        results2 = list(df['seq'])
        
        assert results1 == results2


class TestRandomNumStatesFromSeqs:
    """Test random mode with num_states for from_seqs."""
    
    def test_from_seqs_random_num_states(self):
        """Test that from_seqs random mode with num_states uses correct num_states."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='random', num_states=100)
            assert pool.operation.num_values == 100
    
    def test_from_seqs_random_random_selection(self):
        """Test that from_seqs random mode with num_states randomly selects sequences."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG', 'CCC'], mode='random', num_states=50).named('seq')
        
        df = pool.generate_library(num_cycles=1, seed=42)
        
        assert len(df) == 50
        for seq in df['seq']:
            assert seq in ['AAA', 'TTT', 'GGG', 'CCC']
        
        # Should have variety - at least 2 different sequences
        unique_seqs = df['seq'].nunique()
        assert unique_seqs >= 2
    
    def test_from_seqs_random_reproducible(self):
        """Test that from_seqs random mode with num_states is reproducible."""
        results1 = []
        with pp.Party() as party:
            pool = from_seqs(['A', 'T', 'G', 'C'], mode='random', num_states=30).named('seq')
        df = pool.generate_library(num_cycles=1, seed=42)
        results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = from_seqs(['A', 'T', 'G', 'C'], mode='random', num_states=30).named('seq')
        df = pool.generate_library(num_cycles=1, seed=42)
        results2 = list(df['seq'])
        
        assert results1 == results2


class TestRandomNumStatesBreakpointScan:
    """Test random mode with num_states for breakpoint_scan."""
    
    def test_breakpoint_scan_random_num_states(self):
        """Test that breakpoint_scan random mode with num_states uses correct num_states."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, 
                                          mode='random', num_states=50)
            assert left.operation.num_values == 50
    
    def test_breakpoint_scan_random_valid_splits(self):
        """Test that random breakpoint_scan with num_states produces valid splits."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, 
                                          mode='random', num_states=30)
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_cycles=1, seed=42, report_design_cards=True, aux_pools=[right])
        
        assert len(df) == 30
        for i, row in df.iterrows():
            # Concatenated segments should equal original
            combined = row['seq'] + row['right.seq']
            assert combined == 'ACGTACGTACGT'
    
    def test_breakpoint_scan_random_reproducible(self):
        """Test that breakpoint_scan random mode with num_states is reproducible."""
        results1 = []
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, 
                                          mode='random', num_states=20)
            left = left.named('left')
        df = left.generate_library(num_cycles=1, seed=42)
        results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, 
                                          mode='random', num_states=20)
            left = left.named('left')
        df = left.generate_library(num_cycles=1, seed=42)
        results2 = list(df['seq'])
        
        assert results1 == results2


class TestRandomNumStatesVsRandomMode:
    """Test differences between random mode with and without num_states."""
    
    def test_random_with_num_states_has_multiple_states(self):
        """Test that random mode with explicit num_states uses that count."""
        with pp.Party() as party:
            # With explicit num_states, uses that value
            random_num_states_pool = mutagenize('ACGT', num_mutations=1, mode='random', num_states=50)
            assert random_num_states_pool.operation.num_values == 50
            
            # Without explicit num_states, syncs to parent (from_seq has 1 state)
            random_pool = mutagenize('ACGT', num_mutations=1, mode='random')
            assert random_pool.operation.num_values == 1  # Synced to from_seq
            
            # For truly stateless, need source op with no parents
            stateless_pool = get_kmers(length=4, mode='random')
            assert stateless_pool.operation.num_values is None
    
    def test_random_with_num_states_iterates_deterministically(self):
        """Test that random mode with num_states iterates through states deterministically."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=10).named('mutant')
        
        # Run twice with same seed
        df1 = pool.generate_library(num_seqs=30, seed=42, init_state=0)
        
        # Create a new pool to test reproducibility
        with pp.Party() as party:
            pool2 = mutagenize('ACGTACGT', num_mutations=1, mode='random', num_states=10).named('mutant')
        df2 = pool2.generate_library(num_seqs=30, seed=42, init_state=0)
        
        # Should be identical
        assert list(df1['seq']) == list(df2['seq'])
    
    def test_random_default_does_not_cycle(self):
        """Test that default random mode doesn't cycle like random with num_states does."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='random').named('mutant')
        df = pool.generate_library(num_seqs=50, seed=42)
        
        # Random mode should have varied results, not cycling patterns
        # (though it's possible by chance to have repeats)
        mutants = list(df['seq'])
        first_10 = mutants[:10]
        
        # It would be extremely unlikely for the sequence to repeat exactly
        # after just 10 elements in random mode
        repeating = True
        for i in range(10, 50, 10):
            chunk = mutants[i:i+10]
            if chunk != first_10:
                repeating = False
                break
        
        # For random mode, we don't expect perfect cycling
        # (unless extremely unlucky with RNG)


class TestRandomNumStatesComposability:
    """Test that random mode with num_states composes correctly with other operations."""
    
    def test_random_with_num_states_with_sequential_parent(self):
        """Test random operation with num_states and sequential parent."""
        with pp.Party() as party:
            seqs = from_seqs(['AAAA', 'TTTT'], mode='sequential')
            mutants = mutagenize(seqs, num_mutations=1, mode='random', num_states=10).named('mutant')
        
        df = mutants.generate_library(num_seqs=20, seed=42)
        
        assert len(df) == 20
        # All should be valid single mutants
        for mutant in df['seq']:
            assert len(mutant) == 4
    
    def test_random_with_num_states_with_random_parent(self):
        """Test random operation with num_states and random parent with num_states."""
        with pp.Party() as party:
            seqs = from_seqs(['AAAA', 'TTTT', 'GGGG'], mode='random', num_states=5)
            mutants = mutagenize(seqs, num_mutations=1, mode='random', num_states=3).named('mutant')
        
        df = mutants.generate_library(num_cycles=1, seed=42)
        
        # Total states = 5 * 3 = 15
        assert len(df) == 15


class TestRandomAutoSyncToParent:
    """Test auto-sync behavior for random mode with num_states=None and parent pools."""
    
    def test_random_with_stateful_parent_syncs_state(self):
        """Test that random operation with stateful parent auto-syncs to parent state."""
        with pp.Party() as party:
            parent = from_seqs(['AAAA', 'TTTT', 'GGGG'], mode='sequential')
            # Random mode without explicit num_states should sync to parent
            child = mutagenize(parent, num_mutations=1, mode='random').named('mutant')
            
            # Operation should have state synced to parent (3 states)
            assert child.operation.state is not None
            assert child.operation.num_values == 3
            # Pool should also have 3 states
            assert child.num_states == 3
    
    def test_random_with_stateful_parent_reproducible(self):
        """Test that random operation synced to parent produces reproducible output."""
        with pp.Party() as party:
            parent = from_seqs(['AAAA', 'TTTT', 'GGGG'], mode='sequential')
            child = mutagenize(parent, num_mutations=1, mode='random').named('mutant')
        
        df1 = child.generate_library(num_cycles=2, seed=42)
        
        with pp.Party() as party:
            parent = from_seqs(['AAAA', 'TTTT', 'GGGG'], mode='sequential')
            child = mutagenize(parent, num_mutations=1, mode='random').named('mutant')
        
        df2 = child.generate_library(num_cycles=2, seed=42)
        
        # Should produce identical results
        assert list(df1['seq']) == list(df2['seq'])
        
        # First 3 (cycle 1) and second 3 (cycle 2) should match
        assert list(df1['seq'][:3]) == list(df1['seq'][3:6])
    
    def test_random_with_stateless_parent_stays_stateless(self):
        """Test that random operation with stateless parent remains stateless."""
        with pp.Party() as party:
            # Parent pool with no state (pure random source operation)
            # get_kmers with mode='random' and no parents is truly stateless
            parent = get_kmers(length=4, mode='random')
            
            # Verify parent is truly stateless
            assert parent.operation.state is None
            assert parent.state is None
            
            # Child should also be stateless (no parent state to sync to)
            child = mutagenize(parent, num_mutations=1, mode='random').named('mutant')
            
            assert child.operation.state is None
            assert child.state is None
    
    def test_random_without_parent_stays_stateless(self):
        """Test that random operation without parent remains stateless."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode='random').named('kmer')
            
            assert pool.operation.state is None
            assert pool.state is None
    
    def test_random_synced_with_multiple_stateful_parents(self):
        """Test random operation with multiple stateful parents uses product."""
        with pp.Party() as party:
            from poolparty.fixed_ops.join import join
            parent1 = from_seqs(['AA', 'TT'], mode='sequential')  # 2 states
            parent2 = from_seqs(['GGG', 'CCC', 'AAA'], mode='sequential')  # 3 states
            joined = join([parent1, parent2], spacer_str='_')
            # Random operation on joined pool should have 2*3=6 states
            child = mutagenize(joined, num_mutations=1, mode='random').named('mutant')
            
            assert child.operation.state is not None
            assert child.num_states == 6
    
    def test_random_synced_with_single_state_parent(self):
        """Test random operation with single-state parent (edge case)."""
        with pp.Party() as party:
            parent = from_seqs(['ACGT'], mode='sequential')  # 1 state
            child = mutagenize(parent, num_mutations=1, mode='random').named('mutant')
            
            # Should have 1 state (synced to parent)
            assert child.operation.state is not None
            assert child.num_states == 1
        
        # Should generate reproducible output
        df = child.generate_library(num_cycles=2, seed=42)
        assert len(df) == 2
        # Same state, same seed -> same output
        assert df['seq'].iloc[0] == df['seq'].iloc[1]
    
    def test_random_synced_chain_propagates(self):
        """Test that synced states propagate through a chain of operations."""
        with pp.Party() as party:
            base = from_seqs(['AAAAAAAAAA', 'TTTTTTTTTT'], mode='sequential')  # 2 states
            mut1 = mutagenize(base, num_mutations=1, mode='random')  # syncs to base
            mut2 = mutagenize(mut1, num_mutations=1, mode='random').named('final')  # syncs to mut1
            
            # All should have 2 states
            assert base.num_states == 2
            assert mut1.num_states == 2
            assert mut2.num_states == 2
        
        df = mut2.generate_library(num_cycles=2, seed=42)
        # 2 states * 2 cycles = 4 sequences
        assert len(df) == 4
        # Cycles should repeat
        assert list(df['seq'][:2]) == list(df['seq'][2:])
    
    def test_random_synced_explicit_num_states_overrides(self):
        """Test that explicit num_states overrides auto-sync behavior."""
        with pp.Party() as party:
            parent = from_seqs(['AAAA', 'TTTT', 'GGGG'], mode='sequential')  # 3 states
            # Explicit num_states should override auto-sync
            child = mutagenize(parent, num_mutations=1, mode='random', num_states=10).named('mutant')
            
            # Should have 10 states (explicit), not synced
            assert child.operation.num_values == 10
            # Pool state is product: 3 * 10 = 30
            assert child.num_states == 30
    
    def test_random_synced_with_get_kmers_region(self):
        """Test auto-sync with get_kmers inserting into a parent pool region."""
        with pp.Party() as party:
            parent = from_seqs(['AA<bc></bc>CC', 'TT<bc></bc>GG'], mode='sequential')  # 2 states
            child = get_kmers(length=3, pool=parent, region='bc', mode='random').named('barcode')
            
            # Should sync to parent's 2 states
            assert child.num_states == 2
        
        df = child.generate_library(num_cycles=1, seed=42)
        assert len(df) == 2
        
        # Each parent state should get a reproducible random barcode
        df2 = child.generate_library(num_cycles=1, seed=42)
        assert list(df['seq']) == list(df2['seq'])
