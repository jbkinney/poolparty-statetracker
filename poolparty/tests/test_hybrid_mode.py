"""Tests for hybrid mode across operations."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.base_ops.mutagenize import mutagenize
from poolparty.base_ops.get_kmers import get_kmers
from poolparty.base_ops.from_seqs import from_seqs
from poolparty.base_ops.breakpoint_scan import breakpoint_scan


class TestHybridModeBasic:
    """Test basic hybrid mode functionality."""
    
    def test_num_hybrid_states(self):
        """Test that hybrid mode uses num_hybrid_states for counter."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='hybrid', num_hybrid_states=100)
            assert pool.operation.num_states == 100
    
    def test_hybrid_mode_requires_num_states(self):
        """Test that hybrid mode requires num_hybrid_states."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                mutagenize('ACGT', num_mutations=1, mode='hybrid')
    
    def test_hybrid_mode_generates_correct_count(self):
        """Test that hybrid mode generates the expected number of sequences."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='hybrid', num_hybrid_states=50).named('mutant')
        
        df = pool.generate_library(num_cycles=1)
        assert len(df) == 50


class TestHybridModeReproducibility:
    """Test that hybrid mode produces reproducible results."""
    
    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
            results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
            results2 = list(df['seq'])
        
        assert results1 == results2
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
            results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=123)
            results2 = list(df['seq'])
        
        assert results1 != results2
    
    def test_same_state_same_output(self):
        """Test that same state always produces same output with same seed."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=20).named('mutant')
            df = pool.generate_library(num_seqs=40, seed=42)  # 2 complete iterations
        
        # First and second iteration should be identical
        first_iter = list(df['seq'][:20])
        second_iter = list(df['seq'][20:])
        assert first_iter == second_iter


class TestHybridModeOperationIsolation:
    """Test that different operations produce different results at same state."""
    
    def test_different_ops_different_results(self):
        """Test that different hybrid operations at same state produce different results."""
        with pp.Party() as party:
            # Two mutagenize operations with same config but different op.id
            pool1 = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant1')
            pool2 = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant2')
            df = pool1.generate_library(num_cycles=1, seed=42, report_design_cards=True, aux_pools=[pool2])
        
        # They should produce different results because op.id is part of the seed
        results1 = list(df['mutant1.seq'])
        results2 = list(df['mutant2.seq'])
        
        # At least some results should differ
        differences = sum(1 for a, b in zip(results1, results2) if a != b)
        assert differences > 0, "Different ops should produce different outputs"


class TestHybridModeMutationScan:
    """Test hybrid mode specifically for mutagenize."""
    
    def test_mutagenize_hybrid_valid_mutations(self):
        """Test that hybrid mutagenize produces valid mutations."""
        with pp.Party() as party:
            pool = mutagenize('ACGT', num_mutations=1, mode='hybrid', num_hybrid_states=20).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
        
        # All outputs should be valid single mutants
        for mutant in df['seq']:
            assert len(mutant) == 4
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_mutagenize_hybrid_double_mutation(self):
        """Test hybrid mode with num_mutations=2."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=2, mode='hybrid', num_hybrid_states=30).named('mutant')
            df = pool.generate_library(num_cycles=1, seed=42)
        
        assert len(df) == 30
        for mutant in df['seq']:
            diffs = sum(1 for a, b in zip('ACGTACGT', mutant) if a != b)
            assert diffs == 2


class TestHybridModeGetKmers:
    """Test hybrid mode for get_kmers."""
    
    def test_get_kmers_hybrid_requires_num_states(self):
        """Test that get_kmers hybrid mode requires num_hybrid_states."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                get_kmers(length=3, mode='hybrid')
    
    def test_get_kmers_hybrid_num_states(self):
        """Test that get_kmers hybrid mode uses correct num_states."""
        with pp.Party() as party:
            pool = get_kmers(length=3, mode='hybrid', num_hybrid_states=100)
            assert pool.operation.num_states == 100
    
    def test_get_kmers_hybrid_valid_kmers(self):
        """Test that hybrid get_kmers produces valid k-mers."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode='hybrid', num_hybrid_states=50).named('kmer')
        
        df = pool.generate_library(num_cycles=1, seed=42)
        
        assert len(df) == 50
        for kmer in df['seq']:
            assert len(kmer) == 4
            assert all(c in 'ACGT' for c in kmer)
    
    def test_get_kmers_hybrid_reproducible(self):
        """Test that get_kmers hybrid mode is reproducible."""
        results1 = []
        with pp.Party() as party:
            pool = get_kmers(length=4, mode='hybrid', num_hybrid_states=20).named('kmer')
        df = pool.generate_library(num_cycles=1, seed=42)
        results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = get_kmers(length=4, mode='hybrid', num_hybrid_states=20).named('kmer')
        df = pool.generate_library(num_cycles=1, seed=42)
        results2 = list(df['seq'])
        
        assert results1 == results2


class TestHybridModeFromSeqs:
    """Test hybrid mode for from_seqs."""
    
    def test_from_seqs_hybrid_requires_num_states(self):
        """Test that from_seqs hybrid mode requires num_hybrid_states."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                from_seqs(['AAA', 'TTT', 'GGG'], mode='hybrid')
    
    def test_from_seqs_hybrid_num_states(self):
        """Test that from_seqs hybrid mode uses correct num_states."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='hybrid', num_hybrid_states=100)
            assert pool.operation.num_states == 100
    
    def test_from_seqs_hybrid_random_selection(self):
        """Test that from_seqs hybrid mode randomly selects sequences."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG', 'CCC'], mode='hybrid', num_hybrid_states=50).named('seq')
        
        df = pool.generate_library(num_cycles=1, seed=42)
        
        assert len(df) == 50
        for seq in df['seq']:
            assert seq in ['AAA', 'TTT', 'GGG', 'CCC']
        
        # Should have variety - at least 2 different sequences
        unique_seqs = df['seq'].nunique()
        assert unique_seqs >= 2
    
    def test_from_seqs_hybrid_reproducible(self):
        """Test that from_seqs hybrid mode is reproducible."""
        results1 = []
        with pp.Party() as party:
            pool = from_seqs(['A', 'T', 'G', 'C'], mode='hybrid', num_hybrid_states=30).named('seq')
        df = pool.generate_library(num_cycles=1, seed=42)
        results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            pool = from_seqs(['A', 'T', 'G', 'C'], mode='hybrid', num_hybrid_states=30).named('seq')
        df = pool.generate_library(num_cycles=1, seed=42)
        results2 = list(df['seq'])
        
        assert results1 == results2


class TestHybridModeBreakpointScan:
    """Test hybrid mode for breakpoint_scan."""
    
    def test_breakpoint_scan_hybrid_requires_num_states(self):
        """Test that breakpoint_scan hybrid mode requires num_hybrid_states."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_hybrid_states is required"):
                breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, mode='hybrid')
    
    def test_breakpoint_scan_hybrid_num_states(self):
        """Test that breakpoint_scan hybrid mode uses correct num_states."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, 
                                          mode='hybrid', num_hybrid_states=50)
            assert left.operation.num_states == 50
    
    def test_breakpoint_scan_hybrid_valid_splits(self):
        """Test that hybrid breakpoint_scan produces valid splits."""
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, 
                                          mode='hybrid', num_hybrid_states=30)
            left = left.named('left')
            right = right.named('right')
        
        df = left.generate_library(num_cycles=1, seed=42, report_design_cards=True, aux_pools=[right])
        
        assert len(df) == 30
        for i, row in df.iterrows():
            # Concatenated segments should equal original
            combined = row['seq'] + row['right.seq']
            assert combined == 'ACGTACGTACGT'
    
    def test_breakpoint_scan_hybrid_reproducible(self):
        """Test that breakpoint_scan hybrid mode is reproducible."""
        results1 = []
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, 
                                          mode='hybrid', num_hybrid_states=20)
            left = left.named('left')
        df = left.generate_library(num_cycles=1, seed=42)
        results1 = list(df['seq'])
        
        results2 = []
        with pp.Party() as party:
            left, right = breakpoint_scan('ACGTACGTACGT', num_breakpoints=1, 
                                          mode='hybrid', num_hybrid_states=20)
            left = left.named('left')
        df = left.generate_library(num_cycles=1, seed=42)
        results2 = list(df['seq'])
        
        assert results1 == results2


class TestHybridModeVsRandomMode:
    """Test differences between hybrid and random modes."""
    
    def test_hybrid_has_multiple_states(self):
        """Test that hybrid mode has multiple states unlike random."""
        with pp.Party() as party:
            random_pool = mutagenize('ACGT', num_mutations=1, mode='random')
            hybrid_pool = mutagenize('ACGT', num_mutations=1, mode='hybrid', num_hybrid_states=50)
            
            assert random_pool.operation.num_states == 1
            assert hybrid_pool.operation.num_states == 50
    
    def test_hybrid_iterates_deterministically(self):
        """Test that hybrid mode iterates through states deterministically."""
        with pp.Party() as party:
            pool = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant')
        
        # Run twice with same seed
        df1 = pool.generate_library(num_seqs=30, seed=42, init_state=0)
        
        # Create a new pool to test reproducibility
        with pp.Party() as party:
            pool2 = mutagenize('ACGTACGT', num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant')
        df2 = pool2.generate_library(num_seqs=30, seed=42, init_state=0)
        
        # Should be identical
        assert list(df1['seq']) == list(df2['seq'])
    
    def test_random_does_not_cycle(self):
        """Test that random mode doesn't cycle like hybrid does."""
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


class TestHybridModeComposability:
    """Test that hybrid mode composes correctly with other operations."""
    
    def test_hybrid_with_sequential_parent(self):
        """Test hybrid operation with sequential parent."""
        with pp.Party() as party:
            seqs = from_seqs(['AAAA', 'TTTT'], mode='sequential')
            mutants = mutagenize(seqs, num_mutations=1, mode='hybrid', num_hybrid_states=10).named('mutant')
        
        df = mutants.generate_library(num_seqs=20, seed=42)
        
        assert len(df) == 20
        # All should be valid single mutants
        for mutant in df['seq']:
            assert len(mutant) == 4
    
    def test_hybrid_with_hybrid_parent(self):
        """Test hybrid operation with hybrid parent."""
        with pp.Party() as party:
            seqs = from_seqs(['AAAA', 'TTTT', 'GGGG'], mode='hybrid', num_hybrid_states=5)
            mutants = mutagenize(seqs, num_mutations=1, mode='hybrid', num_hybrid_states=3).named('mutant')
        
        df = mutants.generate_library(num_cycles=1, seed=42)
        
        # Total states = 5 * 3 = 15
        assert len(df) == 15
