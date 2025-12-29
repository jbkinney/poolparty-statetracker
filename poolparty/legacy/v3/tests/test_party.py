"""Tests for the Party context manager and core architecture."""

import pytest
import poolparty as pp
from poolparty import reset_op_id_counter


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


class TestBasicUsage:
    """Test basic Party usage patterns."""
    
    def test_simple_from_seqs(self):
        """Test creating a simple pool from sequences."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT', 'GGG'])
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=3)
        assert len(df) == 3
        assert 'seq' in df.columns
        assert list(df['seq']) == ['AAA', 'TTT', 'GGG']
    
    def test_get_kmers_sequential(self):
        """Test generating k-mers in sequential mode."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=2, alphabet='AB', mode='sequential')
            party.output(pool, name='kmer')
        
        df = party.generate(num_complete_iterations=1)
        assert len(df) == 4  # 2^2 = 4 k-mers
        assert list(df['kmer']) == ['AA', 'AB', 'BA', 'BB']
    
    def test_get_kmers_random(self):
        """Test generating k-mers in random mode."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=5, alphabet='dna', mode='random')
            party.output(pool, name='kmer')
        
        df = party.generate(num_seqs=10, seed=42)
        assert len(df) == 10
        # All should be valid 5-mers of DNA
        for kmer in df['kmer']:
            assert len(kmer) == 5
            assert all(c in 'ACGT' for c in kmer)
    
    def test_concatenation(self):
        """Test concatenating pools."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            oligo = left + '...' + right
            party.output(oligo, name='oligo')
        
        df = party.generate(num_seqs=1)
        assert df['oligo'].iloc[0] == 'AAA...TTT'
    
    def test_concatenation_with_kmer(self):
        """Test concatenating a sequence with a random barcode."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            barcode = pp.get_kmers(length=4, alphabet='dna', mode='random')
            oligo = seq + '...' + barcode
            party.output(oligo, name='oligo')
        
        df = party.generate(num_seqs=5, seed=42)
        assert len(df) == 5
        for oligo in df['oligo']:
            assert oligo.startswith('ACGT...')
            assert len(oligo) == 4 + 3 + 4  # seq + ... + barcode


class TestMutationScan:
    """Test mutation scanning operations."""
    
    def test_single_mutation_sequential(self):
        """Test single mutation in sequential mode."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=1, mode='sequential')
            party.output(mutants, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        # 4 positions * 3 mutations each = 12 mutants
        assert len(df) == 12
        
        # Check all are single mutations
        for mutant in df['mutant']:
            assert len(mutant) == 4
            diffs = sum(1 for a, b in zip('ACGT', mutant) if a != b)
            assert diffs == 1
    
    def test_double_mutation_sequential(self):
        """Test double mutation in sequential mode."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=2, mode='sequential')
            party.output(mutants, name='mutant')
        
        df = party.generate(num_complete_iterations=1)
        # C(4,2) * 3^2 = 6 * 9 = 54 mutants
        assert len(df) == 54
    
    def test_mutation_scan_random(self):
        """Test mutation scan in random mode."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGTACGT', k=1, mode='random')
            party.output(mutants, name='mutant')
        
        df = party.generate(num_seqs=10, seed=42)
        assert len(df) == 10


class TestBreakpointScan:
    """Test breakpoint scanning operations."""
    
    def test_single_breakpoint(self):
        """Test splitting with one breakpoint."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1, mode='sequential')
            party.output(left, name='left')
            party.output(right, name='right')
        
        df = party.generate(num_complete_iterations=1)
        # 3 possible breakpoint positions
        assert len(df) == 3
        
        # Check all splits are valid
        for _, row in df.iterrows():
            assert row['left'] + row['right'] == 'ACGT'
    
    def test_double_breakpoint(self):
        """Test splitting with two breakpoints."""
        with pp.Party() as party:
            left, mid, right = pp.breakpoint_scan('ACGTACGT', num_breakpoints=2, 
                                                   mode='sequential')
            party.output(left, name='left')
            party.output(mid, name='mid')
            party.output(right, name='right')
        
        df = party.generate(num_complete_iterations=1)
        
        # Check all splits are valid
        for _, row in df.iterrows():
            assert row['left'] + row['mid'] + row['right'] == 'ACGTACGT'
    
    def test_breakpoint_with_mutation(self):
        """Test combining breakpoint scan with mutation."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGTACGT', num_breakpoints=1, 
                                              mode='sequential')
            mutated_right = pp.mutation_scan(right, k=1, mode='sequential')
            oligo = left + mutated_right
            party.output(oligo, name='oligo')
        
        df = party.generate(num_seqs=10)
        assert len(df) == 10


class TestStatePersistence:
    """Test that Party maintains state between generate() calls."""
    
    def test_state_continues(self):
        """Test that sequential iteration continues across generate() calls."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D'], mode='sequential')
            party.output(pool, name='seq')
        
        df1 = party.generate(num_seqs=2)
        df2 = party.generate(num_seqs=2)
        
        assert list(df1['seq']) == ['A', 'B']
        assert list(df2['seq']) == ['C', 'D']
    
    def test_reset(self):
        """Test that reset() resets the state."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            party.output(pool, name='seq')
        
        df1 = party.generate(num_seqs=2)
        party.reset()
        df2 = party.generate(num_seqs=2)
        
        assert list(df1['seq']) == ['A', 'B']
        assert list(df2['seq']) == ['A', 'B']  # Reset to start
    
    def test_init_state(self):
        """Test starting from a specific state."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D'], mode='sequential')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=2, init_state=2)
        assert list(df['seq']) == ['C', 'D']


class TestMixedModes:
    """Test combining sequential and random operations."""
    
    def test_sequential_with_random_barcode(self):
        """Test sequential mutations with random barcodes."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=1, mode='sequential')
            barcode = pp.get_kmers(length=4, alphabet='dna', mode='random')
            oligo = mutants + '.' + barcode
            party.output(oligo, name='oligo')
        
        df = party.generate(num_complete_iterations=1, seed=42)
        assert len(df) == 12  # 12 single mutants
        
        # Check that barcodes are varied (random)
        barcodes = [oligo.split('.')[-1] for oligo in df['oligo']]
        assert len(set(barcodes)) > 1  # Not all the same


class TestDesignCards:
    """Test design card metadata."""
    
    def test_mutation_scan_metadata(self):
        """Test that mutation scan includes position and char metadata."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGT', k=1, mode='sequential')
            party.output(mutants, name='mutant')
        
        df = party.generate(num_seqs=3)
        
        assert 'mutation_scan.positions' in df.columns
        assert 'mutation_scan.wt_chars' in df.columns
        assert 'mutation_scan.mut_chars' in df.columns
    
    def test_from_seqs_metadata(self):
        """Test that from_seqs includes name and index metadata."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'], names=['seq_a', 'seq_b'])
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=2)
        
        assert 'from_seqs.seq_name' in df.columns
        assert 'from_seqs.seq_index' in df.columns
        assert list(df['from_seqs.seq_name']) == ['seq_a', 'seq_b']


class TestSliceOp:
    """Test slicing operations."""
    
    def test_slice_with_range(self):
        """Test slicing with a range."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            sliced = pool[0:4]
            party.output(sliced, name='sliced')
        
        df = party.generate(num_seqs=1)
        assert df['sliced'].iloc[0] == 'ACGT'
    
    def test_slice_negative_index(self):
        """Test slicing with negative index."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            last = pool[-1]
            party.output(last, name='last')
        
        df = party.generate(num_seqs=1)
        assert df['last'].iloc[0] == 'T'
    
    def test_slice_with_step(self):
        """Test slicing with step."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            every_other = pool[::2]
            party.output(every_other, name='every_other')
        
        df = party.generate(num_seqs=1)
        assert df['every_other'].iloc[0] == 'ACEG'
    
    def test_slice_reverse(self):
        """Test reversing a sequence with slice."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            reversed_seq = pool[::-1]
            party.output(reversed_seq, name='reversed')
        
        df = party.generate(num_seqs=1)
        assert df['reversed'].iloc[0] == 'TGCA'
    
    def test_subseq_function(self):
        """Test the subseq factory function."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            sliced = pp.subseq(pool, slice(2, 6))
            party.output(sliced, name='middle')
        
        df = party.generate(num_seqs=1)
        assert df['middle'].iloc[0] == 'GTAC'
    
    def test_slice_with_mutation(self):
        """Test combining slice with mutation."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            first_half = pool[0:4]
            mutated = pp.mutation_scan(first_half, k=1, mode='sequential')
            party.output(mutated, name='mutated')
        
        df = party.generate(num_seqs=3)
        # All should be 4 characters
        for seq in df['mutated']:
            assert len(seq) == 4


class TestErrors:
    """Test error handling."""
    
    def test_no_outputs_error(self):
        """Test error when no outputs are defined."""
        with pp.Party() as party:
            pp.from_seqs(['AAA'])
        
        with pytest.raises(ValueError, match="No outputs defined"):
            party.generate(num_seqs=1)
    
    def test_nested_party_error(self):
        """Test error for nested Party contexts."""
        with pp.Party():
            with pytest.raises(RuntimeError, match="Nested Party contexts"):
                with pp.Party():
                    pass
    
    def test_num_seqs_required(self):
        """Test error when neither num_seqs nor num_complete_iterations given."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            party.output(pool)
        
        with pytest.raises(ValueError, match="Must specify"):
            party.generate()

