"""Tests for the Slice operation."""

import pytest
import poolparty as pp
from poolparty import reset_op_id_counter
from poolparty.operations.slice_op import SliceOp, subseq


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


class TestSubseqFactory:
    """Test subseq factory function."""
    
    def test_returns_pool(self):
        """Test that subseq returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            sliced = subseq(pool, slice(0, 4))
            assert sliced is not None
            assert hasattr(sliced, 'operation')
    
    def test_creates_slice_op(self):
        """Test that subseq creates a SliceOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            sliced = subseq(pool, slice(0, 4))
            assert isinstance(sliced.operation, SliceOp)
    
    def test_subseq_with_int(self):
        """Test subseq with integer index."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = subseq(pool, 0)
            party.output(sliced, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'A'


class TestPoolGetitemOperator:
    """Test Pool.__getitem__ operator (same as subseq)."""
    
    def test_getitem_equivalent_to_subseq(self):
        """Test that pool[key] is equivalent to subseq(pool, key)."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            via_getitem = pool[0:4]
            party.output(via_getitem, name='seq')
        
        df1 = party.generate(num_seqs=1)
        
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            via_subseq = subseq(pool, slice(0, 4))
            party.output(via_subseq, name='seq')
        
        df2 = party.generate(num_seqs=1)
        
        assert df1['seq'].iloc[0] == df2['seq'].iloc[0]


class TestSliceIntegerIndexing:
    """Test integer indexing."""
    
    def test_positive_index_first(self):
        """Test positive index for first character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = pool[0]
            party.output(sliced, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'A'
    
    def test_positive_index_middle(self):
        """Test positive index for middle character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = pool[2]
            party.output(sliced, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'C'
    
    def test_positive_index_last(self):
        """Test positive index for last character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = pool[5]
            party.output(sliced, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'F'
    
    def test_negative_index_last(self):
        """Test negative index for last character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = pool[-1]
            party.output(sliced, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'F'
    
    def test_negative_index_first(self):
        """Test negative index for first character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = pool[-6]
            party.output(sliced, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'A'
    
    def test_negative_index_middle(self):
        """Test negative index for middle character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = pool[-3]
            party.output(sliced, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'D'


class TestSliceRanges:
    """Test slice range operations."""
    
    def test_start_to_end(self):
        """Test slice with start and end."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[2:6]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'CDEF'
    
    def test_from_start(self):
        """Test slice from start."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[:4]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABCD'
    
    def test_to_end(self):
        """Test slice to end."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[4:]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'EFGH'
    
    def test_full_slice(self):
        """Test full slice (copy)."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[:]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABCDEFGH'
    
    def test_negative_range(self):
        """Test slice with negative indices."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[-4:-1]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'EFG'
    
    def test_mixed_positive_negative(self):
        """Test slice with mixed positive and negative indices."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[1:-1]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'BCDEFG'


class TestSliceWithStep:
    """Test slice with step parameter."""
    
    def test_step_two(self):
        """Test slice with step=2."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[::2]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACEG'
    
    def test_step_three(self):
        """Test slice with step=3."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGHI'])
            sliced = pool[::3]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ADG'
    
    def test_step_with_start(self):
        """Test slice with step and start."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[1::2]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'BDFH'
    
    def test_step_with_start_and_end(self):
        """Test slice with step, start, and end."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[1:7:2]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'BDF'


class TestSliceReverse:
    """Test reverse slicing."""
    
    def test_full_reverse(self):
        """Test full reverse with [::-1]."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCD'])
            sliced = pool[::-1]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'DCBA'
    
    def test_reverse_with_start_end(self):
        """Test reverse with start and end."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[6:2:-1]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'GFED'
    
    def test_reverse_step_two(self):
        """Test reverse with step=-2."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[::-2]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'HFDB'


class TestSliceEdgeCases:
    """Test edge cases for slicing."""
    
    def test_empty_slice(self):
        """Test slice that results in empty string."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCD'])
            sliced = pool[2:2]  # Empty range
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == ''
    
    def test_out_of_bounds_slice(self):
        """Test slice with out of bounds indices (Python handles gracefully)."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCD'])
            sliced = pool[0:100]  # Beyond length
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABCD'
    
    def test_negative_out_of_bounds(self):
        """Test slice with negative out of bounds start."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCD'])
            sliced = pool[-100:2]  # Way before start
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AB'
    
    def test_single_char_sequence(self):
        """Test slicing single character sequence."""
        with pp.Party() as party:
            pool = pp.from_seqs(['X'])
            sliced = pool[0]
            party.output(sliced, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'X'


class TestSliceFixedMode:
    """Test that SliceOp is always fixed mode."""
    
    def test_mode_is_fixed(self):
        """Test that slice creates fixed mode operation."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = pool[0:2]
            assert sliced.operation.mode == 'fixed'
    
    def test_num_states_is_one(self):
        """Test that slice has num_states=1."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = pool[0:2]
            assert sliced.operation.num_states == 1
    
    def test_variability_from_parent(self):
        """Test that variability comes from parent pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAAA', 'BBBB', 'CCCC'], mode='sequential')
            sliced = pool[0:2]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=3)
        assert list(df['seq']) == ['AA', 'BB', 'CC']


class TestSliceDesignCards:
    """Test SliceOp design cards."""
    
    def test_no_design_card_keys(self):
        """Test that slice has no design card keys."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = pool[0:2]
            assert len(sliced.operation.design_card_keys) == 0
    
    def test_parent_design_cards_preserved(self):
        """Test that parent design cards are still included."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'], names=['test_seq'])
            sliced = pool[0:2]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert 'from_seqs.seq_name' in df.columns


class TestSliceCompute:
    """Test SliceOp compute method directly."""
    
    def test_compute_with_slice(self):
        """Test compute with slice key."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = subseq(pool, slice(0, 2))
        
        result = sliced.operation.compute(['ACGT'], 0, None)
        assert result['seq_0'] == 'AC'
    
    def test_compute_with_int(self):
        """Test compute with integer key."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = subseq(pool, 0)
        
        result = sliced.operation.compute(['ACGT'], 0, None)
        assert result['seq_0'] == 'A'
    
    def test_compute_returns_string(self):
        """Test that compute always returns string."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = pool[0]
        
        result = sliced.operation.compute(['ACGT'], 0, None)
        assert isinstance(result['seq_0'], str)


class TestSliceChaining:
    """Test chaining slice operations."""
    
    def test_double_slice(self):
        """Test slicing a sliced pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            first_slice = pool[2:6]  # 'CDEF'
            second_slice = first_slice[1:3]  # 'DE'
            party.output(second_slice, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'DE'
    
    def test_slice_then_concatenate(self):
        """Test slicing then concatenating."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            first = pool[0:2]  # 'AB'
            last = pool[-2:]  # 'GH'
            combined = first + '...' + last
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AB...GH'


class TestSliceWithOtherOperations:
    """Test slice with other operation types."""
    
    def test_slice_mutation_scan_output(self):
        """Test slicing mutation scan output."""
        with pp.Party() as party:
            mutants = pp.mutation_scan('ACGTACGT', k=1, mode='sequential')
            first_half = mutants[0:4]
            party.output(first_half, name='seq')
        
        df = party.generate(num_seqs=5)
        for seq in df['seq']:
            assert len(seq) == 4
    
    def test_slice_breakpoint_output(self):
        """Test slicing breakpoint output."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGTACGT', num_breakpoints=1, mode='sequential')
            # Slice the first char from the right segment
            right_first = right[0]
            party.output(right_first, name='char')
        
        df = party.generate(num_seqs=3)
        for char in df['char']:
            assert len(char) == 1
    
    def test_slice_kmer_output(self):
        """Test slicing k-mer output."""
        with pp.Party() as party:
            kmers = pp.get_kmers(length=8, alphabet='dna', mode='random')
            first_half = kmers[0:4]
            party.output(first_half, name='seq')
        
        df = party.generate(num_seqs=10, seed=42)
        for seq in df['seq']:
            assert len(seq) == 4
            assert all(c in 'ACGT' for c in seq)


class TestSliceCustomName:
    """Test SliceOp name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = subseq(pool, slice(0, 2))
            assert sliced.operation.name == 'slice'
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = subseq(pool, slice(0, 2), name='my_slice')
            assert sliced.operation.name == 'my_slice'
    
    def test_pool_getitem_uses_default_name(self):
        """Test that Pool.__getitem__ uses default slice name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = pool[0:2]
            assert sliced.operation.name == 'slice'

