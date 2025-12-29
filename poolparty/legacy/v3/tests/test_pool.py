"""Tests for the Pool class."""

import pytest
import poolparty as pp
from poolparty import reset_op_id_counter
from poolparty.pool import Pool


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


class TestPoolCreation:
    """Test Pool creation and basic attributes."""
    
    def test_pool_has_operation(self):
        """Test that Pool has operation attribute."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            assert pool.operation is not None
            assert hasattr(pool.operation, 'compute')
    
    def test_pool_output_index_default(self):
        """Test that output_index defaults to 0."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            assert pool.output_index == 0
    
    def test_pool_name_attribute(self):
        """Test Pool name attribute."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            # Name comes from operation by default
            assert pool.name is None
    
    def test_pool_parents_property(self):
        """Test Pool parents property returns operation's parent_pools."""
        with pp.Party() as party:
            seq = pp.from_seqs(['AAA'])
            mutants = pp.mutation_scan(seq, k=1)
            
            # mutants pool should have seq as parent
            assert len(mutants.parents) == 1
            assert mutants.parents[0] is seq


class TestPoolRepr:
    """Test Pool __repr__ formatting."""
    
    def test_repr_basic(self):
        """Test basic repr output."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'], name='test_pool')
            repr_str = repr(pool)
            assert 'Pool' in repr_str
            # The repr uses the operation's name attribute
            assert 'test_pool' in repr_str
    
    def test_repr_multi_output(self):
        """Test repr for multi-output operation."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            left_repr = repr(left)
            right_repr = repr(right)
            
            # Should show output index
            assert 'out=0' in left_repr
            assert 'out=1' in right_repr


class TestPoolAddOperator:
    """Test Pool __add__ operator for concatenation."""
    
    def test_pool_plus_pool(self):
        """Test Pool + Pool concatenation."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            combined = left + right
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAATTT'
    
    def test_pool_plus_string(self):
        """Test Pool + string concatenation."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            combined = pool + '...'
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA...'
    
    def test_chained_add(self):
        """Test chained + operators."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            c = pp.from_seqs(['GGG'])
            combined = a + b + c
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAATTTGGG'
    
    def test_pool_plus_string_plus_pool(self):
        """Test Pool + string + Pool."""
        with pp.Party() as party:
            left = pp.from_seqs(['AAA'])
            right = pp.from_seqs(['TTT'])
            combined = left + '...' + right
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA...TTT'


class TestPoolRaddOperator:
    """Test Pool __radd__ operator for string + Pool."""
    
    def test_string_plus_pool(self):
        """Test string + Pool concatenation."""
        with pp.Party() as party:
            pool = pp.from_seqs(['TTT'])
            combined = '...' + pool
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == '...TTT'
    
    def test_string_plus_pool_plus_string(self):
        """Test string + Pool + string."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            combined = '[' + pool + ']'
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == '[AAA]'


class TestPoolMulOperator:
    """Test Pool __mul__ and __rmul__ operators for repetition."""
    
    def test_pool_times_int(self):
        """Test Pool * int repetition."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AB'])
            repeated = pool * 3
            party.output(repeated, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABABAB'
    
    def test_int_times_pool(self):
        """Test int * Pool repetition."""
        with pp.Party() as party:
            pool = pp.from_seqs(['XY'])
            repeated = 2 * pool
            party.output(repeated, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'XYXY'
    
    def test_pool_times_one(self):
        """Test Pool * 1 returns equivalent result."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            repeated = pool * 1
            party.output(repeated, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAA'


class TestPoolGetitemOperator:
    """Test Pool __getitem__ operator for slicing."""
    
    def test_positive_index(self):
        """Test Pool[positive_int] indexing."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            first = pool[0]
            party.output(first, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'A'
    
    def test_negative_index(self):
        """Test Pool[negative_int] indexing."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            last = pool[-1]
            party.output(last, name='char')
        
        df = party.generate(num_seqs=1)
        assert df['char'].iloc[0] == 'T'
    
    def test_slice_start_stop(self):
        """Test Pool[start:stop] slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            sliced = pool[0:4]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGT'
    
    def test_slice_with_step(self):
        """Test Pool[::step] slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[::2]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACEG'
    
    def test_slice_reverse(self):
        """Test Pool[::-1] reverse slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            reversed_seq = pool[::-1]
            party.output(reversed_seq, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'TGCA'
    
    def test_slice_negative_indices(self):
        """Test slicing with negative indices."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = pool[-4:-1]
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'EFG'


class TestPoolOperatorChaining:
    """Test chaining multiple operators together."""
    
    def test_slice_then_add(self):
        """Test slicing then concatenating."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            left = pool[0:4]
            right = pool[4:8]
            combined = left + '...' + right
            party.output(combined, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACGT...ACGT'
    
    def test_add_then_slice(self):
        """Test concatenating then slicing."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAAA'])
            b = pp.from_seqs(['TTTT'])
            combined = a + b
            first_half = combined[0:4]
            party.output(first_half, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'AAAA'
    
    def test_multiply_then_slice(self):
        """Test multiplication then slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AB'])
            repeated = pool * 4  # 'ABABABAB'
            sliced = repeated[2:6]  # 'ABAB'
            party.output(sliced, name='seq')
        
        df = party.generate(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABAB'


class TestPoolMultiOutput:
    """Test Pool behavior with multi-output operations."""
    
    def test_breakpoint_creates_multiple_pools(self):
        """Test that breakpoint_scan creates multiple pools."""
        with pp.Party() as party:
            pools = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            assert len(pools) == 2
            assert all(isinstance(p, Pool) for p in pools)
    
    def test_multi_output_different_indices(self):
        """Test that multi-output pools have different indices."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            assert left.output_index == 0
            assert right.output_index == 1
    
    def test_multi_output_same_operation(self):
        """Test that multi-output pools reference same operation."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            assert left.operation is right.operation

