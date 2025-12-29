"""Tests for slice operations - both state slicing and sequence slicing."""

import pytest
import poolparty as pp
from poolparty.operations.seq_slice import SeqSliceOp, seq_slice
from poolparty.operations.state_slice import StateSliceOp, state_slice


# =============================================================================
# Tests for seq_slice - SEQUENCE slicing (slicing characters in a sequence)
# =============================================================================

class TestSeqSliceFactory:
    """Test seq_slice factory function."""
    
    def test_returns_pool(self):
        """Test that seq_slice returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            sliced = seq_slice(pool, slice(0, 4))
            assert sliced is not None
            assert hasattr(sliced, 'operation')
    
    def test_creates_seq_slice_op(self):
        """Test that seq_slice creates a SeqSliceOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGTACGT'])
            sliced = seq_slice(pool, slice(0, 4))
            assert isinstance(sliced.operation, SeqSliceOp)
    
    def test_seq_slice_with_int(self):
        """Test seq_slice with integer index."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = seq_slice(pool, 0).named('char')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'A'


class TestSeqSliceIntegerIndexing:
    """Test integer indexing for sequence slicing."""
    
    def test_positive_index_first(self):
        """Test positive index for first character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = seq_slice(pool, 0).named('char')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'A'
    
    def test_positive_index_middle(self):
        """Test positive index for middle character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = seq_slice(pool, 2).named('char')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'C'
    
    def test_negative_index_last(self):
        """Test negative index for last character."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEF'])
            sliced = seq_slice(pool, -1).named('char')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'F'


class TestSeqSliceRanges:
    """Test slice range operations for sequence slicing."""
    
    def test_start_to_end(self):
        """Test slice with start and end."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = seq_slice(pool, slice(2, 6)).named('sl')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'CDEF'
    
    def test_from_start(self):
        """Test slice from start."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = seq_slice(pool, slice(None, 4)).named('sl')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABCD'
    
    def test_to_end(self):
        """Test slice to end."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = seq_slice(pool, slice(4, None)).named('sl')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'EFGH'
    
    def test_full_slice(self):
        """Test full slice (copy)."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = seq_slice(pool, slice(None, None)).named('sl')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'ABCDEFGH'


class TestSeqSliceWithStep:
    """Test slice with step parameter for sequence slicing."""
    
    def test_step_two(self):
        """Test slice with step=2."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCDEFGH'])
            sliced = seq_slice(pool, slice(None, None, 2)).named('sl')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'ACEG'
    
    def test_reverse(self):
        """Test reverse slice with step=-1."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ABCD'])
            sliced = seq_slice(pool, slice(None, None, -1)).named('sl')
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'DCBA'


class TestSeqSliceCompute:
    """Test SeqSliceOp compute methods directly."""
    
    def test_compute_with_slice(self):
        """Test compute with slice key."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = seq_slice(pool, slice(0, 2))
        
        card = sliced.operation.compute_design_card(['ACGT'])
        result = sliced.operation.compute_seq_from_card(['ACGT'], card)
        assert result['seq_0'] == 'AC'
    
    def test_compute_with_int(self):
        """Test compute with integer key."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = seq_slice(pool, 0)
        
        card = sliced.operation.compute_design_card(['ACGT'])
        result = sliced.operation.compute_seq_from_card(['ACGT'], card)
        assert result['seq_0'] == 'A'


class TestSeqSliceCustomName:
    """Test SeqSliceOp name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = seq_slice(pool, slice(0, 2))
            assert sliced.operation.name.startswith('op[')
            assert ':seq_slice' in sliced.operation.name
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['ACGT'])
            sliced = seq_slice(pool, slice(0, 2), op_name='my_slice')
            assert sliced.operation.name == 'my_slice'


# =============================================================================
# Tests for state_slice - STATE slicing (selecting subset of states)
# =============================================================================

class TestStateSliceFactory:
    """Test state_slice factory function."""
    
    def test_returns_pool(self):
        """Test that state_slice returns a Pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'])
            sliced = state_slice(pool, slice(1, 4))
            assert sliced is not None
            assert hasattr(sliced, 'operation')
    
    def test_creates_state_slice_op(self):
        """Test that state_slice creates a StateSliceOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'])
            sliced = state_slice(pool, slice(1, 4))
            assert isinstance(sliced.operation, StateSliceOp)


class TestStateSliceNumStates:
    """Test state slicing affects num_states correctly."""
    
    def test_slice_reduces_num_states(self):
        """Test that state slicing reduces num_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'])  # 5 states
            sliced = state_slice(pool, slice(1, 4))  # 3 states
            assert sliced.num_states == 3
    
    def test_slice_with_step(self):
        """Test state slicing with step."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E', 'F'])  # 6 states
            sliced = state_slice(pool, slice(None, None, 2))  # Every other: A, C, E -> 3 states
            assert sliced.num_states == 3


class TestStateSliceOutput:
    """Test state slicing output."""
    
    def test_correct_states_selected(self):
        """Test that correct states are selected."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'])
            sliced = state_slice(pool, slice(1, 4)).named('sl')  # B, C, D
        
        df = sliced.generate_seqs(num_complete_iterations=1)
        assert list(df['seq']) == ['B', 'C', 'D']
    
    def test_from_start(self):
        """Test state slice from start."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'])
            sliced = state_slice(pool, slice(None, 3)).named('sl')  # A, B, C
        
        df = sliced.generate_seqs(num_complete_iterations=1)
        assert list(df['seq']) == ['A', 'B', 'C']
    
    def test_to_end(self):
        """Test state slice to end."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D', 'E'])
            sliced = state_slice(pool, slice(2, None)).named('sl')  # C, D, E
        
        df = sliced.generate_seqs(num_complete_iterations=1)
        assert list(df['seq']) == ['C', 'D', 'E']


class TestStateSliceCustomName:
    """Test StateSliceOp name parameter."""
    
    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])
            sliced = state_slice(pool, slice(0, 2))
            assert sliced.operation.name.startswith('op[')
            assert ':state_slice' in sliced.operation.name
    
    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])
            sliced = state_slice(pool, slice(0, 2), op_name='my_state_slice')
            assert sliced.operation.name == 'my_state_slice'


# =============================================================================
# Tests for Pool.__getitem__ - should now do STATE slicing
# =============================================================================

class TestPoolGetitemOperator:
    """Test Pool.__getitem__ operator (now does STATE slicing)."""
    
    def test_getitem_does_state_slice(self):
        """Test that pool[key] does state slicing, not sequence slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AA', 'BB', 'CC', 'DD', 'EE'])  # 5 states
            sliced = pool[1:4]  # Should select states 1, 2, 3
            assert sliced.num_states == 3
    
    def test_getitem_with_slice(self):
        """Test pool[start:stop] for state slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AA', 'BB', 'CC', 'DD', 'EE'])
            sliced = pool[1:4].named('sl')  # States 1, 2, 3 -> BB, CC, DD
        
        df = sliced.generate_seqs(num_complete_iterations=1)
        assert list(df['seq']) == ['BB', 'CC', 'DD']
    
    def test_getitem_with_int(self):
        """Test pool[index] for single state selection."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AA', 'BB', 'CC'])
            sliced = pool[1].named('sl')  # State 1 -> BB
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'BB'
    
    def test_getitem_negative_index(self):
        """Test pool[-1] for last state."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AA', 'BB', 'CC'])
            sliced = pool[-1].named('sl')  # Last state -> CC
        
        df = sliced.generate_seqs(num_seqs=1)
        assert df['seq'].iloc[0] == 'CC'
    
    def test_getitem_uses_state_slice_op(self):
        """Test that Pool.__getitem__ creates StateSliceOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])
            sliced = pool[0:2]
            assert isinstance(sliced.operation, StateSliceOp)
    
    def test_getitem_default_name(self):
        """Test that Pool.__getitem__ uses default state_slice name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])
            sliced = pool[0:2]
            assert sliced.operation.name.startswith('op[')
            assert ':state_slice' in sliced.operation.name
