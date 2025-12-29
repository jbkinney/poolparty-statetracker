"""Tests for the FromSeq operation."""

import pytest
import poolparty as pp
from poolparty.operations.from_seq import FromSeqOp, from_seq


class TestFromSeqFactory:
    """Test from_seq factory function."""
    
    def test_returns_pool(self):
        """Test that from_seq returns a Pool."""
        with pp.Party() as party:
            pool = from_seq('AAA')
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_from_seq_op(self):
        """Test that from_seq creates a FromSeqOp."""
        with pp.Party() as party:
            pool = from_seq('AAA')
            assert isinstance(pool.operation, FromSeqOp)
    
    def test_mode_is_fixed(self):
        """Test that the operation mode is always 'fixed'."""
        with pp.Party() as party:
            pool = from_seq('AAA')
            assert pool.operation.mode == 'fixed'


class TestFromSeqGeneration:
    """Test sequence generation from FromSeq."""
    
    def test_generates_same_sequence(self):
        """Test that the same sequence is generated repeatedly."""
        with pp.Party() as party:
            pool = from_seq('ATGC').named('seq')
        
        df = pool.generate_seqs(num_seqs=5)
        assert list(df['seq']) == ['ATGC', 'ATGC', 'ATGC', 'ATGC', 'ATGC']
    
    def test_num_states_is_one(self):
        """Test that num_states is always 1."""
        with pp.Party() as party:
            pool = from_seq('AAA')
            assert pool.operation.num_states == 1
            assert pool.num_states == 1


class TestFromSeqDesignCards:
    """Test FromSeq design card output."""
    
    def test_seq_in_output(self):
        """Test seq is in output."""
        with pp.Party() as party:
            pool = from_seq('ATGC', op_name='my_seq').named('mypool')
        
        df = pool.generate_seqs(num_seqs=1)
        assert 'mypool.op.key.seq' in df.columns
        # Use iloc with column index to handle potential duplicate columns
        seq_col_idx = list(df.columns).index('mypool.op.key.seq')
        assert df.iloc[0, seq_col_idx] == 'ATGC'
    
    def test_design_card_keys_defined(self):
        """Test design_card_keys is defined correctly."""
        with pp.Party() as party:
            pool = from_seq('AAA')
            assert 'seq' in pool.operation.design_card_keys


class TestFromSeqCustomName:
    """Test FromSeq operation and pool name parameters."""
    
    def test_default_operation_name(self):
        """Test default operation name is op[{id}]:from_seq."""
        with pp.Party() as party:
            pool = from_seq('AAA')
            # Default name is op[{id}]:from_seq
            assert pool.operation.name.startswith('op[')
            assert ':from_seq' in pool.operation.name
    
    def test_custom_operation_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = from_seq('AAA', op_name='my_sequence')
            assert pool.operation.name == 'my_sequence'
    
    def test_custom_pool_name(self):
        """Test custom pool name."""
        with pp.Party() as party:
            pool = from_seq('AAA', pool_name='my_pool')
            assert pool.name == 'my_pool'
    
    def test_custom_name_in_design_card(self):
        """Test custom op name appears in design card columns."""
        with pp.Party() as party:
            pool = from_seq('ATGC', op_name='my_seq').named('mypool')
        
        df = pool.generate_seqs(num_seqs=1)
        assert 'mypool.op.key.seq' in df.columns


class TestFromSeqCompute:
    """Test FromSeq compute methods directly."""
    
    def test_compute_returns_sequence(self):
        """Test compute returns the sequence."""
        with pp.Party() as party:
            pool = from_seq('ATGC')
        
        card = pool.operation.compute_design_card([])
        result = pool.operation.compute_seq_from_card([], card)
        assert result['seq_0'] == 'ATGC'
        assert card['seq'] == 'ATGC'

