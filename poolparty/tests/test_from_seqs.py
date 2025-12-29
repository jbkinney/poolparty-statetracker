"""Tests for the FromSeqs operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty.operations.from_seqs import FromSeqsOp, from_seqs


class TestFromSeqsFactory:
    """Test from_seqs factory function."""
    
    def test_returns_pool(self):
        """Test that from_seqs returns a Pool."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'])
            assert pool is not None
            assert hasattr(pool, 'operation')
    
    def test_creates_from_seqs_op(self):
        """Test that from_seqs creates a FromSeqsOp."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'])
            assert isinstance(pool.operation, FromSeqsOp)
    
    def test_with_names(self):
        """Test from_seqs with custom names."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT'], names=['seq_a', 'seq_b'], op_name='seqs').named('mypool')
        
        df = pool.generate_seqs(num_seqs=2)
        assert list(df['mypool.op.key.seq_name']) == ['seq_a', 'seq_b']


class TestFromSeqsSequentialMode:
    """Test FromSeqs in sequential mode."""
    
    def test_sequential_iteration(self):
        """Test sequential iteration through sequences."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B', 'C', 'D'], mode='sequential').named('mypool')
        
        df = pool.generate_seqs(num_seqs=4)
        assert list(df['seq']) == ['A', 'B', 'C', 'D']
    
    def test_sequential_cycling(self):
        """Test that sequential mode cycles when exhausted."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B'], mode='sequential').named('mypool')
        
        df = pool.generate_seqs(num_seqs=5)
        assert list(df['seq']) == ['A', 'B', 'A', 'B', 'A']
    
    def test_sequential_num_states(self):
        """Test num_states equals number of sequences."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B', 'C'], mode='sequential')
            assert pool.operation.num_states == 3


class TestFromSeqsRandomMode:
    """Test FromSeqs in random mode."""
    
    def test_random_sampling(self):
        """Test random sampling from sequences."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B', 'C', 'D'], mode='random').named('mypool')
        
        df = pool.generate_seqs(num_seqs=100, seed=42)
        assert len(df) == 100
        # All should be from the input sequences
        assert all(s in 'ABCD' for s in df['seq'])
    
    def test_random_uniform_distribution(self):
        """Test that random sampling is roughly uniform."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B'], mode='random').named('mypool')
        
        df = pool.generate_seqs(num_seqs=1000, seed=42)
        counts = df['seq'].value_counts()
        # Should be roughly 50/50 (within 10%)
        assert 400 < counts['A'] < 600
        assert 400 < counts['B'] < 600
    
    def test_random_num_states_is_one(self):
        """Test that random mode has num_states=1."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B', 'C'], mode='random')
            assert pool.operation.num_states == 1


class TestFromSeqsFixedMode:
    """Test FromSeqs in fixed mode."""
    
    def test_fixed_single_seq(self):
        """Test fixed mode with single sequence."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], mode='fixed').named('mypool')
        
        df = pool.generate_seqs(num_seqs=3)
        assert list(df['seq']) == ['AAA', 'AAA', 'AAA']
    
    def test_fixed_multiple_seqs_raises(self):
        """Test that fixed mode with multiple seqs raises error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="exactly 1 sequence"):
                from_seqs(['A', 'B'], mode='fixed')
    
    def test_fixed_num_states_is_one(self):
        """Test that fixed mode has num_states=1."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], mode='fixed')
            assert pool.operation.num_states == 1


class TestFromSeqsNames:
    """Test FromSeqs name handling."""
    
    def test_default_names(self):
        """Test default names are seq_0, seq_1, etc."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG'], op_name='seqs').named('mypool')
        
        df = pool.generate_seqs(num_seqs=3)
        assert list(df['mypool.op.key.seq_name']) == ['seq_0', 'seq_1', 'seq_2']
    
    def test_custom_names(self):
        """Test custom names."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT'], names=['first', 'second'], op_name='seqs').named('mypool')
        
        df = pool.generate_seqs(num_seqs=2)
        assert list(df['mypool.op.key.seq_name']) == ['first', 'second']


class TestFromSeqsDesignCards:
    """Test FromSeqs design card output."""
    
    def test_seq_name_in_output(self):
        """Test seq_name is in output."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], names=['test'], op_name='seqs').named('mypool')
        
        df = pool.generate_seqs(num_seqs=1)
        assert 'mypool.op.key.seq_name' in df.columns
        assert df['mypool.op.key.seq_name'].iloc[0] == 'test'
    
    def test_seq_index_in_output(self):
        """Test seq_index is in output."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B', 'C'], op_name='seqs').named('mypool')
        
        df = pool.generate_seqs(num_seqs=3)
        assert 'mypool.op.key.seq_index' in df.columns
        assert list(df['mypool.op.key.seq_index']) == [0, 1, 2]
    
    def test_design_card_keys_defined(self):
        """Test design_card_keys is defined correctly."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'])
            assert 'seq_name' in pool.operation.design_card_keys
            assert 'seq_index' in pool.operation.design_card_keys


class TestFromSeqsErrors:
    """Test FromSeqs error handling."""
    
    def test_empty_seqs_error(self):
        """Test error for empty sequences list."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="must not be empty"):
                from_seqs([])
    
    def test_mismatched_names_length(self):
        """Test error for mismatched names length."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="same length"):
                from_seqs(['A', 'B', 'C'], names=['x', 'y'])


class TestFromSeqsCompute:
    """Test FromSeqs compute methods directly."""
    
    def test_compute_sequential(self):
        """Test compute in sequential mode returns based on counter state."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='sequential')
        
        # Set counter state to 0
        pool.operation.counter._state = 0
        card = pool.operation.compute_design_card([])
        result = pool.operation.compute_seq_from_card([], card)
        assert result['seq_0'] == 'AAA'
        assert card['seq_index'] == 0
        
        # Set counter state to 1
        pool.operation.counter._state = 1
        card = pool.operation.compute_design_card([])
        result = pool.operation.compute_seq_from_card([], card)
        assert result['seq_0'] == 'TTT'
        assert card['seq_index'] == 1
    
    def test_compute_random(self):
        """Test compute in random mode uses RNG."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='random')
        
        rng = np.random.default_rng(42)
        card = pool.operation.compute_design_card([], rng)
        result = pool.operation.compute_seq_from_card([], card)
        assert result['seq_0'] in ['AAA', 'TTT', 'GGG']


class TestFromSeqsCustomName:
    """Test FromSeqs operation and pool name parameters."""
    
    def test_default_operation_name(self):
        """Test default operation name is op[{id}]:from_seqs."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'])
            # Default name is op[{id}]:from_seqs
            assert pool.operation.name.startswith('op[')
            assert ':from_seqs' in pool.operation.name
    
    def test_custom_operation_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], op_name='my_sequences')
            assert pool.operation.name == 'my_sequences'
    
    def test_custom_pool_name(self):
        """Test custom pool name."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], pool_name='my_pool')
            assert pool.name == 'my_pool'
    
    def test_custom_name_in_design_card(self):
        """Test custom op name appears in design card columns."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], op_name='my_seqs').named('mypool')
        
        df = pool.generate_seqs(num_seqs=1)
        assert 'mypool.op.key.seq_name' in df.columns
        assert 'mypool.op.key.seq_index' in df.columns
