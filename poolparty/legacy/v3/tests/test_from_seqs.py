"""Tests for the FromSeqs operation."""

import pytest
import numpy as np
import poolparty as pp
from poolparty import reset_op_id_counter
from poolparty.operations.from_seqs import FromSeqsOp, from_seqs


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


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
            pool = from_seqs(['AAA', 'TTT'], names=['seq_a', 'seq_b'])
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=2)
        assert list(df['from_seqs.seq_name']) == ['seq_a', 'seq_b']


class TestFromSeqsSequentialMode:
    """Test FromSeqs in sequential mode."""
    
    def test_sequential_iteration(self):
        """Test sequential iteration through sequences."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B', 'C', 'D'], mode='sequential')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=4)
        assert list(df['seq']) == ['A', 'B', 'C', 'D']
    
    def test_sequential_cycling(self):
        """Test that sequential mode cycles when exhausted."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B'], mode='sequential')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=5)
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
            pool = from_seqs(['A', 'B', 'C', 'D'], mode='random')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=100, seed=42)
        assert len(df) == 100
        # All should be from the input sequences
        assert all(s in 'ABCD' for s in df['seq'])
    
    def test_random_uniform_distribution(self):
        """Test that random sampling is roughly uniform."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B'], mode='random')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=1000, seed=42)
        counts = df['seq'].value_counts()
        # Should be roughly 50/50 (within 10%)
        assert 400 < counts['A'] < 600
        assert 400 < counts['B'] < 600
    
    def test_random_requires_rng(self):
        """Test that random mode requires RNG."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B'], mode='random')
        
        with pytest.raises(RuntimeError, match="Random mode requires RNG"):
            pool.operation.compute([], 0, None)


class TestFromSeqsWithProbabilities:
    """Test FromSeqs with custom probabilities."""
    
    def test_weighted_sampling(self):
        """Test sampling with custom probabilities."""
        with pp.Party() as party:
            # A should appear ~90%, B ~10%
            pool = from_seqs(['A', 'B'], probs=[0.9, 0.1], mode='random')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=1000, seed=42)
        counts = df['seq'].value_counts()
        assert counts['A'] > 800
        assert counts['B'] < 200
    
    def test_unnormalized_probs(self):
        """Test that probabilities are normalized."""
        with pp.Party() as party:
            # Should normalize to 50/50
            pool = from_seqs(['A', 'B'], probs=[10.0, 10.0], mode='random')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=1000, seed=42)
        counts = df['seq'].value_counts()
        assert 400 < counts['A'] < 600
    
    def test_single_nonzero_prob(self):
        """Test that single nonzero prob always returns that seq."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B', 'C'], probs=[0.0, 1.0, 0.0], mode='random')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=10, seed=42)
        assert all(s == 'B' for s in df['seq'])


class TestFromSeqsNames:
    """Test FromSeqs name handling."""
    
    def test_default_names(self):
        """Test default names are seq_0, seq_1, etc."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG'])
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=3)
        assert list(df['from_seqs.seq_name']) == ['seq_0', 'seq_1', 'seq_2']
    
    def test_custom_names(self):
        """Test custom names."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT'], names=['first', 'second'])
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=2)
        assert list(df['from_seqs.seq_name']) == ['first', 'second']


class TestFromSeqsDesignCards:
    """Test FromSeqs design card output."""
    
    def test_seq_name_in_output(self):
        """Test seq_name is in output."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], names=['test'])
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=1)
        assert 'from_seqs.seq_name' in df.columns
        assert df['from_seqs.seq_name'].iloc[0] == 'test'
    
    def test_seq_index_in_output(self):
        """Test seq_index is in output."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B', 'C'])
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=3)
        assert 'from_seqs.seq_index' in df.columns
        assert list(df['from_seqs.seq_index']) == [0, 1, 2]
    
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
        with pytest.raises(ValueError, match="must not be empty"):
            from_seqs([])
    
    def test_mismatched_names_length(self):
        """Test error for mismatched names length."""
        with pytest.raises(ValueError, match="same length"):
            from_seqs(['A', 'B', 'C'], names=['x', 'y'])
    
    def test_negative_probs_error(self):
        """Test error for negative probabilities."""
        with pytest.raises(ValueError, match="non-negative"):
            from_seqs(['A', 'B'], probs=[-0.5, 0.5])
    
    def test_mismatched_probs_length(self):
        """Test error for mismatched probs length."""
        with pytest.raises(ValueError, match="same length"):
            from_seqs(['A', 'B', 'C'], probs=[0.5, 0.5])


class TestFromSeqsCompute:
    """Test FromSeqs compute method directly."""
    
    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='sequential')
        
        result = pool.operation.compute([], 0, None)
        assert result['seq_0'] == 'AAA'
        assert result['seq_index'] == 0
        
        result = pool.operation.compute([], 1, None)
        assert result['seq_0'] == 'TTT'
        assert result['seq_index'] == 1
    
    def test_compute_random(self):
        """Test compute in random mode."""
        with pp.Party() as party:
            pool = from_seqs(['AAA', 'TTT', 'GGG'], mode='random')
        
        rng = np.random.default_rng(42)
        result = pool.operation.compute([], 0, rng)
        assert result['seq_0'] in ['AAA', 'TTT', 'GGG']
    
    def test_compute_with_probs(self):
        """Test compute with probabilities."""
        with pp.Party() as party:
            pool = from_seqs(['A', 'B'], probs=[1.0, 0.0], mode='random')
        
        rng = np.random.default_rng(42)
        # Should always return 'A' since B has 0 probability
        for _ in range(10):
            result = pool.operation.compute([], 0, rng)
            assert result['seq_0'] == 'A'


class TestFromSeqsCustomName:
    """Test FromSeqs operation name parameter."""
    
    def test_default_operation_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'])
            assert pool.operation.name == 'from_seqs'
    
    def test_custom_operation_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], name='my_sequences')
            assert pool.operation.name == 'my_sequences'
    
    def test_custom_name_in_design_card(self):
        """Test custom name appears in design card columns."""
        with pp.Party() as party:
            pool = from_seqs(['AAA'], name='my_seqs')
            party.output(pool, name='seq')
        
        df = party.generate(num_seqs=1)
        assert 'my_seqs.seq_name' in df.columns
        assert 'my_seqs.seq_index' in df.columns

