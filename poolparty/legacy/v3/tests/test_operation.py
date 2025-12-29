"""Tests for the Operation base class."""

import pytest
import numpy as np
import poolparty as pp
from poolparty import reset_op_id_counter
from poolparty.operation import Operation
from poolparty.pool import Pool


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset operation ID counter before each test."""
    reset_op_id_counter()
    yield
    reset_op_id_counter()


class TestOperationIdCounter:
    """Test Operation ID counter behavior."""
    
    def test_ids_start_at_zero(self):
        """Test that operation IDs start at 0."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            assert pool.operation.id == 0
    
    def test_ids_increment(self):
        """Test that operation IDs increment."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(['AAA'])
            pool2 = pp.from_seqs(['TTT'])
            assert pool1.operation.id == 0
            assert pool2.operation.id == 1
    
    def test_reset_op_id_counter(self):
        """Test that reset_op_id_counter works."""
        with pp.Party() as party:
            pp.from_seqs(['AAA'])
            pp.from_seqs(['TTT'])
        
        reset_op_id_counter()
        
        with pp.Party() as party:
            pool = pp.from_seqs(['GGG'])
            assert pool.operation.id == 0
    
    def test_ids_unique_across_operations(self):
        """Test that IDs are unique across different operation types."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])  # id=0
            mutants = pp.mutation_scan(seq, k=1)  # id=1
            barcode = pp.get_kmers(length=4)  # id=2
            combined = mutants + barcode  # id=3 (literal) + id=4 (concat)
            
            assert seq.operation.id == 0
            assert mutants.operation.id == 1
            assert barcode.operation.id == 2


class TestOperationAttributes:
    """Test Operation attribute access."""
    
    def test_parent_pools_attribute(self):
        """Test parent_pools attribute."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutation_scan(seq, k=1)
            
            assert len(seq.operation.parent_pools) == 0
            assert len(mutants.operation.parent_pools) == 1
    
    def test_num_states_attribute(self):
        """Test num_states attribute."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])
            assert pool.operation.num_states == 3
    
    def test_mode_attribute(self):
        """Test mode attribute."""
        with pp.Party() as party:
            seq_pool = pp.from_seqs(['AAA'], mode='sequential')
            random_pool = pp.get_kmers(length=4, mode='random')
            
            assert seq_pool.operation.mode == 'sequential'
            assert random_pool.operation.mode == 'random'
    
    def test_name_attribute(self):
        """Test name attribute."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'], name='my_seqs')
            assert pool.operation.name == 'my_seqs'
    
    def test_default_name(self):
        """Test default name is class name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            assert pool.operation.name == 'from_seqs'


class TestOperationModeValidation:
    """Test Operation mode validation."""
    
    def test_valid_modes(self):
        """Test that valid modes are accepted."""
        with pp.Party() as party:
            pp.from_seqs(['AAA'], mode='sequential')
            pp.get_kmers(length=4, mode='random')
            concat = pp.from_seqs(['A']) + pp.from_seqs(['B'])
            assert concat.operation.mode == 'fixed'
    
    def test_invalid_mode_error(self):
        """Test that invalid mode raises error."""
        # Beartype validates the Literal type before our code runs
        with pytest.raises(Exception):  # BeartypeCallHintParamViolation
            # Create a direct Operation with invalid mode
            op = Operation(
                parent_pools=[],
                num_states=1,
                mode='invalid',  # type: ignore
            )


class TestValidateNumStates:
    """Test Operation.validate_num_states class method."""
    
    def test_valid_num_states(self):
        """Test valid num_states passes through."""
        result = Operation.validate_num_states(100, 'sequential')
        assert result == 100
    
    def test_num_states_one(self):
        """Test num_states=1 is valid."""
        result = Operation.validate_num_states(1, 'sequential')
        assert result == 1
    
    def test_num_states_minus_one(self):
        """Test num_states=-1 is valid (for infinite)."""
        result = Operation.validate_num_states(-1, 'random')
        assert result == -1
    
    def test_invalid_num_states_zero(self):
        """Test num_states=0 raises error."""
        with pytest.raises(ValueError, match="num_states must be >= 1 or -1"):
            Operation.validate_num_states(0, 'sequential')
    
    def test_invalid_num_states_negative(self):
        """Test num_states < -1 raises error."""
        with pytest.raises(ValueError, match="num_states must be >= 1 or -1"):
            Operation.validate_num_states(-2, 'sequential')
    
    def test_exceeds_max_sequential_error(self):
        """Test exceeding max in sequential mode raises error."""
        huge_num = Operation.max_num_sequential_states + 1
        with pytest.raises(ValueError, match="exceeds max_num_sequential_states"):
            Operation.validate_num_states(huge_num, 'sequential')
    
    def test_exceeds_max_random_returns_minus_one(self):
        """Test exceeding max in random mode returns -1."""
        huge_num = Operation.max_num_sequential_states + 1
        result = Operation.validate_num_states(huge_num, 'random')
        assert result == -1
    
    def test_at_max_is_valid(self):
        """Test exactly max_num_sequential_states is valid."""
        result = Operation.validate_num_states(Operation.max_num_sequential_states, 'sequential')
        assert result == Operation.max_num_sequential_states


class TestOperationCompute:
    """Test Operation compute method."""
    
    def test_base_compute_raises(self):
        """Test that base Operation.compute raises NotImplementedError."""
        op = Operation(
            parent_pools=[],
            num_states=1,
            mode='fixed',
        )
        
        with pytest.raises(NotImplementedError, match="Subclasses must implement"):
            op.compute([], 0, None)
    
    def test_subclass_compute_works(self):
        """Test that subclass compute works."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'])
            
            result = pool.operation.compute([], 0, None)
            assert result['seq_0'] == 'AAA'
            
            result = pool.operation.compute([], 1, None)
            assert result['seq_0'] == 'TTT'


class TestOperationRepr:
    """Test Operation __repr__ method."""
    
    def test_repr_format(self):
        """Test repr format."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'], name='test_op')
            repr_str = repr(pool.operation)
            
            assert 'FromSeqsOp' in repr_str
            assert 'id=0' in repr_str
            assert "mode='sequential'" in repr_str
            assert "name='test_op'" in repr_str
    
    def test_repr_different_modes(self):
        """Test repr shows different modes correctly."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'], mode='sequential')
            random = pp.get_kmers(length=4, mode='random')
            concat = seq + '.'
            
            assert "'sequential'" in repr(seq.operation)
            assert "'random'" in repr(random.operation)
            assert "'fixed'" in repr(concat.operation)


class TestOperationRng:
    """Test Operation RNG handling."""
    
    def test_rng_none_by_default(self):
        """Test that RNG is None by default."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4, mode='random')
            # Before generate, rng should be None
            assert pool.operation.rng is None
    
    def test_rng_set_by_party(self):
        """Test that Party sets RNG for random operations."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4, mode='random')
            party.output(pool, name='kmer')
        
        # After generate, rng should be set
        party.generate(num_seqs=1, seed=42)
        assert pool.operation.rng is not None
        assert isinstance(pool.operation.rng, np.random.Generator)
    
    def test_random_compute_requires_rng(self):
        """Test that random mode compute requires RNG."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4, mode='random')
        
        with pytest.raises(RuntimeError, match="Random mode requires RNG"):
            pool.operation.compute([], 0, None)


class TestOperationDesignCards:
    """Test Operation design_card_keys."""
    
    def test_from_seqs_design_cards(self):
        """Test FromSeqsOp design card keys."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            assert 'seq_name' in pool.operation.design_card_keys
            assert 'seq_index' in pool.operation.design_card_keys
    
    def test_get_kmers_design_cards(self):
        """Test GetKmersOp design card keys."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4)
            assert 'kmer_index' in pool.operation.design_card_keys
    
    def test_concatenate_no_design_cards(self):
        """Test ConcatenateOp has no design card keys."""
        with pp.Party() as party:
            concat = pp.from_seqs(['A']) + pp.from_seqs(['B'])
            assert len(concat.operation.design_card_keys) == 0


class TestOperationNumOutputs:
    """Test Operation num_outputs attribute."""
    
    def test_single_output_default(self):
        """Test most operations have num_outputs=1."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            assert pool.operation.num_outputs == 1
    
    def test_breakpoint_multi_output(self):
        """Test BreakpointScanOp has multiple outputs."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGT', num_breakpoints=1)
            assert left.operation.num_outputs == 2
    
    def test_breakpoint_three_outputs(self):
        """Test BreakpointScanOp with 2 breakpoints has 3 outputs."""
        with pp.Party() as party:
            left, mid, right = pp.breakpoint_scan('ACGTACGT', num_breakpoints=2)
            assert left.operation.num_outputs == 3

