"""Tests for the Operation base class."""

import pytest
import numpy as np
import poolparty as pp
from poolparty import join
from poolparty.operation import Operation
from poolparty.pool import Pool


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
    
    def test_op_ids_reset_in_new_party(self):
        """Test that operation IDs reset in each new Party context."""
        with pp.Party() as party:
            pp.from_seqs(['AAA'])
            pp.from_seqs(['TTT'])
        
        # In a new Party context, IDs should reset to 0
        with pp.Party() as party:
            pool = pp.from_seqs(['GGG'])
            assert pool.operation.id == 0
    
    def test_ids_unique_across_operations(self):
        """Test that IDs are unique across different operation types."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])  # id=0
            mutants = pp.mutagenize(seq, num_mutations=1)  # id=1
            barcode = pp.get_kmers(length=4)  # id=2
            
            assert seq.operation.id == 0
            assert mutants.operation.id == 1
            assert barcode.operation.id == 2


class TestOperationAttributes:
    """Test Operation attribute access."""
    
    def test_parent_pools_attribute(self):
        """Test parent_pools attribute."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutagenize(seq, num_mutations=1)
            
            assert len(seq.operation.parent_pools) == 0
            assert len(mutants.operation.parent_pools) == 1
    
    def test_num_states_attribute(self):
        """Test num_states attribute."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
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
            pool = pp.from_seqs(['AAA'], op_name='my_seqs')
            assert pool.operation.name == 'my_seqs'
    
    def test_default_name(self):
        """Test default name includes factory name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            assert pool.operation.name == 'op[0]:from_seqs'


class TestOperationModeValidation:
    """Test Operation mode validation."""
    
    def test_valid_modes(self):
        """Test that valid modes are accepted."""
        with pp.Party() as party:
            pp.from_seqs(['AAA'], mode='sequential')
            pp.get_kmers(length=4, mode='random')
            combined = join([pp.from_seqs(['A']), pp.from_seqs(['B'])])
            assert combined.operation.mode == 'fixed'
    
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
    
    def test_num_states_inf(self):
        """Test num_states=np.inf is valid (for infinite)."""
        import numpy as np
        result = Operation.validate_num_states(np.inf, 'random')
        assert result == np.inf
    
    def test_invalid_num_states_zero(self):
        """Test num_states=0 raises error."""
        with pytest.raises(ValueError, match="num_states must be >= 1 or np.inf"):
            Operation.validate_num_states(0, 'sequential')
    
    def test_invalid_num_states_negative(self):
        """Test num_states=-1 raises error."""
        with pytest.raises(ValueError, match="num_states must be >= 1 or np.inf"):
            Operation.validate_num_states(-1, 'sequential')
    
    def test_exceeds_max_sequential_error(self):
        """Test exceeding max in sequential mode raises error."""
        huge_num = Operation.max_num_sequential_states + 1
        with pytest.raises(ValueError, match="exceeds max_num_sequential_states"):
            Operation.validate_num_states(huge_num, 'sequential')
    
    def test_exceeds_max_random_returns_inf(self):
        """Test exceeding max in random mode returns np.inf."""
        import numpy as np
        huge_num = Operation.max_num_sequential_states + 1
        result = Operation.validate_num_states(huge_num, 'random')
        assert result == np.inf
    
    def test_at_max_is_valid(self):
        """Test exactly max_num_sequential_states is valid."""
        result = Operation.validate_num_states(Operation.max_num_sequential_states, 'sequential')
        assert result == Operation.max_num_sequential_states


class TestOperationCompute:
    """Test Operation compute_design_card and compute_seq_from_card methods."""
    
    def test_base_compute_design_card_raises(self):
        """Test that base Operation.compute_design_card raises NotImplementedError."""
        with pp.Party() as party:
            op = Operation(
                parent_pools=[],
                num_states=1,
                mode='fixed',
            )
            
            with pytest.raises(NotImplementedError, match="Subclasses must implement"):
                op.compute_design_card([])
    
    def test_base_compute_seq_from_card_raises(self):
        """Test that base Operation.compute_seq_from_card raises NotImplementedError."""
        with pp.Party() as party:
            op = Operation(
                parent_pools=[],
                num_states=1,
                mode='fixed',
            )
            
            with pytest.raises(NotImplementedError, match="Subclasses must implement"):
                op.compute_seq_from_card([], {})
    
    def test_subclass_compute_works(self):
        """Test that subclass compute_design_card and compute_seq_from_card work."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'], mode='sequential')
        
        # Set counter state and compute
        pool.operation.counter._state = 0
        card = pool.operation.compute_design_card([])
        result = pool.operation.compute_seq_from_card([], card)
        assert result['seq_0'] == 'AAA'
        
        pool.operation.counter._state = 1
        card = pool.operation.compute_design_card([])
        result = pool.operation.compute_seq_from_card([], card)
        assert result['seq_0'] == 'TTT'


class TestOperationRepr:
    """Test Operation __repr__ method."""
    
    def test_repr_format(self):
        """Test repr format."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'], op_name='test_op', mode='sequential')
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
            combined = join([seq, '.'])
            
            assert "'sequential'" in repr(seq.operation)
            assert "'random'" in repr(random.operation)
            assert "'fixed'" in repr(combined.operation)


class TestOperationRng:
    """Test Operation RNG handling."""
    
    def test_rng_none_by_default(self):
        """Test that RNG is None by default."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4, mode='random')
            # Before generate, rng should be None
            assert pool.operation.rng is None
    
    def test_rng_set_by_generate(self):
        """Test that generate_library sets RNG for random operations."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4, mode='random').named('kmer')
        
        # After generate, rng should be set
        pool.generate_library(num_seqs=1, seed=42)
        assert pool.operation.rng is not None
        assert isinstance(pool.operation.rng, np.random.Generator)
    
    def test_sequential_rng_is_none(self):
        """Test that sequential mode RNG remains None after generate."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential').named('seq')
        
        pool.generate_library(num_seqs=3, seed=42)
        assert pool.operation.rng is None


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
    
    def test_join_no_design_cards(self):
        """Test JoinOp has no design card keys."""
        with pp.Party() as party:
            combined = join([pp.from_seqs(['A']), pp.from_seqs(['B'])])
            assert len(combined.operation.design_card_keys) == 0


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


class TestOperationCopy:
    """Test Operation.copy() method."""
    
    def test_copy_creates_new_operation(self):
        """Test that copy() creates a new operation instance."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'])
            original_op = pool.operation
            copied_op = original_op.copy()
            
            assert copied_op is not original_op
            assert type(copied_op) is type(original_op)
    
    def test_copy_gets_new_id(self):
        """Test that copied operation gets a new ID."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'])
            original_op = pool.operation
            copied_op = original_op.copy()
            
            assert copied_op.id != original_op.id
    
    def test_copy_preserves_parameters(self):
        """Test that copy() preserves operation parameters."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT', 'CCC'])
            original_op = pool.operation
            copied_op = original_op.copy()
            
            assert copied_op.num_states == original_op.num_states
            assert copied_op.mode == original_op.mode
    
    def test_copy_with_custom_name(self):
        """Test that copy() accepts custom name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'])
            copied_op = pool.operation.copy(name='my_copied_op')
            
            assert copied_op.name == 'my_copied_op'
    
    def test_copy_gets_new_counter(self):
        """Test that copied operation has its own counter."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA', 'TTT'])
            original_op = pool.operation
            copied_op = original_op.copy()
            
            assert copied_op.counter is not original_op.counter
            assert copied_op.counter.num_states == original_op.counter.num_states
    
    def test_copy_from_seqs_op(self):
        """Test copying FromSeqsOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'], mode='sequential')
            copied_op = pool.operation.copy()
            
            # Verify copied op produces same results
            copied_op.counter._state = 0
            card = copied_op.compute_design_card([])
            result = copied_op.compute_seq_from_card([], card)
            assert result['seq_0'] == 'A'
    
    def test_copy_mutagenize_op(self):
        """Test copying MutagenizeOp."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.copy()
            
            assert copied_op.parent_pools == mutants.operation.parent_pools
            assert copied_op.num_states == mutants.operation.num_states
    
    def test_copy_get_kmers_op(self):
        """Test copying GetKmersOp."""
        with pp.Party() as party:
            kmers = pp.get_kmers(length=3, mode='sequential')
            copied_op = kmers.operation.copy()
            
            assert copied_op.num_states == kmers.operation.num_states
    
    def test_copy_join_op(self):
        """Test copying JoinOp."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
            copied_op = combined.operation.copy()
            
            assert copied_op.parent_pools == combined.operation.parent_pools
    
    def test_copy_stack_op(self):
        """Test copying StackOp."""
        with pp.Party() as party:
            a = pp.from_seqs(['A', 'B'])
            b = pp.from_seqs(['X', 'Y'])
            stacked = a + b
            copied_op = stacked.operation.copy()
            
            assert copied_op.parent_pools == stacked.operation.parent_pools
    
    def test_copy_repeat_op(self):
        """Test copying RepeatOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B'])
            repeated = pool * 3
            copied_op = repeated.operation.copy()
            
            assert copied_op.parent_pools == repeated.operation.parent_pools
    
    def test_copy_state_slice_op(self):
        """Test copying StateSliceOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C', 'D'])
            sliced = pool[1:3]
            copied_op = sliced.operation.copy()
            
            assert copied_op.parent_pools == sliced.operation.parent_pools
    
    def test_copy_slice_seq_op(self):
        """Test copying SliceSeqOp."""
        with pp.Party() as party:
            from poolparty.fixed_ops.slice_seq import slice_seq
            pool = pp.from_seqs(['ACGT'])
            sliced = slice_seq(pool, slice(1, 3))
            copied_op = sliced.operation.copy()
            
            assert copied_op.parent_pools == sliced.operation.parent_pools
    
    def test_copy_breakpoint_scan_op(self):
        """Test copying BreakpointScanOp."""
        with pp.Party() as party:
            left, right = pp.breakpoint_scan('ACGTACGT', num_breakpoints=1)
            copied_op = left.operation.copy()
            
            assert copied_op.parent_pools == left.operation.parent_pools
            assert copied_op.num_outputs == left.operation.num_outputs
    
    def test_base_operation_get_copy_params_raises(self):
        """Test that base Operation._get_copy_params() raises NotImplementedError."""
        with pp.Party() as party:
            op = Operation(
                parent_pools=[],
                num_states=1,
                mode='fixed',
            )
            
            with pytest.raises(NotImplementedError, match="must implement _get_copy_params"):
                op._get_copy_params()
    
    def test_copy_default_name_uses_suffix(self):
        """Test that copy() uses self.name + '.copy' as default name."""
        with pp.Party() as party:
            pool = pp.from_seqs(['AAA'], op_name='my_op')
            copied_op = pool.operation.copy()
            
            assert copied_op.name == 'my_op.copy'


class TestOperationDeepCopy:
    """Test Operation.deepcopy() method."""
    
    def test_deepcopy_creates_new_operation(self):
        """Test that deepcopy() creates a new operation instance."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.deepcopy()
            
            assert copied_op is not mutants.operation
            assert type(copied_op) is type(mutants.operation)
    
    def test_deepcopy_gets_new_id(self):
        """Test that deepcopied operation gets a new ID."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.deepcopy()
            
            assert copied_op.id != mutants.operation.id
    
    def test_deepcopy_creates_new_parent_pools(self):
        """Test that deepcopy() creates new parent pools (not same references)."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.deepcopy()
            
            # The parent pools should be different objects
            assert copied_op.parent_pools[0] is not mutants.operation.parent_pools[0]
    
    def test_deepcopy_with_custom_name(self):
        """Test that deepcopy() accepts custom name."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.deepcopy(name='my_deepcopy')
            
            assert copied_op.name == 'my_deepcopy'
    
    def test_deepcopy_default_name_uses_suffix(self):
        """Test that deepcopy() uses self.name + '.copy' as default name."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutagenize(seq, num_mutations=1, op_name='my_mutants')
            copied_op = mutants.operation.deepcopy()
            
            assert copied_op.name == 'my_mutants.copy'
    
    def test_deepcopy_preserves_parameters(self):
        """Test that deepcopy() preserves operation parameters."""
        with pp.Party() as party:
            seq = pp.from_seqs(['ACGT'])
            mutants = pp.mutagenize(seq, num_mutations=2, mode='sequential')
            copied_op = mutants.operation.deepcopy()
            
            assert copied_op.num_states == mutants.operation.num_states
            assert copied_op.mode == mutants.operation.mode
    
    def test_deepcopy_recursive_chain(self):
        """Test deepcopy on a chain of operations."""
        with pp.Party() as party:
            a = pp.from_seqs(['ACGT'])
            b = pp.mutagenize(a, num_mutations=1)
            c = pp.mutagenize(b, num_mutations=1)
            
            copied_op = c.operation.deepcopy()
            
            # c's parent should be a new copy of b
            assert copied_op.parent_pools[0] is not b
            # b's parent (inside the copy) should be a new copy of a
            assert copied_op.parent_pools[0].operation.parent_pools[0] is not a
    
    def test_deepcopy_no_parents(self):
        """Test deepcopy on operation with no parents."""
        with pp.Party() as party:
            pool = pp.from_seqs(['A', 'B', 'C'])
            copied_op = pool.operation.deepcopy()
            
            assert copied_op is not pool.operation
            assert len(copied_op.parent_pools) == 0
    
    def test_deepcopy_multiple_parents(self):
        """Test deepcopy on operation with multiple parents."""
        with pp.Party() as party:
            a = pp.from_seqs(['AAA'])
            b = pp.from_seqs(['TTT'])
            combined = join([a, b])
            copied_op = combined.operation.deepcopy()
            
            # Both parents should be new copies
            assert copied_op.parent_pools[0] is not a
            assert copied_op.parent_pools[1] is not b
            assert len(copied_op.parent_pools) == 2
    
    def test_deepcopy_stack_op(self):
        """Test deepcopy on StackOp."""
        with pp.Party() as party:
            a = pp.from_seqs(['A', 'B'])
            b = pp.from_seqs(['X', 'Y'])
            stacked = a + b
            copied_op = stacked.operation.deepcopy()
            
            # Parent pools should be new copies
            assert copied_op.parent_pools[0] is not a
            assert copied_op.parent_pools[1] is not b
