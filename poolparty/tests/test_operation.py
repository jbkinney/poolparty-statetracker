"""Tests for the Operation base class."""

import numpy as np
import pytest

import poolparty as pp
from poolparty import join
from poolparty.operation import Operation


class TestOperationIdState:
    """Test Operation ID counter behavior."""

    def test_ids_start_at_zero(self):
        """Test that operation IDs start at 0."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            assert pool.operation.id == 0

    def test_ids_increment(self):
        """Test that operation IDs increment."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(["AAA"])
            pool2 = pp.from_seqs(["TTT"])
            assert pool1.operation.id == 0
            assert pool2.operation.id == 1

    def test_op_ids_reset_in_new_party(self):
        """Test that operation IDs reset in each new Party context."""
        with pp.Party() as party:
            pp.from_seqs(["AAA"])
            pp.from_seqs(["TTT"])

        # In a new Party context, IDs should reset to 0
        with pp.Party() as party:
            pool = pp.from_seqs(["GGG"])
            assert pool.operation.id == 0

    def test_ids_unique_across_operations(self):
        """Test that IDs are unique across different operation types."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])  # id=0
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
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)

            assert len(seq.operation.parent_pools) == 0
            assert len(mutants.operation.parent_pools) == 1

    def test_num_states_attribute(self):
        """Test num_states attribute."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            assert pool.operation.num_states == 3

    def test_mode_attribute(self):
        """Test mode attribute."""
        with pp.Party() as party:
            seq_pool = pp.from_seqs(["AAA"], mode="sequential")
            random_pool = pp.get_kmers(length=4, mode="random")

            assert seq_pool.operation.mode == "sequential"
            assert random_pool.operation.mode == "random"

    def test_name_attribute(self):
        """Test name attribute."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"]).named("my_seqs")
            assert pool.name == "my_seqs"

    def test_default_name(self):
        """Test default name includes factory name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            assert pool.operation.name == "op[0]:from_seqs"


class TestOperationModeValidation:
    """Test Operation mode validation."""

    def test_valid_modes(self):
        """Test that valid modes are accepted."""
        with pp.Party() as party:
            pp.from_seqs(["AAA"], mode="sequential")
            pp.get_kmers(length=4, mode="random")
            combined = join([pp.from_seqs(["A"]), pp.from_seqs(["B"])])
            assert combined.operation.mode == "fixed"

    def test_invalid_mode_error(self):
        """Test that invalid mode is accepted (no runtime validation after beartype removal)."""
        # Note: Operation class no longer has beartype for performance reasons.
        # Invalid modes are only validated at factory function level (which still have beartype).
        # Direct Operation instantiation with invalid mode will not raise an error.
        with pp.Party():
            op = Operation(
                parent_pools=[],
                num_states=1,
                mode="invalid",  # type: ignore
            )
            # Should not raise - validation happens at factory function level, not class level
            assert op.mode == "invalid"


class TestValidateNumStates:
    """Test Operation.validate_num_states class method."""

    def test_valid_num_states(self):
        """Test valid num_states passes through."""
        result = Operation.validate_num_states(100, "sequential")
        assert result == 100

    def test_num_states_one(self):
        """Test num_values=1 is valid."""
        result = Operation.validate_num_states(1, "sequential")
        assert result == 1

    def test_num_states_inf(self):
        """Test num_values=np.inf is valid (for infinite)."""
        result = Operation.validate_num_states(np.inf, "random")
        assert result == np.inf

    def test_invalid_num_states_zero(self):
        """Test num_states=0 raises error."""
        with pytest.raises(ValueError, match="num_states must be >= 1"):
            Operation.validate_num_states(0, "sequential")

    def test_invalid_num_states_negative(self):
        """Test num_states=-1 raises error."""
        with pytest.raises(ValueError, match="num_states must be >= 1"):
            Operation.validate_num_states(-1, "sequential")

    def test_exceeds_max_sequential_error(self):
        """Test exceeding max in sequential mode raises error."""
        huge_num = Operation.max_num_sequential_states + 1
        with pytest.raises(ValueError, match="exceeds max_num_sequential_states"):
            Operation.validate_num_states(huge_num, "sequential")

    def test_exceeds_max_random_returns_inf(self):
        """Test exceeding max in random mode returns np.inf."""
        huge_num = Operation.max_num_sequential_states + 1
        result = Operation.validate_num_states(huge_num, "random")
        assert result == np.inf

    def test_at_max_is_valid(self):
        """Test exactly max_num_sequential_states is valid."""
        result = Operation.validate_num_states(Operation.max_num_sequential_states, "sequential")
        assert result == Operation.max_num_sequential_states


class TestOperationCompute:
    """Test Operation compute method."""

    def test_base_compute_raises(self):
        """Test that base Operation.compute raises NotImplementedError."""
        with pp.Party() as party:
            op = Operation(
                parent_pools=[],
                num_states=1,
                mode="fixed",
            )

            with pytest.raises(NotImplementedError, match="Subclasses must implement"):
                op.compute([])

    def test_subclass_compute_works(self):
        """Test that subclass compute works."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"], mode="sequential")

        # Set counter state and compute
        pool.operation.state._value = 0
        output_seq, card = pool.operation.compute([])
        assert output_seq.string == "AAA"

        pool.operation.state._value = 1
        output_seq, card = pool.operation.compute([])
        assert output_seq.string == "TTT"


class TestOperationRepr:
    """Test Operation __repr__ method."""

    def test_repr_format(self):
        """Test repr format."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"], mode="sequential").named("test_op")
            repr_str = repr(pool.operation)

            assert "FromSeqsOp" in repr_str
            assert "id=0" in repr_str
            assert "mode='sequential'" in repr_str
            # Operation name is auto-generated when op_name is removed
            assert "name=" in repr_str

    def test_repr_different_modes(self):
        """Test repr shows different modes correctly."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"], mode="sequential")
            random = pp.get_kmers(length=4, mode="random")
            combined = join([seq, "."])

            assert "'sequential'" in repr(seq.operation)
            assert "'random'" in repr(random.operation)
            assert "'fixed'" in repr(combined.operation)


class TestOperationRng:
    """Test Operation RNG handling."""

    def test_rng_none_by_default(self):
        """Test that RNG is None by default."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4, mode="random")
            # Before generate, rng should be None
            assert pool.operation.rng is None

    def test_rng_is_none_after_generate(self):
        """Test that op.rng remains None after generate (RNG created per-call)."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4, mode="random").named("kmer")

        # After generate, rng should still be None (RNG is created per-call in _compute_one)
        pool.generate_library(num_seqs=1, seed=42)
        assert pool.operation.rng is None

    def test_sequential_rng_is_none(self):
        """Test that sequential mode RNG remains None after generate."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("seq")

        pool.generate_library(num_seqs=3, seed=42)
        assert pool.operation.rng is None


class TestOperationDesignCards:
    """Test Operation design_card_keys."""

    def test_from_seqs_design_cards(self):
        """Test FromSeqsOp design card keys."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            assert "seq_name" in pool.operation.design_card_keys
            assert "seq_index" in pool.operation.design_card_keys

    def test_get_kmers_design_cards(self):
        """Test GetKmersOp design card keys."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=4)
            assert "kmer_index" in pool.operation.design_card_keys

    def test_join_no_design_cards(self):
        """Test JoinOp has no design card keys."""
        with pp.Party() as party:
            combined = join([pp.from_seqs(["A"]), pp.from_seqs(["B"])])
            assert len(combined.operation.design_card_keys) == 0


class TestOperationCopy:
    """Test Operation.copy() method."""

    def test_copy_creates_new_operation(self):
        """Test that copy() creates a new operation instance."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"])
            original_op = pool.operation
            copied_op = original_op.copy()

            assert copied_op is not original_op
            assert type(copied_op) is type(original_op)

    def test_copy_gets_new_id(self):
        """Test that copied operation gets a new ID."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"])
            original_op = pool.operation
            copied_op = original_op.copy()

            assert copied_op.id != original_op.id

    def test_copy_preserves_parameters(self):
        """Test that copy() preserves operation parameters."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT", "CCC"])
            original_op = pool.operation
            copied_op = original_op.copy()

            assert copied_op.num_states == original_op.num_states
            assert copied_op.mode == original_op.mode

    def test_copy_with_custom_name(self):
        """Test that copy() accepts custom name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            copied_op = pool.operation.copy(name="my_copied_op")

            assert copied_op.name == "my_copied_op"

    def test_copy_gets_new_counter(self):
        """Test that copied operation has its own counter."""
        with pp.Party() as party:
            # Use sequential mode to ensure operations have state
            pool = pp.from_seqs(["AAA", "TTT"], mode="sequential")
            original_op = pool.operation
            copied_op = original_op.copy()

            assert copied_op.state is not original_op.state
            assert copied_op.state.num_values == original_op.state.num_values

    def test_copy_from_seqs_op(self):
        """Test copying FromSeqsOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            copied_op = pool.operation.copy()

            # Verify copied op produces same results
            copied_op.state._value = 0
            output_seq, card = copied_op.compute([])
            assert output_seq.string == "A"

    def test_copy_mutagenize_op(self):
        """Test copying MutagenizeOp."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.copy()

            assert copied_op.parent_pools == mutants.operation.parent_pools
            assert copied_op.num_states == mutants.operation.num_states

    def test_copy_get_kmers_op(self):
        """Test copying GetKmersOp."""
        with pp.Party() as party:
            kmers = pp.get_kmers(length=3, mode="sequential")
            copied_op = kmers.operation.copy()

            assert copied_op.num_states == kmers.operation.num_states

    def test_copy_join_op(self):
        """Test copying JoinOp."""
        with pp.Party() as party:
            a = pp.from_seqs(["AAA"])
            b = pp.from_seqs(["TTT"])
            combined = join([a, b])
            copied_op = combined.operation.copy()

            assert copied_op.parent_pools == combined.operation.parent_pools

    def test_copy_stack_op(self):
        """Test copying StackOp."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state (required for stacking)
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            stacked = a + b
            copied_op = stacked.operation.copy()

            assert copied_op.parent_pools == stacked.operation.parent_pools

    def test_copy_repeat_op(self):
        """Test copying RepeatOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"])
            repeated = pool * 3
            copied_op = repeated.operation.copy()

            assert copied_op.parent_pools == repeated.operation.parent_pools

    def test_copy_state_slice_op(self):
        """Test copying StateSliceOp."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state (required for slicing)
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sliced = pool[1:3]
            copied_op = sliced.operation.copy()

            assert copied_op.parent_pools == sliced.operation.parent_pools

    def test_copy_slice_seq_op(self):
        """Test copying SliceSeqOp."""
        with pp.Party() as party:
            from poolparty.fixed_ops.slice_seq import slice_seq

            pool = pp.from_seqs(["ACGT"])
            sliced = slice_seq(pool, start=1, stop=3)
            copied_op = sliced.operation.copy()

            assert copied_op.parent_pools == sliced.operation.parent_pools

    def test_base_operation_get_copy_params_works(self):
        """Test that base Operation now has a working _get_copy_params() implementation."""
        with pp.Party() as party:
            op = Operation(
                parent_pools=[],
                num_states=1,
                mode="fixed",
            )
            # Should no longer raise - now returns auto-generated params
            params = op._get_copy_params()
            assert isinstance(params, dict)
            assert params["name"] is None

    def test_copy_default_name_uses_suffix(self):
        """Test that copy() uses self.name + '.copy' as default name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"]).named("my_op")
            copied_op = pool.operation.copy()

            # Operation name is auto-generated when op_name is removed
        assert copied_op.name.endswith(".copy")


class TestOperationDeepCopy:
    """Test Operation.deepcopy() method."""

    def test_deepcopy_creates_new_operation(self):
        """Test that deepcopy() creates a new operation instance."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.deepcopy()

            assert copied_op is not mutants.operation
            assert type(copied_op) is type(mutants.operation)

    def test_deepcopy_gets_new_id(self):
        """Test that deepcopied operation gets a new ID."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.deepcopy()

            assert copied_op.id != mutants.operation.id

    def test_deepcopy_creates_new_parent_pools(self):
        """Test that deepcopy() creates new parent pools (not same references)."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.deepcopy()

            # The parent pools should be different objects
            assert copied_op.parent_pools[0] is not mutants.operation.parent_pools[0]

    def test_deepcopy_with_custom_name(self):
        """Test that deepcopy() accepts custom name."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied_op = mutants.operation.deepcopy(name="my_deepcopy")

            assert copied_op.name == "my_deepcopy"

    def test_deepcopy_preserves_parameters(self):
        """Test that deepcopy() preserves operation parameters."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=2, mode="sequential")
            copied_op = mutants.operation.deepcopy()

            assert copied_op.num_states == mutants.operation.num_states
            assert copied_op.mode == mutants.operation.mode

    def test_deepcopy_recursive_chain(self):
        """Test deepcopy on a chain of operations."""
        with pp.Party() as party:
            a = pp.from_seqs(["ACGT"])
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
            pool = pp.from_seqs(["A", "B", "C"])
            copied_op = pool.operation.deepcopy()

            assert copied_op is not pool.operation
            assert len(copied_op.parent_pools) == 0

    def test_deepcopy_multiple_parents(self):
        """Test deepcopy on operation with multiple parents."""
        with pp.Party() as party:
            a = pp.from_seqs(["AAA"])
            b = pp.from_seqs(["TTT"])
            combined = join([a, b])
            copied_op = combined.operation.deepcopy()

            # Both parents should be new copies
            assert copied_op.parent_pools[0] is not a
            assert copied_op.parent_pools[1] is not b
            assert len(copied_op.parent_pools) == 2

    def test_deepcopy_stack_op(self):
        """Test deepcopy on StackOp."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state (required for stacking)
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            stacked = a + b
            copied_op = stacked.operation.deepcopy()

            # Parent pools should be new copies
            assert copied_op.parent_pools[0] is not a
            assert copied_op.parent_pools[1] is not b
