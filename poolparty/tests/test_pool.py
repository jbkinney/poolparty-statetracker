"""Tests for the Pool class.

Pool operators now work on States:
- pool1 + pool2: Stack (union of states)
- pool * n: Repeat (repeat states n times)
- pool[start:stop]: State slice (select subset of states)

For sequence operations, use join(), slice_seq(), etc.
"""

import pytest

import poolparty as pp
from poolparty import join
from poolparty.pool import Pool
from poolparty.state_ops.repeat import RepeatOp
from poolparty.state_ops.stack import StackOp
from poolparty.state_ops.state_slice import StateSliceOp


class TestPoolCreation:
    """Test Pool creation and basic attributes."""

    def test_pool_has_operation(self):
        """Test that Pool has operation attribute."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            assert pool.operation is not None
            assert hasattr(pool.operation, "compute")

    def test_pool_name_attribute(self):
        """Test Pool name attribute uses default pool[{id}] format."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            # Default pool name is pool[{id}]
            assert pool.name == "pool[0]"

    def test_pool_parents_property(self):
        """Test Pool parents property returns operation's parent_pools."""
        with pp.Party() as party:
            seq = pp.from_seqs(["AAA"])
            mutants = pp.mutagenize(seq, num_mutations=1)

            # mutants pool should have seq as parent
            assert len(mutants.parents) == 1
            assert mutants.parents[0] is seq

    def test_pool_num_states(self):
        """Test Pool num_states property delegates to counter."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            assert pool.num_states == 3


class TestPoolCopy:
    """Test Pool.copy() method."""

    def test_copy_creates_new_pool(self):
        """Test that copy() creates a new pool instance."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"])
            copied = pool.copy()

            assert copied is not pool
            assert isinstance(copied, Pool)

    def test_copy_creates_new_operation(self):
        """Test that copy() creates a new operation."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"])
            copied = pool.copy()

            assert copied.operation is not pool.operation

    def test_copy_gets_new_id(self):
        """Test that copied pool gets a new ID."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"])
            copied = pool.copy()

            assert copied._id != pool._id

    def test_copy_preserves_num_states(self):
        """Test that copy() preserves num_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            copied = pool.copy()

            assert copied.num_states == pool.num_states

    def test_copy_with_custom_name(self):
        """Test that copy() accepts custom name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            copied = pool.copy(name="my_copied_pool")

            assert copied.name == "my_copied_pool"

    def test_copy_references_same_parent_pools(self):
        """Test that copied pool's operation references same parent_pools."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied = mutants.copy()

            # Both should reference the same parent pool
            assert copied.operation.parent_pools == mutants.operation.parent_pools
            assert copied.operation.parent_pools[0] is seq

    def test_copy_produces_same_sequences(self):
        """Test that copied pool produces the same sequences."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("original")
            copied = pool.copy(name="copied")

        df_original = pool.generate_library(
            num_cycles=1, seed=0, init_state=0
        )
        df_copied = copied.generate_library(
            num_cycles=1, seed=0, init_state=0
        )

        # Both should produce the same sequences
        assert list(df_original["seq"]) == list(df_copied["seq"])

    def test_copy_independent_generation(self):
        """Test that copied pools can generate independently."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential").named("original")
            copied = pool.copy(name="copied")

        # Generate from original
        df1 = pool.generate_library(num_seqs=2, init_state=0)
        # Generate from copy (should start fresh)
        df2 = copied.generate_library(num_seqs=2, init_state=0)

        assert list(df1["seq"]) == ["A", "B"]
        assert list(df2["seq"]) == ["A", "B"]

    def test_copy_mutagenize_pool(self):
        """Test copying a mutagenize pool."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"], mode="sequential")
            mutants = pp.mutagenize(seq, num_mutations=1, mode="sequential").named("mutants")
            copied = mutants.copy(name="copied")

        df_mutants = mutants.generate_library(
            num_seqs=5, seed=42, init_state=0
        )
        df_copied = copied.generate_library(
            num_seqs=5, seed=42, init_state=0
        )

        # Both should produce the same sequences
        assert list(df_mutants["seq"]) == list(df_copied["seq"])

    def test_copy_stacked_pool(self):
        """Test copying a stacked pool."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            stacked = (a + b).named("stacked")
            copied = stacked.copy(name="copied")

        df_stacked = stacked.generate_library(num_cycles=1, init_state=0)
        df_copied = copied.generate_library(num_cycles=1, init_state=0)

        assert list(df_stacked["seq"]) == list(df_copied["seq"])

    def test_copy_repeated_pool(self):
        """Test copying a repeated pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            repeated = (pool * 2).named("repeated")
            copied = repeated.copy(name="copied")

        df_repeated = repeated.generate_library(
            num_cycles=1, init_state=0
        )
        df_copied = copied.generate_library(num_cycles=1, init_state=0)

        assert list(df_repeated["seq"]) == list(df_copied["seq"])

    def test_copy_sliced_pool(self):
        """Test copying a sliced pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sliced = pool[1:3]
            sliced.name = "sliced"
            copied = sliced.copy(name="copied")

        df_sliced = sliced.generate_library(num_cycles=1, init_state=0)
        df_copied = copied.generate_library(num_cycles=1, init_state=0)

        assert list(df_sliced["seq"]) == list(df_copied["seq"])

    def test_copy_default_name_uses_suffix(self):
        """Test that copy() uses self.name + '.copy' as default name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"]).named("my_pool")
            copied = pool.copy()

            assert copied.name == "my_pool.copy"


class TestPoolDeepCopy:
    """Test Pool.deepcopy() method."""

    def test_deepcopy_creates_new_pool(self):
        """Test that deepcopy() creates a new pool instance."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied = mutants.deepcopy()

            assert copied is not mutants
            assert isinstance(copied, Pool)

    def test_deepcopy_creates_new_operation(self):
        """Test that deepcopy() creates a new operation."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied = mutants.deepcopy()

            assert copied.operation is not mutants.operation

    def test_deepcopy_creates_new_parent_pools(self):
        """Test that deepcopy() creates new parent pools."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            mutants = pp.mutagenize(seq, num_mutations=1)
            copied = mutants.deepcopy()

            # The parent pool should be a different object
            assert copied.parents[0] is not seq

    def test_deepcopy_gets_new_id(self):
        """Test that deepcopied pool gets a new ID."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"])
            copied = pool.deepcopy()

            assert copied._id != pool._id

    def test_deepcopy_with_custom_name(self):
        """Test that deepcopy() accepts custom name."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            copied = pool.deepcopy(name="my_deepcopy")

            assert copied.name == "my_deepcopy"

    def test_deepcopy_produces_same_sequences(self):
        """Test that deepcopied pool produces the same sequences."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("original")
            copied = pool.deepcopy(name="copied")

        df_original = pool.generate_library(
            num_cycles=1, seed=0, init_state=0
        )
        df_copied = copied.generate_library(
            num_cycles=1, seed=0, init_state=0
        )

        # Both should produce the same sequences
        assert list(df_original["seq"]) == list(df_copied["seq"])

    def test_deepcopy_independent_dag(self):
        """Test that deepcopy creates a fully independent DAG."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"]).named("seq")
            mutants = pp.mutagenize(seq, num_mutations=1, mode="sequential").named("mutants")
            copied = mutants.deepcopy(name="copied")

            # Verify the copied pool's parent is different from original
            original_parent = mutants.parents[0]
            copied_parent = copied.parents[0]

            assert copied_parent is not original_parent
            assert copied_parent.operation is not original_parent.operation

    def test_deepcopy_chain(self):
        """Test deepcopy on a chain of pools."""
        with pp.Party() as party:
            a = pp.from_seqs(["ACGT"]).named("a")
            b = pp.mutagenize(a, num_mutations=1, mode="sequential").named("b")
            c = pp.mutagenize(b, num_mutations=1, mode="sequential").named("c")
            copied = c.deepcopy(name="copied")

        # Verify the entire chain is copied
        assert copied.parents[0] is not b  # c's parent
        assert copied.parents[0].parents[0] is not a  # b's parent

    def test_deepcopy_stacked_pool(self):
        """Test deepcopy on a stacked pool."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential").named("a")
            b = pp.from_seqs(["X", "Y"], mode="sequential").named("b")
            stacked = (a + b).named("stacked")
            copied = stacked.deepcopy(name="copied")

        # Both parents should be new copies
        assert copied.parents[0] is not a
        assert copied.parents[1] is not b

        # But should produce same sequences
        df_stacked = stacked.generate_library(num_cycles=1, init_state=0)
        df_copied = copied.generate_library(num_cycles=1, init_state=0)
        assert list(df_stacked["seq"]) == list(df_copied["seq"])

    def test_deepcopy_mutagenize_produces_same(self):
        """Test deepcopy of mutagenize produces same sequences."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"], mode="sequential")
            mutants = pp.mutagenize(seq, num_mutations=1, mode="sequential").named("mutants")
            copied = mutants.deepcopy(name="copied")

        df_mutants = mutants.generate_library(
            num_seqs=5, seed=42, init_state=0
        )
        df_copied = copied.generate_library(
            num_seqs=5, seed=42, init_state=0
        )

        # Both should produce the same sequences
        assert list(df_mutants["seq"]) == list(df_copied["seq"])


class TestPoolRepr:
    """Test Pool __repr__ formatting."""

    def test_repr_basic(self):
        """Test basic repr output."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"]).named("test_pool")
            repr_str = repr(pool)
            assert "Pool" in repr_str
            # The repr uses the operation's name attribute
            assert "test_pool" in repr_str

    def test_repr_shows_num_states(self):
        """Test repr shows num_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            repr_str = repr(pool)
            assert "num_states=3" in repr_str


# =============================================================================
# Tests for Pool + Pool (Stack operation)
# =============================================================================


class TestPoolAddOperator:
    """Test Pool __add__ operator for stacking (union of states)."""

    def test_pool_plus_pool_creates_stack_op(self):
        """Test Pool + Pool creates StackOp."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state (required for stacking)
            a = pp.from_seqs(["AAA"], mode="sequential")
            b = pp.from_seqs(["TTT"], mode="sequential")
            stacked = a + b
            assert isinstance(stacked.operation, StackOp)

    def test_pool_plus_pool_num_states(self):
        """Test Pool + Pool num_states is sum of states."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential")  # 2 states
            b = pp.from_seqs(["X", "Y", "Z"], mode="sequential")  # 3 states
            stacked = a + b
            assert stacked.num_states == 5  # 2 + 3

    def test_pool_plus_pool_generates_union(self):
        """Test Pool + Pool generates union of sequences."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            stacked = (a + b).named("stacked")

        df = stacked.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["A", "B", "X", "Y"]

    def test_chained_add(self):
        """Test chained + operators."""
        with pp.Party() as party:
            a = pp.from_seqs(["A"], mode="sequential")
            b = pp.from_seqs(["B"], mode="sequential")
            c = pp.from_seqs(["C"], mode="sequential")
            stacked = (a + b + c).named("stacked")

        df = stacked.generate_library(num_cycles=1)
        # With statecounter ordering preserved, sequences follow input order A, B, C
        assert list(df["seq"]) == ["A", "B", "C"]

    def test_pool_plus_string_raises_error(self):
        """Test Pool + string raises error (use join instead)."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            with pytest.raises(Exception):  # beartype raises roar.BeartypeCallHintParamViolation
                _ = pool + "..."


# =============================================================================
# Tests for Pool * n (Repeat operation)
# =============================================================================


class TestPoolMulOperator:
    """Test Pool __mul__ and __rmul__ operators for repetition."""

    def test_pool_times_int_creates_repeat_op(self):
        """Test Pool * int creates RepeatOp."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"])
            repeated = pool * 3
            assert isinstance(repeated.operation, RepeatOp)

    def test_pool_times_int_num_states(self):
        """Test Pool * n num_states is original * n."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential")  # 2 states
            repeated = pool * 3
            assert repeated.num_states == 6  # 2 * 3

    def test_pool_times_int_generates_repeated(self):
        """Test Pool * n generates repeated sequences."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            repeated = (pool * 2).named("rep")

        df = repeated.generate_library(num_cycles=1)
        # With first_counter_slowest default, original pool cycles slowest
        assert list(df["seq"]) == ["A", "A", "B", "B"]

    def test_int_times_pool(self):
        """Test int * Pool repetition."""
        with pp.Party() as party:
            pool = pp.from_seqs(["X", "Y"], mode="sequential")
            repeated = (2 * pool).named("rep")

        df = repeated.generate_library(num_cycles=1)
        # With first_counter_slowest default, original pool cycles slowest
        assert list(df["seq"]) == ["X", "X", "Y", "Y"]

    def test_pool_times_one(self):
        """Test Pool * 1 returns equivalent result."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            repeated = pool * 1
            assert repeated.num_states == 3


# =============================================================================
# Tests for Pool[key] (State slice operation)
# =============================================================================


class TestPoolGetitemOperator:
    """Test Pool __getitem__ operator for state slicing."""

    def test_getitem_creates_state_slice_op(self):
        """Test Pool[key] creates StateSliceOp."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state (required for slicing)
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sliced = pool[1:3]
            assert isinstance(sliced.operation, StateSliceOp)

    def test_getitem_slice_num_states(self):
        """Test Pool[start:stop] reduces num_states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")  # 4 states
            sliced = pool[1:3]  # 2 states
            assert sliced.num_states == 2

    def test_getitem_slice_output(self):
        """Test Pool[start:stop] selects correct states."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sliced = pool[1:3].named("sl")  # States 1 and 2 -> B, C

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["B", "C"]

    def test_getitem_int(self):
        """Test Pool[int] for single state selection."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            single = pool[1].named("single")  # State 1 -> B

        df = single.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "B"

    def test_getitem_negative_index(self):
        """Test Pool[-1] for last state."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            last = pool[-1].named("last")  # Last state -> C

        df = last.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "C"

    def test_getitem_with_step(self):
        """Test Pool[::step] slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E", "F"], mode="sequential")  # 6 states
            sliced = pool[::2].named("sl")  # States 0, 2, 4 -> A, C, E

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["A", "C", "E"]


# =============================================================================
# Tests for join (sequence joining)
# =============================================================================


class TestConcatFunction:
    """Test join function for sequence joining."""

    def test_join_two_pools(self):
        """Test join([pool1, pool2])."""
        with pp.Party() as party:
            a = pp.from_seqs(["AAA"])
            b = pp.from_seqs(["TTT"])
            combined = join([a, b]).named("comb")

        df = combined.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AAATTT"

    def test_join_with_string(self):
        """Test join([pool, string])."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            combined = join([pool, "---", "TTT"]).named("comb")

        df = combined.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AAA---TTT"

    def test_join_multiple_pools(self):
        """Test join with multiple pools."""
        with pp.Party() as party:
            a = pp.from_seqs(["A"])
            b = pp.from_seqs(["B"])
            c = pp.from_seqs(["C"])
            combined = join([a, "-", b, "-", c]).named("comb")

        df = combined.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "A-B-C"


# =============================================================================
# Tests for operator chaining
# =============================================================================


class TestPoolOperatorChaining:
    """Test chaining multiple operators together."""

    def test_add_then_slice(self):
        """Test stacking then state slicing."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential")  # 2 states
            b = pp.from_seqs(["X", "Y"], mode="sequential")  # 2 states
            stacked = a + b  # 4 states: A, B, X, Y
            sliced = stacked[1:3].named("sl")  # States 1, 2 -> B, X

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["B", "X"]

    def test_multiply_then_slice(self):
        """Test repetition then state slicing."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential")  # 2 states
            # With first_counter_slowest, 6 states: A, A, A, B, B, B
            repeated = pool * 3
            sliced = repeated[2:5].named("sl")  # States 2, 3, 4 -> A, B, B

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["A", "B", "B"]

    def test_slice_then_add(self):
        """Test state slicing then stacking."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")  # 4 states
            first = pool[:2]  # A, B
            last = pool[-2:]  # C, D
            combined = (first + last).named("comb")  # A, B, C, D

        df = combined.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["A", "B", "C", "D"]


# =============================================================================
# Tests for multi-output operations
# =============================================================================

# =============================================================================
# Tests for Pool.generate_library()
# =============================================================================


class TestPoolGenerate:
    """Test Pool.generate_library() method for direct sequence generation."""

    def test_generate_basic(self):
        """Test basic generate() call with num_seqs."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("X")

        df = pool.generate_library(num_seqs=3)
        assert len(df) == 3
        assert "seq" in df.columns
        assert list(df["seq"]) == ["A", "B", "C"]

    def test_generate_num_cycles(self):
        """Test generate() with num_cycles."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential").named("X")

        df = pool.generate_library(num_cycles=2)
        assert len(df) == 4
        assert list(df["seq"]) == ["A", "B", "A", "B"]

    def test_generate_basic_with_mutagenize(self):
        """Test generate() with mutagenize operation."""
        with pp.Party() as party:
            a = pp.from_seqs(["AAA", "TTT"]).named("A")
            b = pp.mutagenize(a, num_mutations=1, mode="sequential").named("B")

        df = b.generate_library(num_cycles=1)
        # Should have name and seq columns (default output)
        assert "seq" in df.columns
        assert "name" in df.columns
        # Sequences should be mutated versions of AAA and TTT
        for seq in df["seq"]:
            assert len(seq) == 3

    def test_generate_stacked_pools(self):
        """Test generate() with stacked pools."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential").named("A")
            b = pp.from_seqs(["X", "Y"], mode="sequential").named("B")
            stacked = a + b

        df = stacked.generate_library(num_cycles=1)
        # Default output should only have name and seq columns
        assert set(df.columns) == {"name", "seq"}

    def test_generate_with_seed(self):
        """Test generate() with seed for reproducibility."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"]).named("X")

        df1 = pool.generate_library(num_seqs=3, seed=42)
        df2 = pool.generate_library(num_seqs=3, seed=42, init_state=0)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_generate_with_init_state(self):
        """Test generate() with init_state."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("X")

        df = pool.generate_library(num_seqs=2, init_state=1)
        assert list(df["seq"]) == ["B", "C"]

    def test_generate_state_continuation(self):
        """Test that generate() continues from last state."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential").named("X")

        df1 = pool.generate_library(num_seqs=2, init_state=0)
        df2 = pool.generate_library(
            num_seqs=2
        )  # Should continue from state 2

        assert list(df1["seq"]) == ["A", "B"]
        assert list(df2["seq"]) == ["C", "D"]

    def test_generate_with_cards_parameter(self):
        """Test that generate() respects cards parameter for design cards."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"], cards=["seq_index"])
            mutants = pp.mutagenize(pool, num_mutations=1, cards=["positions"])

        df = mutants.generate_library(num_seqs=5)
        # Should have design card columns only when requested via cards
        seq_index_cols = [c for c in df.columns if "seq_index" in c]
        positions_cols = [c for c in df.columns if "positions" in c]
        assert len(seq_index_cols) > 0
        assert len(positions_cols) > 0

    def test_generate_name_column_first(self):
        """Test that 'name' column appears first."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"]).named("A")
            mutants = pp.mutagenize(pool, num_mutations=1).named("B")

        df = mutants.generate_library(num_seqs=3)
        assert df.columns[0] == "name"
        assert df.columns[1] == "seq"

    def test_generate_default_minimal_columns(self):
        """Test that default output has only name and seq columns."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state (required for stacking)
            a = pp.from_seqs(["A"], mode="sequential").named("A")
            b = pp.from_seqs(["B"], mode="sequential").named("B")
            c = pp.from_seqs(["C"], mode="sequential").named("C")
            combined = a + b + c

        df = combined.generate_library(num_seqs=3)
        # Default output should only have name and seq columns
        assert set(df.columns) == {"name", "seq"}


# =============================================================================
# Tests for Pool.generate_library() record_counters
# =============================================================================


class TestPoolDefaultOutput:
    """Test Pool.generate_library() default output (name and seq only)."""

    def test_default_output_minimal_columns(self):
        """Test that default output only has name and seq columns."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"]).named("X")

        df = pool.generate_library(num_seqs=3)
        # Should only have name and seq columns
        assert set(df.columns) == {"name", "seq"}

    def test_design_cards_require_opt_in(self):
        """Test that design cards require explicit cards parameter."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("X")

        df = pool.generate_library(num_seqs=3)
        # Should not have any design card columns
        design_card_cols = [c for c in df.columns if "." in c and c not in ["name", "seq"]]
        assert len(design_card_cols) == 0

    def test_cards_parameter_adds_columns(self):
        """Test that cards parameter adds requested columns."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential", cards=["seq_index"]).named("X")

        df = pool.generate_library(num_cycles=1)
        # Should have seq_index column
        index_cols = [c for c in df.columns if "seq_index" in c]
        assert len(index_cols) > 0


# Note: Old tests for report_op_keys, pools_to_report, and state -1 behavior
# have been removed as these features are deprecated. Design cards are now
# opt-in via the `cards` parameter on factory functions.


