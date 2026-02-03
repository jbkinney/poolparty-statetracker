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
            num_cycles=1, seed=0, init_state=0, report_design_cards=True
        )
        df_copied = copied.generate_library(
            num_cycles=1, seed=0, init_state=0, report_design_cards=True
        )

        assert list(df_original["original.seq"]) == list(df_copied["copied.seq"])

    def test_copy_independent_generation(self):
        """Test that copied pools can generate independently."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential").named("original")
            copied = pool.copy(name="copied")

        # Generate from original
        df1 = pool.generate_library(num_seqs=2, init_state=0, report_design_cards=True)
        # Generate from copy (should start fresh)
        df2 = copied.generate_library(num_seqs=2, init_state=0, report_design_cards=True)

        assert list(df1["original.seq"]) == ["A", "B"]
        assert list(df2["copied.seq"]) == ["A", "B"]

    def test_copy_mutagenize_pool(self):
        """Test copying a mutagenize pool."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"], mode="sequential")
            mutants = pp.mutagenize(seq, num_mutations=1, mode="sequential").named("mutants")
            copied = mutants.copy(name="copied")

        df_mutants = mutants.generate_library(
            num_seqs=5, seed=42, init_state=0, report_design_cards=True
        )
        df_copied = copied.generate_library(
            num_seqs=5, seed=42, init_state=0, report_design_cards=True
        )

        assert list(df_mutants["mutants.seq"]) == list(df_copied["copied.seq"])

    def test_copy_stacked_pool(self):
        """Test copying a stacked pool."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            stacked = (a + b).named("stacked")
            copied = stacked.copy(name="copied")

        df_stacked = stacked.generate_library(num_cycles=1, init_state=0, report_design_cards=True)
        df_copied = copied.generate_library(num_cycles=1, init_state=0, report_design_cards=True)

        assert list(df_stacked["stacked.seq"]) == list(df_copied["copied.seq"])

    def test_copy_repeated_pool(self):
        """Test copying a repeated pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            repeated = (pool * 2).named("repeated")
            copied = repeated.copy(name="copied")

        df_repeated = repeated.generate_library(
            num_cycles=1, init_state=0, report_design_cards=True
        )
        df_copied = copied.generate_library(num_cycles=1, init_state=0, report_design_cards=True)

        assert list(df_repeated["repeated.seq"]) == list(df_copied["copied.seq"])

    def test_copy_sliced_pool(self):
        """Test copying a sliced pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential")
            sliced = pool[1:3]
            sliced.name = "sliced"
            copied = sliced.copy(name="copied")

        df_sliced = sliced.generate_library(num_cycles=1, init_state=0, report_design_cards=True)
        df_copied = copied.generate_library(num_cycles=1, init_state=0, report_design_cards=True)

        assert list(df_sliced["sliced.seq"]) == list(df_copied["copied.seq"])

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
            num_cycles=1, seed=0, init_state=0, report_design_cards=True
        )
        df_copied = copied.generate_library(
            num_cycles=1, seed=0, init_state=0, report_design_cards=True
        )

        assert list(df_original["original.seq"]) == list(df_copied["copied.seq"])

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
        df_stacked = stacked.generate_library(num_cycles=1, init_state=0, report_design_cards=True)
        df_copied = copied.generate_library(num_cycles=1, init_state=0, report_design_cards=True)
        assert list(df_stacked["stacked.seq"]) == list(df_copied["copied.seq"])

    def test_deepcopy_mutagenize_produces_same(self):
        """Test deepcopy of mutagenize produces same sequences."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"], mode="sequential")
            mutants = pp.mutagenize(seq, num_mutations=1, mode="sequential").named("mutants")
            copied = mutants.deepcopy(name="copied")

        df_mutants = mutants.generate_library(
            num_seqs=5, seed=42, init_state=0, report_design_cards=True
        )
        df_copied = copied.generate_library(
            num_seqs=5, seed=42, init_state=0, report_design_cards=True
        )

        assert list(df_mutants["mutants.seq"]) == list(df_copied["copied.seq"])


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

        df = pool.generate_library(num_seqs=3, report_design_cards=True)
        assert len(df) == 3
        assert "X.seq" in df.columns
        assert list(df["X.seq"]) == ["A", "B", "C"]

    def test_generate_num_cycles(self):
        """Test generate() with num_cycles."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential").named("X")

        df = pool.generate_library(num_cycles=2, report_design_cards=True)
        assert len(df) == 4
        assert list(df["X.seq"]) == ["A", "B", "A", "B"]

    def test_generate_with_aux_pools(self):
        """Test generate() with aux_pools."""
        with pp.Party() as party:
            a = pp.from_seqs(["AAA", "TTT"]).named("A")
            b = pp.mutagenize(a, num_mutations=1, mode="sequential").named("B")

        df = b.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a])
        assert "B.seq" in df.columns
        assert "A.seq" in df.columns
        # A.seq should contain the parent sequences
        assert "AAA" in list(df["A.seq"]) or "TTT" in list(df["A.seq"])

    def test_generate_multiple_aux_pools(self):
        """Test generate() with multiple aux_pools."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential").named("A")
            b = pp.from_seqs(["X", "Y"], mode="sequential").named("B")
            stacked = a + b

        df = stacked.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a, b])
        # stacked pool uses StackOp class name since no name given
        seq_cols = [c for c in df.columns if c.endswith(".seq")]
        assert len(seq_cols) == 3  # main pool + 2 aux pools
        assert "A.seq" in df.columns
        assert "B.seq" in df.columns

    def test_generate_with_seed(self):
        """Test generate() with seed for reproducibility."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"]).named("X")

        df1 = pool.generate_library(num_seqs=3, seed=42, report_design_cards=True)
        df2 = pool.generate_library(num_seqs=3, seed=42, init_state=0, report_design_cards=True)
        assert list(df1["X.seq"]) == list(df2["X.seq"])

    def test_generate_with_init_state(self):
        """Test generate() with init_state."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("X")

        df = pool.generate_library(num_seqs=2, init_state=1, report_design_cards=True)
        assert list(df["X.seq"]) == ["B", "C"]

    def test_generate_state_continuation(self):
        """Test that generate() continues from last state."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential").named("X")

        df1 = pool.generate_library(num_seqs=2, init_state=0, report_design_cards=True)
        df2 = pool.generate_library(
            num_seqs=2, report_design_cards=True
        )  # Should continue from state 2

        assert list(df1["X.seq"]) == ["A", "B"]
        assert list(df2["X.seq"]) == ["C", "D"]

    def test_generate_includes_design_cards(self):
        """Test that generate() includes design card columns."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"])
            mutants = pp.mutagenize(pool, num_mutations=1)

        df = mutants.generate_library(num_seqs=5, report_design_cards=True)
        # Should have design card columns from mutagenize
        design_cols = [c for c in df.columns if "." in c]
        assert len(design_cols) > 0

    def test_generate_seq_column_first(self):
        """Test that 'seq' column appears first, then pool's .seq column."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"]).named("A")
            mutants = pp.mutagenize(pool, num_mutations=1).named("B")

        df = mutants.generate_library(num_seqs=3, report_design_cards=True)
        assert df.columns[0] == "seq"
        assert df.columns[1] == "B.seq"

    def test_generate_aux_columns_order(self):
        """Test that seq columns are sorted by reverse topological order."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state (required for stacking)
            a = pp.from_seqs(["A"], mode="sequential").named("A")
            b = pp.from_seqs(["B"], mode="sequential").named("B")
            c = pp.from_seqs(["C"], mode="sequential").named("C")
            combined = a + b + c

        df = combined.generate_library(num_seqs=3, report_design_cards=True, aux_pools=[a, b])
        # First column should be 'seq', then .seq columns
        assert df.columns[0] == "seq"
        assert df.columns[1].endswith(".seq")
        # All pools in pools_to_report='all' should have seq columns
        seq_cols = [c for c in df.columns if c.endswith(".seq")]
        # a+b+c creates intermediate pools: a+b then (a+b)+c
        # So: combined, intermediate(a+b), A, B, C = 5 pools
        assert len(seq_cols) == 5
        assert "A.seq" in seq_cols
        assert "B.seq" in seq_cols
        assert "C.seq" in seq_cols


# =============================================================================
# Tests for Pool.generate_library() record_counters
# =============================================================================


class TestPoolGenerateRecordStates:
    """Test Pool.generate_library() with report_pool_states parameter."""

    def test_report_pool_states_false_no_columns(self):
        """Test that report_pool_states=False produces no pool state columns."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"]).named("X")

        df = pool.generate_library(
            num_seqs=3, report_design_cards=True, report_pool_states=False, report_op_states=False
        )
        counter_cols = [c for c in df.columns if c.endswith(".state")]
        assert len(counter_cols) == 0

    def test_report_pool_states_true_adds_columns(self):
        """Test that report_pool_states=True adds counter state columns."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("X")

        df = pool.generate_library(num_seqs=3, report_design_cards=True, report_pool_states=True)
        counter_cols = [c for c in df.columns if c.endswith(".state")]
        assert len(counter_cols) > 0

    def test_report_pool_states_column_format(self):
        """Test counter column naming format."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"]).named("X")

        df = pool.generate_library(num_seqs=3, report_design_cards=True, report_pool_states=True)
        counter_cols = [c for c in df.columns if c.endswith(".state")]

        # All counter columns should end with '.state'
        # Named counters: 'B.state' or 'B.op.state'
        # Unnamed counters: 'id_N'
        for col in counter_cols:
            assert col.endswith(".state")

    def test_report_pool_states_state_values(self):
        """Test that counter states are recorded correctly."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("X")

        df = pool.generate_library(num_cycles=1, report_design_cards=True, report_pool_states=True)
        counter_cols = [c for c in df.columns if c.endswith(".state")]

        # The root counter should iterate 0, 1, 2
        assert len(df) == 3
        # At least one counter column should have states 0, 1, 2
        found_matching = False
        for col in counter_cols:
            if list(df[col]) == [0, 1, 2]:
                found_matching = True
                break
        assert found_matching, "Expected to find a counter with states [0, 1, 2]"

    def test_report_pool_states_column_order(self):
        """Test that counter columns appear after output cols, before design cards."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"]).named("A")
            mutants = pp.mutagenize(pool, num_mutations=1).named("B")

        df = mutants.generate_library(num_seqs=5, report_design_cards=True, report_pool_states=True)

        # 'seq' should be first, then 'B.seq'
        assert df.columns[0] == "seq"
        assert df.columns[1] == "B.seq"

        counter_cols = [c for c in df.columns if c.endswith(".state")]
        # Design card cols have '.key.' (operation key columns)
        design_cols = [c for c in df.columns if ".key." in c]

        # State columns should come before design card columns
        if counter_cols and design_cols:
            first_counter_idx = min(list(df.columns).index(c) for c in counter_cols)
            first_design_idx = min(list(df.columns).index(c) for c in design_cols)
            assert first_counter_idx < first_design_idx

    def test_report_pool_states_with_aux_pools(self):
        """Test report_pool_states works with aux_pools."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential").named("A")
            b = a + a  # Stack creates a sum counter

        df = b.generate_library(
            num_cycles=1, report_design_cards=True, aux_pools=[a], report_pool_states=True
        )

        # Check for seq columns (main pool uses StackOp name, aux uses 'A')
        seq_cols = [c for c in df.columns if c.endswith(".seq")]
        assert len(seq_cols) == 2
        assert "A.seq" in df.columns
        counter_cols = [c for c in df.columns if c.endswith(".state")]
        assert len(counter_cols) > 0

    def test_report_pool_states_named_counter(self):
        """Test that named counters use their name in column."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state
            pool = pp.from_seqs(["A", "B"], mode="sequential").named("my_pool")

        df = pool.generate_library(num_seqs=2, report_design_cards=True, report_pool_states=True)
        counter_cols = [c for c in df.columns if c.endswith(".state")]

        # Should find columns like 'my_pool.state' or 'my_pool.op.state'
        assert len(counter_cols) > 0
        # Check at least one column contains 'my_pool'
        assert any("my_pool" in col for col in counter_cols)


# =============================================================================
# Tests for Pool.generate_library() report_op_keys parameter
# =============================================================================


class TestPoolGenerateRecordKeys:
    """Test Pool.generate_library() with report_op_keys parameter."""

    def test_report_op_keys_default_true(self):
        """Test that report_op_keys=True (default) includes design card columns."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"]).named("A")
            mutants = pp.mutagenize(pool, num_mutations=1).named("B")

        df = mutants.generate_library(num_seqs=5, report_design_cards=True)
        # Should have design card columns (contain '.key.')
        key_cols = [c for c in df.columns if ".key." in c]
        assert len(key_cols) > 0

    def test_report_op_keys_false_excludes_columns(self):
        """Test that report_op_keys=False excludes design card columns."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"]).named("A")
            mutants = pp.mutagenize(pool, num_mutations=1).named("B")

        df = mutants.generate_library(num_seqs=5, report_design_cards=True, report_op_keys=False)
        # Should NOT have design card columns
        key_cols = [c for c in df.columns if ".key." in c]
        assert len(key_cols) == 0

    def test_report_op_keys_false_still_has_seq(self):
        """Test that report_op_keys=False still includes sequence columns."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT"]).named("A")
            mutants = pp.mutagenize(pool, num_mutations=1).named("B")

        df = mutants.generate_library(num_seqs=5, report_design_cards=True, report_op_keys=False)
        assert "B.seq" in df.columns

    def test_report_op_keys_false_with_report_pool_states(self):
        """Test that report_op_keys=False works with report_pool_states=True."""
        with pp.Party() as party:
            # Use sequential mode to ensure pool has state
            pool = pp.from_seqs(["AAA", "TTT"], mode="sequential").named("A")
            mutants = pp.mutagenize(pool, num_mutations=1, mode="sequential").named("B")

        df = mutants.generate_library(
            num_seqs=5, report_design_cards=True, report_op_keys=False, report_pool_states=True
        )

        # Should have seq and state columns but no key columns
        assert "B.seq" in df.columns
        state_cols = [c for c in df.columns if c.endswith(".state")]
        key_cols = [c for c in df.columns if ".key." in c]
        assert len(state_cols) > 0
        assert len(key_cols) == 0


# =============================================================================
# Tests for Pool.generate_library() pools_to_report parameter
# =============================================================================


class TestPoolGeneratePoolsToRecord:
    """Test Pool.generate_library() with pools_to_report parameter."""

    def test_pools_to_report_all_default(self):
        """Test that pools_to_report='all' (default) includes all pools' info."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state
            a = pp.from_seqs(["AAA", "TTT"], mode="sequential").named("A")
            b = pp.mutagenize(a, num_mutations=1, mode="sequential").named("B")

        df = b.generate_library(num_seqs=5, report_design_cards=True)

        # Should have state columns from both A and B
        state_cols = [c for c in df.columns if c.endswith(".state")]
        assert any("A" in col for col in state_cols)
        assert any("B" in col for col in state_cols)

        # Should have key columns from both A and B operations (auto-generated names)
        key_cols = [c for c in df.columns if ".key." in c]
        # Check that we have key columns for both operations
        assert any("from_seqs" in col or "seq_" in col for col in key_cols)
        assert any("mutagenize" in col or "mut_pos" in col for col in key_cols)

    def test_pools_to_report_self(self):
        """Test that pools_to_report='self' only includes self pool's info."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state
            a = pp.from_seqs(["AAA", "TTT"], mode="sequential").named("A")
            b = pp.mutagenize(a, num_mutations=1, mode="sequential").named("B")

        df = b.generate_library(num_seqs=5, report_design_cards=True, pools_to_report="self")

        # Should have state columns only from B (self)
        state_cols = [c for c in df.columns if c.endswith(".state")]
        assert any("B" in col for col in state_cols)
        assert not any("A.state" in col for col in state_cols)

        # Should have key columns only from B's operation (self)
        key_cols = [c for c in df.columns if ".key." in c]
        assert any(b.operation.name in col for col in key_cols)
        assert not any(a.operation.name + ".key." in col for col in key_cols)

    def test_pools_to_report_explicit_list(self):
        """Test that pools_to_report=[pool] filters to those pools."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state
            a = pp.from_seqs(["AAA", "TTT"], mode="sequential").named("A")
            b = pp.mutagenize(a, num_mutations=1, mode="sequential").named("B")

        # Record only A's info
        df = b.generate_library(num_seqs=5, report_design_cards=True, pools_to_report=[a])

        # Should have state columns only from A
        state_cols = [c for c in df.columns if c.endswith(".state")]
        assert any("A" in col for col in state_cols)

        # Should have key columns from A's operation (from_seqs keys)
        key_cols = [c for c in df.columns if ".key." in c]
        assert any("seq_" in col or "from_seqs" in col for col in key_cols)

        # Should have state columns only from A
        state_cols = [c for c in df.columns if c.endswith(".state")]
        assert any("A" in col for col in state_cols)
        assert not any("B.state" in col for col in state_cols)

        # Should have key columns only from A's operation
        key_cols = [c for c in df.columns if ".key." in c]
        assert any(a.operation.name in col for col in key_cols)
        assert not any(b.operation.name + ".key." in col for col in key_cols)

    def test_pools_to_report_with_report_pool_states_false(self):
        """Test that pools_to_report respects report_pool_states=False and report_op_states=False."""
        with pp.Party() as party:
            a = pp.from_seqs(["AAA", "TTT"]).named("A")
            b = pp.mutagenize(a, num_mutations=1).named("B")

        df = b.generate_library(
            num_seqs=5,
            report_design_cards=True,
            pools_to_report="self",
            report_pool_states=False,
            report_op_states=False,
        )

        # Should have no state columns
        state_cols = [c for c in df.columns if c.endswith(".state")]
        assert len(state_cols) == 0

        # Should still have key columns from B's operation
        key_cols = [c for c in df.columns if ".key." in c]
        assert any(b.operation.name in col for col in key_cols)

    def test_pools_to_report_with_report_op_keys_false(self):
        """Test that pools_to_report respects report_op_keys=False."""
        with pp.Party() as party:
            # Use sequential mode to ensure pools have state
            a = pp.from_seqs(["AAA", "TTT"], mode="sequential").named("A")
            b = pp.mutagenize(a, num_mutations=1, mode="sequential").named("B")

        df = b.generate_library(
            num_seqs=5, report_design_cards=True, pools_to_report="self", report_op_keys=False
        )

        # Should have no key columns
        key_cols = [c for c in df.columns if ".key." in c]
        assert len(key_cols) == 0

        # Should still have state columns from B
        state_cols = [c for c in df.columns if c.endswith(".state")]
        assert any("B" in col for col in state_cols)

    def test_pools_to_report_self_still_has_seq(self):
        """Test that pools_to_report='self' still includes sequence columns."""
        with pp.Party() as party:
            a = pp.from_seqs(["AAA", "TTT"]).named("A")
            b = pp.mutagenize(a, num_mutations=1).named("B")

        df = b.generate_library(num_seqs=5, report_design_cards=True, pools_to_report="self")
        assert "B.seq" in df.columns


# =============================================================================
# Tests for state -1 returning None for sequences
# =============================================================================


class TestPoolStateMinusOneReturnsNone:
    """Test that pools with state=None return None for sequences."""

    def test_inactive_pool_seq_is_none_in_stack(self):
        """Test that stacked pool with state=None has None sequence."""
        import pandas as pd

        with pp.Party() as party:
            a = pp.from_seqs(["AAAAA", "TTTTT"], mode="sequential").named("A")
            b = pp.from_seqs(["CCCCC"], mode="sequential").named("B")
            c = (a + b).named("C")

        df = c.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a, b])

        # C has 3 states: 0, 1 from A and 2 from B
        # When A is active (states 0, 1), B should be inactive (state=None -> NaN)
        # When B is active (state 2), A should be inactive (state=None -> NaN)

        # Check B.seq is None when B.state is None/NaN
        for i, row in df.iterrows():
            if pd.isna(row["B.state"]):
                assert pd.isna(row["B.seq"]), f"Row {i}: B.seq should be None when B.state=None"
            else:
                assert row["B.seq"] == "CCCCC", f"Row {i}: B.seq should be 'CCCCC' when active"

        # Check A.seq is None when A.state is None/NaN
        for i, row in df.iterrows():
            if pd.isna(row["A.state"]):
                assert pd.isna(row["A.seq"]), f"Row {i}: A.seq should be None when A.state=None"
            else:
                assert row["A.seq"] in ["AAAAA", "TTTTT"], (
                    f"Row {i}: A.seq should be valid when active"
                )

    def test_inactive_pool_seq_none_with_mutagenize(self):
        """Test inactive pool returns None with downstream mutagenize."""
        import pandas as pd

        with pp.Party() as party:
            a = pp.from_seqs(["AAAAA", "TTTTT"], mode="sequential").named("A")
            b = pp.from_seqs(["CCCCC"], mode="sequential").named("B")
            c = (a + b).named("C")
            d = pp.mutagenize(c, num_mutations=1, mode="sequential").named("D")

        df = d.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a, b, c])

        # Verify B.seq is None when B.state is None/NaN
        b_inactive_rows = df[pd.isna(df["B.state"])]
        assert len(b_inactive_rows) > 0, "Should have rows where B is inactive"
        assert b_inactive_rows["B.seq"].isna().all(), "B.seq should be None when B.state=None"

        # Verify A.seq is None when A.state is None/NaN
        a_inactive_rows = df[pd.isna(df["A.state"])]
        assert len(a_inactive_rows) > 0, "Should have rows where A is inactive"
        assert a_inactive_rows["A.seq"].isna().all(), "A.seq should be None when A.state=None"

    def test_active_pool_seq_is_not_none(self):
        """Test that active pools (state != None) have valid sequences."""
        with pp.Party() as party:
            a = pp.from_seqs(["AAA", "TTT"], mode="sequential").named("A")
            b = pp.from_seqs(["CCC"], mode="sequential").named("B")
            c = (a + b).named("C")

        df = c.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a, b])

        # Check that C.seq is never None (C is always active as the output pool)
        assert df["C.seq"].notna().all(), "C.seq should never be None"

        # Check that when pools are active, their sequences are valid
        a_active_rows = df[df["A.state"].notna()]
        assert a_active_rows["A.seq"].notna().all(), "A.seq should not be None when active"

        b_active_rows = df[df["B.state"].notna()]
        assert b_active_rows["B.seq"].notna().all(), "B.seq should not be None when active"

    def test_triple_stack_inactive_pools(self):
        """Test state=None behavior with three stacked pools."""
        import pandas as pd

        with pp.Party() as party:
            a = pp.from_seqs(["A"], mode="sequential").named("A")
            b = pp.from_seqs(["B"], mode="sequential").named("B")
            c = pp.from_seqs(["C"], mode="sequential").named("C")
            stacked = (a + b + c).named("stacked")

        df = stacked.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a, b, c])

        # 3 states total, order preserves input A, B, C
        assert len(df) == 3

        # Row 0: A active, B/C inactive
        assert df.loc[0, "A.seq"] == "A"
        assert pd.isna(df.loc[0, "B.seq"])
        assert pd.isna(df.loc[0, "C.seq"])

        # Row 1: B active, A/C inactive
        assert pd.isna(df.loc[1, "A.seq"])
        assert df.loc[1, "B.seq"] == "B"
        assert pd.isna(df.loc[1, "C.seq"])

        # Row 2: C active, A/B inactive
        assert pd.isna(df.loc[2, "A.seq"])
        assert pd.isna(df.loc[2, "B.seq"])
        assert df.loc[2, "C.seq"] == "C"

    def test_inactive_operation_design_keys_are_none(self):
        """Test that operation design card keys are None when op.state.value=None."""
        import pandas as pd

        with pp.Party() as party:
            a = pp.from_seqs(["AAAAA", "TTTTT"], mode="sequential").named("A")
            b = pp.from_seqs(["CCCCC"], mode="sequential").named("B")
            c = (a + b).named("C")

        df = c.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a, b])

        # from_seqs has design_card_keys: ['seq_name', 'seq_index']
        # Find key columns using actual operation names (auto-generated)
        op_a_name = a.operation.name
        op_b_name = b.operation.name

        op_a_key_cols = [col for col in df.columns if f"{op_a_name}.key." in col]
        op_b_key_cols = [col for col in df.columns if f"{op_b_name}.key." in col]

        assert len(op_a_key_cols) > 0, f"Should have {op_a_name} key columns"
        assert len(op_b_key_cols) > 0, f"Should have {op_b_name} key columns"

        # Find state columns
        op_a_state_col = f"{op_a_name}.state"
        op_b_state_col = f"{op_b_name}.state"

        # Check that when op_A.state is None/NaN, its keys are None/NaN
        for i, row in df.iterrows():
            if pd.isna(row[op_a_state_col]):
                for col in op_a_key_cols:
                    assert pd.isna(row[col]), (
                        f"Row {i}: {col} should be None when {op_a_state_col}=None"
                    )
            else:
                for col in op_a_key_cols:
                    assert pd.notna(row[col]), (
                        f"Row {i}: {col} should not be None when {op_a_name} is active"
                    )

        # Check that when op_B.state is None/NaN, its keys are None/NaN
        for i, row in df.iterrows():
            if pd.isna(row[op_b_state_col]):
                for col in op_b_key_cols:
                    assert pd.isna(row[col]), (
                        f"Row {i}: {col} should be None when {op_b_state_col}=None"
                    )
            else:
                for col in op_b_key_cols:
                    assert pd.notna(row[col]), (
                        f"Row {i}: {col} should not be None when {op_b_name} is active"
                    )

    def test_inactive_mutagenize_keys_are_none(self):
        """Test that mutagenize design card keys are None when parent is inactive."""
        import pandas as pd

        with pp.Party() as party:
            a = pp.from_seqs(["AAAAA"], mode="sequential").named("A")
            b = pp.from_seqs(["CCCCC"], mode="sequential").named("B")
            c = (a + b).named("C")
            d = pp.mutagenize(c, num_mutations=1, mode="sequential").named("D")

        df = d.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a, b, c])

        # Find key columns using actual operation names (auto-generated)
        op_a_name = a.operation.name
        op_b_name = b.operation.name

        op_a_key_cols = [col for col in df.columns if f"{op_a_name}.key." in col]
        op_b_key_cols = [col for col in df.columns if f"{op_b_name}.key." in col]

        op_a_state_col = f"{op_a_name}.state"
        op_b_state_col = f"{op_b_name}.state"

        # Check op_A keys
        for i, row in df.iterrows():
            if pd.isna(row[op_a_state_col]):
                for col in op_a_key_cols:
                    assert pd.isna(row[col]), (
                        f"Row {i}: {col} should be None when {op_a_state_col}=None"
                    )

        # Check op_B keys
        for i, row in df.iterrows():
            if pd.isna(row[op_b_state_col]):
                for col in op_b_key_cols:
                    assert pd.isna(row[col]), (
                        f"Row {i}: {col} should be None when {op_b_state_col}=None"
                    )

    def test_active_operation_keys_are_not_none(self):
        """Test that active operation design card keys are not None."""
        import pandas as pd

        with pp.Party() as party:
            a = pp.from_seqs(["AAA", "TTT"], mode="sequential").named("A")
            b = pp.from_seqs(["CCC"], mode="sequential").named("B")
            c = (a + b).named("C")

        df = c.generate_library(num_cycles=1, report_design_cards=True, aux_pools=[a, b])

        # Find key columns using actual operation names (auto-generated)
        op_a_name = a.operation.name
        op_b_name = b.operation.name

        op_a_key_cols = [col for col in df.columns if f"{op_a_name}.key." in col]
        op_b_key_cols = [col for col in df.columns if f"{op_b_name}.key." in col]

        op_a_state_col = f"{op_a_name}.state"
        op_b_state_col = f"{op_b_name}.state"

        # Check op_A keys are not None when active
        for i, row in df.iterrows():
            if pd.notna(row[op_a_state_col]):
                for col in op_a_key_cols:
                    assert pd.notna(row[col]), (
                        f"Row {i}: {col} should not be None when {op_a_name} is active"
                    )

        # Check op_B keys are not None when active
        for i, row in df.iterrows():
            if pd.notna(row[op_b_state_col]):
                for col in op_b_key_cols:
                    assert pd.notna(row[col]), (
                        f"Row {i}: {col} should not be None when {op_b_name} is active"
                    )
