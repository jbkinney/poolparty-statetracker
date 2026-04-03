"""Tests for the Party context manager and core architecture.

Pool operators now work on States:
- pool1 + pool2: Stack (union of states)
- pool * n: Repeat (repeat states n times)
- pool[start:stop]: State slice (select subset of states)

For sequence operations, use join(), slice_seq(), etc.
"""

import pytest

import poolparty as pp
import statetracker as st
from poolparty import join
from poolparty.fixed_ops.slice_seq import slice_seq


class TestBasicUsage:
    """Test basic Party usage patterns."""

    def test_simple_from_seqs(self):
        """Test creating a simple pool from sequences."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA", "TTT", "GGG"], mode="sequential").named("seq")

        df = pool.generate_library(num_seqs=3)
        assert len(df) == 3
        assert "seq" in df.columns
        assert list(df["seq"]) == ["AAA", "TTT", "GGG"]

    def test_get_kmers_sequential(self):
        """Test generating k-mers in sequential mode."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=2, mode="sequential").named("kmer")

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 16  # 4^2 = 16 k-mers for DNA
        assert df["seq"].iloc[0] == "AA"

    def test_get_kmers_random(self):
        """Test generating k-mers in random mode."""
        with pp.Party() as party:
            pool = pp.get_kmers(length=5, mode="random").named("kmer")

        df = pool.generate_library(num_seqs=10, seed=42)
        assert len(df) == 10
        # All should be valid 5-mers of DNA
        for kmer in df["seq"]:
            assert len(kmer) == 5
            assert all(c in "ACGT" for c in kmer)

    def test_join_pools(self):
        """Test joining pools using join()."""
        with pp.Party() as party:
            left = pp.from_seqs(["AAA"])
            right = pp.from_seqs(["TTT"])
            oligo = join([left, "...", right]).named("oligo")

        df = oligo.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "AAA...TTT"

    def test_join_with_kmer(self):
        """Test joining a sequence with a random barcode."""
        with pp.Party() as party:
            seq_pool = pp.from_seqs(["ACGT"])
            barcode = pp.get_kmers(length=4, mode="random")
            oligo = join([seq_pool, "...", barcode]).named("oligo")

        df = oligo.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5
        for s in df["seq"]:
            assert s.startswith("ACGT...")
            assert len(s) == 4 + 3 + 4  # seq + ... + barcode


class TestMutationScan:
    """Test mutation scanning operations."""

    def test_single_mutation_sequential(self):
        """Test single mutation in sequential mode."""
        with pp.Party() as party:
            mutants = pp.mutagenize("ACGT", num_mutations=1, mode="sequential").named("mutant")

        df = mutants.generate_library(num_cycles=1)
        # 4 positions * 3 mutations each = 12 mutants
        assert len(df) == 12

        # Check all are single mutations
        for mutant in df["seq"]:
            assert len(mutant) == 4
            diffs = sum(1 for a, b in zip("ACGT", mutant) if a != b)
            assert diffs == 1

    def test_double_mutation_sequential(self):
        """Test double mutation in sequential mode."""
        with pp.Party() as party:
            mutants = pp.mutagenize("ACGT", num_mutations=2, mode="sequential").named("mutant")

        df = mutants.generate_library(num_cycles=1)
        # C(4,2) * 3^2 = 6 * 9 = 54 mutants
        assert len(df) == 54

    def test_mutagenize_random(self):
        """Test mutation scan in random mode."""
        with pp.Party() as party:
            mutants = pp.mutagenize("ACGTACGT", num_mutations=1, mode="random").named("mutant")

        df = mutants.generate_library(num_seqs=10, seed=42)
        assert len(df) == 10


class TestStatePersistence:
    """Test that Pool maintains state between generate_library() calls."""

    def test_state_continues(self):
        """Test that sequential iteration continues across generate_library() calls."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential").named("seq")

        df1 = pool.generate_library(num_seqs=2)
        df2 = pool.generate_library(num_seqs=2)

        assert list(df1["seq"]) == ["A", "B"]
        assert list(df2["seq"]) == ["C", "D"]

    def test_reset_via_init_state(self):
        """Test that init_state=0 resets the state."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("seq")

        df1 = pool.generate_library(num_seqs=2)
        df2 = pool.generate_library(num_seqs=2, init_state=0)

        assert list(df1["seq"]) == ["A", "B"]
        assert list(df2["seq"]) == ["A", "B"]  # Reset to start

    def test_init_state(self):
        """Test starting from a specific state."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D"], mode="sequential").named("seq")

        df = pool.generate_library(num_seqs=2, init_state=2)
        assert list(df["seq"]) == ["C", "D"]


class TestMixedModes:
    """Test combining sequential and random operations."""

    def test_sequential_with_random_barcode(self):
        """Test sequential mutations with random barcodes."""
        with pp.Party() as party:
            mutants = pp.mutagenize("ACGT", num_mutations=1, mode="sequential")
            barcode = pp.get_kmers(length=4, mode="random")
            oligo = join([mutants, ".", barcode]).named("oligo")

        df = oligo.generate_library(num_cycles=1, seed=42)
        assert len(df) == 12  # 12 single mutants

        # Check that barcodes are varied (random)
        barcodes = [s.split(".")[-1] for s in df["seq"]]
        assert len(set(barcodes)) > 1  # Not all the same


class TestDesignCards:
    """Test design card metadata with opt-in cards parameter."""

    def test_mutagenize_metadata(self):
        """Test that mutation scan includes position and char metadata when cards requested."""
        with pp.Party() as party:
            mutants = pp.mutagenize("ACGT", num_mutations=1, mode="sequential", cards=["positions", "wt_chars", "mut_chars"]).named("mutant")

        op_name = mutants.operation.name
        df = mutants.generate_library(num_seqs=3)

        assert f"{op_name}.positions" in df.columns
        assert f"{op_name}.wt_chars" in df.columns
        assert f"{op_name}.mut_chars" in df.columns

    def test_from_seqs_metadata(self):
        """Test that from_seqs includes name and index metadata when cards requested."""
        with pp.Party() as party:
            pool = pp.from_seqs(
                ["AAA", "TTT"], seq_names=["seq_a", "seq_b"], mode="sequential", cards=["seq_name", "seq_index"]
            ).named("myseq")

        df = pool.generate_library(num_seqs=2)

        # Find design card columns (operation name is auto-generated)
        seq_name_cols = [c for c in df.columns if "seq_name" in c]
        seq_index_cols = [c for c in df.columns if "seq_index" in c]
        assert len(seq_name_cols) > 0
        assert len(seq_index_cols) > 0
        assert list(df[seq_name_cols[0]]) == ["seq_a", "seq_b"]


class TestSliceSeqOp:
    """Test sequence slicing operations using slice_seq()."""

    def test_slice_seq_with_range(self):
        """Test sequence slicing with a range."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGTACGT"])
            sliced = slice_seq(pool, start=0, stop=4).named("sliced")

        df = sliced.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACGT"

    def test_slice_seq_negative_index(self):
        """Test sequence slicing with negative index."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGTACGT"])
            last = slice_seq(pool, start=-1).named("last")

        df = last.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "T"

    def test_slice_seq_with_step(self):
        """Test sequence slicing with step."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ABCDEFGH"])
            every_other = slice_seq(pool, step=2).named("every_other")

        df = every_other.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "ACEG"

    def test_slice_seq_reverse(self):
        """Test reversing a sequence with slice."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGT"])
            reversed_seq = slice_seq(pool, step=-1).named("reversed")

        df = reversed_seq.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "TGCA"

    def test_slice_seq_with_mutation(self):
        """Test combining slice_seq with mutation."""
        with pp.Party() as party:
            pool = pp.from_seqs(["ACGTACGT"])
            first_half = slice_seq(pool, start=0, stop=4)
            mutated = pp.mutagenize(first_half, num_mutations=1, mode="sequential").named("mutated")

        df = mutated.generate_library(num_seqs=3)
        # All should be 4 characters
        for s in df["seq"]:
            assert len(s) == 4


class TestStateSliceOp:
    """Test state slicing operations using pool[start:stop]."""

    def test_state_slice_with_range(self):
        """Test state slicing with a range."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C", "D", "E"], mode="sequential")
            sliced = pool[1:4].named("seq")  # States 1, 2, 3 -> B, C, D

        df = sliced.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["B", "C", "D"]

    def test_state_slice_single(self):
        """Test state slicing with single index."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            single = pool[1].named("seq")  # State 1 -> B

        df = single.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "B"

    def test_state_slice_negative(self):
        """Test state slicing with negative index."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")
            last = pool[-1].named("seq")  # Last state -> C

        df = last.generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "C"


class TestStackOp:
    """Test stacking operations using pool1 + pool2."""

    def test_stack_two_pools(self):
        """Test stacking two pools."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential")
            b = pp.from_seqs(["X", "Y"], mode="sequential")
            stacked = (a + b).named("seq")

        df = stacked.generate_library(num_cycles=1)
        assert list(df["seq"]) == ["A", "B", "X", "Y"]

    def test_stack_num_states(self):
        """Test that stack num_states is sum of parent states."""
        with pp.Party() as party:
            a = pp.from_seqs(["A", "B"], mode="sequential")  # 2 states
            b = pp.from_seqs(["X", "Y", "Z"], mode="sequential")  # 3 states
            stacked = a + b
            assert stacked.num_states == 5


class TestRepeatOp:
    """Test repeat operations using pool * n."""

    def test_repeat_pool(self):
        """Test repeating a pool."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential")
            repeated = (pool * 2).named("seq")

        df = repeated.generate_library(num_cycles=1)
        # With first_counter_slowest default, original pool cycles slowest
        assert list(df["seq"]) == ["A", "A", "B", "B"]

    def test_repeat_num_states(self):
        """Test that repeat num_states is original * n."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential")  # 3 states
            repeated = pool * 4
            assert repeated.num_states == 12


class TestErrors:
    """Test error handling."""

    def test_nested_party_allowed(self):
        """Test that nested Party contexts are allowed (they stack)."""
        with pp.Party() as outer:
            pool1 = pp.from_seqs(["AAA"])
            with pp.Party() as inner:
                # Inner party is now active
                assert pp.get_active_party() is inner
                pool2 = pp.from_seqs(["AAA"])
            # Outer party restored after exiting inner
            assert pp.get_active_party() is outer

    def test_pool_plus_string_raises(self):
        """Test that pool + string raises error (use join instead)."""
        with pp.Party() as party:
            pool = pp.from_seqs(["AAA"])
            with pytest.raises(Exception):  # beartype raises roar.BeartypeCallHintParamViolation
                _ = pool + "TTT"


class TestStateManagerIntegration:
    """Test StateManager integration with Party."""

    def test_state_manager_accessible(self):
        """Test that party.state_manager is accessible."""
        with pp.Party() as party:
            assert party.state_manager is not None
            assert isinstance(party.state_manager, pp.StateManager)

    def test_counters_registered_from_seqs(self):
        """Test that counters are registered when operations are created."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("seq")

            # Should have registered counters
            names = party.state_manager.get_all_names()
            assert len(names) > 0

    def test_counters_registered_mutagenize(self):
        """Test that mutagenize counters are registered."""
        with pp.Party() as party:
            mutants = pp.mutagenize("ACGT", num_mutations=1, mode="sequential").named("mutant")

            # Should have multiple counters registered
            names = party.state_manager.get_all_names()
            assert len(names) > 0

    def test_test_iteration_works(self, capsys):
        """Test that test_iteration() works on registered counters."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("seq")

            # get_iteration_df should work on the pool's counter
            df = party.state_manager.get_iteration_df(
                pool.state,
                states=None,
            )

            # Should have iterated through all states
            assert len(df) == pool.state.num_values

    def test_print_graph_works(self, capsys):
        """Test that print_graph() shows the counter DAG."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("seq")

            # Should not raise an error
            party.state_manager.print_graph()

            # Check that something was printed
            captured = capsys.readouterr()
            assert len(captured.out) > 0

    def test_state_manager_restored_on_exit(self):
        """Test that previous Manager is restored when exiting Party context."""
        default_mgr = st.Manager._active_manager
        with pp.Party() as party:
            assert st.Manager._active_manager is party.state_manager
            assert st.Manager._active_manager is not default_mgr

        # Previous (default) manager is restored, not cleared to None
        assert st.Manager._active_manager is default_mgr

    def test_complex_dag_counters_registered(self):
        """Test that counters from a complex DAG are all registered."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"], mode="sequential")
            mutants = pp.mutagenize(seq, num_mutations=1, mode="sequential")
            barcode = pp.get_kmers(length=4, mode="random")
            oligo = pp.join([mutants, "---", barcode]).named("oligo")

            # Should have multiple counters registered
            names = party.state_manager.get_all_names()
            # At minimum: from_seqs counter, mutagenize counter, get_kmers counter,
            # join counter, and their pool counters
            assert len(names) >= 4


class TestPrintGraph:
    """Test Party.print_graph() and Pool.print_dag() visualization."""

    def test_print_graph_simple_clean(self, capsys):
        """Test print_graph() with clean style (default)."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("mypool")

        party.print_graph()  # clean is default
        captured = capsys.readouterr()

        # Should contain pool name followed by (pool, ...)
        assert "mypool (pool," in captured.out
        # Should show n= for num_states in clean mode
        assert "n=" in captured.out

    def test_print_graph_minimal(self, capsys):
        """Test print_graph() with minimal style."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("mypool")

        party.print_graph(style="minimal")
        captured = capsys.readouterr()

        # Should contain pool name followed by (pool)
        assert "mypool (pool)" in captured.out
        # Should contain op name followed by [op]
        assert "[op]" in captured.out
        # Should not contain n=
        assert "n=" not in captured.out

    def test_print_graph_repr(self, capsys):
        """Test print_graph() with repr style."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("mypool")

        party.print_graph(style="repr")
        captured = capsys.readouterr()

        # Should contain full repr format
        assert "Pool(" in captured.out
        assert "name='mypool'" in captured.out
        assert "num_states=" in captured.out

    def test_print_graph_chain(self, capsys):
        """Test print_graph() with a chain of pools."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"], mode="sequential").named("seq")
            mutants = pp.mutagenize(seq, num_mutations=1, mode="sequential").named("mutants")

        party.print_graph()
        captured = capsys.readouterr()

        # Should contain both pool names
        assert "mutants (pool," in captured.out
        assert "seq (pool," in captured.out
        # Should show tree structure with connectors
        assert "└──" in captured.out

    def test_print_graph_multi_parent(self, capsys):
        """Test print_graph() with multiple parent pools (join)."""
        with pp.Party() as party:
            left = pp.from_seqs(["AAA"], mode="sequential").named("left")
            right = pp.from_seqs(["TTT"], mode="sequential").named("right")
            oligo = pp.join([left, right]).named("oligo")

        party.print_graph()
        captured = capsys.readouterr()

        # Should contain all pool names
        assert "oligo (pool," in captured.out
        assert "left (pool," in captured.out
        assert "right (pool," in captured.out
        # Should show branching structure
        assert "├──" in captured.out or "└──" in captured.out

    def test_print_graph_multiple_roots(self, capsys):
        """Test print_graph() with multiple root pools."""
        with pp.Party() as party:
            pool1 = pp.from_seqs(["A", "B"], mode="sequential").named("pool1")
            pool2 = pp.from_seqs(["X", "Y"], mode="sequential").named("pool2")

        party.print_graph()
        captured = capsys.readouterr()

        # Should contain both pool names
        assert "pool1 (pool," in captured.out
        assert "pool2 (pool," in captured.out

    def test_print_graph_shows_mode(self, capsys):
        """Test print_graph() shows operation mode."""
        with pp.Party() as party:
            seq_pool = pp.from_seqs(["ACGT"], mode="sequential").named("seq")
            mutants = pp.mutagenize(seq_pool, num_mutations=1, mode="sequential").named("mutants")

        party.print_graph()
        captured = capsys.readouterr()

        # Should show sequential mode
        assert "mode=sequential" in captured.out

    def test_print_graph_no_pools(self, capsys):
        """Test print_graph() with no pools registered."""
        with pp.Party() as party:
            pass

        party.print_graph()
        captured = capsys.readouterr()

        assert "(no pools registered)" in captured.out

    def test_pool_print_dag(self, capsys):
        """Test Pool.print_dag() method directly."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("mypool")
            pool.print_dag()  # clean is default

        captured = capsys.readouterr()

        # Should start with pool name
        assert captured.out.startswith("mypool (pool,")
        # Should show n=3
        assert "n=3" in captured.out

    def test_pool_print_dag_repr(self, capsys):
        """Test Pool.print_dag() with repr style."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("mypool")
            pool.print_dag(style="repr")

        captured = capsys.readouterr()

        # Should start with Pool or DnaPool repr
        assert captured.out.startswith("Pool(") or captured.out.startswith("DnaPool(")
        assert "num_states=3" in captured.out

    def test_operation_print_dag(self, capsys):
        """Test Operation.print_dag() method directly."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("mypool")
            pool.operation.print_dag()  # clean is default

        captured = capsys.readouterr()

    def test_operation_print_dag_repr(self, capsys):
        """Test Operation.print_dag() with repr style."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B", "C"], mode="sequential").named("mypool")
            pool.operation.print_dag(style="repr")

        captured = capsys.readouterr()

        # Should start with operation class name
        assert captured.out.startswith("FromSeqsOp(")

    def test_print_graph_with_repeat(self, capsys):
        """Test print_graph() with repeat operation."""
        with pp.Party() as party:
            pool = pp.from_seqs(["A", "B"], mode="sequential").named("base")
            repeated = (pool * 2).named("repeated")

        party.print_graph()
        captured = capsys.readouterr()

        # Should contain both pool names
        assert "repeated (pool," in captured.out
        assert "base (pool," in captured.out
        # Repeated should have n=4
        assert "n=4" in captured.out

    def test_print_graph_with_stack(self, capsys):
        """Test print_graph() with stack operation."""
        with pp.Party() as party:
            a = pp.from_seqs(["A"], mode="sequential").named("a")
            b = pp.from_seqs(["B"], mode="sequential").named("b")
            stacked = (a + b).named("stacked")

        party.print_graph()
        captured = capsys.readouterr()

        # Should contain all pool names
        assert "stacked (pool," in captured.out
        assert "a (pool," in captured.out
        assert "b (pool," in captured.out
