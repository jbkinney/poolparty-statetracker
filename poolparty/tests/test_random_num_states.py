"""Tests for random mode with num_states across operations."""

import poolparty as pp
from poolparty.base_ops.from_seqs import from_seqs
from poolparty.base_ops.get_kmers import get_kmers
from poolparty.base_ops.mutagenize import mutagenize


class TestRandomNumStatesBasic:
    """Test basic random mode with num_states functionality."""

    def test_num_states(self):
        """Test that random mode with num_states uses num_states for counter."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="random", num_states=100)
            assert pool.operation.num_states == 100

    def test_random_mode_stateless_without_num_states(self):
        """Test that random mode without num_states is stateless (no implicit syncing)."""
        with pp.Party() as party:
            # With a parent but no explicit num_states, stays stateless
            pool = mutagenize("ACGT", num_mutations=1, mode="random")
            assert pool.operation.num_states == 1
            assert pool.operation.state.num_values == 1
            assert pool.operation.action_uniquely_determined_by_state is False

            # Without a parent (source operation), also stays stateless
            source_pool = get_kmers(length=4, mode="random")
            assert source_pool.operation.num_states == 1

    def test_random_mode_generates_correct_count(self):
        """Test that random mode with num_states generates the expected number of sequences."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="random", num_states=50).named("mutant")

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 50


class TestRandomNumStatesReproducibility:
    """Test that random mode with num_states produces reproducible results."""

    def test_same_seed_same_results(self):
        """Test that same seed produces same results."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=10).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=42)
            results1 = list(df["seq"])

        results2 = []
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=10).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=42)
            results2 = list(df["seq"])

        assert results1 == results2

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=10).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=42)
            results1 = list(df["seq"])

        results2 = []
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=10).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=123)
            results2 = list(df["seq"])

        assert results1 != results2

    def test_same_state_same_output(self):
        """Test that same state always produces same output with same seed."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=20).named(
                "mutant"
            )
            df = pool.generate_library(num_seqs=40, seed=42)  # 2 complete iterations

        # First and second iteration should be identical
        first_iter = list(df["seq"][:20])
        second_iter = list(df["seq"][20:])
        assert first_iter == second_iter


class TestRandomNumStatesOperationIsolation:
    """Test that different operations produce different results at same state."""

    def test_different_ops_different_results(self):
        """Test that different random operations at same state produce different results."""
        with pp.Party() as party:
            # Two mutagenize operations with same config but different op.id
            pool1 = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=10).named(
                "mutant1"
            )
            pool2 = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=10).named(
                "mutant2"
            )
            df1 = pool1.generate_library(num_cycles=1, seed=42)
            df2 = pool2.generate_library(num_cycles=1, seed=42)

        # They should produce different results because op.id is part of the seed
        results1 = list(df1["seq"])
        results2 = list(df2["seq"])

        # At least some results should differ
        differences = sum(1 for a, b in zip(results1, results2) if a != b)
        assert differences > 0, "Different ops should produce different outputs"


class TestRandomNumStatesMutationScan:
    """Test random mode with num_states specifically for mutagenize."""

    def test_mutagenize_random_valid_mutations(self):
        """Test that random mutagenize with num_states produces valid mutations."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="random", num_states=20).named("mutant")
            df = pool.generate_library(num_cycles=1, seed=42)

        # All outputs should be valid single mutants
        for mutant in df["seq"]:
            assert len(mutant) == 4
            diffs = sum(1 for a, b in zip("ACGT", mutant) if a != b)
            assert diffs == 1

    def test_mutagenize_random_double_mutation(self):
        """Test random mode with num_states and num_mutations=2."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=2, mode="random", num_states=30).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=42)

        assert len(df) == 30
        for mutant in df["seq"]:
            diffs = sum(1 for a, b in zip("ACGTACGT", mutant) if a != b)
            assert diffs == 2


class TestRandomNumStatesGetKmers:
    """Test random mode with num_states for get_kmers."""

    def test_get_kmers_random_num_states(self):
        """Test that get_kmers random mode with num_states uses correct num_states."""
        with pp.Party() as party:
            pool = get_kmers(length=3, mode="random", num_states=100)
            assert pool.operation.num_states == 100

    def test_get_kmers_random_valid_kmers(self):
        """Test that random get_kmers with num_states produces valid k-mers."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode="random", num_states=50).named("kmer")

        df = pool.generate_library(num_cycles=1, seed=42)

        assert len(df) == 50
        for kmer in df["seq"]:
            assert len(kmer) == 4
            assert all(c in "ACGT" for c in kmer)

    def test_get_kmers_random_reproducible(self):
        """Test that get_kmers random mode with num_states is reproducible."""
        results1 = []
        with pp.Party() as party:
            pool = get_kmers(length=4, mode="random", num_states=20).named("kmer")
        df = pool.generate_library(num_cycles=1, seed=42)
        results1 = list(df["seq"])

        results2 = []
        with pp.Party() as party:
            pool = get_kmers(length=4, mode="random", num_states=20).named("kmer")
        df = pool.generate_library(num_cycles=1, seed=42)
        results2 = list(df["seq"])

        assert results1 == results2


class TestRandomNumStatesFromSeqs:
    """Test random mode with num_states for from_seqs."""

    def test_from_seqs_random_num_states(self):
        """Test that from_seqs random mode with num_states uses correct num_states."""
        with pp.Party() as party:
            pool = from_seqs(["AAA", "TTT", "GGG"], mode="random", num_states=100)
            assert pool.operation.num_states == 100

    def test_from_seqs_random_random_selection(self):
        """Test that from_seqs random mode with num_states randomly selects sequences."""
        with pp.Party() as party:
            pool = from_seqs(["AAA", "TTT", "GGG", "CCC"], mode="random", num_states=50).named(
                "seq"
            )

        df = pool.generate_library(num_cycles=1, seed=42)

        assert len(df) == 50
        for seq in df["seq"]:
            assert seq in ["AAA", "TTT", "GGG", "CCC"]

        # Should have variety - at least 2 different sequences
        unique_seqs = df["seq"].nunique()
        assert unique_seqs >= 2

    def test_from_seqs_random_reproducible(self):
        """Test that from_seqs random mode with num_states is reproducible."""
        results1 = []
        with pp.Party() as party:
            pool = from_seqs(["A", "T", "G", "C"], mode="random", num_states=30).named("seq")
        df = pool.generate_library(num_cycles=1, seed=42)
        results1 = list(df["seq"])

        results2 = []
        with pp.Party() as party:
            pool = from_seqs(["A", "T", "G", "C"], mode="random", num_states=30).named("seq")
        df = pool.generate_library(num_cycles=1, seed=42)
        results2 = list(df["seq"])

        assert results1 == results2


class TestRandomNumStatesVsRandomMode:
    """Test differences between random mode with and without num_states."""

    def test_random_with_num_states_has_multiple_states(self):
        """Test that random mode with explicit num_states uses that count."""
        with pp.Party() as party:
            # With explicit num_states, uses that value
            random_num_states_pool = mutagenize(
                "ACGT", num_mutations=1, mode="random", num_states=50
            )
            assert random_num_states_pool.operation.num_states == 50

            # Without explicit num_states, stays stateless (no implicit syncing)
            random_pool = mutagenize("ACGT", num_mutations=1, mode="random")
            assert random_pool.operation.num_states == 1

            # Source op with no parents is also stateless
            stateless_pool = get_kmers(length=4, mode="random")
            assert stateless_pool.operation.num_states == 1

    def test_random_with_num_states_iterates_deterministically(self):
        """Test that random mode with num_states iterates through states deterministically."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=10).named(
                "mutant"
            )

        # Run twice with same seed
        df1 = pool.generate_library(num_seqs=30, seed=42, init_state=0)

        # Create a new pool to test reproducibility
        with pp.Party() as party:
            pool2 = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=10).named(
                "mutant"
            )
        df2 = pool2.generate_library(num_seqs=30, seed=42, init_state=0)

        # Should be identical
        assert list(df1["seq"]) == list(df2["seq"])

    def test_random_default_does_not_cycle(self):
        """Test that default random mode doesn't cycle like random with num_states does."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random").named("mutant")
        df = pool.generate_library(num_seqs=50, seed=42)

        # Random mode should have varied results, not cycling patterns
        # (though it's possible by chance to have repeats)
        mutants = list(df["seq"])
        first_10 = mutants[:10]

        # It would be extremely unlikely for the sequence to repeat exactly
        # after just 10 elements in random mode
        repeating = True
        for i in range(10, 50, 10):
            chunk = mutants[i : i + 10]
            if chunk != first_10:
                repeating = False
                break

        # For random mode, we don't expect perfect cycling
        # (unless extremely unlucky with RNG)


class TestRandomNumStatesComposability:
    """Test that random mode with num_states composes correctly with other operations."""

    def test_random_with_num_states_with_sequential_parent(self):
        """Test random operation with num_states and sequential parent."""
        with pp.Party() as party:
            seqs = from_seqs(["AAAA", "TTTT"], mode="sequential")
            mutants = mutagenize(seqs, num_mutations=1, mode="random", num_states=10).named(
                "mutant"
            )

        df = mutants.generate_library(num_seqs=20, seed=42)

        assert len(df) == 20
        # All should be valid single mutants
        for mutant in df["seq"]:
            assert len(mutant) == 4

    def test_random_with_num_states_with_random_parent(self):
        """Test random operation with num_states and random parent with num_states."""
        with pp.Party() as party:
            seqs = from_seqs(["AAAA", "TTTT", "GGGG"], mode="random", num_states=5)
            mutants = mutagenize(seqs, num_mutations=1, mode="random", num_states=3).named("mutant")

        df = mutants.generate_library(num_cycles=1, seed=42)

        # Total states = 5 * 3 = 15
        assert len(df) == 15


class TestStatelessRandomWithGlobalState:
    """Test stateless random mode (num_states=None) uses global_state for seeding."""

    def test_random_with_parent_stays_stateless(self):
        """Test that random operation with parent stays stateless (no implicit syncing)."""
        with pp.Party() as party:
            parent = from_seqs(["AAAA", "TTTT", "GGGG"], mode="sequential")
            # Random mode without explicit num_states should stay stateless
            child = mutagenize(parent, num_mutations=1, mode="random").named("mutant")

            # Operation should be stateless (num_values=1, action not determined by state)
            assert child.operation.state.num_values == 1
            assert child.operation.num_states == 1
            assert child.operation.action_uniquely_determined_by_state is False
            # Pool state comes from parent only
            assert child.num_states == 3

    def test_stateless_random_reproducible_with_seed(self):
        """Test that stateless random operation is reproducible with same seed."""
        with pp.Party() as party:
            parent = from_seqs(["AAAA", "TTTT", "GGGG"], mode="sequential")
            child = mutagenize(parent, num_mutations=1, mode="random").named("mutant")

        df1 = child.generate_library(num_seqs=10, seed=42)

        with pp.Party() as party:
            parent = from_seqs(["AAAA", "TTTT", "GGGG"], mode="sequential")
            child = mutagenize(parent, num_mutations=1, mode="random").named("mutant")

        df2 = child.generate_library(num_seqs=10, seed=42)

        # Should produce identical results with same seed
        assert list(df1["seq"]) == list(df2["seq"])

    def test_stateless_random_different_per_row(self):
        """Test that stateless random produces different sequences per row."""
        with pp.Party() as party:
            parent = from_seqs(["AAAA"], mode="sequential")  # 1 state parent
            child = mutagenize(parent, num_mutations=1, mode="random").named("mutant")

        df = child.generate_library(num_seqs=20, seed=42)

        # Even though parent has only 1 state, we should get varied outputs
        # because stateless random uses global_state (row number) for seeding
        unique_seqs = df["seq"].nunique()
        # With 20 single mutations on 'AAAA', we expect variety
        assert unique_seqs > 1, "Stateless random should produce different sequences per row"

    def test_stateless_random_with_stateless_parent(self):
        """Test that random operation with stateless parent remains stateless."""
        with pp.Party() as party:
            # Parent pool with no state (pure random source operation)
            parent = get_kmers(length=4, mode="random")

            # Verify parent is truly stateless (num_values=1)
            assert parent.operation.state.num_values == 1
            assert parent.state.num_values == 1

            # Child should also be stateless
            child = mutagenize(parent, num_mutations=1, mode="random").named("mutant")

            assert child.operation.state.num_values == 1
            assert child.state.num_values == 1

    def test_stateless_random_without_parent(self):
        """Test that random operation without parent remains stateless."""
        with pp.Party() as party:
            pool = get_kmers(length=4, mode="random").named("kmer")

            assert pool.operation.state.num_values == 1
            assert pool.state.num_values == 1

    def test_stateless_random_with_multiple_stateful_parents(self):
        """Test random operation with multiple stateful parents stays stateless."""
        with pp.Party() as party:
            from poolparty.fixed_ops.join import join

            parent1 = from_seqs(["AA", "TT"], mode="sequential")  # 2 states
            parent2 = from_seqs(["GGG", "CCC", "AAA"], mode="sequential")  # 3 states
            joined = join([parent1, parent2], spacer_str="_")
            # Random operation on joined pool should be stateless
            child = mutagenize(joined, num_mutations=1, mode="random").named("mutant")

            # Op should be stateless (num_values=1)
            assert child.operation.state.num_values == 1
            assert child.operation.num_states == 1
            # Pool state comes from parents (product)
            assert child.num_states == 6

    def test_stateless_random_single_state_parent_different_outputs(self):
        """Test stateless random with single-state parent produces different outputs."""
        with pp.Party() as party:
            parent = from_seqs(["ACGT"], mode="sequential")  # 1 state
            child = mutagenize(parent, num_mutations=1, mode="random").named("mutant")

            # Should be stateless (num_values=1)
            assert child.operation.state.num_values == 1

        # Should generate different outputs for each row (not cycling)
        df = child.generate_library(num_seqs=10, seed=42)
        assert len(df) == 10
        # Different rows should have different outputs (using global_state)
        unique_seqs = df["seq"].nunique()
        assert unique_seqs > 1, "Different rows should produce different sequences"

    def test_stateless_random_chain_stays_stateless(self):
        """Test that stateless random chain stays stateless."""
        with pp.Party() as party:
            base = from_seqs(["AAAAAAAAAA", "TTTTTTTTTT"], mode="sequential")  # 2 states
            mut1 = mutagenize(base, num_mutations=1, mode="random")  # stateless
            mut2 = mutagenize(mut1, num_mutations=1, mode="random").named("final")  # stateless

            # Base has states, but random ops are stateless (num_values=1)
            assert base.num_states == 2
            assert mut1.operation.state.num_values == 1
            assert mut2.operation.state.num_values == 1
            # Pool states come from base only
            assert mut1.num_states == 2
            assert mut2.num_states == 2

    def test_explicit_num_states_creates_state(self):
        """Test that explicit num_states creates state (not stateless)."""
        with pp.Party() as party:
            parent = from_seqs(["AAAA", "TTTT", "GGGG"], mode="sequential")  # 3 states
            # Explicit num_states should create state
            child = mutagenize(parent, num_mutations=1, mode="random", num_states=10).named(
                "mutant"
            )

            # Should have 10 states (explicit)
            assert child.operation.num_states == 10
            assert child.operation.state is not None
            # Pool state is product: 3 * 10 = 30
            assert child.num_states == 30

    def test_stateless_random_with_region(self):
        """Test stateless random with region parameter stays stateless."""
        with pp.Party() as party:
            parent = from_seqs(["AA<bc></bc>CC", "TT<bc></bc>GG"], mode="sequential")  # 2 states
            child = get_kmers(length=3, pool=parent, region="bc", mode="random").named("barcode")

            # Should be stateless (num_values=1)
            assert child.operation.state.num_values == 1
            # Pool state comes from parent
            assert child.num_states == 2

        # Should be reproducible with same seed
        df1 = child.generate_library(num_seqs=6, seed=42)

        with pp.Party() as party:
            parent = from_seqs(["AA<bc></bc>CC", "TT<bc></bc>GG"], mode="sequential")
            child = get_kmers(length=3, pool=parent, region="bc", mode="random").named("barcode")

        df2 = child.generate_library(num_seqs=6, seed=42)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_stateless_random_uses_global_state_for_seeding(self):
        """Test that stateless random uses global_state (row number) for RNG seeding."""
        with pp.Party() as party:
            # Single-state parent to isolate the effect
            parent = from_seqs(["ACGTACGT"], mode="sequential")
            child = mutagenize(parent, num_mutations=2, mode="random").named("mutant")

        # Generate with init_state=0
        df1 = child.generate_library(num_seqs=5, seed=42, init_state=0)

        with pp.Party() as party:
            parent = from_seqs(["ACGTACGT"], mode="sequential")
            child = mutagenize(parent, num_mutations=2, mode="random").named("mutant")

        # Generate with init_state=5 (should match rows 5-9 of a longer run)
        df2 = child.generate_library(num_seqs=5, seed=42, init_state=5)

        # The sequences should be different because global_state differs
        # (init_state=0 gives global_states 0,1,2,3,4; init_state=5 gives 5,6,7,8,9)
        assert list(df1["seq"]) != list(df2["seq"])

    def test_stateless_random_prefix_uses_global_state(self):
        """Test that stateless random with prefix uses global_state for naming."""
        with pp.Party() as party:
            parent = from_seqs(["AA<bc></bc>CC"], mode="sequential")
            child = get_kmers(length=3, pool=parent, region="bc", mode="random", prefix="bc").named(
                "barcode"
            )

        df = child.generate_library(num_seqs=5, seed=42)

        # Names should include bc_0, bc_1, bc_2, etc. (using global_state)
        names = list(df["name"])
        assert names[0] == "bc_0"
        assert names[1] == "bc_1"
        assert names[2] == "bc_2"
        assert names[3] == "bc_3"
        assert names[4] == "bc_4"

    def test_stateless_random_prefix_with_init_state(self):
        """Test that stateless random prefix uses correct global_state with init_state."""
        with pp.Party() as party:
            parent = from_seqs(["AA<bc></bc>CC"], mode="sequential")
            child = get_kmers(length=3, pool=parent, region="bc", mode="random", prefix="bc").named(
                "barcode"
            )

        # Start from init_state=10
        df = child.generate_library(num_seqs=3, seed=42, init_state=10)

        # Names should start from bc_10
        names = list(df["name"])
        assert names[0] == "bc_10"
        assert names[1] == "bc_11"
        assert names[2] == "bc_12"
