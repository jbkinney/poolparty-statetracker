"""Tests for the Mutagenize operation."""

import numpy as np
import pytest

import poolparty as pp
from poolparty.base_ops.mutagenize import MutagenizeOp, mutagenize


class TestMutagenizeFactory:
    """Test mutagenize factory function."""

    def test_returns_pool(self):
        """mutagenize returns a Pool object."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1)
            assert pool is not None
            assert hasattr(pool, "operation")

    def test_creates_mutagenize_op(self):
        """Pool's operation is MutagenizeOp."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1)
            assert isinstance(pool.operation, MutagenizeOp)

    def test_accepts_string_input_with_num(self):
        """Factory accepts a string with num_mutations."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1).named("mutant")

        df = pool.generate_library(num_seqs=1)
        assert len(df["seq"].iloc[0]) == 4

    def test_accepts_string_input_with_rate(self):
        """Factory accepts a string with mutation_rate."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.1).named("mutant")

        df = pool.generate_library(num_seqs=1)
        assert len(df["seq"].iloc[0]) == 4

    def test_accepts_pool_input(self):
        """Factory accepts an existing Pool as input."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ACGT"])
            pool = mutagenize(seq, num_mutations=1).named("mutant")

        df = pool.generate_library(num_seqs=1)
        assert len(df["seq"].iloc[0]) == 4


class TestMutagenizeParameterValidation:
    """Test parameter validation."""

    def test_requires_num_or_rate(self):
        """Must provide either num_mutations or mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(
                ValueError, match="Either num_mutations or mutation_rate must be provided"
            ):
                mutagenize("ACGT")

    def test_exclusive_num_and_rate(self):
        """Cannot provide both num_mutations and mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Only one of num_mutations or mutation_rate"):
                mutagenize("ACGT", num_mutations=1, mutation_rate=0.1)

    def test_num_mutations_minimum(self):
        """num_mutations must be >= 1."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations must be >= 1"):
                mutagenize("ACGT", num_mutations=0)

    def test_negative_num_mutations(self):
        """Negative num_mutations raises error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations must be >= 1"):
                mutagenize("ACGT", num_mutations=-1)

    def test_negative_rate_error(self):
        """mutation_rate < 0 raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize("ACGT", mutation_rate=-0.1)

    def test_rate_greater_than_one_error(self):
        """mutation_rate > 1 raises ValueError."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize("ACGT", mutation_rate=1.5)

    def test_sequential_mode_requires_num_mutations(self):
        """mode='sequential' not allowed with mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(
                ValueError, match="mode='sequential' is not supported with mutation_rate"
            ):
                mutagenize("ACGT", mutation_rate=0.1, mode="sequential")

    def test_k_greater_than_length_error(self):
        """Error when num_mutations > sequence length."""
        with pytest.raises(ValueError, match="Cannot apply 3 mutations"):
            with pp.Party() as party:
                pool = mutagenize("AC", num_mutations=3, mode="sequential")
                party.output(pool, name="mutant")


# =============================================================================
# Tests for num_mutations mode
# =============================================================================


class TestMutagenizeSingleMutation:
    """Test single mutation (num_mutations=1) behavior."""

    def test_single_mutation_count(self):
        """Test correct number of single mutants."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential").named("mutant")

        df = pool.generate_library(num_cycles=1)
        # 4 positions * 3 mutations each = 12
        assert len(df) == 12

    def test_single_mutation_correctness(self):
        """Test that all single mutants have exactly 1 difference."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential").named("mutant")

        df = pool.generate_library(num_cycles=1)
        for mutant in df["seq"]:
            diffs = sum(1 for a, b in zip("ACGT", mutant) if a != b)
            assert diffs == 1

    def test_single_mutation_preserves_length(self):
        """Test that mutations preserve sequence length."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="sequential").named("mutant")

        df = pool.generate_library(num_seqs=10)
        for mutant in df["seq"]:
            assert len(mutant) == 8


class TestMutagenizeMultipleMutations:
    """Test multiple mutations (num_mutations > 1) behavior."""

    def test_double_mutation_count(self):
        """Test correct number of double mutants."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=2, mode="sequential").named("mutant")

        df = pool.generate_library(num_cycles=1)
        # C(4,2) * 3^2 = 6 * 9 = 54
        assert len(df) == 54

    def test_double_mutation_correctness(self):
        """Test that all double mutants have exactly 2 differences."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=2, mode="sequential").named("mutant")

        df = pool.generate_library(num_cycles=1)
        for mutant in df["seq"]:
            diffs = sum(1 for a, b in zip("ACGT", mutant) if a != b)
            assert diffs == 2

    def test_triple_mutation(self):
        """Test triple mutations."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=3, mode="sequential").named("mutant")

        df = pool.generate_library(num_cycles=1)
        # C(4,3) * 3^3 = 4 * 27 = 108
        assert len(df) == 108

        for mutant in df["seq"]:
            diffs = sum(1 for a, b in zip("ACGT", mutant) if a != b)
            assert diffs == 3


class TestMutagenizeSequentialMode:
    """Test Mutagenize in sequential mode (num_mutations only)."""

    def test_sequential_enumeration(self):
        """Test sequential enumeration of mutations."""
        with pp.Party() as party:
            pool = mutagenize("AC", num_mutations=1, mode="sequential").named("mutant")

        df = pool.generate_library(num_cycles=1)
        # 2 positions * 3 mutations = 6 mutants
        assert len(df) == 6

    def test_sequential_cycling(self):
        """Test that sequential mode cycles."""
        with pp.Party() as party:
            pool = mutagenize("AC", num_mutations=1, mode="sequential").named("mutant")

        df = pool.generate_library(num_seqs=12)  # 2 complete cycles
        first_half = list(df["seq"][:6])
        second_half = list(df["seq"][6:])
        assert first_half == second_half

    def test_num_states_k1(self):
        """Test num_states for num_mutations=1."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential")
            # 4 positions * 3 mutations = 12
            assert pool.operation.num_states == 12

    def test_num_states_k2(self):
        """Test num_states for num_mutations=2."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=2, mode="sequential")
            # C(4,2) * 3^2 = 6 * 9 = 54
            assert pool.operation.num_states == 54


class TestMutagenizeRandomModeWithNum:
    """Test Mutagenize in random mode with num_mutations."""

    def test_random_sampling(self):
        """Test random sampling of mutations."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random").named("mutant")

        df = pool.generate_library(num_seqs=100, seed=42)
        assert len(df) == 100

        # All should be valid single mutants
        for mutant in df["seq"]:
            assert len(mutant) == 8
            diffs = sum(1 for a, b in zip("ACGTACGT", mutant) if a != b)
            assert diffs == 1

    def test_random_variability(self):
        """Test that random mode with explicit num_states produces varied outputs."""
        with pp.Party() as party:
            # Use explicit num_states to get varied outputs
            pool = mutagenize("ACGTACGT", num_mutations=1, mode="random", num_states=100).named(
                "mutant"
            )

        df = pool.generate_library(num_cycles=1, seed=42)
        unique_mutants = df["seq"].nunique()
        assert unique_mutants > 10  # Should have variety

    def test_random_is_stateless_without_num_states(self):
        """Test that random mode without num_states is stateless (no implicit syncing)."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="random")
            # Stays stateless (no implicit syncing to parent)
            assert pool.operation.num_states == 1
            # Operations always have state now, stateless ones have num_values=1
            assert pool.operation.state.num_values == 1
            assert pool.operation.action_uniquely_determined_by_state is False


# =============================================================================
# Tests for mutation_rate mode
# =============================================================================


class TestMutagenizeRandomModeWithRate:
    """Test random mode with mutation_rate."""

    def test_random_mode_is_stateless(self):
        """Random mode without num_states is stateless (no implicit syncing)."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.1, mode="random")
            # Stays stateless (no implicit syncing to parent)
            assert pool.operation.num_states == 1
            # Operations always have state now, stateless ones have num_values=1
            assert pool.operation.state.num_values == 1
            assert pool.operation.action_uniquely_determined_by_state is False

    def test_random_mode_produces_valid_output(self):
        """Random mode generates valid mutated sequences."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", mutation_rate=0.2, mode="random", num_states=100).named(
                "mutant"
            )

        df = pool.generate_library(num_cycles=1, seed=42)
        assert len(df) == 100

        # All should be valid sequences
        for mutant in df["seq"]:
            assert len(mutant) == 8
            assert all(c in "ACGT" for c in mutant)

    def test_random_mode_variability(self):
        """Random mode with explicit num_states produces varied outputs."""
        with pp.Party() as party:
            # Use explicit num_states to get varied outputs
            pool = mutagenize("ACGTACGT", mutation_rate=0.3, mode="random", num_states=100).named(
                "mutant"
            )

        df = pool.generate_library(num_cycles=1, seed=42)
        unique_mutants = df["seq"].nunique()
        assert unique_mutants > 10  # Should have variety

    def test_mutations_preserve_length(self):
        """Output sequences have same length as input."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", mutation_rate=0.2).named("mutant")

        df = pool.generate_library(num_seqs=10, seed=42)
        for mutant in df["seq"]:
            assert len(mutant) == 8

    def test_mutations_stay_in_alphabet(self):
        """All characters in output are from the alphabet."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.3).named("mutant")

        df = pool.generate_library(num_seqs=20, seed=42)
        for mutant in df["seq"]:
            assert all(c in "ACGT" for c in mutant)

    def test_mutations_differ_from_wildtype(self):
        """Mutated positions have different characters than original."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.5, cards=["positions", "wt_chars", "mut_chars"]).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_seqs=50, seed=42)

        for _, row in df.iterrows():
            mutant = row["seq"]
            positions = row[f"{op_name}.positions"]
            wt_chars = row[f"{op_name}.wt_chars"]
            mut_chars = row[f"{op_name}.mut_chars"]

            # Check that mutations differ from wild-type
            for pos, wt, mut in zip(positions, wt_chars, mut_chars):
                assert "ACGT"[pos] == wt  # Original has wt at position
                assert mutant[pos] == mut  # Mutant has mut at position
                assert wt != mut  # Mutation differs from wild-type


class TestMutagenizeRandomNumStatesWithRate:
    """Test random mode with num_states and mutation_rate."""

    def test_random_uses_num_states(self):
        """Random mode with num_states sets num_states correctly."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.1, mode="random", num_states=100)
            assert pool.operation.num_states == 100

    def test_random_generates_correct_count(self):
        """Random mode with num_states generates num_states sequences per iteration."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.2, mode="random", num_states=50).named(
                "mutant"
            )

        df = pool.generate_library(num_cycles=1, seed=42)
        assert len(df) == 50

    def test_random_num_states_reproducible_with_seed(self):
        """Same seed produces identical results in random mode with num_states."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", mutation_rate=0.2, mode="random", num_states=10).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=42)
            results1 = list(df["seq"])

        results2 = []
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", mutation_rate=0.2, mode="random", num_states=10).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=42)
            results2 = list(df["seq"])

        assert results1 == results2

    def test_random_num_states_different_seeds_different_results(self):
        """Different seeds produce different results in random mode with num_states."""
        results1 = []
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", mutation_rate=0.2, mode="random", num_states=10).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=42)
            results1 = list(df["seq"])

        results2 = []
        with pp.Party() as party:
            pool = mutagenize("ACGTACGT", mutation_rate=0.2, mode="random", num_states=10).named(
                "mutant"
            )
            df = pool.generate_library(num_cycles=1, seed=123)
            results2 = list(df["seq"])

        assert results1 != results2


class TestMutagenizeRateStatistics:
    """Test statistical properties of mutation_rate mode."""

    def test_zero_mutations_possible(self):
        """Low mutation_rate can produce zero mutations."""
        with pp.Party() as party:
            pool = mutagenize("ACGTACGTACGT", mutation_rate=0.001, mode="random", cards=["positions"]).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_seqs=100, seed=42)

        # With very low rate, some sequences should have zero mutations
        zero_mut_count = 0
        for _, row in df.iterrows():
            positions = row[f"{op_name}.positions"]
            if len(positions) == 0:
                zero_mut_count += 1

        # Should have at least some zero-mutation sequences
        assert zero_mut_count > 0

    def test_average_mutations_matches_rate(self):
        """Average number of mutations ≈ mutation_rate * seq_len over many samples."""
        mutation_rate = 0.1
        seq_len = 100
        num_samples = 1000

        with pp.Party() as party:
            # Use explicit num_states to get varied outputs for statistical testing
            pool = mutagenize(
                "A" * seq_len, mutation_rate=mutation_rate, mode="random", num_states=num_samples, cards=["positions"]
            ).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_cycles=1, seed=42)

        total_mutations = 0
        for _, row in df.iterrows():
            positions = row[f"{op_name}.positions"]
            total_mutations += len(positions)

        avg_mutations = total_mutations / num_samples
        expected_avg = mutation_rate * seq_len

        # Allow 20% tolerance for statistical variation
        assert abs(avg_mutations - expected_avg) < 0.2 * expected_avg

    def test_high_rate_many_mutations(self):
        """High mutation_rate produces many mutations per sequence."""
        mutation_rate = 0.8
        seq_len = 50

        with pp.Party() as party:
            # Use explicit num_states to get varied outputs
            pool = mutagenize(
                "A" * seq_len, mutation_rate=mutation_rate, mode="random", num_states=100, cards=["positions"]
            ).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_cycles=1, seed=42)

        total_mutations = 0
        for _, row in df.iterrows():
            positions = row[f"{op_name}.positions"]
            total_mutations += len(positions)

        avg_mutations = total_mutations / 100
        expected_avg = mutation_rate * seq_len

        # With high rate, average should be close to expected
        assert avg_mutations > expected_avg * 0.7  # At least 70% of expected


# =============================================================================
# Common tests (both modes)
# =============================================================================


class TestMutagenizeDNA:
    """Test DNA mutagenesis."""

    def test_dna_with_num(self):
        """DNA mutations use only ACGT (num_mutations)."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential").named("mutant")

        df = pool.generate_library(num_cycles=1)
        for mutant in df["seq"]:
            assert all(c in "ACGT" for c in mutant)

    def test_dna_with_rate(self):
        """DNA mutations use only ACGT (mutation_rate)."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.3, mode="random").named("mutant")

        df = pool.generate_library(num_seqs=20, seed=42)
        for mutant in df["seq"]:
            assert all(c in "ACGT" for c in mutant)


class TestMutagenizeDesignCards:
    """Test design card output."""

    def test_positions_in_output(self):
        """Design card contains 'positions' column when cards requested."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential", cards=["positions"]).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_seqs=4)
        assert f"{op_name}.positions" in df.columns

    def test_wt_chars_in_output(self):
        """Design card contains 'wt_chars' column when cards requested."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential", cards=["wt_chars"]).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_seqs=4)
        assert f"{op_name}.wt_chars" in df.columns

    def test_mut_chars_in_output(self):
        """Design card contains 'mut_chars' column when cards requested."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential", cards=["mut_chars"]).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_seqs=4)
        assert f"{op_name}.mut_chars" in df.columns

    def test_design_card_consistency_with_num(self):
        """Design card values match actual mutations in sequence (num_mutations)."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential", cards=["positions", "wt_chars", "mut_chars"]).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_seqs=12)

        for _, row in df.iterrows():
            mutant = row["seq"]
            positions = row[f"{op_name}.positions"]
            wt_chars = row[f"{op_name}.wt_chars"]
            mut_chars = row[f"{op_name}.mut_chars"]

            # Verify the mutation is at the stated position
            for pos, wt, mut in zip(positions, wt_chars, mut_chars):
                assert "ACGT"[pos] == wt  # Original has wt at position
                assert mutant[pos] == mut  # Mutant has mut at position

    def test_design_card_consistency_with_rate(self):
        """Design card values match actual mutations in sequence (mutation_rate)."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.5, mode="random", cards=["positions", "wt_chars", "mut_chars"]).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_seqs=20, seed=42)

        for _, row in df.iterrows():
            mutant = row["seq"]
            positions = row[f"{op_name}.positions"]
            wt_chars = row[f"{op_name}.wt_chars"]
            mut_chars = row[f"{op_name}.mut_chars"]

            # Verify the mutation is at the stated position
            for pos, wt, mut in zip(positions, wt_chars, mut_chars):
                assert "ACGT"[pos] == wt  # Original has wt at position
                assert mutant[pos] == mut  # Mutant has mut at position


class TestMutagenizeMutationMap:
    """Test the mutation map correctness."""

    def test_mutations_are_different_from_wt(self):
        """Test that mutations are always different from wild-type."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential", cards=["positions", "wt_chars", "mut_chars"]).named("mutant")

        op_name = pool.operation.name
        df = pool.generate_library(num_cycles=1)

        for _, row in df.iterrows():
            positions = row[f"{op_name}.positions"]
            wt_chars = row[f"{op_name}.wt_chars"]
            mut_chars = row[f"{op_name}.mut_chars"]

            for wt, mut in zip(wt_chars, mut_chars):
                assert wt != mut

    def test_all_mutations_covered(self):
        """Test that all possible mutations are covered."""
        with pp.Party() as party:
            pool = mutagenize("A", num_mutations=1, mode="sequential").named("mutant")

        df = pool.generate_library(num_cycles=1)
        # A can mutate to C, G, T
        mutants = set(df["seq"])
        assert mutants == {"C", "G", "T"}


class TestMutagenizeCompute:
    """Test compute methods directly."""

    def test_compute_sequential(self):
        """Test compute in sequential mode."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="sequential")

        pool.operation.state._value = 0
        output_seq, card = pool.operation.compute([pp.types.Seq.from_string("ACGT")])
        assert len(output_seq.string) == 4
        assert card["positions"] is not None

    def test_compute_random_with_num(self):
        """Test compute in random mode with num_mutations."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="random")

        rng = np.random.default_rng(42)
        output_seq, card = pool.operation.compute([pp.types.Seq.from_string("ACGT")], rng)
        assert len(output_seq.string) == 4

        # Verify exactly one mutation
        diffs = sum(1 for a, b in zip("ACGT", output_seq.string) if a != b)
        assert diffs == 1

    def test_compute_random_with_rate(self):
        """Test compute in random mode with mutation_rate."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", mutation_rate=0.5, mode="random")

        rng = np.random.default_rng(42)
        output_seq, card = pool.operation.compute([pp.types.Seq.from_string("ACGT")], rng)
        assert len(output_seq.string) == 4


class TestMutagenizeWithParentPool:
    """Test Mutagenize with various parent pool configurations."""

    def test_with_sequential_parent(self):
        """Test mutation scan with sequential parent pool."""
        with pp.Party() as party:
            seqs = pp.from_seqs(["AAA", "TTT"], mode="sequential")
            mutants = mutagenize(seqs, num_mutations=1, mode="sequential").named("mutant")

        df = mutants.generate_library(num_seqs=10)
        # Should see mutations of both AAA and TTT
        assert len(df) == 10



class TestMutagenizeCustomName:
    """Test name parameter."""

    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1)
            assert pool.operation.name.startswith("op[")
            assert ":mutagenize" in pool.operation.name


class TestMutagenizeRandomNumStatesWithNum:
    """Test random mode with num_states and num_mutations."""

    def test_random_uses_num_states_with_num(self):
        """Random mode with num_states sets num_states correctly."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, mode="random", num_states=100)
            assert pool.operation.num_states == 100


class TestMutagenizeAllowedChars:
    """Test allowed_chars parameter for position-specific mutation constraints."""

    def test_allowed_chars_basic(self):
        """Test allowed_chars restricts mutations to specified bases."""
        with pp.Party() as party:
            # Only allow purines (R=AG) at all positions - sequence must be all purines
            pool = mutagenize(
                "AAGG", num_mutations=1, allowed_chars="RRRR", mode="sequential"
            ).named("mutant")

        df = pool.generate_library(num_cycles=1)
        # Check that all mutant chars are purines (A or G)
        for seq in df["seq"]:
            for i, (orig, mut) in enumerate(zip("AAGG", seq)):
                if orig != mut:
                    assert mut in "AG", f"Position {i}: expected purine, got {mut}"

    def test_allowed_chars_pyrimidines_only(self):
        """Test allowing only pyrimidines (Y=CT) at all positions."""
        with pp.Party() as party:
            # Sequence must be all pyrimidines to match Y constraint
            pool = mutagenize(
                "CCTT", num_mutations=1, allowed_chars="YYYY", mode="sequential"
            ).named("mutant")

        df = pool.generate_library(num_cycles=1)
        for seq in df["seq"]:
            for i, (orig, mut) in enumerate(zip("CCTT", seq)):
                if orig != mut:
                    assert mut in "CT", f"Position {i}: expected pyrimidine, got {mut}"

    def test_allowed_chars_position_specific(self):
        """Test position-specific constraints."""
        with pp.Party() as party:
            # Use ACGT with position-specific IUPAC codes that include each wt
            # Position 0: R (A,G) - wt=A is valid, can mutate to G
            # Position 1: Y (C,T) - wt=C is valid, can mutate to T
            # Position 2: S (G,C) - wt=G is valid, can mutate to C
            # Position 3: W (A,T) - wt=T is valid, can mutate to A
            pool = mutagenize(
                "ACGT", num_mutations=1, allowed_chars="RYSW", mode="sequential"
            ).named("mutant")

        df = pool.generate_library(num_cycles=1)
        # Each position has 1 mutation option, total = 4 states
        assert len(df) == 4
        seqs = set(df["seq"])
        assert seqs == {"GCGT", "ATGT", "ACCT", "ACGA"}

    def test_allowed_chars_with_n_is_default(self):
        """Test that 'N' (all bases) is equivalent to no restriction."""
        with pp.Party() as party:
            pool1 = mutagenize("ACGT", num_mutations=1, mode="sequential").named("mutant1")
        df1 = pool1.generate_library(num_cycles=1)

        with pp.Party() as party:
            pool2 = mutagenize(
                "ACGT", num_mutations=1, allowed_chars="NNNN", mode="sequential"
            ).named("mutant2")
        df2 = pool2.generate_library(num_cycles=1)

        assert len(df1) == len(df2)
        assert set(df1["seq"]) == set(df2["seq"])

    def test_allowed_chars_random_mode(self):
        """Test allowed_chars works in random mode."""
        with pp.Party() as party:
            # Use all-purine sequence for R constraint
            pool = mutagenize("AGAG", num_mutations=1, allowed_chars="RRRR", mode="random").named(
                "mutant"
            )

        df = pool.generate_library(num_seqs=50, seed=42)
        for seq in df["seq"]:
            for i, (orig, mut) in enumerate(zip("AGAG", seq)):
                if orig != mut:
                    assert mut in "AG", f"Position {i}: expected purine, got {mut}"

    def test_allowed_chars_with_mutation_rate(self):
        """Test allowed_chars works with mutation_rate."""
        with pp.Party() as party:
            # Use all-purine sequence for R constraint
            pool = mutagenize(
                "AGAGAGAG", mutation_rate=0.5, allowed_chars="RRRRRRRR", mode="random"
            ).named("mutant")

        df = pool.generate_library(num_seqs=50, seed=42)
        for seq in df["seq"]:
            for i, (orig, mut) in enumerate(zip("AGAGAGAG", seq)):
                if orig != mut:
                    assert mut in "AG", f"Position {i}: expected purine, got {mut}"

    def test_allowed_chars_sequential_state_count(self):
        """Test that sequential mode calculates correct state count with allowed_chars."""
        with pp.Party() as party:
            # 'RRRR' at AAGG: R={A,G}, each position has 1 mutation option
            # A->G (1), A->G (1), G->A (1), G->A (1)
            # Total for k=1: 4 positions * 1 option = 4
            pool = mutagenize(
                "AAGG", num_mutations=1, allowed_chars="RRRR", mode="sequential"
            ).named("mutant")

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 4

    def test_allowed_chars_double_mutations(self):
        """Test allowed_chars with num_mutations=2."""
        with pp.Party() as party:
            # 'RR': positions 0,1 allow A,G only
            # wt=AG: pos0 A->G (1), pos1 G->A (1)
            # k=2: C(2,2)*1*1 = 1 state
            pool = mutagenize("AG", num_mutations=2, allowed_chars="RR", mode="sequential").named(
                "mutant"
            )

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 1
        assert df["seq"].iloc[0] == "GA"

    def test_allowed_chars_invalid_iupac_error(self):
        """Test that invalid IUPAC characters raise an error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="invalid IUPAC"):
                mutagenize("ACGT", num_mutations=1, allowed_chars="XXXX")

    def test_allowed_chars_length_mismatch_error(self):
        """Test that length mismatch raises an error."""
        with pp.Party() as party:
            pool = mutagenize("ACGT", num_mutations=1, allowed_chars="NN", mode="sequential").named(
                "mutant"
            )
            with pytest.raises(ValueError, match="length"):
                pool.generate_library(num_cycles=1)

    def test_allowed_chars_in_copy_params(self):
        """Test allowed_chars is included in _get_copy_params."""
        with pp.Party() as party:
            pool = mutagenize("AAGG", num_mutations=1, allowed_chars="RRRR")
            params = pool.operation._get_copy_params()
        assert params["allowed_chars"] == "RRRR"

    def test_allowed_chars_skip_non_mutable_positions(self):
        """Test that positions with no valid mutations are skipped."""
        with pp.Party() as party:
            # Position 0: only A allowed, wt=A, no valid mutations -> skip
            # Position 1: N allowed, wt=C, valid mutations: A,G,T
            pool = mutagenize("AC", num_mutations=1, allowed_chars="AN", mode="sequential").named(
                "mutant"
            )

        df = pool.generate_library(num_cycles=1)
        # Only position 1 is mutable with 3 options
        assert len(df) == 3
        seqs = set(df["seq"])
        assert seqs == {"AA", "AG", "AT"}

    def test_allowed_chars_lowercase_sequence(self):
        """Test allowed_chars with lowercase input sequence."""
        with pp.Party() as party:
            # Use all-purine lowercase sequence
            pool = mutagenize(
                "agag", num_mutations=1, allowed_chars="RRRR", mode="sequential"
            ).named("mutant")

        df = pool.generate_library(num_cycles=1)
        # Mutant chars should be lowercase (preserving case)
        for seq in df["seq"]:
            for i, (orig, mut) in enumerate(zip("agag", seq)):
                if orig != mut:
                    assert mut in "ag", f"Position {i}: expected lowercase purine, got {mut}"

    def test_allowed_chars_validation_error(self):
        """Test that incompatible sequence raises validation error."""
        with pp.Party() as party:
            # R={A,G}, but sequence has C which is not in R
            pool = mutagenize("ACGT", num_mutations=1, allowed_chars="RRRR", mode="random").named(
                "mutant"
            )
            with pytest.raises(ValueError, match="not in allowed_chars"):
                pool.generate_library(num_seqs=1, seed=42)
