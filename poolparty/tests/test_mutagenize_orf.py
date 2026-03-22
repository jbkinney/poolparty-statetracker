"""Tests for the MutagenizeOrf operation."""

import pytest

import poolparty as pp
from poolparty.codon_table import CodonTable
from poolparty.orf_ops.mutagenize_orf import MutagenizeOrfOp, mutagenize_orf


class TestCodonTable:
    """Test CodonTable class."""

    def test_standard_codon_table(self):
        """Test standard codon table initialization."""
        ct = CodonTable("standard")
        assert len(ct.all_codons) == 64
        assert len(ct.stop_codons) == 3
        assert ct.codon_to_aa["ATG"] == "M"
        assert "ATG" in ct.aa_to_codons["M"]

    def test_mutation_lookup_exists(self):
        """Test that mutation lookup contains all types."""
        ct = CodonTable("standard")
        expected_types = [
            "any_codon",
            "nonsynonymous_first",
            "nonsynonymous_random",
            "missense_only_first",
            "missense_only_random",
            "synonymous",
            "nonsense",
        ]
        for mt in expected_types:
            assert mt in ct.mutation_lookup

    def test_any_codon_mutations(self):
        """Test any_codon returns 63 alternatives."""
        ct = CodonTable("standard")
        for codon in ct.all_codons:
            alts = ct.get_mutations(codon, "any_codon")
            assert len(alts) == 63
            assert codon not in alts

    def test_synonymous_mutations(self):
        """Test synonymous mutations are correct."""
        ct = CodonTable("standard")
        # Methionine has only one codon - no synonymous mutations
        assert len(ct.get_mutations("ATG", "synonymous")) == 0
        # Leucine has 6 codons - 5 synonymous mutations
        assert len(ct.get_mutations("CTG", "synonymous")) == 5

    def test_nonsense_mutations(self):
        """Test nonsense mutations return stop codons."""
        ct = CodonTable("standard")
        # Non-stop codon should get 3 stop options
        assert len(ct.get_mutations("ATG", "nonsense")) == 3
        # Stop codon should get no options
        assert len(ct.get_mutations("TAA", "nonsense")) == 0

    def test_is_uniform(self):
        """Test uniformity detection."""
        ct = CodonTable("standard")
        # Truly uniform across all 64 codons
        assert ct.is_uniform("any_codon") is True
        assert ct.is_uniform("nonsynonymous_first") is True
        # Non-uniform
        assert ct.is_uniform("synonymous") is False
        assert ct.is_uniform("nonsynonymous_random") is False
        assert ct.is_uniform("missense_only_random") is False
        # missense_only_first and nonsense are uniform for non-stop codons only
        # (tested via UNIFORM_MUTATION_TYPES dict in mutagenize_orf.py)

    def test_custom_codon_table(self):
        """Test custom codon table."""
        custom = {"A": ["GCT", "GCC"], "M": ["ATG"]}
        ct = CodonTable(custom)
        assert len(ct.all_codons) == 3
        assert ct.codon_to_aa["GCT"] == "A"


class TestMutagenizeOrfFactory:
    """Test mutagenize_orf factory function."""

    def test_returns_pool(self):
        """mutagenize_orf returns a Pool object."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1)
            assert pool is not None
            assert hasattr(pool, "operation")

    def test_creates_mutagenize_orf_op(self):
        """Pool's operation is MutagenizeOrfOp."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1)
            assert isinstance(pool.operation, MutagenizeOrfOp)

    def test_accepts_string_input(self):
        """Factory accepts a string with num_mutations."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1).named("mutant")

        df = pool.generate_library(num_seqs=1)
        assert len(df["seq"].iloc[0]) == 9

    def test_accepts_pool_input(self):
        """Factory accepts an existing Pool as input."""
        with pp.Party() as party:
            seq = pp.from_seqs(["ATGAAATTT"])
            pool = mutagenize_orf(seq, num_mutations=1).named("mutant")

        df = pool.generate_library(num_seqs=1)
        assert len(df["seq"].iloc[0]) == 9


class TestMutagenizeOrfParameterValidation:
    """Test parameter validation."""

    def test_requires_num_or_rate(self):
        """Must provide either num_mutations or mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(
                ValueError, match="Either num_mutations or mutation_rate must be provided"
            ):
                mutagenize_orf("ATGAAATTT")

    def test_exclusive_num_and_rate(self):
        """Cannot provide both num_mutations and mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="Only one of num_mutations or mutation_rate"):
                mutagenize_orf("ATGAAATTT", num_mutations=1, mutation_rate=0.1)

    def test_num_mutations_minimum(self):
        """num_mutations must be >= 1."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations must be >= 1"):
                mutagenize_orf("ATGAAATTT", num_mutations=0)

    def test_mutation_rate_range(self):
        """mutation_rate must be between 0 and 1."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize_orf("ATGAAATTT", mutation_rate=-0.1)
            with pytest.raises(ValueError, match="mutation_rate must be between 0 and 1"):
                mutagenize_orf("ATGAAATTT", mutation_rate=1.5)

    def test_invalid_mutation_type(self):
        """Invalid mutation_type raises error."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="mutation_type must be one of"):
                mutagenize_orf("ATGAAATTT", num_mutations=1, mutation_type="invalid")

    def test_sequential_mode_requires_num_mutations(self):
        """mode='sequential' not allowed with mutation_rate."""
        with pp.Party() as party:
            with pytest.raises(
                ValueError, match="mode='sequential' is not supported with mutation_rate"
            ):
                mutagenize_orf("ATGAAATTT", mutation_rate=0.1, mode="sequential")

    def test_sequential_mode_requires_uniform_type(self):
        """mode='sequential' requires uniform mutation type."""
        with pp.Party() as party:
            with pytest.raises(
                ValueError, match="mode='sequential' requires a uniform mutation type"
            ):
                mutagenize_orf(
                    "ATGAAATTT", num_mutations=1, mutation_type="synonymous", mode="sequential"
                )

    def test_orf_length_not_divisible_by_3_allowed(self):
        """ORF length not divisible by 3 is allowed - partial codons are not mutable."""
        with pp.Party():
            # 5 bp with frame=1 gives (5-0)//3 = 1 complete codon
            pool = mutagenize_orf("ATGAA", num_mutations=1)
            assert pool.operation.num_codons == 1
            # 5 bp with frame=2 gives (5-1)//3 = 1 complete codon
            pool2 = mutagenize_orf("ATGAA", num_mutations=1, frame=2)
            assert pool2.operation.num_codons == 1
            # 5 bp with frame=3 gives (5-2)//3 = 1 complete codon
            pool3 = mutagenize_orf("ATGAA", num_mutations=1, frame=3)
            assert pool3.operation.num_codons == 1

    def test_num_mutations_exceeds_eligible(self):
        """Error when num_mutations > number of eligible positions."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="num_mutations.*exceeds.*eligible"):
                mutagenize_orf("ATGAAA", num_mutations=3)  # Only 2 codons


class TestMutagenizeOrfRegion:
    """Test region parameter handling."""

    def test_region_interval(self):
        """Test region parameter with [start, end] interval."""
        # Sequence with 5' UTR (GGG) + ORF (ATGAAA) + 3' UTR (CCC)
        seq = "GGGATGAAACCC"
        with pp.Party() as party:
            pool = mutagenize_orf(seq, [3, 9], num_mutations=1).named("mutant")

        df = pool.generate_library(num_seqs=10, seed=42)
        for mutant in df["seq"]:
            # UTRs should be preserved
            assert mutant[:3] == "GGG"
            assert mutant[-3:] == "CCC"
            # Total length preserved
            assert len(mutant) == 12

    def test_region_marker_name(self):
        """Test region parameter with marker name."""
        # Sequence with ORF marked by tags
        seq = "GGG<orf>ATGAAA</orf>CCC"
        with pp.Party() as party:
            pool = mutagenize_orf(seq, "orf", num_mutations=1, frame=1).named("mutant")

        df = pool.generate_library(num_seqs=10, seed=42)
        for mutant in df["seq"]:
            # UTRs should be preserved
            assert mutant[:3] == "GGG"
            assert mutant.endswith("CCC")
            # Tags should be preserved
            assert "<orf>" in mutant
            assert "</orf>" in mutant

    def test_region_validation(self):
        """Test region validation."""
        with pp.Party() as party:
            # region start out of range
            with pytest.raises(ValueError, match="region start must be >= 0"):
                mutagenize_orf("ATGAAA", [-1, 6], num_mutations=1)

            # region end exceeds length
            with pytest.raises(ValueError, match="region end.*cannot exceed sequence length"):
                mutagenize_orf("ATGAAA", [0, 10], num_mutations=1)

            # region start >= end
            with pytest.raises(ValueError, match="region start.*must be < end"):
                mutagenize_orf("ATGAAATTT", [6, 3], num_mutations=1)

            # region must have exactly 2 elements
            with pytest.raises(ValueError, match="region must have exactly 2 elements"):
                mutagenize_orf("ATGAAATTT", [0, 3, 6], num_mutations=1)


class TestMutagenizeOrfCodonPositions:
    """Test codon position selection."""

    def test_codon_positions_explicit(self):
        """Test explicit codon_positions parameter."""
        # 4 codons: ATG AAA TTT GGG
        seq = "ATGAAATTTGGG"
        with pp.Party() as party:
            # Only allow mutations at codons 1 and 2 (AAA and TTT)
            pool = mutagenize_orf(
                seq, num_mutations=1, codon_positions=[1, 2], mode="sequential"
            ).named("mutant")

        df = pool.generate_library(num_cycles=1)
        # Should only mutate at positions 1 and 2
        for mutant in df["seq"]:
            # First codon (ATG) should be unchanged
            assert mutant[:3] == "ATG"
            # Last codon (GGG) should be unchanged
            assert mutant[-3:] == "GGG"

    def test_codon_positions_slice(self):
        """Test codon_positions with slice parameter."""
        # 6 codons
        seq = "ATGAAATTTGGGCCCAAA"
        with pp.Party() as party:
            # Only codons 0, 2, 4 (using slice with step=2)
            pool = mutagenize_orf(seq, num_mutations=1, codon_positions=slice(0, 6, 2)).named(
                "mutant"
            )
            # Should have 3 eligible positions
            assert pool.operation.num_eligible == 3
            assert pool.operation.eligible_positions == [0, 2, 4]

    def test_codon_positions_validation(self):
        """Test codon position validation."""
        with pp.Party() as party:
            # Position out of range
            with pytest.raises(ValueError, match="codon_positions value.*is out of range"):
                mutagenize_orf("ATGAAA", num_mutations=1, codon_positions=[5])

            # Duplicate positions
            with pytest.raises(ValueError, match="must not contain duplicates"):
                mutagenize_orf("ATGAAATTT", num_mutations=1, codon_positions=[0, 0, 1])


class TestMutagenizeOrfMutationTypes:
    """Test different mutation types."""

    def test_any_codon_type(self):
        """Test any_codon mutation type."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                "ATGAAA", num_mutations=1, mutation_type="any_codon", mode="sequential"
            ).named("mutant")

        # 2 codons * 63 alternatives = 126 states
        assert pool.operation.num_states == 126

    def test_missense_only_first_type(self):
        """Test missense_only_first mutation type (default)."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAA", num_mutations=1, mode="sequential", cards=["wt_aas", "mut_aas"]).named("mutant")

        df = pool.generate_library(num_cycles=1)
        ct = CodonTable("standard")

        # Find design card columns (operation name is auto-generated)
        wt_aas_col = [c for c in df.columns if "wt_aas" in c][0]
        mut_aas_col = [c for c in df.columns if "mut_aas" in c][0]
        for _, row in df.iterrows():
            wt_aas = row[wt_aas_col]
            mut_aas = row[mut_aas_col]
            for wt_aa, mut_aa in zip(wt_aas, mut_aas):
                # Should be different AA
                assert wt_aa != mut_aa
                # Should not be stop
                assert mut_aa != "*"

    def test_nonsense_type(self):
        """Test nonsense mutation type."""
        with pp.Party() as party:
            pool = mutagenize_orf(
                "ATGAAA", num_mutations=1, mutation_type="nonsense", mode="sequential", cards=["mut_aas"]
            ).named("mutant")

        df = pool.generate_library(num_cycles=1)

        # Find design card columns (operation name is auto-generated)
        mut_aas_col = [c for c in df.columns if "mut_aas" in c][0]
        for _, row in df.iterrows():
            mut_aas = row[mut_aas_col]
            for mut_aa in mut_aas:
                # All mutations should be stops
                assert mut_aa == "*"

    def test_synonymous_type(self):
        """Test synonymous mutation type (random mode only)."""
        # Use a codon with synonymous options (Leucine CTG has 5 alternatives)
        with pp.Party() as party:
            pool = mutagenize_orf(
                "CTGCTG", num_mutations=1, mutation_type="synonymous", mode="random", cards=["wt_aas", "mut_aas"]
            ).named("mutant")

        df = pool.generate_library(num_seqs=20, seed=42)
        ct = CodonTable("standard")

        # Find design card columns (operation name is auto-generated)
        wt_aas_col = [c for c in df.columns if "wt_aas" in c][0]
        mut_aas_col = [c for c in df.columns if "mut_aas" in c][0]
        for _, row in df.iterrows():
            wt_aas = row[wt_aas_col]
            mut_aas = row[mut_aas_col]
            for wt_aa, mut_aa in zip(wt_aas, mut_aas):
                # AA should be the same (synonymous)
                assert wt_aa == mut_aa


class TestMutagenizeOrfSequentialMode:
    """Test sequential mode enumeration."""

    def test_sequential_single_mutation_count(self):
        """Test correct number of single mutants in sequential mode."""
        with pp.Party() as party:
            # 3 codons, missense_only_first has 19 alternatives (20 AAs - 1 current - stop)
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="sequential").named("mutant")

        # 3 positions * 19 alternatives = 57
        assert pool.operation.num_states == 57

        df = pool.generate_library(num_cycles=1)
        assert len(df) == 57

    def test_sequential_double_mutation_count(self):
        """Test correct number of double mutants in sequential mode."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=2, mode="sequential").named("mutant")

        # C(3,2) * 19^2 = 3 * 361 = 1083
        assert pool.operation.num_states == 1083

    def test_sequential_mutations_correctness(self):
        """Test that sequential mutations are applied correctly."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="sequential", cards=["codon_positions", "wt_codons", "mut_codons"]).named("mutant")

        df = pool.generate_library(num_cycles=1)

        # Find design card columns (operation name is auto-generated)
        positions_col = [c for c in df.columns if "codon_positions" in c][0]
        wt_codons_col = [c for c in df.columns if "wt_codons" in c][0]
        mut_codons_col = [c for c in df.columns if "mut_codons" in c][0]
        for _, row in df.iterrows():
            mutant = row["seq"]
            positions = row[positions_col]
            wt_codons = row[wt_codons_col]
            mut_codons = row[mut_codons_col]

            # Verify mutations are applied
            for pos, wt, mut in zip(positions, wt_codons, mut_codons):
                # Get codon from mutant sequence
                codon_start = pos * 3
                mutant_codon = mutant[codon_start : codon_start + 3]
                assert mutant_codon == mut
                assert wt != mut  # Should be different


class TestMutagenizeOrfRandomMode:
    """Test random mode."""

    def test_random_mode_with_num_mutations(self):
        """Test random mode with fixed num_mutations."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTTGGG", num_mutations=2, mode="random", cards=["codon_positions"]).named("mutant")

        df = pool.generate_library(num_seqs=50, seed=42)

        # Find design card columns (operation name is auto-generated)
        positions_col = [c for c in df.columns if "codon_positions" in c][0]
        for _, row in df.iterrows():
            positions = row[positions_col]
            # Should have exactly 2 mutations
            assert len(positions) == 2

    def test_random_mode_with_mutation_rate(self):
        """Test random mode with mutation_rate."""
        with pp.Party() as party:
            # Use explicit num_states to get varied outputs
            pool = mutagenize_orf(
                "ATGAAATTTGGGCCCAAA", mutation_rate=0.5, mode="random", num_states=100, cards=["codon_positions"]
            ).named("mutant")

        df = pool.generate_library(num_cycles=1, seed=42)

        # Find design card columns (operation name is auto-generated)
        positions_col = [c for c in df.columns if "codon_positions" in c][0]
        # Should have variable number of mutations
        num_mutations_list = [len(row[positions_col]) for _, row in df.iterrows()]
        # With 6 codons and 50% rate, should see some variation
        assert len(set(num_mutations_list)) > 1

    def test_random_variability(self):
        """Test that random mode with explicit num_states produces varied outputs."""
        with pp.Party() as party:
            # Use explicit num_states to get varied outputs
            pool = mutagenize_orf(
                "ATGAAATTTGGG", num_mutations=1, mode="random", num_states=50
            ).named("mutant")

        df = pool.generate_library(num_cycles=1, seed=42)
        unique_mutants = df["seq"].nunique()
        assert unique_mutants > 5  # Should have variety


class TestMutagenizeOrfRandomWithNumStates:
    """Test random mode with explicit num_states."""

    def test_random_uses_num_states(self):
        """Random mode with num_states sets num_states correctly."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="random", num_states=100)
            assert pool.operation.num_states == 100

    def test_random_generates_correct_count(self):
        """Random mode with num_states generates num_states sequences per iteration."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="random", num_states=25).named(
                "mutant"
            )

        df = pool.generate_library(num_cycles=1, seed=42)
        assert len(df) == 25


class TestMutagenizeOrfDesignCards:
    """Test design card output."""

    def test_design_card_columns(self):
        """Design card contains expected columns when cards requested."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="sequential", cards=["codon_positions", "wt_codons", "mut_codons", "wt_aas", "mut_aas"]).named("mutant")

        df = pool.generate_library(num_seqs=4)

        # Find design card columns (operation name is auto-generated)
        assert len([c for c in df.columns if "codon_positions" in c]) > 0
        assert len([c for c in df.columns if "wt_codons" in c]) > 0
        assert len([c for c in df.columns if "mut_codons" in c]) > 0
        assert len([c for c in df.columns if "wt_aas" in c]) > 0
        assert len([c for c in df.columns if "mut_aas" in c]) > 0

    def test_design_card_consistency(self):
        """Design card values match actual mutations."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, mode="sequential", cards=["codon_positions", "wt_codons", "mut_codons", "wt_aas", "mut_aas"]).named("mutant")

        df = pool.generate_library(num_seqs=20)
        ct = CodonTable("standard")

        # Find design card columns (operation name is auto-generated)
        positions_col = [c for c in df.columns if "codon_positions" in c][0]
        wt_codons_col = [c for c in df.columns if "wt_codons" in c][0]
        mut_codons_col = [c for c in df.columns if "mut_codons" in c][0]
        wt_aas_col = [c for c in df.columns if "wt_aas" in c][0]
        mut_aas_col = [c for c in df.columns if "mut_aas" in c][0]
        for _, row in df.iterrows():
            mutant = row["seq"]
            positions = row[positions_col]
            wt_codons = row[wt_codons_col]
            mut_codons = row[mut_codons_col]
            wt_aas = row[wt_aas_col]
            mut_aas = row[mut_aas_col]

            for pos, wt_c, mut_c, wt_aa, mut_aa in zip(
                positions, wt_codons, mut_codons, wt_aas, mut_aas
            ):
                # Check codon in mutant sequence
                codon_start = pos * 3
                actual_codon = mutant[codon_start : codon_start + 3]
                assert actual_codon == mut_c

                # Check AA translations
                assert ct.codon_to_aa.get(wt_c.upper()) == wt_aa
                assert ct.codon_to_aa.get(mut_c.upper()) == mut_aa


class TestMutagenizeOrfPreservesLength:
    """Test that mutations preserve sequence length."""

    def test_length_preserved(self):
        """Mutations preserve sequence length."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTTGGG", num_mutations=2).named("mutant")

        df = pool.generate_library(num_seqs=20, seed=42)
        for mutant in df["seq"]:
            assert len(mutant) == 12

    def test_length_preserved_with_flanks(self):
        """Mutations preserve length with ORF boundaries."""
        seq = "GGGATGAAACCC"  # 3bp UTR + 6bp ORF + 3bp UTR
        with pp.Party() as party:
            pool = mutagenize_orf(seq, [3, 9], num_mutations=1).named("mutant")

        df = pool.generate_library(num_seqs=10, seed=42)
        for mutant in df["seq"]:
            assert len(mutant) == 12


class TestMutagenizeOrfCustomName:
    """Test name parameter."""

    def test_default_name(self):
        """Test default operation name."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1)
            assert pool.operation.name.startswith("op[")
            assert ":mutagenize_orf" in pool.operation.name

    def test_custom_name(self):
        """Test custom operation name."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1).named("my_orf_mutations")
            assert pool.name == "my_orf_mutations"


class TestMutagenizeOrfStyle:
    """Test style parameter."""

    def test_style_applied_to_mutations(self):
        """Test that style is applied to mutated codons."""
        with pp.Party() as party:
            pool = mutagenize_orf("ATGAAATTT", num_mutations=1, style="red").named("mutant")

        df = pool.generate_library(num_seqs=5, seed=42)
        # Just verify it runs without error - style is applied internally
        assert len(df) == 5

    def test_style_with_region_marker(self):
        """Test style with marker-based region."""
        seq = "GGG<orf>ATGAAA</orf>CCC"
        with pp.Party() as party:
            pool = mutagenize_orf(seq, "orf", num_mutations=1, style="cyan", frame=1).named(
                "mutant"
            )

        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5


class TestMutagenizeOrfFrame:
    """Test frame parameter."""

    def test_frame_negative_one(self):
        """Test frame=-1 extracts codons from end (reverse direction)."""
        seq = "ATGAAATTT"  # 3 codons: ATG AAA TTT
        with pp.Party() as party:
            # With frame=-1, codons are read from end to start: TTT AAA ATG
            pool = mutagenize_orf(seq, num_mutations=1, frame=-1).named("mutant")

        df = pool.generate_library(num_seqs=10, seed=42)
        # Just verify it runs without error
        assert len(df) == 10
        for mutant in df["seq"]:
            assert len(mutant) == 9

    def test_frame_validation_zero(self):
        """frame=0 is not a valid value."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="frame must be one of"):
                mutagenize_orf("ATGAAATTT", num_mutations=1, frame=0)

    def test_frame_validation_out_of_range(self):
        """frame values outside +-3 are invalid."""
        with pp.Party() as party:
            with pytest.raises(ValueError, match="frame must be one of"):
                mutagenize_orf("ATGAAATTT", num_mutations=1, frame=4)
            with pytest.raises(ValueError, match="frame must be one of"):
                mutagenize_orf("ATGAAATTT", num_mutations=1, frame=-4)

    def test_frame_positive_values(self):
        """Test that positive frame values work (forward direction)."""
        seq = "ATGAAATTT"
        with pp.Party() as party:
            for frame in [1, 2, 3]:
                pool = mutagenize_orf(seq, num_mutations=1, frame=frame).named(f"mutant_{frame}")
                df = pool.generate_library(num_seqs=3, seed=42)
                assert len(df) == 3

    def test_frame_negative_values(self):
        """Test that negative frame values work (reverse direction)."""
        seq = "ATGAAATTT"
        with pp.Party() as party:
            for frame in [-1, -2, -3]:
                pool = mutagenize_orf(seq, num_mutations=1, frame=frame).named(f"mutant_{frame}")
                df = pool.generate_library(num_seqs=3, seed=42)
                assert len(df) == 3


class TestMutagenizeOrfPoolMethod:
    """Test Pool.mutagenize_orf() method."""

    def test_pool_method(self):
        """Test calling mutagenize_orf as Pool method."""
        with pp.Party() as party:
            pool = pp.from_seq("ATGAAATTT").mutagenize_orf(num_mutations=1).named("mutant")

        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5

    def test_pool_method_with_region(self):
        """Test Pool method with region parameter."""
        with pp.Party() as party:
            pool = (
                pp.from_seq("GGG<orf>ATGAAA</orf>CCC")
                .mutagenize_orf("orf", num_mutations=1, frame=1)
                .named("mutant")
            )

        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5
        for mutant in df["seq"]:
            assert mutant[:3] == "GGG"
            assert "<orf>" in mutant
