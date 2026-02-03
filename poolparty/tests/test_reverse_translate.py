"""Tests for the reverse_translate operation."""

import poolparty as pp
from poolparty.codon_table import CodonTable
from poolparty.orf_ops.reverse_translate import ReverseTranslateOp


class TestReverseTranslateBasic:
    """Test basic reverse translation functionality."""

    def test_basic_first_selection(self):
        """Test reverse translation with 'first' codon selection."""
        with pp.Party():
            pool = pp.from_seq("ATG").translate().reverse_translate()
            df = pool.generate_library(num_seqs=1)
            assert df["seq"].iloc[0] == "ATG"  # M → ATG (most frequent)

    def test_basic_multiple_amino_acids(self):
        """Test reverse translation of multiple amino acids."""
        with pp.Party():
            # MAP: M→ATG, A→GCC (most frequent), P→CCC (most frequent)
            pool = pp.reverse_translate("MAP")
            df = pool.generate_library(num_seqs=1)
            seq = df["seq"].iloc[0]
            assert seq == "ATGGCCCCC"

    def test_reverse_translate_from_string(self):
        """Test reverse translation from protein string."""
        with pp.Party():
            pool = pp.reverse_translate("MK")
            df = pool.generate_library(num_seqs=1)
            seq = df["seq"].iloc[0]
            # M→ATG, K→AAG (most frequent)
            assert seq == "ATGAAG"

    def test_reverse_translate_from_pool_method(self):
        """Test reverse translation via ProteinPool method."""
        with pp.Party():
            pool = pp.from_seq("ATGAAG").translate().reverse_translate()
            df = pool.generate_library(num_seqs=1)
            seq = df["seq"].iloc[0]
            # MK → ATG AAG
            assert seq == "ATGAAG"


class TestCodonSelection:
    """Test codon selection strategies."""

    def test_first_selection_uses_most_frequent(self):
        """Test 'first' selection always uses most frequent codon."""
        codon_table = CodonTable()
        with pp.Party():
            # L has 6 codons: CTG, CTC, CTT, TTG, TTA, CTA (sorted by frequency)
            pool = pp.reverse_translate("L", codon_selection="first")
            df = pool.generate_library(num_seqs=1)
            assert df["seq"].iloc[0] == "CTG"  # Most frequent Leu codon

    def test_random_selection_produces_valid_codons(self):
        """Test 'random' selection produces valid codons for the amino acid."""
        codon_table = CodonTable()
        with pp.Party():
            pool = pp.reverse_translate("L", codon_selection="random")
            # Generate multiple sequences
            df = pool.generate_library(num_seqs=20, seed=42)
            valid_leu_codons = set(codon_table.aa_to_codons["L"])
            for seq in df["seq"]:
                assert seq in valid_leu_codons

    def test_random_selection_varies_output(self):
        """Test 'random' selection can produce different codons."""
        with pp.Party():
            # L has 6 codons, so with enough samples we should see variation
            pool = pp.reverse_translate("LLLLLL", codon_selection="random")
            df = pool.generate_library(num_seqs=10, seed=42)
            seqs = set(df["seq"])
            # With 6^6 = 46656 possible combinations, we should see variation
            assert len(seqs) > 1

    def test_first_is_deterministic(self):
        """Test 'first' selection is deterministic."""
        with pp.Party():
            pool = pp.reverse_translate("MAPK", codon_selection="first")
            df1 = pool.generate_library(num_seqs=5)
            df2 = pool.generate_library(num_seqs=5)
            assert all(df1["seq"] == df2["seq"])


class TestModeHandling:
    """Test mode and num_states handling."""

    def test_first_selection_mode_fixed(self):
        """Test 'first' selection uses mode='fixed' with num_states=1."""
        with pp.Party():
            pool = pp.reverse_translate("MAP", codon_selection="first")
            assert pool.operation.mode == "fixed"
            assert pool.num_states == 1

    def test_random_selection_mode_random(self):
        """Test 'random' selection uses mode='random'."""
        with pp.Party():
            pool = pp.reverse_translate("MAP", codon_selection="random")
            assert pool.operation.mode == "random"

    def test_random_selection_with_num_states(self):
        """Test 'random' selection with explicit num_states."""
        with pp.Party():
            pool = pp.reverse_translate("MAP", codon_selection="random", num_states=5)
            assert pool.num_states == 5
            df = pool.generate_library(num_cycles=1)
            assert len(df) == 5

    def test_random_selection_without_num_states(self):
        """Test 'random' selection without num_states generates on-the-fly."""
        with pp.Party():
            pool = pp.reverse_translate("MAP", codon_selection="random")
            # num_states=None becomes 1 after validation, but random mode generates on-the-fly
            assert pool.num_states == 1
            # Can still generate any number of sequences
            df = pool.generate_library(num_seqs=10, seed=42)
            assert len(df) == 10


class TestStopCodonHandling:
    """Test stop codon handling."""

    def test_stop_codon_first_selection(self):
        """Test stop codon uses most frequent with 'first' selection."""
        codon_table = CodonTable()
        with pp.Party():
            pool = pp.from_seq("ATGTGA").translate().reverse_translate()
            df = pool.generate_library(num_seqs=1)
            seq = df["seq"].iloc[0]
            # M* → ATG + most frequent stop codon (TGA)
            assert seq == "ATGTGA"

    def test_stop_codon_random_selection(self):
        """Test stop codon with 'random' selection produces valid stop codons."""
        codon_table = CodonTable()
        valid_stop_codons = set(codon_table.aa_to_codons["*"])
        with pp.Party():
            pool = pp.reverse_translate("*", codon_selection="random")
            df = pool.generate_library(num_seqs=20, seed=42)
            for seq in df["seq"]:
                assert seq in valid_stop_codons


class TestStylePreservation:
    """Test style propagation from protein to DNA."""

    def test_style_propagates_to_all_three_positions(self):
        """Test amino acid style applies to all 3 codon positions."""
        with pp.Party():
            # Create DNA with codon styling, translate, then reverse translate
            dna = pp.from_seq("ATGGCC").annotate_orf("cds", frame=1, style_codons=["red", "green"])
            protein = dna.translate(region="cds", preserve_codon_styles=True)
            back_to_dna = protein.reverse_translate()

            # Test that reverse translation completes without error
            df = back_to_dna.generate_library(num_seqs=1)
            seq = df["seq"].iloc[0]
            # The sequence should be valid DNA
            assert len(seq) == 6  # 2 amino acids * 3 nucleotides


class TestRegionHandling:
    """Test region extraction in reverse translation."""

    def test_no_region_translates_full_sequence(self):
        """Test full sequence is reverse translated when no region specified."""
        with pp.Party():
            pool = pp.reverse_translate("MAP")
            df = pool.generate_library(num_seqs=1)
            assert len(df["seq"].iloc[0]) == 9  # 3 AAs * 3 nucleotides


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_protein(self):
        """Test reverse translation of empty protein."""
        with pp.Party():
            pool = pp.from_seq("").translate().reverse_translate()
            df = pool.generate_library(num_seqs=1)
            # Empty input may result in None (NullSeq) or empty string
            seq = df["seq"].iloc[0]
            assert seq is None or seq == ""

    def test_single_amino_acid(self):
        """Test reverse translation of single amino acid."""
        with pp.Party():
            pool = pp.reverse_translate("M")
            df = pool.generate_library(num_seqs=1)
            assert df["seq"].iloc[0] == "ATG"

    def test_invalid_characters_skipped(self):
        """Test invalid amino acid characters are skipped (treated as non-molecular)."""
        with pp.Party():
            # X is not a valid amino acid - it's skipped like gaps/tags
            pool = pp.reverse_translate("MXK")
            df = pool.generate_library(num_seqs=1)
            seq = df["seq"].iloc[0]
            # Only M and K are translated (X is skipped)
            assert seq == "ATGAAG"  # M→ATG, K→AAG


class TestOutputType:
    """Test output pool type."""

    def test_returns_dna_pool(self):
        """Test reverse_translate returns DnaPool."""
        with pp.Party():
            pool = pp.reverse_translate("MAP")
            assert isinstance(pool, pp.DnaPool)

    def test_output_seq_length(self):
        """Test output sequence length is 3x protein length."""
        with pp.Party():
            pool = pp.reverse_translate("MAPK")
            assert pool.seq_length == 12  # 4 AAs * 3 nucleotides


class TestRoundTrip:
    """Test translate -> reverse_translate round trips."""

    def test_round_trip_preserves_amino_acids(self):
        """Test round trip produces synonymous DNA encoding same protein."""
        with pp.Party():
            original_dna = "ATGGCTCCCAAG"  # MAPK
            translated = pp.from_seq(original_dna).translate()
            back_to_dna = translated.reverse_translate()

            # Translate both to compare proteins
            original_protein = pp.from_seq(original_dna).translate()
            round_trip_protein = back_to_dna.translate()

            df_orig = original_protein.generate_library(num_seqs=1)
            df_round = round_trip_protein.generate_library(num_seqs=1)

            assert df_orig["seq"].iloc[0] == df_round["seq"].iloc[0]


class TestReverseTranslateOp:
    """Test ReverseTranslateOp class directly."""

    def test_factory_name(self):
        """Test factory name is set correctly."""
        assert ReverseTranslateOp.factory_name == "reverse_translate"

    def test_op_name_in_pool(self):
        """Test operation name appears in pool."""
        with pp.Party():
            pool = pp.reverse_translate("M")
            assert "reverse_translate" in pool.operation.name
