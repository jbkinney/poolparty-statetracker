"""Tests for poolparty DNA utilities."""

import pytest

import poolparty as pp
from poolparty.utils.dna_utils import (
    BASES,
    COMPLEMENT,
    IGNORE_CHARS,
    IUPAC_CHARS,
    IUPAC_TO_DNA,
    VALID_CHARS,
    complement,
    get_length_without_tags,
    get_molecular_positions,
    get_mutations,
    get_seq_length,
    reverse_complement,
)


class TestConstants:
    """Test DNA constants."""

    def test_bases(self):
        """Test BASES constant."""
        assert BASES == ["A", "C", "G", "T"]

    def test_complement(self):
        """Test COMPLEMENT constant."""
        assert COMPLEMENT["A"] == "T"
        assert COMPLEMENT["T"] == "A"
        assert COMPLEMENT["G"] == "C"
        assert COMPLEMENT["C"] == "G"
        # Lowercase
        assert COMPLEMENT["a"] == "t"
        assert COMPLEMENT["t"] == "a"
        assert COMPLEMENT["g"] == "c"
        assert COMPLEMENT["c"] == "g"

    def test_iupac_to_dna(self):
        """Test IUPAC_TO_DNA constant."""
        assert IUPAC_TO_DNA["A"] == ["A"]
        assert IUPAC_TO_DNA["N"] == ["A", "C", "G", "T"]
        assert set(IUPAC_TO_DNA["R"]) == {"A", "G"}
        assert set(IUPAC_TO_DNA["Y"]) == {"C", "T"}
        # Lowercase
        assert IUPAC_TO_DNA["a"] == ["a"]
        assert IUPAC_TO_DNA["n"] == ["a", "c", "g", "t"]

    def test_ignore_chars(self):
        """Test IGNORE_CHARS contains expected characters."""
        assert "-" in IGNORE_CHARS
        assert "." in IGNORE_CHARS
        assert " " in IGNORE_CHARS
        # These are safe in sequence content (only special in tag attribute syntax)
        assert "/" in IGNORE_CHARS
        assert "'" in IGNORE_CHARS
        assert '"' in IGNORE_CHARS
        assert "=" in IGNORE_CHARS
        # Only < and > are prohibited (they delimit XML tags)
        assert "<" not in IGNORE_CHARS
        assert ">" not in IGNORE_CHARS

    def test_valid_chars(self):
        """Test VALID_CHARS constant."""
        assert "A" in VALID_CHARS
        assert "C" in VALID_CHARS
        assert "G" in VALID_CHARS
        assert "T" in VALID_CHARS
        assert "a" in VALID_CHARS
        assert "c" in VALID_CHARS
        assert "g" in VALID_CHARS
        assert "t" in VALID_CHARS
        assert "-" not in VALID_CHARS
        assert "." not in VALID_CHARS

    def test_iupac_chars(self):
        """Test IUPAC_CHARS constant."""
        # All IUPAC codes should be present
        for char in "ACGTRYSWKMBDHVNacgtryswkmbdhvn":
            assert char in IUPAC_CHARS


class TestComplementFunction:
    """Test complement function."""

    def test_complement_bases(self):
        """Test complement of DNA bases."""
        assert complement("A") == "T"
        assert complement("T") == "A"
        assert complement("G") == "C"
        assert complement("C") == "G"

    def test_complement_lowercase(self):
        """Test complement preserves case."""
        assert complement("a") == "t"
        assert complement("t") == "a"
        assert complement("g") == "c"
        assert complement("c") == "g"

    def test_complement_non_dna(self):
        """Test non-DNA characters pass through."""
        assert complement("-") == "-"
        assert complement(".") == "."
        assert complement("X") == "X"

    def test_complement_iupac(self):
        """Test complement of IUPAC ambiguity codes."""
        assert complement("R") == "Y"
        assert complement("Y") == "R"
        assert complement("S") == "S"  # Self-complementary
        assert complement("W") == "W"  # Self-complementary
        assert complement("K") == "M"
        assert complement("M") == "K"
        assert complement("B") == "V"
        assert complement("V") == "B"
        assert complement("D") == "H"
        assert complement("H") == "D"
        assert complement("N") == "N"  # Self-complementary

    def test_complement_iupac_lowercase(self):
        """Test complement of lowercase IUPAC codes."""
        assert complement("r") == "y"
        assert complement("y") == "r"
        assert complement("b") == "v"
        assert complement("v") == "b"
        assert complement("n") == "n"


class TestReverseComplement:
    """Test reverse_complement function."""

    def test_reverse_complement_simple(self):
        """Test reverse complement of simple sequence."""
        assert reverse_complement("ACGT") == "ACGT"  # Palindrome
        assert reverse_complement("AAAA") == "TTTT"
        assert reverse_complement("GGGG") == "CCCC"

    def test_reverse_complement_mixed(self):
        """Test reverse complement of mixed sequence."""
        assert reverse_complement("ATGC") == "GCAT"
        assert reverse_complement("AACG") == "CGTT"

    def test_reverse_complement_case(self):
        """Test reverse complement preserves case."""
        assert reverse_complement("AcGt") == "aCgT"

    def test_reverse_complement_with_gaps(self):
        """Test reverse complement with gap characters."""
        assert reverse_complement("AC-GT") == "AC-GT"

    def test_reverse_complement_iupac(self):
        """Test reverse complement with IUPAC codes."""
        assert reverse_complement("ACB.A") == "T.VGT"
        assert reverse_complement("RN") == "NY"
        assert reverse_complement("ACGTN") == "NACGT"


class TestGetMutations:
    """Test get_mutations function."""

    def test_mutations_uppercase(self):
        """Test mutations for uppercase characters."""
        assert set(get_mutations("A")) == {"C", "G", "T"}
        assert set(get_mutations("C")) == {"A", "G", "T"}
        assert set(get_mutations("G")) == {"A", "C", "T"}
        assert set(get_mutations("T")) == {"A", "C", "G"}

    def test_mutations_lowercase(self):
        """Test mutations for lowercase characters."""
        assert set(get_mutations("a")) == {"c", "g", "t"}
        assert set(get_mutations("c")) == {"a", "g", "t"}

    def test_mutations_invalid(self):
        """Test mutations raises for non-DNA characters."""
        with pytest.raises(ValueError, match="non-DNA character"):
            get_mutations("X")
        with pytest.raises(ValueError, match="non-DNA character"):
            get_mutations("-")


class TestMolecularPositions:
    """Test get_molecular_positions function."""

    def test_simple_sequence(self):
        """Test positions in simple sequence."""
        assert get_molecular_positions("ACGT") == [0, 1, 2, 3]

    def test_sequence_with_gaps(self):
        """Test positions skip gap characters."""
        assert get_molecular_positions("AC-GT") == [0, 1, 3, 4]
        assert get_molecular_positions("A-C-G-T") == [0, 2, 4, 6]

    def test_leading_gaps(self):
        """Test positions with leading gaps."""
        assert get_molecular_positions("---ACGT") == [3, 4, 5, 6]

    def test_trailing_gaps(self):
        """Test positions with trailing gaps."""
        assert get_molecular_positions("ACGT---") == [0, 1, 2, 3]

    def test_sequence_with_dots(self):
        """Test positions skip dot characters."""
        assert get_molecular_positions("AC.GT") == [0, 1, 3, 4]

    def test_sequence_with_spaces(self):
        """Test positions skip space characters."""
        assert get_molecular_positions("AC GT") == [0, 1, 3, 4]

    def test_empty_sequence(self):
        """Test positions of empty sequence."""
        assert get_molecular_positions("") == []

    def test_only_gaps(self):
        """Test sequence with only gaps."""
        assert get_molecular_positions("---") == []

    def test_with_markers(self):
        """Test positions exclude marker tags."""
        # Marker tag positions should be excluded
        assert get_molecular_positions("AC<ins/>GT") == [0, 1, 8, 9]


class TestSeqLength:
    """Test get_seq_length function."""

    def test_simple_sequence(self):
        """Test length of simple sequence."""
        assert get_seq_length("ACGT") == 4

    def test_sequence_with_gaps(self):
        """Test length ignores gap characters."""
        assert get_seq_length("AC-GT") == 4
        assert get_seq_length("A-C-G-T") == 4
        assert get_seq_length("---ACGT---") == 4

    def test_sequence_with_dots(self):
        """Test length ignores dot characters."""
        assert get_seq_length("AC.GT") == 4
        assert get_seq_length("...ACGT") == 4

    def test_sequence_with_spaces(self):
        """Test length ignores space characters."""
        assert get_seq_length("AC GT") == 4
        assert get_seq_length("A C G T") == 4

    def test_empty_sequence(self):
        """Test length of empty sequence."""
        assert get_seq_length("") == 0

    def test_only_gaps(self):
        """Test sequence with only gaps."""
        assert get_seq_length("---") == 0


class TestLengthWithoutMarkers:
    """Test get_length_without_tags function."""

    def test_simple_sequence(self):
        """Test length of simple sequence."""
        assert get_length_without_tags("ACGT") == 4

    def test_with_markers(self):
        """Test length excludes marker tags."""
        assert get_length_without_tags("AC<region>TG</region>AA") == 6
        assert get_length_without_tags("AC<ins/>GT") == 4


class TestModuleExports:
    """Test that module exports are correct."""

    def test_bases_exported(self):
        """Test BASES is exported."""
        assert pp.BASES == ["A", "C", "G", "T"]

    def test_complement_exported(self):
        """Test COMPLEMENT is exported."""
        assert pp.COMPLEMENT["A"] == "T"

    def test_iupac_to_dna_exported(self):
        """Test IUPAC_TO_DNA is exported."""
        assert pp.IUPAC_TO_DNA["N"] == ["A", "C", "G", "T"]

    def test_ignore_chars_exported(self):
        """Test IGNORE_CHARS is exported."""
        assert "-" in pp.IGNORE_CHARS

    def test_valid_chars_exported(self):
        """Test VALID_CHARS is exported."""
        assert "A" in pp.VALID_CHARS


class TestDnaSeqIupacSupport:
    """Test DnaSeq IUPAC support and validation."""

    def test_dnaseq_valid_chars_includes_iupac(self):
        """Test DnaSeq.VALID_CHARS includes IUPAC codes."""
        from poolparty.types import DnaSeq

        for char in "ACGTRYSWKMBDHVNacgtryswkmbdhvn":
            assert char in DnaSeq.VALID_CHARS

    def test_dnaseq_rc_with_iupac(self):
        """Test DnaSeq.rc() works with IUPAC codes."""
        from poolparty.types import DnaSeq

        seq = DnaSeq.from_string("ACGTN")
        assert seq.rc().string == "NACGT"

    def test_dnaseq_rc_with_iupac_ambiguity(self):
        """Test DnaSeq.rc() handles IUPAC ambiguity codes correctly."""
        from poolparty.types import DnaSeq

        seq = DnaSeq.from_string("ACRY")
        assert seq.rc().string == "RYGT"

    def test_translate_rejects_iupac(self):
        """Test translate() raises error on IUPAC ambiguity codes."""
        pool = pp.from_seq("ATGRYATGA")
        with pytest.raises(ValueError, match="IUPAC ambiguity codes"):
            pool.translate().generate_library(num_seqs=1)

    def test_translate_accepts_acgt_only(self):
        """Test translate() works with ACGT only."""
        pool = pp.from_seq("ATGAAATGA")
        df = pool.translate().generate_library(num_seqs=1)
        assert df["seq"].iloc[0] == "MK*"

    def test_mutagenize_rejects_iupac(self):
        """Test mutagenize() raises error on IUPAC ambiguity codes."""
        pool = pp.from_seq("ACGTN")
        with pytest.raises(ValueError, match="IUPAC ambiguity codes"):
            pool.mutagenize(num_mutations=1, num_states=1).generate_library(num_seqs=1)

    def test_mutagenize_accepts_acgt_only(self):
        """Test mutagenize() works with ACGT only."""
        pool = pp.from_seq("ACGT")
        df = pool.mutagenize(num_mutations=1, num_states=3, mode="sequential").generate_library(
            num_cycles=1
        )
        assert len(df) == 3
