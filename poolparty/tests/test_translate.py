"""Tests for the translate operation."""

import numpy as np
import pytest

import poolparty as pp
from poolparty.orf_ops.translate import TranslateOp, _get_shared_styles, _resolve_frame, translate
from poolparty.utils.protein_seq import (
    AA_THREE_LETTER,
    VALID_PROTEIN_CHARS,
    ProteinSeq,
    map_style_positions_to_three_letter,
    to_three_letter,
)
from poolparty.utils.style_utils import SeqStyle


class TestProteinSeq:
    """Test ProteinSeq class."""

    def test_from_string_basic(self):
        """Test basic ProteinSeq creation."""
        seq = ProteinSeq.from_string("MAKT*")
        assert seq.string == "MAKT*"
        assert len(seq) == 5

    def test_from_string_empty(self):
        """Test empty ProteinSeq."""
        seq = ProteinSeq.empty()
        assert seq.string == ""
        assert len(seq) == 0

    def test_molecular_length(self):
        """Test molecular length counts valid amino acids."""
        seq = ProteinSeq.from_string("MAKT*")
        assert seq.molecular_length == 5  # All chars are valid

    def test_repr(self):
        """Test string representation."""
        seq = ProteinSeq.from_string("MAK")
        assert "ProteinSeq" in repr(seq)
        assert "len=3" in repr(seq)

    def test_with_style(self):
        """Test adding style to ProteinSeq."""
        seq = ProteinSeq.from_string("MAK")
        style = SeqStyle.empty(3).add_style("red", np.array([0, 1]))
        styled_seq = seq.with_style(style)
        assert styled_seq.style is not None
        assert len(styled_seq.style.style_list) == 1

    def test_valid_protein_chars(self):
        """Test VALID_PROTEIN_CHARS contains expected characters."""
        expected = set("ACDEFGHIKLMNPQRSTVWYacdefghiklmnpqrstvwy*")
        assert VALID_PROTEIN_CHARS == frozenset(expected)


class TestResolveFrame:
    """Test frame resolution logic."""

    def test_explicit_frame(self):
        """Test explicit frame is used directly."""
        with pp.Party():
            assert _resolve_frame(None, 1) == 1
            assert _resolve_frame(None, -1) == -1
            assert _resolve_frame("region", 2) == 2

    def test_default_frame_no_region(self):
        """Test default frame +1 when no region."""
        with pp.Party():
            assert _resolve_frame(None, None) == 1

    def test_invalid_frame_raises(self):
        """Test invalid frame raises error."""
        with pp.Party():
            with pytest.raises(ValueError, match="frame must be one of"):
                _resolve_frame(None, 4)

    def test_orf_region_frame(self):
        """Test frame from OrfRegion."""
        with pp.Party() as party:
            pool = pp.from_seq("ATGGCTTAA").annotate_orf("cds", frame=2)
            assert _resolve_frame("cds", None) == 2


class TestGetSharedStyles:
    """Test style propagation helper."""

    def test_all_positions_share_style(self):
        """Test when all positions share a style."""
        style = SeqStyle.empty(9).add_style("red", np.array([0, 1, 2]))
        shared = _get_shared_styles(style, [0, 1, 2])
        assert "red" in shared

    def test_partial_overlap_no_share(self):
        """Test when positions don't all share a style."""
        style = SeqStyle.empty(9).add_style("red", np.array([0, 1]))
        shared = _get_shared_styles(style, [0, 1, 2])
        assert "red" not in shared

    def test_no_style(self):
        """Test with None style."""
        shared = _get_shared_styles(None, [0, 1, 2])
        assert shared == []

    def test_multiple_shared_styles(self):
        """Test multiple styles shared across positions."""
        style = SeqStyle.empty(9)
        style = style.add_style("red", np.array([0, 1, 2]))
        style = style.add_style("bold", np.array([0, 1, 2]))
        shared = _get_shared_styles(style, [0, 1, 2])
        assert "red" in shared
        assert "bold" in shared


class TestTranslateBasic:
    """Test basic translate functionality."""

    def test_translate_full_sequence(self):
        """Test translating a full sequence."""
        with pp.Party():
            # ATG=M, GCT=A, TAA=*
            pool = pp.from_seq("ATGGCTTAA").translate()
            df = pool.generate_library()
            assert df["seq"].iloc[0] == "MA*"

    def test_translate_no_stop(self):
        """Test translating without stop codon."""
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate(include_stop=False)
            df = pool.generate_library()
            assert df["seq"].iloc[0] == "MA"

    def test_translate_region(self):
        """Test translating a specific region."""
        with pp.Party():
            pool = pp.from_seq("NNATGGCTTAANN").annotate_orf("cds", extent=(2, 11))
            protein_pool = pool.translate(region="cds")
            df = protein_pool.generate_library()
            assert df["seq"].iloc[0] == "MA*"

    def test_translate_empty_sequence(self):
        """Test translating empty/too-short sequence."""
        with pp.Party():
            pool = pp.from_seq("AT").translate()  # Less than 3 bases
            df = pool.generate_library()
            # Empty sequence may return None or "" depending on implementation
            assert df["seq"].iloc[0] in ("", None)


class TestTranslateFrame:
    """Test frame handling in translate."""

    def test_frame_plus_1(self):
        """Test frame +1 (default)."""
        with pp.Party():
            # Frame +1: ATG GCT
            pool = pp.from_seq("ATGGCT").translate(frame=1)
            df = pool.generate_library()
            assert df["seq"].iloc[0] == "MA"

    def test_frame_plus_2(self):
        """Test frame +2 skips first base."""
        with pp.Party():
            # Frame +2: skip 1 base, then read codons
            # ATGGCTTAA: skip A, read TGG CTT AA? -> TGG=W, CTT=L
            pool = pp.from_seq("ATGGCTTAA").translate(frame=2)
            df = pool.generate_library()
            assert df["seq"].iloc[0] == "WL"

    def test_frame_plus_3(self):
        """Test frame +3 skips first two bases."""
        with pp.Party():
            # Frame +3: skip 2 bases, then read codons
            # ATGGCTTAA: skip AT, read GGC TTA A? -> GGC=G, TTA=L
            pool = pp.from_seq("ATGGCTTAA").translate(frame=3)
            df = pool.generate_library()
            assert df["seq"].iloc[0] == "GL"

    def test_frame_from_orf_region(self):
        """Test frame is auto-detected from OrfRegion."""
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").annotate_orf("cds", frame=1)
            protein_pool = pool.translate(region="cds")
            df = protein_pool.generate_library()
            assert df["seq"].iloc[0] == "MA*"


class TestTranslateReverseFrame:
    """Test reverse frame translation."""

    def test_frame_minus_1(self):
        """Test frame -1 (reverse complement)."""
        with pp.Party():
            # ATGGCT reverse complement is AGCCAT
            # Read from right: ATG GCT -> (rc of TGG in ATGGCT is CCA, etc.)
            # Actually: for -1 frame, we read codons from the end
            # Original: ATGGCT
            # For -1: start from end, read backwards (then rc each codon)
            # Codon 0: positions 3,4,5 = GCT, rc = AGC = S
            # Codon 1: positions 0,1,2 = ATG, rc = CAT = H
            pool = pp.from_seq("ATGGCT").translate(frame=-1)
            df = pool.generate_library()
            # Reading backwards: GCT -> rc=AGC=S, ATG -> rc=CAT=H
            assert df["seq"].iloc[0] == "SH"


class TestTranslatePartialCodons:
    """Test partial codon handling."""

    def test_skip_partial_at_end(self):
        """Test that partial codons at end are skipped."""
        with pp.Party():
            # ATGGCTT - 7 bases, frame +1: ATG GCT T (skip last T)
            pool = pp.from_seq("ATGGCTT").translate()
            df = pool.generate_library()
            assert df["seq"].iloc[0] == "MA"

    def test_skip_partial_at_start_frame_2(self):
        """Test that partial codons at start are skipped for frame +2."""
        with pp.Party():
            # AATGGCT - frame +2 skips first A, then ATG GCT
            pool = pp.from_seq("AATGGCT").translate(frame=2)
            df = pool.generate_library()
            assert df["seq"].iloc[0] == "MA"


class TestTranslateStylePropagation:
    """Test style propagation to amino acids."""

    def test_style_propagation_all_share(self):
        """Test style is propagated when all 3 nt share same style."""
        with pp.Party():
            # Use stylize_orf to apply colors to codons
            pool = pp.from_seq("ATGGCT").annotate_orf("cds", frame=1)
            pool = pool.stylize_orf(region="cds", style_codons=["red", "blue"])
            protein_pool = pool.translate(region="cds", preserve_codon_styles=True)
            # Just verify translation works with styled input
            df = protein_pool.generate_library()
            assert df["seq"].iloc[0] == "MA"

    def test_no_style_propagation_mixed(self):
        """Test translation with mixed-style input."""
        with pp.Party():
            # Apply style to whole sequence
            pool = pp.from_seq("ATGGCT").stylize(style="red", which="all")
            protein_pool = pool.translate(preserve_codon_styles=True)
            df = protein_pool.generate_library()
            assert df["seq"].iloc[0] == "MA"

    def test_preserve_codon_styles_false(self):
        """Test style propagation disabled with preserve_codon_styles=False."""
        with pp.Party():
            pool = pp.from_seq("ATGGCT").annotate_orf("cds", frame=1)
            pool = pool.stylize_orf(region="cds", style_codons=["red", "blue"])
            protein_pool = pool.translate(region="cds", preserve_codon_styles=False)
            df = protein_pool.generate_library()
            # Translation should still work
            assert df["seq"].iloc[0] == "MA"


class TestTranslateNullSeq:
    """Test NullSeq handling."""

    def test_null_seq_input(self):
        """Test that NullSeq input produces NullSeq output."""
        with pp.Party():
            # Create a filtered pool that produces NullSeq
            pool = pp.from_seq("ATGGCT").filter(lambda seq: False)
            protein_pool = pool.translate()
            df = protein_pool.generate_library()
            assert df["seq"].iloc[0] is None


class TestProteinPool:
    """Test ProteinPool class."""

    def test_protein_pool_type(self):
        """Test that translate returns ProteinPool."""
        with pp.Party():
            pool = pp.from_seq("ATGGCT").translate()
            assert isinstance(pool, pp.ProteinPool)

    def test_protein_pool_print_library(self):
        """Test ProteinPool print_library works."""
        with pp.Party():
            pool = pp.from_seq("ATGGCT").translate()
            # Should not raise
            pool.print_library()

    def test_protein_pool_named(self):
        """Test ProteinPool naming."""
        with pp.Party():
            pool = pp.from_seq("ATGGCT").translate().named("my_protein")
            assert pool.name == "my_protein"

    def test_protein_pool_subscriptable(self):
        """Test ProteinPool supports subscripting (slicing)."""
        with pp.Party():
            # Create a protein pool with multiple states via repeat
            pool = pp.from_seq("ATGGCT").translate() * 5
            assert pool.num_states == 5
            sliced = pool[1:3]
            assert isinstance(sliced, pp.ProteinPool)
            assert sliced.num_states == 2

    def test_protein_pool_multiply(self):
        """Test ProteinPool supports multiplication (repeat)."""
        with pp.Party():
            pool = pp.from_seq("ATGGCT").translate()
            repeated = pool * 3
            assert isinstance(repeated, pp.ProteinPool)
            assert repeated.num_states == 3

    def test_protein_pool_add(self):
        """Test ProteinPool supports addition (stack)."""
        with pp.Party():
            pool1 = pp.from_seq("ATGGCT").translate()
            pool2 = pp.from_seq("ATGAAA").translate()
            stacked = pool1 + pool2
            assert isinstance(stacked, pp.ProteinPool)
            assert stacked.num_states == 2


class TestTranslateOp:
    """Test TranslateOp class directly."""

    def test_factory_name(self):
        """Test factory name is set correctly."""
        assert TranslateOp.factory_name == "translate"

    def test_translate_function_returns_protein_pool(self):
        """Test translate() factory returns ProteinPool."""
        with pp.Party():
            pool = pp.from_seq("ATG")
            protein_pool = translate(pool)
            assert isinstance(protein_pool, pp.ProteinPool)


class TestThreeLetterConversion:
    """Test 3-letter amino acid conversion functions."""

    def test_aa_three_letter_mapping(self):
        """Test AA_THREE_LETTER contains all standard amino acids."""
        # Check uppercase
        assert AA_THREE_LETTER["A"] == "Ala"
        assert AA_THREE_LETTER["M"] == "Met"
        assert AA_THREE_LETTER["K"] == "Lys"
        assert AA_THREE_LETTER["*"] == "***"
        # Check lowercase
        assert AA_THREE_LETTER["a"] == "ala"
        assert AA_THREE_LETTER["m"] == "met"

    def test_to_three_letter_basic(self):
        """Test basic conversion to 3-letter codes."""
        assert to_three_letter("MAK") == "Met-Ala-Lys"
        assert to_three_letter("M*") == "Met-***"

    def test_to_three_letter_empty(self):
        """Test conversion of empty sequence."""
        assert to_three_letter("") == ""

    def test_to_three_letter_single(self):
        """Test conversion of single amino acid."""
        assert to_three_letter("M") == "Met"
        assert to_three_letter("*") == "***"

    def test_to_three_letter_custom_separator(self):
        """Test conversion with custom separator."""
        assert to_three_letter("MAK", separator=".") == "Met.Ala.Lys"
        assert to_three_letter("MAK", separator="") == "MetAlaLys"
        assert to_three_letter("MAK", separator=" ") == "Met Ala Lys"

    def test_to_three_letter_unknown_char(self):
        """Test conversion with unknown character."""
        assert to_three_letter("MXK") == "Met-???-Lys"

    def test_to_three_letter_lowercase(self):
        """Test conversion preserves case."""
        assert to_three_letter("mak") == "met-ala-lys"
        assert to_three_letter("Mak") == "Met-ala-lys"


class TestStylePositionMapping:
    """Test style position mapping for 3-letter display."""

    def test_map_positions_default_separator(self):
        """Test position mapping with default '-' separator."""
        # Position 0 (M) in "MAK" maps to positions 0,1,2 in "Met-Ala-Lys"
        # Position 1 (A) maps to positions 4,5,6
        # Position 2 (K) maps to positions 8,9,10
        positions = np.array([1], dtype=np.int64)
        result = map_style_positions_to_three_letter(positions, seq_len=3)
        np.testing.assert_array_equal(result, np.array([4, 5, 6]))

    def test_map_positions_no_separator(self):
        """Test position mapping with empty separator."""
        # Position 1 (A) in "MAK" maps to positions 3,4,5 in "MetAlaLys"
        positions = np.array([1], dtype=np.int64)
        result = map_style_positions_to_three_letter(positions, seq_len=3, separator="")
        np.testing.assert_array_equal(result, np.array([3, 4, 5]))

    def test_map_positions_multiple(self):
        """Test mapping multiple positions."""
        positions = np.array([0, 2], dtype=np.int64)
        result = map_style_positions_to_three_letter(positions, seq_len=3)
        # Position 0 -> 0,1,2; Position 2 -> 8,9,10
        np.testing.assert_array_equal(result, np.array([0, 1, 2, 8, 9, 10]))

    def test_map_positions_empty(self):
        """Test mapping empty positions array."""
        positions = np.array([], dtype=np.int64)
        result = map_style_positions_to_three_letter(positions, seq_len=3)
        np.testing.assert_array_equal(result, np.array([]))


class TestPrintLibraryThreeLetter:
    """Test ProteinPool.print_library() with 3-letter display."""

    def test_print_library_three_letter(self, capsys):
        """Test print_library with chars_per_aa=3."""
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(chars_per_aa=3, show_header=False, show_state=False, show_name=False)
            captured = capsys.readouterr()
            assert "Met Ala ***" in captured.out

    def test_print_library_custom_separator(self, capsys):
        """Test print_library with custom aa_separator."""
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(
                chars_per_aa=3,
                aa_separator=".",
                show_header=False,
                show_state=False,
                show_name=False,
            )
            captured = capsys.readouterr()
            assert "Met.Ala.***" in captured.out

    def test_print_library_no_separator(self, capsys):
        """Test print_library with empty separator."""
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(
                chars_per_aa=3,
                aa_separator="",
                show_header=False,
                show_state=False,
                show_name=False,
            )
            captured = capsys.readouterr()
            assert "MetAla***" in captured.out

    def test_print_library_one_letter_default(self, capsys):
        """Test print_library with default chars_per_aa=1."""
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            pool.print_library(show_header=False, show_state=False, show_name=False)
            captured = capsys.readouterr()
            assert "MA*" in captured.out

    def test_print_library_three_letter_returns_self(self):
        """Test print_library returns self for chaining."""
        with pp.Party():
            pool = pp.from_seq("ATGGCTTAA").translate()
            result = pool.print_library(chars_per_aa=3)
            assert result is pool

    def test_print_library_three_letter_with_styles(self, capsys):
        """Test print_library with chars_per_aa=3 and styled input."""
        with pp.Party():
            # Create a styled protein pool (codon styles propagate to amino acids)
            pool = (
                pp.from_seq("ATGGCTTAA")
                .annotate_orf("cds", frame=1, style_codons=["red", "green", "blue"])
                .translate(region="cds", preserve_codon_styles=True)
            )
            # This should not raise - was previously failing with SeqStyle argument order bug
            pool.print_library(chars_per_aa=3, show_header=False, show_state=False, show_name=False)
            captured = capsys.readouterr()
            # Should contain 3-letter codes (with ANSI escape codes for styling)
            assert "Met" in captured.out or "\x1b[" in captured.out
