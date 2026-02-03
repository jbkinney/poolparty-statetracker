"""Tests for the Seq class."""

import numpy as np
import pytest

from poolparty.types import DnaSeq, Seq
from poolparty.utils.style_utils import SeqStyle


def test_seq_creation():
    """Test basic Seq creation."""
    seq = Seq.from_string("ACGT")
    assert seq.string == "ACGT"
    assert len(seq) == 4
    assert len(seq.style.style_list) == 0


def test_seq_with_style():
    """Test Seq creation with style."""
    style = SeqStyle.full(4, "red")
    seq = Seq("ACGT", style)
    assert seq.string == "ACGT"
    assert len(seq.style.style_list) == 1


def test_seq_slicing():
    """Test Seq slicing."""
    style = SeqStyle.full(8, "red")
    seq = Seq("ACGTACGT", style)

    sliced = seq[2:6]
    assert sliced.string == "GTAC"
    assert len(sliced) == 4
    assert len(sliced.style.style_list) == 1  # Style sliced


def test_seq_join():
    """Test joining multiple Seq objects."""
    seq1 = Seq.from_string("ACG")
    seq2 = Seq.from_string("TGC")

    joined = Seq.join([seq1, seq2])
    assert joined.string == "ACGTGC"
    assert len(joined) == 6


def test_seq_join_with_separator():
    """Test joining with separator."""
    seq1 = Seq.from_string("ACG")
    seq2 = Seq.from_string("TGC")

    joined = Seq.join([seq1, seq2], sep="NNN")
    assert joined.string == "ACGNNNTGC"
    assert len(joined) == 9


def test_seq_join_with_styles():
    """Test that styles are properly offset when joining."""
    # Create two Seq with different styles
    style1 = SeqStyle.full(3, "red")
    style2 = SeqStyle.full(3, "blue")
    seq1 = Seq("ACG", style1)
    seq2 = Seq("TGC", style2)

    joined = Seq.join([seq1, seq2])
    assert joined.string == "ACGTGC"
    assert len(joined.style.style_list) == 2

    # Check that positions are correctly offset
    red_positions = joined.style.style_list[0][1]
    blue_positions = joined.style.style_list[1][1]
    assert all(red_positions < 3)
    assert all(blue_positions >= 3)


def test_seq_insert():
    """Test inserting one Seq into another."""
    seq1 = Seq.from_string("ACGTACGT")
    seq2 = Seq.from_string("NNN")

    inserted = seq1.insert(4, seq2)
    assert inserted.string == "ACGTNNNACGT"
    assert len(inserted) == 11


def test_seq_reversed():
    """Test reversing a DnaSeq (reverse complement)."""
    style = SeqStyle.full(4, "red")
    seq = DnaSeq("ACGT", style)

    rev = seq.reversed()
    assert rev.string == "ACGT"  # reverse_complement of ACGT
    # Style positions should be mirrored


def test_seq_reversed_conditional():
    """Test conditional reversal."""
    seq = DnaSeq.from_string("ACGT")

    not_reversed = seq.reversed(do_reverse=False)
    assert not_reversed.string == "ACGT"
    assert not_reversed is seq  # Should return same object


def test_seq_with_style():
    """Test updating style."""
    seq = Seq.from_string("ACGT")
    new_style = SeqStyle.full(4, "blue")
    new_seq = seq.with_style(new_style)

    assert len(seq.style.style_list) == 0
    assert len(new_seq.style.style_list) == 1


def test_seq_add_style():
    """Test adding style."""
    seq = Seq.from_string("ACGT")
    positions = np.array([0, 2], dtype=np.int64)
    new_seq = seq.add_style("red", positions)

    assert len(seq.style.style_list) == 0
    assert len(new_seq.style.style_list) == 1
    assert len(new_seq.style.style_list[0][1]) == 2


def test_seq_empty():
    """Test empty Seq creation."""
    seq = Seq.empty()
    assert seq.string == ""
    assert len(seq) == 0


def test_seq_immutability():
    """Test that Seq is immutable."""
    seq = Seq.from_string("ACGT")

    with pytest.raises(AttributeError):
        seq.string = "TTTT"
