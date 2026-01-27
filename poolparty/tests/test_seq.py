"""Tests for the Seq class."""
import pytest
import numpy as np
import poolparty as pp
from poolparty.types import Seq
from poolparty.utils.style_utils import SeqStyle


def test_seq_creation():
    """Test basic Seq creation."""
    seq = Seq.from_string('ACGT')
    assert seq.string == 'ACGT'
    assert len(seq) == 4
    assert seq.name is None
    assert len(seq.style.style_list) == 0


def test_seq_with_style():
    """Test Seq creation with style."""
    style = SeqStyle.full(4, 'red')
    seq = Seq('ACGT', style, 'test')
    assert seq.string == 'ACGT'
    assert len(seq.style.style_list) == 1
    assert seq.name == 'test'


def test_seq_slicing():
    """Test Seq slicing."""
    style = SeqStyle.full(8, 'red')
    seq = Seq('ACGTACGT', style, 'test')
    
    sliced = seq[2:6]
    assert sliced.string == 'GTAC'
    assert len(sliced) == 4
    assert sliced.name == 'test'  # Name preserved
    assert len(sliced.style.style_list) == 1  # Style sliced


def test_seq_join():
    """Test joining multiple Seq objects."""
    seq1 = Seq.from_string('ACG', name='a')
    seq2 = Seq.from_string('TGC', name='b')
    
    joined = Seq.join([seq1, seq2])
    assert joined.string == 'ACGTGC'
    assert joined.name == 'a.b'
    assert len(joined) == 6


def test_seq_join_with_separator():
    """Test joining with separator."""
    seq1 = Seq.from_string('ACG', name='a')
    seq2 = Seq.from_string('TGC', name='b')
    
    joined = Seq.join([seq1, seq2], sep='NNN')
    assert joined.string == 'ACGNNNTGC'
    assert joined.name == 'a.b'
    assert len(joined) == 9


def test_seq_join_with_styles():
    """Test that styles are properly offset when joining."""
    # Create two Seq with different styles
    style1 = SeqStyle.full(3, 'red')
    style2 = SeqStyle.full(3, 'blue')
    seq1 = Seq('ACG', style1, 'a')
    seq2 = Seq('TGC', style2, 'b')
    
    joined = Seq.join([seq1, seq2])
    assert joined.string == 'ACGTGC'
    assert len(joined.style.style_list) == 2
    
    # Check that positions are correctly offset
    red_positions = joined.style.style_list[0][1]
    blue_positions = joined.style.style_list[1][1]
    assert all(red_positions < 3)
    assert all(blue_positions >= 3)


def test_seq_insert():
    """Test inserting one Seq into another."""
    seq1 = Seq.from_string('ACGTACGT')
    seq2 = Seq.from_string('NNN')
    
    inserted = seq1.insert(4, seq2)
    assert inserted.string == 'ACGTNNNACGT'
    assert len(inserted) == 11


def test_seq_reversed():
    """Test reversing a Seq."""
    style = SeqStyle.full(4, 'red')
    seq = Seq('ACGT', style, 'test')
    
    rev = seq.reversed()
    assert rev.string == 'ACGT'  # reverse_complement of ACGT
    assert rev.name == 'test'
    # Style positions should be mirrored


def test_seq_reversed_conditional():
    """Test conditional reversal."""
    seq = Seq.from_string('ACGT', name='test')
    
    not_reversed = seq.reversed(do_reverse=False)
    assert not_reversed.string == 'ACGT'
    assert not_reversed is seq  # Should return same object


def test_seq_with_name():
    """Test updating name."""
    seq = Seq.from_string('ACGT', name='old')
    new_seq = seq.with_name('new')
    
    assert seq.name == 'old'
    assert new_seq.name == 'new'
    assert new_seq.string == 'ACGT'


def test_seq_with_style():
    """Test updating style."""
    seq = Seq.from_string('ACGT')
    new_style = SeqStyle.full(4, 'blue')
    new_seq = seq.with_style(new_style)
    
    assert len(seq.style.style_list) == 0
    assert len(new_seq.style.style_list) == 1


def test_seq_add_style():
    """Test adding style."""
    seq = Seq.from_string('ACGT')
    positions = np.array([0, 2], dtype=np.int64)
    new_seq = seq.add_style('red', positions)
    
    assert len(seq.style.style_list) == 0
    assert len(new_seq.style.style_list) == 1
    assert len(new_seq.style.style_list[0][1]) == 2


def test_seq_combine_names():
    """Test combining names from multiple Seq objects."""
    seq1 = Seq.from_string('A', name='a')
    seq2 = Seq.from_string('C', name='b')
    seq3 = Seq.from_string('G', name=None)
    
    combined = Seq.combine_names([seq1, seq2, seq3])
    assert combined == 'a.b'
    
    # All None
    combined_none = Seq.combine_names([seq3])
    assert combined_none is None


def test_seq_empty():
    """Test empty Seq creation."""
    seq = Seq.empty()
    assert seq.string == ''
    assert len(seq) == 0
    assert seq.name is None


def test_seq_immutability():
    """Test that Seq is immutable."""
    seq = Seq.from_string('ACGT', name='test')
    
    with pytest.raises(AttributeError):
        seq.string = 'TTTT'
    
    with pytest.raises(AttributeError):
        seq.name = 'new'


def test_seq_join_name_combinations():
    """Test various name combinations when joining Seq objects."""
    # Both with names
    seq1 = Seq.from_string('ACG', name='a')
    seq2 = Seq.from_string('TGC', name='b')
    joined = Seq.join([seq1, seq2])
    assert joined.name == 'a.b'
    
    # One with name, one without
    seq3 = Seq.from_string('AAA', name=None)
    joined2 = Seq.join([seq1, seq3])
    assert joined2.name == 'a'
    
    # Both without names
    seq4 = Seq.from_string('CCC', name=None)
    joined3 = Seq.join([seq3, seq4])
    assert joined3.name is None
    
    # Three with names
    seq5 = Seq.from_string('GGG', name='c')
    joined4 = Seq.join([seq1, seq2, seq5])
    assert joined4.name == 'a.b.c'


def test_seq_join_clears_prefix_suffix_names():
    """Test that clearing names from prefix/suffix only keeps middle name."""
    prefix = Seq.from_string('AAA', name='prefix')
    middle = Seq.from_string('CCC', name='middle')
    suffix = Seq.from_string('GGG', name='suffix')
    
    # Join with cleared prefix/suffix names - only middle name should remain
    joined = Seq.join([prefix.with_name(None), middle, suffix.with_name(None)])
    assert joined.name == 'middle'
    assert joined.string == 'AAACCCGGG'


def test_seq_combine_names_empty_list():
    """Test combine_names with empty list."""
    combined = Seq.combine_names([])
    assert combined is None


def test_seq_combine_names_mixed():
    """Test combine_names with mix of named and unnamed Seq objects."""
    seqs = [
        Seq.from_string('A', name='first'),
        Seq.from_string('C', name=None),
        Seq.from_string('G', name='third'),
        Seq.from_string('T', name=None),
    ]
    combined = Seq.combine_names(seqs)
    assert combined == 'first.third'


def test_seq_with_name_clear():
    """Test clearing a name."""
    seq = Seq.from_string('ACGT', name='old')
    cleared = seq.with_name(None)
    
    assert seq.name == 'old'
    assert cleared.name is None
    assert cleared.string == 'ACGT'
