"""Tests for dinucleotide shuffle utility and integration with shuffle_seq / shuffle_scan."""

import re
from collections import Counter

import numpy as np
import pytest

import poolparty as pp
from poolparty.base_ops.shuffle_seq import shuffle_seq
from poolparty.scan_ops.shuffle_scan import shuffle_scan
from poolparty.utils.shuffle_utils import dinucleotide_shuffle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dinuc_counts(seq: str) -> Counter:
    """Count all dinucleotides in a sequence."""
    return Counter(seq[i : i + 2] for i in range(len(seq) - 1))


def mono_counts(seq: str) -> Counter:
    return Counter(seq)


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


# ===========================================================================
# Unit tests for dinucleotide_shuffle()
# ===========================================================================


class TestDinucleotideShuffleUnit:
    """Pure-function tests for the Euler-path dinucleotide shuffle."""

    def test_preserves_length(self):
        rng = np.random.default_rng(0)
        seq = "ACGTACGTACGT"
        result = dinucleotide_shuffle(seq, rng)
        assert len(result) == len(seq)

    def test_preserves_dinucleotide_frequencies(self):
        rng = np.random.default_rng(1)
        seq = "ACGTACGTAACC"
        for _ in range(20):
            result = dinucleotide_shuffle(seq, rng)
            assert dinuc_counts(result) == dinuc_counts(seq)

    def test_preserves_mononucleotide_frequencies(self):
        rng = np.random.default_rng(2)
        seq = "AACCGGTTAACCGGTT"
        for _ in range(20):
            result = dinucleotide_shuffle(seq, rng)
            assert mono_counts(result) == mono_counts(seq)

    def test_first_and_last_char_fixed(self):
        rng = np.random.default_rng(3)
        seq = "ACGTACGTACGT"
        for _ in range(50):
            result = dinucleotide_shuffle(seq, rng)
            assert result[0] == seq[0]
            assert result[-1] == seq[-1]

    def test_empty_string(self):
        rng = np.random.default_rng(4)
        assert dinucleotide_shuffle("", rng) == ""

    def test_single_char(self):
        rng = np.random.default_rng(5)
        assert dinucleotide_shuffle("A", rng) == "A"

    def test_two_chars(self):
        rng = np.random.default_rng(6)
        assert dinucleotide_shuffle("AC", rng) == "AC"

    def test_homopolymer(self):
        rng = np.random.default_rng(7)
        assert dinucleotide_shuffle("AAAA", rng) == "AAAA"
        assert dinucleotide_shuffle("CCCCCC", rng) == "CCCCCC"

    def test_non_dna_alphabet(self):
        rng = np.random.default_rng(8)
        seq = "ABCABCABC"
        result = dinucleotide_shuffle(seq, rng)
        assert len(result) == len(seq)
        assert mono_counts(result) == mono_counts(seq)
        assert dinuc_counts(result) == dinuc_counts(seq)

    def test_produces_different_sequences(self):
        """With enough trials, the shuffle should produce at least one different sequence."""
        rng = np.random.default_rng(9)
        # Use a sequence with enough dinucleotide branching to allow multiple paths
        seq = "AACGACGTACGAACCGT"
        results = {dinucleotide_shuffle(seq, rng) for _ in range(100)}
        assert len(results) > 1

    def test_reproducibility_with_same_seed(self):
        seq = "ACGTACGTACGT"
        r1 = dinucleotide_shuffle(seq, np.random.default_rng(42))
        r2 = dinucleotide_shuffle(seq, np.random.default_rng(42))
        assert r1 == r2


# ===========================================================================
# Integration tests for shuffle_seq with shuffle_type="dinuc"
# ===========================================================================


class TestShuffleSeqDinuc:
    """Integration tests for pp.shuffle_seq(shuffle_type='dinuc')."""

    def test_basic_dinuc_shuffle(self):
        with pp.Party():
            pool = shuffle_seq("ACGTACGTACGT", shuffle_type="dinuc").named("s")
        df = pool.generate_library(num_seqs=10, seed=42)
        original = "ACGTACGTACGT"
        orig_dinucs = dinuc_counts(original)
        for seq in df["seq"]:
            assert len(seq) == len(original)
            assert dinuc_counts(seq) == orig_dinucs

    def test_dinuc_with_named_region(self):
        with pp.Party():
            pool = shuffle_seq(
                "AA<r>ACGTACGT</r>TT", region="r", shuffle_type="dinuc"
            ).named("s")
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            clean = strip_tags(seq)
            assert clean[:2] == "AA"
            assert clean[-2:] == "TT"
            middle = clean[2:-2]
            assert dinuc_counts(middle) == dinuc_counts("ACGTACGT")

    def test_dinuc_with_interval_region(self):
        with pp.Party():
            pool = shuffle_seq("ACGTACGTACGT", region=[2, 10], shuffle_type="dinuc").named("s")
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert len(seq) == 12
            assert seq[:2] == "AC"
            assert seq[10:] == "GT"
            region = seq[2:10]
            assert dinuc_counts(region) == dinuc_counts("GTACGTAC")

    def test_dinuc_tags_preserved(self):
        """Tags are preserved when shuffling a tagged sequence."""
        with pp.Party():
            pool = shuffle_seq(
                "AA<cre>ACGTACGT</cre>TT", shuffle_type="dinuc"
            ).named("s")
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert "<cre>" in seq
            assert "</cre>" in seq

    def test_dinuc_interval_crossing_tag(self):
        """Interval region crossing a tag boundary works correctly."""
        with pp.Party():
            pool = shuffle_seq(
                "AA<cre>ACGTACGT</cre>TT", region=[1, 11], shuffle_type="dinuc"
            ).named("s")
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert "<cre>" in seq
            assert "</cre>" in seq
            clean = strip_tags(seq)
            assert len(clean) == 12

    def test_dinuc_mixin_method(self):
        """Test using the Pool.shuffle_seq mixin method."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT").shuffle_seq(shuffle_type="dinuc").named("s")
        df = pool.generate_library(num_seqs=5, seed=42)
        orig_dinucs = dinuc_counts("ACGTACGTACGT")
        for seq in df["seq"]:
            assert dinuc_counts(seq) == orig_dinucs

    def test_mono_still_default(self):
        """Verify that default shuffle_type='mono' is unchanged."""
        with pp.Party():
            pool = shuffle_seq("ACGTACGT").named("s")
        df = pool.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert mono_counts(seq) == mono_counts("ACGTACGT")

    def test_design_card_has_permutation(self):
        """Design card should contain 'permutation' key for dinuc shuffle."""
        with pp.Party():
            pool = shuffle_seq("ACGTACGT", shuffle_type="dinuc")
            rng = np.random.default_rng(42)
            output_seq, card = pool.operation.compute(
                [pp.types.Seq.from_string("ACGTACGT")], rng
            )
        assert "permutation" in card
        assert len(card["permutation"]) == 8


# ===========================================================================
# Integration tests for shuffle_scan with shuffle_type="dinuc"
# ===========================================================================


class TestShuffleScanDinuc:
    """Integration tests for pp.shuffle_scan(shuffle_type='dinuc')."""

    def test_basic_dinuc_scan(self):
        with pp.Party():
            pool = shuffle_scan(
                "ACGTACGTACGT", shuffle_length=6, shuffle_type="dinuc", mode="sequential"
            ).named("s")
        df = pool.generate_library(num_cycles=1, seed=42)
        assert len(df) > 0
        for seq in df["seq"]:
            assert len(seq) == 12

    def test_dinuc_scan_preserves_composition(self):
        seq_str = "ACGTACGTACGT"
        with pp.Party():
            pool = shuffle_scan(
                seq_str,
                shuffle_length=6,
                shuffle_type="dinuc",
                mode="sequential",
                shuffles_per_position=5,
            ).named("s")
        df = pool.generate_library(num_cycles=1, seed=42)
        for seq in df["seq"]:
            assert mono_counts(seq) == mono_counts(seq_str)

    def test_dinuc_scan_mixin(self):
        """Test using the Pool.shuffle_scan mixin method."""
        with pp.Party():
            pool = (
                pp.from_seq("ACGTACGTACGT")
                .shuffle_scan(shuffle_length=4, shuffle_type="dinuc")
                .named("s")
            )
        df = pool.generate_library(num_seqs=5, seed=42)
        assert len(df) == 5
        for seq in df["seq"]:
            assert len(seq) == 12

    def test_mono_scan_still_default(self):
        """Verify that default shuffle_type='mono' is unchanged for scan."""
        with pp.Party():
            pool = shuffle_scan(
                "ACGTACGTAC", shuffle_length=4, mode="sequential"
            ).named("s")
        df = pool.generate_library(num_cycles=1, seed=42)
        assert len(df) > 0
