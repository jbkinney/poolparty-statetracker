"""Tests for the flip operation."""

import re

import pytest

import poolparty as pp
from poolparty.base_ops.flip import flip
from poolparty.utils.dna_utils import reverse_complement


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


# ===========================================================================
# Sequential mode
# ===========================================================================


class TestFlipSequential:

    def test_default_two_states(self):
        """Sequential mode produces exactly 2 states: forward then rc."""
        seq = "ACGTAA"
        with pp.Party():
            pool = flip(seq, mode="sequential").named("s")
            df = pool.generate_library(num_cycles=1)

        assert len(df) == 2
        assert df["seq"].iloc[0] == seq
        assert df["seq"].iloc[1] == reverse_complement(seq)

    def test_num_states_override_cycles(self):
        """num_states=4 cycles: fwd, rc, fwd, rc."""
        seq = "ACGTAA"
        rc_seq = reverse_complement(seq)
        with pp.Party():
            pool = flip(seq, mode="sequential", num_states=4).named("s")
            df = pool.generate_library(num_cycles=1)

        assert len(df) == 4
        assert list(df["seq"]) == [seq, rc_seq, seq, rc_seq]

    def test_num_states_one(self):
        """num_states=1 in sequential mode gives only forward."""
        seq = "ACGTAA"
        with pp.Party():
            pool = flip(seq, mode="sequential", num_states=1).named("s")
            df = pool.generate_library(num_cycles=1)

        assert len(df) == 1
        assert df["seq"].iloc[0] == seq


# ===========================================================================
# Random mode
# ===========================================================================


class TestFlipRandom:

    def test_random_produces_mix(self):
        """Random mode with rc_prob=0.5 produces both orientations."""
        seq = "ACGTAA"
        rc_seq = reverse_complement(seq)
        with pp.Party():
            pool = flip(seq, mode="random", num_states=100).named("s")
            df = pool.generate_library(num_cycles=1)

        seqs = set(df["seq"])
        assert seq in seqs
        assert rc_seq in seqs

    def test_rc_prob_zero_always_forward(self):
        """rc_prob=0.0 always produces forward."""
        seq = "ACGTAA"
        with pp.Party():
            pool = flip(seq, mode="random", rc_prob=0.0, num_states=50).named("s")
            df = pool.generate_library(num_cycles=1)

        assert all(df["seq"] == seq)

    def test_rc_prob_one_always_rc(self):
        """rc_prob=1.0 always produces rc."""
        seq = "ACGTAA"
        rc_seq = reverse_complement(seq)
        with pp.Party():
            pool = flip(seq, mode="random", rc_prob=1.0, num_states=50).named("s")
            df = pool.generate_library(num_cycles=1)

        assert all(df["seq"] == rc_seq)

    def test_num_states_controls_row_count(self):
        """Random mode with num_states=N generates N rows."""
        with pp.Party():
            pool = flip("ACGT", mode="random", num_states=7).named("s")
            df = pool.generate_library(num_cycles=1)

        assert len(df) == 7

    def test_pure_random_one_row(self):
        """Random mode with num_states=None generates 1 row per cycle."""
        with pp.Party():
            pool = flip("ACGT", mode="random").named("s")
            df = pool.generate_library(num_cycles=1)

        assert len(df) == 1


# ===========================================================================
# Design card
# ===========================================================================


class TestFlipDesignCard:

    def test_sequential_cards(self):
        """Design card has 'flip' column with correct values."""
        seq = "ACGTAA"
        with pp.Party():
            pool = flip(seq, mode="sequential", cards=["flip"]).named("s")
            op_name = pool.operation.name
            df = pool.generate_library(num_cycles=1)

        col = f"{op_name}.flip"
        assert col in df.columns
        assert df[col].iloc[0] == "forward"
        assert df[col].iloc[1] == "rc"

    def test_random_card_values(self):
        """Random mode cards reflect actual orientation chosen."""
        seq = "ACGTAA"
        rc_seq = reverse_complement(seq)
        with pp.Party():
            pool = flip(
                seq, mode="random", num_states=50, cards=["flip"]
            ).named("s")
            op_name = pool.operation.name
            df = pool.generate_library(num_cycles=1)

        col = f"{op_name}.flip"
        for _, row in df.iterrows():
            if row["seq"] == seq:
                assert row[col] == "forward"
            else:
                assert row["seq"] == rc_seq
                assert row[col] == "rc"


# ===========================================================================
# Region support
# ===========================================================================


class TestFlipRegion:

    def test_named_region(self):
        """Flip with named region rc's only the region content."""
        with pp.Party():
            pool = pp.from_seq("AAA<cre>ACGT</cre>TTT")
            flipped = flip(pool, region="cre", mode="sequential").named("s")
            df = flipped.generate_library(num_cycles=1)

        assert len(df) == 2
        fwd_seq = strip_tags(df["seq"].iloc[0])
        rc_seq = strip_tags(df["seq"].iloc[1])
        assert fwd_seq == "AAAACGTTTT"
        assert rc_seq == "AAAACGTTTT"  # ACGT rc is ACGT (palindrome)

    def test_named_region_non_palindrome(self):
        """Flip with named region on non-palindromic content."""
        with pp.Party():
            pool = pp.from_seq("AAA<cre>ACGTAA</cre>TTT")
            flipped = flip(pool, region="cre", mode="sequential").named("s")
            df = flipped.generate_library(num_cycles=1)

        assert len(df) == 2
        fwd_seq = strip_tags(df["seq"].iloc[0])
        rc_seq = strip_tags(df["seq"].iloc[1])
        assert fwd_seq == "AAAACGTAATTT"
        assert rc_seq == "AAATTACGTTTT"

    def test_interval_region(self):
        """Flip with interval region."""
        seq = "AAAACGTAATTT"  # region [3,9] = ACGTAA
        with pp.Party():
            flipped = flip(seq, region=[3, 9], mode="sequential").named("s")
            df = flipped.generate_library(num_cycles=1)

        assert len(df) == 2
        assert df["seq"].iloc[0] == seq
        assert df["seq"].iloc[1] == "AAATTACGTTTT"


# ===========================================================================
# Style
# ===========================================================================


class TestFlipStyle:

    def test_style_applied_only_on_rc(self):
        """Style is applied when reverse complementing, not on forward."""
        seq = "ACGT"
        with pp.Party():
            pool = flip(seq, mode="sequential", style="red").named("s")
            df = pool.generate_library(num_cycles=1)

        assert len(df) == 2
        assert df["seq"].iloc[0] == seq
        assert df["seq"].iloc[1] == reverse_complement(seq)


# ===========================================================================
# Tags
# ===========================================================================


class TestFlipTags:

    def test_tags_stripped_on_rc(self):
        """Tags are stripped when reverse complementing (same as rc op)."""
        with pp.Party():
            pool = pp.from_seq("A<m>CG</m>T")
            flipped = flip(pool, mode="sequential").named("s")
            df = flipped.generate_library(num_cycles=1)

        assert df["seq"].iloc[1] == "ACGT"  # rc of ACGT is ACGT (palindrome)

    def test_tags_stripped_non_palindrome(self):
        """Tags stripped on rc path for non-palindromic sequence."""
        with pp.Party():
            pool = pp.from_seq("AA<m>CG</m>TT")
            flipped = flip(pool, mode="sequential").named("s")
            df = flipped.generate_library(num_cycles=1)

        fwd = df["seq"].iloc[0]
        rc_out = df["seq"].iloc[1]
        assert "<m>" in fwd
        assert "<m>" not in rc_out
        assert rc_out == reverse_complement("AACGTT")


# ===========================================================================
# Mixin method
# ===========================================================================


class TestFlipMixin:

    def test_mixin_method(self):
        """pool.flip() works via fluent API."""
        seq = "ACGTAA"
        with pp.Party():
            pool = pp.from_seq(seq).flip(mode="sequential").named("s")
            df = pool.generate_library(num_cycles=1)

        assert len(df) == 2
        assert df["seq"].iloc[0] == seq
        assert df["seq"].iloc[1] == reverse_complement(seq)

    def test_mixin_with_rc_prob(self):
        """pool.flip(rc_prob=1.0) always rc's."""
        seq = "ACGTAA"
        rc_seq = reverse_complement(seq)
        with pp.Party():
            pool = pp.from_seq(seq).flip(
                mode="random", rc_prob=1.0, num_states=10
            ).named("s")
            df = pool.generate_library(num_cycles=1)

        assert all(df["seq"] == rc_seq)


# ===========================================================================
# Validation
# ===========================================================================


class TestFlipValidation:

    def test_rc_prob_negative_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="rc_prob must be between 0 and 1"):
                flip("ACGT", rc_prob=-0.1)

    def test_rc_prob_above_one_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="rc_prob must be between 0 and 1"):
                flip("ACGT", rc_prob=1.5)

    def test_fixed_mode_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="mode='fixed' is not supported"):
                flip("ACGT", mode="fixed")
