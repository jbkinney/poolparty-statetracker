"""Tests for region_multiscan: sequential mode, spacing, per-insert positions, design card."""

import pytest

import poolparty as pp
from poolparty.multiscan_ops import deletion_multiscan, insertion_multiscan, replacement_multiscan
from poolparty.region_ops import region_multiscan
from poolparty.utils.scan_utils import enumerate_multiscan_combinations


# ---------------------------------------------------------------------------
# enumerate_multiscan_combinations utility
# ---------------------------------------------------------------------------


class TestEnumerateMultiscanCombinations:
    """Test the combinatorial enumeration utility."""

    def test_shared_positions_no_spacing(self):
        combos = enumerate_multiscan_combinations([0, 1, 2, 3], 2, region_length=0)
        assert len(combos) == 6  # C(4,2)

    def test_shared_positions_with_min_spacing(self):
        # region_length=2, min_spacing=1 → gap between starts = next - (prev+2) >= 1
        combos = enumerate_multiscan_combinations(
            [0, 1, 2, 3, 4, 5], 2, region_length=2, min_spacing=1
        )
        for c in combos:
            assert c[1] - (c[0] + 2) >= 1

    def test_shared_positions_with_max_spacing(self):
        combos = enumerate_multiscan_combinations(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 2, region_length=0, max_spacing=3
        )
        for c in combos:
            assert c[1] - c[0] <= 3

    def test_shared_positions_min_and_max_spacing(self):
        combos = enumerate_multiscan_combinations(
            list(range(20)), 2, region_length=3, min_spacing=2, max_spacing=5
        )
        for c in combos:
            gap = c[1] - (c[0] + 3)
            assert 2 <= gap <= 5

    def test_per_insert_positions(self):
        combos = enumerate_multiscan_combinations(
            [[0, 2, 4], [1, 3, 5]], 2, region_length=0
        )
        assert len(combos) > 0
        for c in combos:
            assert c[0] != c[1]

    def test_per_insert_positions_with_spacing(self):
        combos = enumerate_multiscan_combinations(
            [[0, 5, 10], [3, 8, 13]], 2, region_length=3, min_spacing=1
        )
        for c in combos:
            # In ordered mode (default), c[0] < c[1]
            gap = c[1] - (c[0] + 3)
            assert gap >= 1

    def test_no_valid_combos_raises(self):
        with pytest.raises(ValueError, match="No valid"):
            enumerate_multiscan_combinations([0, 1], 2, region_length=3, min_spacing=5)

    def test_single_insertion(self):
        combos = enumerate_multiscan_combinations([0, 1, 2], 1, region_length=0)
        assert len(combos) == 3
        assert all(len(c) == 1 for c in combos)

    def test_safety_cap(self):
        with pytest.raises(ValueError, match="exceeds"):
            enumerate_multiscan_combinations(list(range(100)), 10, region_length=0, max_combinations=10)

    def test_unordered_generates_both_orderings(self):
        """Unordered mode keeps both (a,b) and (b,a) as distinct assignments."""
        combos = enumerate_multiscan_combinations(
            [0, 1, 2, 3], 2, region_length=0, insertion_mode="unordered"
        )
        # P(4,2) = 12 ordered pairs with distinct positions
        assert len(combos) == 12
        # Both orderings should be present
        assert (0, 1) in combos
        assert (1, 0) in combos

    def test_ordered_is_default(self):
        """Default mode is ordered, producing sorted combos only."""
        combos = enumerate_multiscan_combinations([0, 1, 2, 3], 2, region_length=0)
        for c in combos:
            assert c == tuple(sorted(c))


# ---------------------------------------------------------------------------
# Sequential mode
# ---------------------------------------------------------------------------


class TestRegionMultiscanSequential:
    """Test sequential mode for region_multiscan."""

    def test_ordered_sequential_zero_length(self):
        """Sequential ordered mode enumerates all C(N,K) combinations."""
        with pp.Party():
            # 5 chars → 6 positions for zero-length, C(6,2)=15
            result = region_multiscan(
                "ACGTG", tag_names=["a", "b"], num_insertions=2,
                region_length=0, mode="sequential",
            )
        df = result.generate_library(num_cycles=1)
        assert len(df) == 15

    def test_ordered_sequential_nonzero_length(self):
        """Sequential mode with region_length > 0."""
        with pp.Party():
            # 6 chars, region_length=2 → 5 valid start positions, C(5,2)=10
            # But min_spacing=0 means non-overlapping: starts must differ by >= 2
            result = region_multiscan(
                "ACGTAC", tag_names=["a", "b"], num_insertions=2,
                region_length=2, mode="sequential",
            )
        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            assert "<a>" in seq
            assert "<b>" in seq
        # Each combo must have non-overlapping regions
        assert len(df) > 0

    def test_sequential_deterministic(self):
        """Same state produces same output."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGT", tag_names=["x", "y"], num_insertions=2,
                region_length=0, mode="sequential",
            )
        df1 = result.generate_library(num_cycles=1)
        df2 = result.generate_library(num_cycles=1)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_sequential_with_positions(self):
        """Sequential mode with restricted positions."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGT", tag_names=["x", "y"], num_insertions=2,
                region_length=0, positions=[0, 2, 4, 6],
                mode="sequential",
            )
        df = result.generate_library(num_cycles=1)
        assert len(df) == 6  # C(4,2)=6

    def test_sequential_single_insertion(self):
        """Sequential with 1 insertion degenerates to single scan."""
        with pp.Party():
            result = region_multiscan(
                "ACGT", tag_names=["m"], num_insertions=1,
                region_length=0, mode="sequential",
            )
        df = result.generate_library(num_cycles=1)
        assert len(df) == 5  # positions 0..4

    def test_sequential_unordered(self):
        """Unordered sequential enumerates all position assignments."""
        with pp.Party():
            # 4 chars, zero-length, 5 positions, 2 inserts
            # Ordered: C(5,2) = 10
            # Unordered: P(5,2) = 20 (all ordered pairs with distinct positions)
            result = region_multiscan(
                "ACGT", tag_names=["a", "b"], num_insertions=2,
                region_length=0, insertion_mode="unordered",
                mode="sequential",
            )
        df = result.generate_library(num_cycles=1)
        assert len(df) == 20


# ---------------------------------------------------------------------------
# Spacing constraints
# ---------------------------------------------------------------------------


class TestRegionMultiscanSpacing:
    """Test min_spacing and max_spacing."""

    def test_min_spacing_sequential(self):
        """min_spacing reduces valid combinations in sequential mode."""
        with pp.Party():
            no_spacing = region_multiscan(
                "ACGTACGTAC", tag_names=["a", "b"], num_insertions=2,
                region_length=2, mode="sequential",
            )
        df_no = no_spacing.generate_library(num_cycles=1)

        with pp.Party():
            with_spacing = region_multiscan(
                "ACGTACGTAC", tag_names=["a", "b"], num_insertions=2,
                region_length=2, min_spacing=3, mode="sequential",
            )
        df_with = with_spacing.generate_library(num_cycles=1)
        assert len(df_with) < len(df_no)

    def test_max_spacing_sequential(self):
        """max_spacing limits how far apart regions can be."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGTACGTACGTACGT", tag_names=["a", "b"], num_insertions=2,
                region_length=0, max_spacing=3, mode="sequential",
                cards=["starts"],
            )
        df = result.generate_library(num_cycles=1)
        col = _find_card_col(df, ".starts")
        for _, row in df.iterrows():
            positions = row[col]
            assert positions[1] - positions[0] <= 3

    def test_min_spacing_random(self):
        """Spacing constraints are enforced in random mode."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGTACGTACGT", tag_names=["a", "b"], num_insertions=2,
                region_length=3, min_spacing=2,
                cards=["starts"],
            )
        df = result.generate_library(num_seqs=20, seed=42)
        col = _find_card_col(df, ".starts")
        for _, row in df.iterrows():
            positions = row[col]
            gap = positions[1] - (positions[0] + 3)
            assert gap >= 2


# ---------------------------------------------------------------------------
# Per-insert positions
# ---------------------------------------------------------------------------


class TestRegionMultiscanPerInsertPositions:
    """Test per-insert positions (list of lists)."""

    def test_per_insert_sequential(self):
        """Per-insert positions with sequential mode."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGTAC", tag_names=["a", "b"], num_insertions=2,
                region_length=0, positions=[[0, 2, 4], [5, 7, 9]],
                mode="sequential",
                cards=["starts"],
            )
        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        col = _find_card_col(df, ".starts")
        for _, row in df.iterrows():
            positions = row[col]
            # At least one position should come from the specified sets
            assert any(p in [0, 2, 4, 5, 7, 9] for p in positions)

    def test_per_insert_random(self):
        """Per-insert positions with random mode."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGTACGTACGT", tag_names=["a", "b"], num_insertions=2,
                region_length=0, positions=[[0, 4, 8], [2, 6, 10]],
            )
        df = result.generate_library(num_seqs=20, seed=42)
        assert len(df) == 20

    def test_per_insert_wrong_length_raises(self):
        """Per-insert positions with wrong length raises."""
        with pytest.raises(ValueError, match="must equal num_insertions"):
            enumerate_multiscan_combinations(
                [[0, 1], [2, 3], [4, 5]], 2, region_length=0
            )

    def test_per_insert_unordered_works(self):
        """Per-insert positions with unordered mode now works (assignment-based)."""
        with pp.Party():
            # Overlapping position lists so unordered generates more combos
            result_unordered = region_multiscan(
                "ACGTACGTAC", tag_names=["a", "b"], num_insertions=2,
                region_length=0, positions=[[0, 5], [2, 7]],
                insertion_mode="unordered", mode="sequential",
            )
        df_unordered = result_unordered.generate_library(num_cycles=1)

        with pp.Party():
            result_ordered = region_multiscan(
                "ACGTACGTAC", tag_names=["a", "b"], num_insertions=2,
                region_length=0, positions=[[0, 5], [2, 7]],
                insertion_mode="ordered", mode="sequential",
            )
        df_ordered = result_ordered.generate_library(num_cycles=1)

        # Unordered should produce at least as many combos as ordered
        assert len(df_unordered) >= len(df_ordered)
        # With overlapping ranges, unordered should produce strictly more
        assert len(df_unordered) > len(df_ordered)


# ---------------------------------------------------------------------------
# Design card
# ---------------------------------------------------------------------------


def _find_card_col(df, suffix):
    """Find the design card column ending with the given suffix."""
    matches = [c for c in df.columns if c.endswith(suffix)]
    assert len(matches) == 1, f"Expected 1 column ending with '{suffix}', got {matches}"
    return matches[0]


class TestRegionMultiscanDesignCard:
    """Test the design card output."""

    def test_card_fields_present(self):
        """Design card has all expected fields."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGTAC", tag_names=["a", "b"], num_insertions=2,
                region_length=2, mode="sequential",
                cards=["starts", "ends", "names", "region_seqs", "combination_index"],
            )
        df = result.generate_library(num_cycles=1)
        suffixes = [".starts", ".names", ".region_seqs", ".combination_index"]
        for s in suffixes:
            assert any(c.endswith(s) for c in df.columns), f"Missing column with suffix {s}"

    def test_card_names_ordered(self):
        """In ordered mode, names match regions in order."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGTAC", tag_names=["first", "second"], num_insertions=2,
                region_length=0, mode="sequential",
                cards=["names"],
            )
        df = result.generate_library(num_cycles=1)
        col = _find_card_col(df, ".names")
        for _, row in df.iterrows():
            assert row[col] == ["first", "second"]

    def test_card_combination_index_sequential(self):
        """combination_index is set in sequential mode."""
        with pp.Party():
            result = region_multiscan(
                "ACGT", tag_names=["a", "b"], num_insertions=2,
                region_length=0, mode="sequential",
                cards=["combination_index"],
            )
        df = result.generate_library(num_cycles=1)
        col = _find_card_col(df, ".combination_index")
        indices = list(df[col])
        assert indices == list(range(len(df)))

    def test_card_combination_index_random(self):
        """combination_index is None in random mode."""
        with pp.Party():
            result = region_multiscan(
                "ACGTACGTAC", tag_names=["a", "b"], num_insertions=2,
                region_length=0,
                cards=["combination_index"],
            )
        df = result.generate_library(num_seqs=5, seed=42)
        col = _find_card_col(df, ".combination_index")
        for _, row in df.iterrows():
            assert row[col] is None


# ---------------------------------------------------------------------------
# Custom names for downstream consumers
# ---------------------------------------------------------------------------


class TestDownstreamCustomNames:
    """Test the names parameter for downstream multiscan ops."""

    def test_deletion_multiscan_custom_names(self):
        with pp.Party():
            result = deletion_multiscan(
                "AAAAAAAAAAAAAAAAAA", deletion_length=3, num_deletions=2,
                names=["gap1", "gap2"],
            ).named("result")
        df = result.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            assert seq.count("-") == 6

    def test_replacement_multiscan_custom_names(self):
        with pp.Party():
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(
                "AAAAAAAAAAAAAAAAAA", num_replacements=2,
                replacement_pools=ins, names=["site1", "site2"],
            ).named("result")
        df = result.generate_library(num_seqs=5, seed=42)
        for seq in df["seq"]:
            assert seq.count("G") == 6

    def test_names_wrong_length_raises(self):
        with pp.Party():
            with pytest.raises(ValueError, match="must equal"):
                deletion_multiscan(
                    "AAAAAAAAAAAAAAAAAA", deletion_length=3, num_deletions=2,
                    names=["only_one"],
                )


# ---------------------------------------------------------------------------
# Downstream consumers with sequential mode
# ---------------------------------------------------------------------------


class TestDownstreamSequentialMode:
    """Test sequential mode flowing through downstream consumers."""

    def test_deletion_multiscan_sequential(self):
        with pp.Party():
            result = deletion_multiscan(
                "AAAAAAAAAAAAAAAAAA", deletion_length=3, num_deletions=2,
                mode="sequential",
            ).named("result")
        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        for seq in df["seq"]:
            assert seq.count("-") == 6

    def test_replacement_multiscan_sequential(self):
        with pp.Party():
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(
                "AAAAAAAAAAAAAAAAAA", num_replacements=2,
                replacement_pools=ins, mode="sequential",
            ).named("result")
        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        for seq in df["seq"]:
            assert seq.count("G") == 6

    def test_insertion_multiscan_sequential(self):
        with pp.Party():
            ins = pp.from_seq("TT")
            result = insertion_multiscan(
                "AAAAAAAAAA", num_insertions=2,
                insertion_pools=ins, mode="sequential",
            ).named("result")
        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        for seq in df["seq"]:
            assert seq.count("T") == 4


# ---------------------------------------------------------------------------
# Region constraint (backward compat from earlier fix)
# ---------------------------------------------------------------------------


class TestRegionMultiscanWithRegionConstraint:
    """Test region_multiscan constrained to a tagged region."""

    def test_zero_length_markers_within_region(self):
        with pp.Party():
            bg = pp.region_scan("AAAAACCCCCGGGGG", tag_name="target", region_length=5, positions=[5], mode="sequential")
            result = region_multiscan(bg, tag_names=["m1", "m2"], num_insertions=2, region="target", region_length=0)
        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert "<m1/>" in seq or "<m2/>" in seq

    def test_surrounding_sequence_preserved(self):
        with pp.Party():
            bg = pp.region_scan("AAAAACCCCCGGGGG", tag_name="target", region_length=5, positions=[5], mode="sequential")
            result = region_multiscan(bg, tag_names=["m1"], num_insertions=1, region="target", region_length=0)
        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert seq.startswith("AAAAA")
            assert seq.endswith("GGGGG")


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestRegionMultiscanBackwardCompat:
    """Verify existing random-mode behavior is unchanged."""

    def test_basic_multiscan_random(self):
        with pp.Party():
            result = region_multiscan("AAAAAAAAAAAAAAAAAA", tag_names=["r1", "r2"], num_insertions=2, region_length=3)
        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert "<r1>" in seq
            assert "<r2>" in seq

    def test_deletion_multiscan_random(self):
        with pp.Party():
            result = deletion_multiscan("AAAAAAAAAAAAAAAAAA", deletion_length=3, num_deletions=2).named("result")
        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert seq.count("-") == 6
            assert len(seq) == 18

    def test_replacement_multiscan_random(self):
        with pp.Party():
            ins = pp.from_seq("GGG")
            result = replacement_multiscan(
                "AAAAAAAAAAAAAAAAAA", num_replacements=2, replacement_pools=ins,
            ).named("result")
        df = result.generate_library(num_seqs=10, seed=42)
        for seq in df["seq"]:
            assert seq.count("G") == 6
            assert len(seq) == 18


# ---------------------------------------------------------------------------
# Downstream consistency checks: seq <-> design card <-> combo count
# ---------------------------------------------------------------------------


def _get_card(df, suffix):
    """Get the design card column values for a given suffix."""
    cols = [c for c in df.columns if c.endswith(suffix)]
    assert len(cols) == 1, f"Expected 1 column ending with '{suffix}', got {cols}"
    return df[cols[0]]


def _find_replacement_positions(seq, original_char, replacement_char, replacement_length):
    """Find 0-based positions where replacement_char runs occur in a sequence of original_char."""
    positions = []
    i = 0
    while i <= len(seq) - replacement_length:
        if all(seq[i + j] == replacement_char for j in range(replacement_length)):
            positions.append(i)
            i += replacement_length
        else:
            i += 1
    return positions


class TestDeletionMultiscanConsistency:
    """Cross-validate deletion_multiscan output: combo count, positions, sequences."""

    def test_sequential_combo_count_with_spacing(self):
        """10-char seq, del_length=2, num_del=2, min_spacing=2.

        Valid starts: 0..8 (9 positions for length-2 deletion).
        Spacing constraint: gap = next_start - (prev_start + 2) >= 2, so next >= prev + 4.
        Valid pairs: (0,4),(0,5),(0,6),(0,7),(0,8),(1,5),(1,6),(1,7),(1,8),
                     (2,6),(2,7),(2,8),(3,7),(3,8),(4,8) = 15 combos.
        """
        with pp.Party():
            result = deletion_multiscan(
                "AAAAAAAAAA", deletion_length=2, num_deletions=2,
                min_spacing=2, mode="sequential",
                cards=["starts"],
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 15

        pos_col = _get_card(df, ".starts")
        for i, (_, row) in enumerate(df.iterrows()):
            positions = pos_col.iloc[i]
            seq = row["seq"]

            assert len(seq) == 10
            assert seq.count("-") == 4

            # Verify deletions appear at reported positions
            for pos in positions:
                assert seq[pos:pos + 2] == "--", (
                    f"Row {i}: expected '--' at pos {pos}, got '{seq[pos:pos+2]}' in '{seq}'"
                )

            # Verify spacing constraint
            gap = positions[1] - (positions[0] + 2)
            assert gap >= 2

    def test_sequential_with_custom_names_and_card(self):
        """Verify custom names appear in design card and sequences are correct."""
        with pp.Party():
            result = deletion_multiscan(
                "AAAAAAAAAAAAAAAA", deletion_length=3, num_deletions=2,
                names=["cut_A", "cut_B"], mode="sequential",
                cards=["names", "starts", "combination_index"],
            ).named("result")

        df = result.generate_library(num_cycles=1)

        names_col = _get_card(df, ".names")
        pos_col = _get_card(df, ".starts")
        idx_col = _get_card(df, ".combination_index")

        for i in range(len(df)):
            assert names_col.iloc[i] == ["cut_A", "cut_B"]
            assert idx_col.iloc[i] == i

            positions = pos_col.iloc[i]
            seq = df.iloc[i]["seq"]
            assert len(seq) == 16
            for pos in positions:
                assert seq[pos:pos + 3] == "---"

    def test_sequential_no_duplicates(self):
        """Every generated sequence is unique in sequential mode."""
        with pp.Party():
            result = deletion_multiscan(
                "AACCGGTTAA", deletion_length=2, num_deletions=2,
                mode="sequential",
            ).named("result")

        df = result.generate_library(num_cycles=1)
        seqs = list(df["seq"])
        assert len(seqs) == len(set(seqs)), "Duplicate sequences in sequential mode"


class TestInsertionMultiscanConsistency:
    """Cross-validate insertion_multiscan output: combo count, positions, sequences."""

    def test_sequential_combo_count(self):
        """8-char seq, 2 zero-length insertions of 'TT'.

        Insertion positions: 0..8 (9 positions for zero-length markers).
        C(9, 2) = 36 combinations.
        """
        with pp.Party():
            ins = pp.from_seq("TT")
            result = insertion_multiscan(
                "AAAAAAAA", num_insertions=2, insertion_pools=ins,
                mode="sequential",
                cards=["starts", "combination_index"],
            ).named("result")

        df = result.generate_library(num_cycles=1)
        assert len(df) == 36

        pos_col = _get_card(df, ".starts")
        idx_col = _get_card(df, ".combination_index")

        for i in range(len(df)):
            positions = pos_col.iloc[i]
            seq = df.iloc[i]["seq"]
            assert idx_col.iloc[i] == i

            # Original 8 A's + 4 T's inserted
            assert len(seq) == 12
            assert seq.count("T") == 4
            assert seq.count("A") == 8

            # Verify T's appear at the reported insertion positions
            # After first insertion at pos[0], the second insertion shifts
            # The positions in the card are in the original (pre-insertion) coordinate space
            # But the output sequence has both insertions applied.
            # We verify by checking that removing all T's gives back the original.
            stripped = seq.replace("T", "")
            assert stripped == "AAAAAAAA"

    def test_sequential_with_spacing_and_distinct_pools(self):
        """2 different insertion pools with min_spacing, verify content at positions."""
        with pp.Party():
            ins_a = pp.from_seq("CC")
            ins_b = pp.from_seq("GG")
            result = insertion_multiscan(
                "AAAAAAAAAA", num_insertions=2,
                insertion_pools=[ins_a, ins_b],
                min_spacing=3, mode="sequential",
                cards=["starts", "names"],
            ).named("result")

        df = result.generate_library(num_cycles=1)

        pos_col = _get_card(df, ".starts")
        names_col = _get_card(df, ".names")

        for i in range(len(df)):
            positions = pos_col.iloc[i]
            seq = df.iloc[i]["seq"]
            names = names_col.iloc[i]

            # Spacing: gap = pos[1] - pos[0] >= 3 (zero-length, so gap = distance)
            assert positions[1] - positions[0] >= 3

            # 10 A's + 2 C's + 2 G's = 14
            assert len(seq) == 14
            assert seq.count("C") == 2
            assert seq.count("G") == 2
            assert seq.count("A") == 10

            # CC always appears before GG (ordered mode, positions sorted)
            cc_pos = seq.index("CC")
            gg_pos = seq.index("GG")
            assert cc_pos < gg_pos

    def test_sequential_exhaustive_small(self):
        """Very small case: 3-char seq, 2 insertions. Verify all combos manually.

        For zero-length insertion, position N = "insert before Nth char".
        Position 3 = "insert after last char".
        """
        with pp.Party():
            ins = pp.from_seq("X")
            result = insertion_multiscan(
                "ABC", num_insertions=2, insertion_pools=ins,
                mode="sequential",
            ).named("result")

        df = result.generate_library(num_cycles=1)
        # C(4,2) = 6 positions: (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        assert len(df) == 6

        expected_seqs = {
            "XAXBC",  # (0,1): before A, before B
            "XABXC",  # (0,2): before A, before C
            "XABCX",  # (0,3): before A, after C
            "AXBXC",  # (1,2): before B, before C
            "AXBCX",  # (1,3): before B, after C
            "ABXCX",  # (2,3): before C, after C
        }
        actual_seqs = set(df["seq"])
        assert actual_seqs == expected_seqs, f"Expected {expected_seqs}, got {actual_seqs}"


class TestReplacementMultiscanConsistency:
    """Cross-validate replacement_multiscan output: combo count, positions, sequences."""

    def test_sequential_combo_count_with_spacing(self):
        """10-char seq, rep_length=2, num_rep=2, min_spacing=1.

        Valid starts: 0..8 (9 positions).
        Non-overlapping: next >= prev + 2 (region_length).
        Spacing: gap = next - (prev + 2) >= 1, so next >= prev + 3.
        Pairs: count manually via the formula.
        """
        with pp.Party():
            rep = pp.from_seq("GG")
            result = replacement_multiscan(
                "AAAAAAAAAA", num_replacements=2, replacement_pools=rep,
                min_spacing=1, mode="sequential",
                cards=["starts", "combination_index"],
            ).named("result")

        df = result.generate_library(num_cycles=1)

        # Verify independently: enumerate valid pairs
        from poolparty.utils.scan_utils import enumerate_multiscan_combinations
        expected_combos = enumerate_multiscan_combinations(
            list(range(9)), 2, region_length=2, min_spacing=1
        )
        assert len(df) == len(expected_combos)

        pos_col = _get_card(df, ".starts")
        idx_col = _get_card(df, ".combination_index")

        for i in range(len(df)):
            positions = pos_col.iloc[i]
            seq = df.iloc[i]["seq"]
            assert idx_col.iloc[i] == i

            assert len(seq) == 10
            assert seq.count("G") == 4
            assert seq.count("A") == 6

            # Verify G's appear at reported positions
            for pos in positions:
                assert seq[pos:pos + 2] == "GG", (
                    f"Row {i}: expected 'GG' at pos {pos}, got '{seq[pos:pos+2]}' in '{seq}'"
                )

            # Spacing check
            gap = positions[1] - (positions[0] + 2)
            assert gap >= 1

    def test_sequential_positions_match_sequence(self):
        """Use a mixed-char background so position verification is unambiguous."""
        with pp.Party():
            rep = pp.from_seq("XX")
            result = replacement_multiscan(
                "ABCDEFGHIJ", num_replacements=2, replacement_pools=rep,
                mode="sequential",
                cards=["starts"],
            ).named("result")

        df = result.generate_library(num_cycles=1)

        pos_col = _get_card(df, ".starts")
        original = "ABCDEFGHIJ"

        for i in range(len(df)):
            positions = pos_col.iloc[i]
            seq = df.iloc[i]["seq"]

            # Length preserved
            assert len(seq) == 10

            # X's at reported positions, original chars elsewhere
            for j in range(10):
                in_replacement = any(
                    pos <= j < pos + 2 for pos in positions
                )
                if in_replacement:
                    assert seq[j] == "X", f"Row {i} pos {j}: expected 'X', got '{seq[j]}'"
                else:
                    assert seq[j] == original[j], (
                        f"Row {i} pos {j}: expected '{original[j]}', got '{seq[j]}'"
                    )

    def test_sequential_region_constraint_consistency(self):
        """Replacement within a tagged region: positions are region-relative."""
        with pp.Party():
            bg = pp.region_scan(
                "AAAAACCCCCCCCCCGGGGG", tag_name="target",
                region_length=10, positions=[5], mode="sequential",
            )
            rep = pp.from_seq("XX")
            result = replacement_multiscan(
                bg, num_replacements=2, replacement_pools=rep,
                region="target", mode="sequential",
                cards=["starts"],
            ).named("result")

        df = result.generate_library(num_cycles=1)

        pos_col = _get_card(df, ".starts")

        for i in range(len(df)):
            seq = df.iloc[i]["seq"]
            positions = pos_col.iloc[i]

            # Flanks are preserved
            assert seq[:5] == "AAAAA"
            assert seq[-5:] == "GGGGG"

            # X count = 4 (2 replacements * length 2)
            assert seq.count("X") == 4

            # Positions should be within [0, 8] (region length 10, rep length 2)
            for pos in positions:
                assert 0 <= pos <= 8


# ---------------------------------------------------------------------------
# Per-region lengths (different-length replacement pools)
# ---------------------------------------------------------------------------


class TestPerRegionLengths:
    """Test support for varying region_length per insertion."""

    def test_enumerate_per_region_lengths(self):
        """enumerate_multiscan_combinations with per-region lengths."""
        # 10-char seq, region_lengths=[2,3], shared positions
        # Valid for region[0] (len 2): 0..8 (9 positions)
        # Valid for region[1] (len 3): 0..7 (8 positions)
        # But shared positions → internally converted to per-insert
        combos = enumerate_multiscan_combinations(
            list(range(9)), 2, region_length=[2, 3]
        )
        assert len(combos) > 0
        for c in combos:
            # In ordered mode, c[0] < c[1]. Insert 0 has length 2.
            # gap = c[1] - (c[0] + 2) >= 0 (non-overlapping)
            assert c[1] - (c[0] + 2) >= 0

    def test_enumerate_per_region_lengths_with_spacing(self):
        """Spacing uses per-region lengths for gap computation."""
        combos = enumerate_multiscan_combinations(
            list(range(20)), 2, region_length=[3, 5], min_spacing=2
        )
        for c in combos:
            gap = c[1] - (c[0] + 3)  # gap uses first region's length
            assert gap >= 2

    def test_region_multiscan_varying_lengths_sequential(self):
        """region_multiscan with varying region_length in sequential mode."""
        with pp.Party():
            result = region_multiscan(
                "AAAAAAAAAA", tag_names=["short", "long"], num_insertions=2,
                region_length=[2, 4], mode="sequential",
            )
        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        for seq in df["seq"]:
            assert "<short>" in seq
            assert "<long>" in seq

    def test_region_multiscan_varying_lengths_random(self):
        """region_multiscan with varying region_length in random mode."""
        with pp.Party():
            result = region_multiscan(
                "AAAAAAAAAAAAAAAA", tag_names=["s", "l"], num_insertions=2,
                region_length=[2, 4],
            )
        df = result.generate_library(num_seqs=10, seed=42)
        assert len(df) == 10
        for seq in df["seq"]:
            assert "<s>" in seq
            assert "<l>" in seq

    def test_varying_lengths_content_correct(self):
        """Tags encompass the correct number of characters for each region length."""
        with pp.Party():
            result = region_multiscan(
                "ABCDEFGHIJ", tag_names=["r2", "r3"], num_insertions=2,
                region_length=[2, 3], mode="sequential",
            )
        df = result.generate_library(num_cycles=1)
        for seq in df["seq"]:
            # Extract content between tags
            import re
            r2_match = re.search(r"<r2>(.+?)</r2>", seq)
            r3_match = re.search(r"<r3>(.+?)</r3>", seq)
            assert r2_match is not None
            assert r3_match is not None
            assert len(r2_match.group(1)) == 2
            assert len(r3_match.group(1)) == 3

    def test_varying_lengths_unordered_works(self):
        """Varying region_length with unordered mode works (assignment-based)."""
        with pp.Party():
            result = region_multiscan(
                "AAAAAAAAAAAAAAAA", tag_names=["s", "l"], num_insertions=2,
                region_length=[2, 4], insertion_mode="unordered",
                mode="sequential",
            )
        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        for seq in df["seq"]:
            assert "<s>" in seq
            assert "<l>" in seq

    def test_varying_lengths_unordered_both_orderings(self):
        """Unordered + varying lengths: both spatial orderings appear."""
        with pp.Party():
            result = region_multiscan(
                "AAAAAAAAAAAAAAAA", tag_names=["s", "l"], num_insertions=2,
                region_length=[2, 4], insertion_mode="unordered",
                mode="sequential",
                cards=["names"],
            )
        df = result.generate_library(num_cycles=1)
        names_col = _find_card_col(df, ".names")
        # In ordered mode, "s" is always leftmost. In unordered,
        # some combos have "l" as the leftmost.
        has_l_left = any(
            row[names_col][0] == "l" for _, row in df.iterrows()
        )
        assert has_l_left, "Unordered mode should include combos where 'l' is leftmost"

    def test_replacement_multiscan_different_lengths(self):
        """replacement_multiscan with pools of different seq_length."""
        with pp.Party():
            pool_5bp = pp.from_seq("GGGGG")   # 5bp
            pool_6bp = pp.from_seq("TTTTTT")  # 6bp
            result = replacement_multiscan(
                "A" * 30, num_replacements=2,
                replacement_pools=[pool_5bp, pool_6bp],
                mode="sequential",
            ).named("result")
        df = result.generate_library(num_cycles=1)
        assert len(df) > 0
        for seq in df["seq"]:
            assert "GGGGG" in seq
            assert "TTTTTT" in seq
            assert len(seq) == 30  # length preserved (5+6 replaced within 30)

    def test_replacement_multiscan_different_lengths_random(self):
        """replacement_multiscan with different lengths in random mode."""
        with pp.Party():
            pool_a = pp.from_seq("GGG")    # 3bp
            pool_b = pp.from_seq("TTTTTT") # 6bp
            result = replacement_multiscan(
                "A" * 20, num_replacements=2,
                replacement_pools=[pool_a, pool_b],
                mode="random", num_states=10,
            ).named("result")
        df = result.generate_library()
        assert len(df) == 10
        for seq in df["seq"]:
            assert "GGG" in seq
            assert "TTTTTT" in seq
            assert len(seq) == 20

    def test_replacement_multiscan_different_lengths_with_spacing(self):
        """Different-length replacements with spacing constraint."""
        with pp.Party():
            pool_a = pp.from_seq("GG")     # 2bp
            pool_b = pp.from_seq("TTTT")   # 4bp
            result = replacement_multiscan(
                "A" * 20, num_replacements=2,
                replacement_pools=[pool_a, pool_b],
                min_spacing=3, mode="sequential",
                cards=["starts"],
            ).named("result")
        df = result.generate_library(num_cycles=1)
        pos_col = _get_card(df, ".starts")
        for i in range(len(df)):
            positions = pos_col.iloc[i]
            # gap = positions[1] - (positions[0] + 2) >= 3
            gap = positions[1] - (positions[0] + 2)
            assert gap >= 3
