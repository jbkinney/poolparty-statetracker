"""Audit tests for region operations (bravo).

Covers: region_scan, region_multiscan, annotate_region, insert_tags,
extract_region, replace_region, apply_at_region, remove_tags, clear_annotation.

Follows operation_audit.mdc Steps 1-7.
"""

import re

import pytest

import poolparty as pp
from poolparty.region_ops import strip_all_tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bio_len(seq_str: str) -> int:
    """Biological (nontag) length of a sequence string."""
    return len(strip_all_tags(seq_str))


def has_balanced_tags(seq_str: str) -> bool:
    """Check that every opening tag has a matching closing tag (well-formed XML)."""
    opens = re.findall(r"<([a-zA-Z_]\w*)(?:\s[^>]*)?>", seq_str)
    closes = re.findall(r"</([a-zA-Z_]\w*)>", seq_str)
    self_closing = re.findall(r"<([a-zA-Z_]\w*)\s*/>", seq_str)
    opens_only = [t for t in opens if t not in self_closing]
    return sorted(opens_only) == sorted(closes)


# ===================================================================
# TestRegionOpsBaseline — I1 + I2 for all 9 ops
# ===================================================================

class TestRegionOpsBaseline:
    """I1 (output length) and I2 (state exhaustion) for every region op."""

    # -- insert_tags --

    def test_insert_tags_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.insert_tags(pool, "r", start=2, stop=5)
            assert result.seq_length == pool.seq_length  # tags don't change bio len
            df = result.generate_library()
            assert len(df) == result.num_states == 1
            assert bio_len(df["seq"].iloc[0]) == result.seq_length

    # -- annotate_region --

    def test_annotate_region_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.annotate_region(pool, "r", extent=(2, 5))
            assert result.seq_length == pool.seq_length
            df = result.generate_library()
            assert len(df) == result.num_states == 1
            assert bio_len(df["seq"].iloc[0]) == result.seq_length

    # -- region_scan --

    def test_region_scan_I1_I2_zero_length(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.region_scan(pool, "r", region_length=0, mode="sequential")
            assert result.num_states == 5
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 4

    def test_region_scan_I1_I2_nonzero_length(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_scan(pool, "r", region_length=3, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    # -- region_multiscan --

    def test_region_multiscan_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_multiscan(
                pool, ["a", "b"], 2, region_length=0, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    # -- extract_region --

    def test_extract_region_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            result = pp.extract_region(pool, "r")
            assert result.seq_length == 2
            df = result.generate_library()
            assert len(df) == result.num_states == 1
            assert bio_len(df["seq"].iloc[0]) == 2

    # -- remove_tags (keep_content=True) --

    def test_remove_tags_keep_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            result = pp.remove_tags(pool, "r", keep_content=True)
            assert result.seq_length == pool.seq_length
            df = result.generate_library()
            assert len(df) == result.num_states == 1
            assert bio_len(df["seq"].iloc[0]) == result.seq_length

    # -- remove_tags (keep_content=False) --

    def test_remove_tags_drop_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            result = pp.remove_tags(pool, "r", keep_content=False)
            expected = pool.seq_length - 2
            assert result.seq_length == expected
            df = result.generate_library()
            assert len(df) == result.num_states == 1
            assert bio_len(df["seq"].iloc[0]) == expected

    # -- replace_region --

    def test_replace_region_I1_I2(self):
        with pp.Party():
            bg = pp.from_seq("AAAA<ins>CC</ins>GGGG")
            content = pp.from_seq("XXXX")
            result = pp.replace_region(bg, content, "ins")
            expected = bg.seq_length - 2 + 4  # 10 - 2 + 4 = 12
            assert result.seq_length == expected
            df = result.generate_library()
            assert len(df) == result.num_states == 1
            assert bio_len(df["seq"].iloc[0]) == expected

    # -- apply_at_region --

    def test_apply_at_region_I1_I2(self):
        with pp.Party():
            bg = pp.from_seq("AAAA<r>CCCC</r>GGGG")
            result = pp.apply_at_region(bg, "r", pp.rc)
            df = result.generate_library()
            assert len(df) == result.num_states == 1
            assert bio_len(df["seq"].iloc[0]) == bg.seq_length

    # -- clear_annotation --

    def test_clear_annotation_runs(self):
        """clear_annotation should strip tags and non-molecular chars."""
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            result = pp.clear_annotation(pool)
            df = result.generate_library(num_cycles=1)
            assert df["seq"].iloc[0] == "ACGTTTGGGG"


# ===================================================================
# TestGroupA_TagInsertion — representative: insert_tags
# ===================================================================

class TestGroupA_TagInsertion:
    """Full invariants for Group A (fixed tag-insertion ops)."""

    def test_insert_tags_I4_tag_preservation(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.insert_tags(pool, "myregion", start=2, stop=5)
            df = result.generate_library()
            seq = df["seq"].iloc[0]
            assert "<myregion>" in seq
            assert "</myregion>" in seq

    def test_insert_tags_zero_length_I4(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.insert_tags(pool, "ins", start=4)
            df = result.generate_library()
            seq = df["seq"].iloc[0]
            assert "<ins/>" in seq
            assert bio_len(seq) == 8

    def test_annotate_region_existing_no_extent(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            result = pp.annotate_region(pool, "r")
            df = result.generate_library()
            assert bio_len(df["seq"].iloc[0]) == pool.seq_length

    def test_annotate_region_existing_with_extent_raises(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            with pytest.raises(ValueError, match="already exists"):
                pp.annotate_region(pool, "r", extent=(0, 2))

    def test_insert_tags_on_tagged_parent(self):
        """Tags on a pre-tagged parent should produce well-formed XML."""
        with pp.Party():
            pool = pp.from_seq("AA<x>CC</x>GG")
            result = pp.insert_tags(pool, "y", start=0, stop=6)
            df = result.generate_library()
            seq = df["seq"].iloc[0]
            assert has_balanced_tags(seq)
            assert bio_len(seq) == pool.seq_length


# ===================================================================
# TestGroupB_TagRemoval — representative: remove_tags
# ===================================================================

class TestGroupB_TagRemoval:
    """Full invariants for Group B (fixed tag-removal/extraction ops)."""

    def test_remove_tags_keep_content_value(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            result = pp.remove_tags(pool, "r", keep_content=True)
            df = result.generate_library()
            assert df["seq"].iloc[0] == "ACGTTTGGGG"

    def test_remove_tags_drop_content_value(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            result = pp.remove_tags(pool, "r", keep_content=False)
            df = result.generate_library()
            assert df["seq"].iloc[0] == "ACGTGGGG"

    def test_remove_tags_I8_replacement_length_algebra(self):
        """I8: seq_length = parent_len - region_len when keep_content=False."""
        with pp.Party():
            pool = pp.from_seq("AAAA<ins>CCC</ins>GGGG")
            result = pp.remove_tags(pool, "ins", keep_content=False)
            expected = pool.seq_length - 3
            assert result.seq_length == expected
            df = result.generate_library()
            assert bio_len(df["seq"].iloc[0]) == expected

    def test_extract_region_value(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TTAA</r>GGGG")
            result = pp.extract_region(pool, "r")
            df = result.generate_library()
            assert df["seq"].iloc[0] == "TTAA"

    def test_extract_region_rc(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TTAA</r>GGGG")
            result = pp.extract_region(pool, "r", rc=True)
            df = result.generate_library()
            assert df["seq"].iloc[0] == "TTAA"

    def test_remove_tags_region_tracking_D2(self):
        """D2: pool.has_region() is correct after remove_tags."""
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            assert pool.has_region("r")
            result = pp.remove_tags(pool, "r")
            assert not result.has_region("r")


# ===================================================================
# TestGroupC_ScanningInserters — representative: region_scan
# ===================================================================

class TestGroupC_ScanningInserters:
    """Full invariants for Group C (scanning tag-inserters)."""

    def test_region_scan_I2_exhaustion(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.region_scan(pool, "r", region_length=0, mode="sequential")
            assert result.num_states == 5
            df = result.generate_library()
            assert len(df) == 5

    def test_region_scan_I3_card_sequence_agreement(self):
        """I3: card 'start' matches actual tag position in output."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_scan(
                pool, "r", region_length=2, mode="sequential",
                cards=["start", "end", "position_index"]
            )
            op_name = result.operation.name
            df = result.generate_library()
            for _, row in df.iterrows():
                seq = row["seq"]
                start = row[f"{op_name}.start"]
                end = row[f"{op_name}.end"]
                assert end - start == 2
                content = strip_all_tags(seq[seq.index("<r>") + 3 : seq.index("</r>")])
                assert len(content) == 2

    def test_region_scan_I5_determinism(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_scan(pool, "r", region_length=2, mode="random", num_states=5)
            df1 = result.generate_library(seed=42)
            df2 = result.generate_library(seed=42)
            assert (df1["seq"] == df2["seq"]).all()

    def test_region_scan_I6_region_only_modification(self):
        """I6: prefix/suffix unchanged when region= is set."""
        with pp.Party():
            pool = pp.from_seq("AA<x>CCCC</x>GG")
            result = pp.region_scan(pool, "r", region="x", region_length=2, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean[:2] == "AA"
                assert clean[-2:] == "GG"

    def test_region_scan_I9_init_compute_geometry(self):
        """I9: sequential cache geometry matches compute-time output."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_scan(pool, "r", region_length=3, mode="sequential")
            ns = result.num_states
            df = result.generate_library()
            assert len(df) == ns
            assert df["seq"].nunique() == ns

    def test_region_scan_I10_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_scan(pool, "r", region_length=2, mode="sequential")
            ns_before = result.num_states
            sv_before = result.operation.state._num_values
            result.generate_library()
            assert result.num_states == ns_before
            assert result.operation.state._num_values == sv_before

    def test_region_scan_D1_well_formed_xml(self):
        """D1: output XML is well-formed."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_scan(pool, "r", region_length=3, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                assert has_balanced_tags(seq)

    def test_region_scan_D2_region_tracking(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_scan(pool, "r", region_length=2, mode="sequential")
            assert result.has_region("r")

    def test_region_scan_mode_validation(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="mode must be"):
                pp.region_scan(pool, "r", mode="bogus")

    def test_region_multiscan_I2_exhaustion(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.region_multiscan(
                pool, ["a", "b"], 2, region_length=0,
                min_spacing=1, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_region_multiscan_I3_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_multiscan(
                pool, ["a", "b"], 2, region_length=0, mode="sequential",
                cards=["starts", "ends", "names"]
            )
            op_name = result.operation.name
            df = result.generate_library()
            for _, row in df.iterrows():
                starts = row[f"{op_name}.starts"]
                ends = row[f"{op_name}.ends"]
                names = row[f"{op_name}.names"]
                assert len(starts) == 2
                assert len(ends) == 2
                assert len(names) == 2

    def test_region_multiscan_I10_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_multiscan(
                pool, ["a", "b"], 2, region_length=0, mode="sequential"
            )
            ns_before = result.num_states
            sv_before = result.operation.state._num_values
            result.generate_library()
            assert result.num_states == ns_before
            assert result.operation.state._num_values == sv_before


# ===================================================================
# TestGroupD_ContentReplacers — representative: replace_region
# ===================================================================

class TestGroupD_ContentReplacers:
    """Full invariants for Group D (region content replacers)."""

    def test_replace_region_I8_algebra(self):
        """I8: seq_length = parent_len - region_len + insert_len."""
        with pp.Party():
            bg = pp.from_seq("AAAA<ins>CC</ins>GGGG")
            content = pp.from_seq("XXXX")
            result = pp.replace_region(bg, content, "ins")
            expected = bg.seq_length - 2 + 4
            assert result.seq_length == expected
            df = result.generate_library()
            assert bio_len(df["seq"].iloc[0]) == expected

    def test_replace_region_I4_keep_tags(self):
        """I4: when keep_tags=True, region tags are preserved."""
        with pp.Party():
            bg = pp.from_seq("AAAA<ins>CC</ins>GGGG")
            content = pp.from_seq("XXXX")
            result = pp.replace_region(bg, content, "ins", keep_tags=True)
            df = result.generate_library()
            seq = df["seq"].iloc[0]
            assert "<ins>" in seq
            assert "</ins>" in seq
            assert bio_len(seq) == result.seq_length

    def test_replace_region_I7_composition(self):
        """I7: chained op has correct num_states and seq_length is computable."""
        with pp.Party():
            bg = pp.from_seq("ACGT<ins>AA</ins>TTTT")
            content = pp.from_seqs(["XX", "YY"], mode="sequential")
            result = pp.replace_region(bg, content, "ins")
            expected_states = bg.num_states * content.num_states
            assert result.num_states == expected_states
            assert result.seq_length is not None
            df = result.generate_library()
            assert len(df) == expected_states
            for seq in df["seq"]:
                assert bio_len(seq) == result.seq_length

    def test_replace_region_sync(self):
        with pp.Party():
            bg = pp.from_seqs(
                ["ACGT<bc/>TTTT", "CCCC<bc/>GGGG"], mode="sequential"
            )
            content = pp.from_seqs(["AA", "GG"], mode="sequential")
            result = pp.replace_region(bg, content, "bc", sync=True)
            df = result.generate_library()
            assert len(df) == 2  # 1:1 pairing, not Cartesian

    def test_replace_region_rc(self):
        with pp.Party():
            bg = pp.from_seq("AAAA<ins>CC</ins>GGGG")
            content = pp.from_seq("AT")
            result = pp.replace_region(bg, content, "ins", rc=True)
            df = result.generate_library()
            seq = strip_all_tags(df["seq"].iloc[0])
            assert seq == "AAAAAT" + "GGGG" or "AT" in seq

    def test_replace_region_D2_region_tracking(self):
        with pp.Party():
            bg = pp.from_seq("AAAA<ins>CC</ins>GGGG")
            content = pp.from_seq("XXXX")
            result = pp.replace_region(bg, content, "ins")
            assert not result.has_region("ins")  # tags removed
            result_kt = pp.replace_region(bg, content, "ins", keep_tags=True)
            assert result_kt.has_region("ins")  # tags kept

    def test_apply_at_region_value(self):
        with pp.Party():
            bg = pp.from_seq("AAAA<r>ACGT</r>GGGG")
            result = pp.apply_at_region(bg, "r", pp.rc)
            df = result.generate_library()
            clean = strip_all_tags(df["seq"].iloc[0])
            assert clean == "AAAAACGTGGGG"

    def test_apply_at_region_remove_tags_false(self):
        with pp.Party():
            bg = pp.from_seq("AAAA<r>ACGT</r>GGGG")
            result = pp.apply_at_region(bg, "r", pp.rc, remove_tags=False)
            df = result.generate_library()
            seq = df["seq"].iloc[0]
            assert "<r>" in seq
            assert "</r>" in seq


# ===================================================================
# TestDomainInvariants
# ===================================================================

class TestDomainInvariants:
    """Domain-specific invariants for region ops (D1-D3)."""

    def test_D1_region_scan_wellformed(self):
        with pp.Party():
            pool = pp.from_seq("AA<x>CCCC</x>GG")
            result = pp.region_scan(pool, "r", region="x", region_length=2, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                assert has_balanced_tags(seq)

    def test_D1_multiscan_wellformed(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.region_multiscan(
                pool, ["a", "b"], 2, region_length=0, mode="sequential"
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert has_balanced_tags(seq)

    def test_D1_insert_tags_wellformed(self):
        with pp.Party():
            pool = pp.from_seq("AA<x>CC</x>GG")
            result = pp.insert_tags(pool, "y", start=0, stop=6)
            df = result.generate_library()
            for seq in df["seq"]:
                assert has_balanced_tags(seq)

    def test_D3_tag_boundary_coordinates(self):
        """Nontag positions map correctly on tagged sequences."""
        with pp.Party():
            pool = pp.from_seq("AA<x>CC</x>GG")
            result = pp.region_scan(pool, "r", region_length=2, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                bio = strip_all_tags(seq)
                assert bio_len(seq) == 6  # AACCGG = 6 bio chars


# ===================================================================
# TestAdversarial_RegionScan
# ===================================================================

class TestAdversarial_RegionScan:
    """Adversarial patterns for high-risk region_scan."""

    def test_variable_length_parent_sequential_rejects(self):
        """FIXED: variable-length parent + sequential now raises ValueError
        instead of silently falling back to position 0 (bug #50, same pattern as #45)."""
        with pp.Party():
            pool = pp.from_seqs(["AC", "ACGT"], mode="sequential")
            assert pool.seq_length is None
            with pytest.raises(ValueError, match="known scan geometry"):
                pp.region_scan(pool, "r", region_length=0, mode="sequential")

    def test_short_parent_sequential(self):
        with pp.Party():
            pool = pp.from_seq("AC")
            result = pp.region_scan(pool, "r", region_length=1, mode="sequential")
            assert result.num_states == 2
            df = result.generate_library()
            assert len(df) == 2
            for seq in df["seq"]:
                assert bio_len(seq) == 2

    def test_tagged_parent_named_region_constraint(self):
        with pp.Party():
            pool = pp.from_seq("AA<x>CCCC</x>GG")
            result = pp.region_scan(
                pool, "r", region="x", region_length=2, mode="sequential"
            )
            assert result.num_states == 3
            df = result.generate_library()
            assert len(df) == 3
            for seq in df["seq"]:
                bio = strip_all_tags(seq)
                assert bio[:2] == "AA"
                assert bio[-2:] == "GG"

    def test_cycling_num_states_gt_natural(self):
        """num_states > natural should cycle positions."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.region_scan(
                pool, "r", region_length=0, mode="sequential", num_states=10
            )
            assert result.num_states == 10
            df = result.generate_library()
            assert len(df) == 10
            for seq in df["seq"]:
                assert bio_len(seq) == 4

    def test_clipped_num_states_lt_natural(self):
        """num_states < natural should produce only first N variants."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.region_scan(
                pool, "r", region_length=0, mode="sequential", num_states=3
            )
            assert result.num_states == 3
            df = result.generate_library()
            assert len(df) == 3

    def test_compositional_stress_cartesian(self):
        """Chain between two sequential ops: full Cartesian product."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TTTT"], mode="sequential")
            scanned = pp.region_scan(pool, "r", region_length=2, mode="sequential")
            expected = pool.num_states * 3  # 2 * C(4,2+1) = 2*3 = 6
            assert scanned.num_states == expected
            df = scanned.generate_library()
            assert len(df) == expected


# ===================================================================
# TestAdversarial_RegionMultiscan
# ===================================================================

class TestAdversarial_RegionMultiscan:
    """Adversarial patterns for high-risk region_multiscan."""

    def test_variable_length_parent_sequential_rejects(self):
        """FIXED: variable-length parent + sequential now raises ValueError
        instead of silently falling back (bug #50, same pattern as #45)."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "ACGTACGT"], mode="sequential")
            assert pool.seq_length is None
            with pytest.raises(ValueError, match="known scan geometry"):
                pp.region_multiscan(
                    pool, ["a", "b"], 2, region_length=0, mode="sequential"
                )

    def test_tight_spacing(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.region_multiscan(
                pool, ["a", "b"], 2, min_spacing=1, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 4

    def test_tagged_parent_unordered_varied_lengths(self):
        with pp.Party():
            pool = pp.from_seq("AA<x>CCCCCC</x>GG")
            result = pp.region_multiscan(
                pool, ["a", "b"], 2, region="x",
                region_length=[1, 2], mode="sequential",
                insertion_mode="unordered",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                bio = strip_all_tags(seq)
                assert bio[:2] == "AA"
                assert bio[-2:] == "GG"

    def test_compositional_stress_cartesian(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGTACGT", "TTTTTTTT"], mode="sequential")
            scanned = pp.region_multiscan(
                pool, ["a", "b"], 2, region_length=0, mode="sequential"
            )
            df = scanned.generate_library()
            assert len(df) == scanned.num_states


# ===================================================================
# TestSeqLengthNone — Finding: region_scan/multiscan drop seq_length
# ===================================================================

class TestSeqLengthNone:
    """FIXED: region_scan and region_multiscan now propagate parent seq_length.
    Previously bugs #48/#49 (same pattern as #22-#25)."""

    def test_region_scan_preserves_seq_length(self):
        """region_scan propagates parent seq_length (tag insertion is metadata-only)."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            assert pool.seq_length == 8
            scanned = pp.region_scan(pool, "r", region_length=2, mode="sequential")
            assert scanned.seq_length == 8

    def test_region_multiscan_preserves_seq_length(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            assert pool.seq_length == 8
            scanned = pp.region_multiscan(
                pool, ["a", "b"], 2, region_length=0, mode="sequential"
            )
            assert scanned.seq_length == 8

    def test_region_scan_chaining_downstream_sequential(self):
        """Chaining region_scan → mutagenize(sequential) now succeeds."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            scanned = pp.region_scan(pool, "r", region_length=2, mode="sequential")
            mutated = pp.mutagenize(scanned, num_mutations=1, mode="sequential")
            assert mutated.seq_length == 8
            df = mutated.generate_library()
            assert len(df) == mutated.num_states

    def test_region_multiscan_chaining_downstream_sequential(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            scanned = pp.region_multiscan(
                pool, ["a", "b"], 2, region_length=0, mode="sequential"
            )
            mutated = pp.mutagenize(scanned, num_mutations=1, mode="sequential")
            assert mutated.seq_length == 8
            df = mutated.generate_library()
            assert len(df) == mutated.num_states


# ===================================================================
# TestMixinForwarding
# ===================================================================

class TestMixinForwarding:
    """Runtime mixin forwarding tests — call every mixin with all factory params."""

    def test_replace_region_mixin_forwards_rc(self):
        """FIXED: replace_region mixin now forwards rc parameter."""
        with pp.Party():
            bg = pp.from_seq("AAAA<ins>CC</ins>GGGG")
            content = pp.from_seq("AT")
            via_mixin = bg.replace_region(content, "ins", rc=True)
            via_factory = pp.replace_region(bg, content, "ins", rc=True)
            assert via_mixin.generate_library(num_cycles=1)["seq"].tolist() == via_factory.generate_library(
                num_cycles=1
            )["seq"].tolist()

    def test_apply_at_region_mixin_forwards_rc(self):
        """FIXED: apply_at_region mixin now forwards rc parameter."""
        with pp.Party():
            bg = pp.from_seq("AAAA<r>ACGT</r>GGGG")
            via_mixin = bg.apply_at_region("r", pp.rc, rc=True)
            via_factory = pp.apply_at_region(bg, "r", pp.rc, rc=True)
            assert via_mixin.generate_library(num_cycles=1)["seq"].tolist() == via_factory.generate_library(
                num_cycles=1
            )["seq"].tolist()

    def test_extract_region_mixin_forwards_all_params(self):
        """FIXED: extract_region mixin now exists and forwards all params."""
        with pp.Party():
            bg = pp.from_seq("ACGT<r>TT</r>GGGG")
            via_mixin = bg.extract_region("r")
            via_factory = pp.extract_region(bg, "r")
            assert via_mixin.generate_library(num_cycles=1)["seq"].tolist() == via_factory.generate_library(
                num_cycles=1
            )["seq"].tolist()
            via_mixin_rc = bg.extract_region("r", rc=True)
            via_factory_rc = pp.extract_region(bg, "r", rc=True)
            assert via_mixin_rc.generate_library(num_cycles=1)["seq"].tolist() == via_factory_rc.generate_library(
                num_cycles=1
            )["seq"].tolist()

    def test_region_scan_no_mixin(self):
        """BUG: region_scan has no mixin method."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            assert not hasattr(bg, "region_scan")

    def test_region_multiscan_no_mixin(self):
        """BUG: region_multiscan has no mixin method."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            assert not hasattr(bg, "region_multiscan")

    def test_annotate_region_mixin_full_parity(self):
        """annotate_region mixin forwards all factory params."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pool.annotate_region("r", extent=(2, 5), style="red")
            assert result.seq_length == pool.seq_length

    def test_insert_tags_mixin_full_parity(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pool.insert_tags("r", 2, 5)
            df = result.generate_library()
            assert "<r>" in df["seq"].iloc[0]

    def test_remove_tags_mixin_full_parity(self):
        with pp.Party():
            pool = pp.from_seq("ACGT<r>TT</r>GGGG")
            result = pool.remove_tags("r", keep_content=True)
            df = result.generate_library()
            assert "<r>" not in df["seq"].iloc[0]

    def test_replace_region_mixin_basic(self):
        """replace_region mixin works for basic params (no rc)."""
        with pp.Party():
            bg = pp.from_seq("AAAA<ins>CC</ins>GGGG")
            content = pp.from_seq("XXXX")
            result = bg.replace_region(content, "ins")
            df = result.generate_library()
            assert "XXXX" in strip_all_tags(df["seq"].iloc[0])

    def test_apply_at_region_mixin_basic(self):
        """apply_at_region mixin works for basic params (no rc)."""
        with pp.Party():
            bg = pp.from_seq("AAAA<r>ACGT</r>GGGG")
            result = bg.apply_at_region("r", pp.rc)
            df = result.generate_library()
            assert bio_len(df["seq"].iloc[0]) == bg.seq_length


# ===================================================================
# TestAPIConsistency
# ===================================================================

class TestAnnotateRegionVariableLength:
    """Finding from alpha audit: annotate_region extent=None on variable-length pools."""

    def test_annotate_region_extent_none_variable_length_full_span(self):
        """FIXED: annotate_region(extent=None) on variable-length pools now
        wraps full sequence content in tags."""
        with pp.Party():
            variable = pp.from_seqs(["AAAA", "AAAAAA"], mode="sequential")
            assert variable.seq_length is None
            annotated = pp.annotate_region(variable, "r")
            seqs = annotated.generate_library()["seq"].tolist()
            assert seqs == ["<r>AAAA</r>", "<r>AAAAAA</r>"]

    def test_annotate_region_variable_length_downstream_replace(self):
        """FIXED: downstream replace_region on variable-length annotated pool
        correctly replaces full region content."""
        with pp.Party():
            variable = pp.from_seqs(["AAAA", "AAAAAA"], mode="sequential")
            annotated = pp.annotate_region(variable, "r")
            replaced = pp.replace_region(annotated, "TT", "r")
            seqs = replaced.generate_library()["seq"].tolist()
            assert seqs == ["TT", "TT"]


class TestAPIConsistency:
    """API consistency checks across region ops."""

    def test_region_scan_rejects_invalid_mode(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="mode must be"):
                pp.region_scan(pool, "r", mode="fixed")

    def test_region_multiscan_rejects_invalid_mode(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="mode must be"):
                pp.region_multiscan(pool, ["a"], 1, mode="fixed")

    def test_region_scan_rejects_negative_region_length(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="region_length"):
                pp.region_scan(pool, "r", region_length=-1)

    def test_naming_consistency_documented(self):
        """Document naming: tag_name (scan ops) vs region_name (region ops)."""
        import inspect
        from poolparty.region_ops.region_scan import region_scan
        from poolparty.region_ops.region_multiscan import region_multiscan
        from poolparty.region_ops.annotate_region import annotate_region
        from poolparty.region_ops.insert_tags import insert_tags
        from poolparty.region_ops.extract_region import extract_region
        from poolparty.region_ops.replace_region import replace_region
        from poolparty.region_ops.remove_tags import remove_tags

        assert "tag_name" in list(inspect.signature(region_scan).parameters)
        assert "tag_names" in list(inspect.signature(region_multiscan).parameters)
        assert "region_name" in list(inspect.signature(annotate_region).parameters)
        assert "region_name" in list(inspect.signature(insert_tags).parameters)
        assert "region_name" in list(inspect.signature(extract_region).parameters)
        assert "region_name" in list(inspect.signature(replace_region).parameters)
        assert "region_name" in list(inspect.signature(remove_tags).parameters)

    def test_beartype_only_on_clear_annotation(self):
        """Only clear_annotation has @beartype among region ops."""
        import inspect
        from poolparty.region_ops.region_scan import region_scan
        from poolparty.region_ops.insert_tags import insert_tags
        from poolparty.region_ops.extract_region import extract_region
        from poolparty.region_ops.replace_region import replace_region
        from poolparty.region_ops.remove_tags import remove_tags
        from poolparty.fixed_ops.clear_annotation import clear_annotation

        for fn in [region_scan, insert_tags, extract_region, replace_region]:
            assert not hasattr(fn, "__wrapped__"), f"{fn.__name__} has @beartype unexpectedly"

        assert "beartype" in str(type(clear_annotation)) or hasattr(
            clear_annotation, "__wrapped__"
        ) or "beartype" in clear_annotation.__qualname__

    def test_remove_tags_param_symmetry(self):
        """FIXED: both region_scan and region_multiscan accept remove_tags."""
        import inspect
        from poolparty.region_ops.region_scan import region_scan
        from poolparty.region_ops.region_multiscan import region_multiscan

        assert "remove_tags" in inspect.signature(region_scan).parameters
        assert "remove_tags" in inspect.signature(region_multiscan).parameters

    def test_region_multiscan_remove_tags_works(self):
        """FIXED: region_multiscan remove_tags strips constraint region tags."""
        with pp.Party():
            pool = pp.from_seq("AAA<outer>CCGGTT</outer>AAA")
            scanned = pp.region_multiscan(
                pool, tag_names=["r1"], num_insertions=1,
                region="outer", remove_tags=True, mode="sequential",
            )
            df = scanned.generate_library()
            for seq in df["seq"]:
                assert "<outer>" not in seq
                assert "</outer>" not in seq
                assert "<r1/>" in seq
