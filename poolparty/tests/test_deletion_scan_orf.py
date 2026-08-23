"""Focused behavior tests for codon-aware ORF deletion scans."""

import pytest

import poolparty as pp
from poolparty.utils.dna_utils import reverse_complement

CARD_MAP = {
    "codon_positions": "codon_positions",
    "wt_codons": "wt_codons",
    "start": "start",
    "end": "end",
}


@pytest.mark.parametrize(
    ("frame", "start", "wt_codon", "expected"),
    [
        (+1, 0, "AAC", "---CCGGGTTTA"),
        (+2, 1, "ACC", "A---CGGGTTTA"),
        (+3, 2, "CCC", "AA---GGGTTTA"),
        (-1, 9, "TAA", "AACCCGGGT---"),
        (-2, 8, "AAA", "AACCCGGG---A"),
        (-3, 7, "AAC", "AACCCGG---TA"),
    ],
)
def test_six_frame_hand_derived_codon_zero_anchor(frame, start, wt_codon, expected):
    with pp.Party():
        pool = pp.deletion_scan_orf(
            "AACCCGGGTTTA",
            deletion_codons=1,
            codon_positions=[0],
            frame=frame,
            mode="sequential",
            cards=CARD_MAP,
        )
        row = pool.generate_library().iloc[0]

    assert row["seq"] == expected
    assert row["codon_positions"] == (0,)
    assert row["wt_codons"] == (wt_codon,)
    assert (row["start"], row["end"]) == (start, start + 3)


@pytest.mark.parametrize("frame", [-1, -2, -3])
@pytest.mark.parametrize("deletion_codons", [1, 2])
@pytest.mark.parametrize("deletion_marker", ["-", None])
def test_negative_frame_matches_positive_on_reverse_complement(
    frame, deletion_codons, deletion_marker
):
    seq = "AACCCGGGTTTAAA"
    kwargs = dict(
        deletion_codons=deletion_codons,
        deletion_marker=deletion_marker,
        codon_positions=[1],
        mode="sequential",
        cards=CARD_MAP,
    )

    with pp.Party():
        negative = pp.deletion_scan_orf(seq, frame=frame, **kwargs)
        negative_row = negative.generate_library().iloc[0]

    with pp.Party():
        positive = pp.deletion_scan_orf(
            reverse_complement(seq), frame=-frame, **kwargs
        )
        positive_row = positive.generate_library().iloc[0]

    assert negative_row["seq"] == reverse_complement(positive_row["seq"])
    assert negative_row["codon_positions"] == positive_row["codon_positions"]
    assert negative_row["wt_codons"] == positive_row["wt_codons"]
    assert negative_row["start"] == len(seq) - positive_row["end"]
    assert negative_row["end"] == len(seq) - positive_row["start"]


def test_named_negative_orf_preserves_flanks_orphans_and_translates():
    with pp.Party():
        base = pp.from_seq("GGAATGAAATTTCCC").annotate_orf(
            "orf", extent=(2, 13), frame=-2
        )
        deleted = base.deletion_scan_orf(
            deletion_codons=1,
            codon_positions=[1],
            region="orf",
            mode="sequential",
            cards=CARD_MAP,
        )
        deleted_row = deleted.generate_library().iloc[0]
        translated_row = deleted.translate(region="orf").generate_library().iloc[0]

    assert deleted_row["seq"] == "GG<orf>AATG---TTTC</orf>CC"
    assert deleted_row["codon_positions"] == (1,)
    assert deleted_row["wt_codons"] == ("TTT",)
    assert (deleted_row["start"], deleted_row["end"]) == (4, 7)
    assert translated_row["seq"] == "KH"


def test_interval_region_supports_first_and_last_deletion_windows():
    with pp.Party():
        pool = pp.deletion_scan_orf(
            "GGAAACCCGGGTTTCC",
            1,
            region=(2, 14),
            frame=1,
            mode="sequential",
            cards=CARD_MAP,
        )
        df = pool.generate_library()

    assert df["seq"].tolist() == [
        "GG---CCCGGGTTTCC",
        "GGAAA---GGGTTTCC",
        "GGAAACCC---TTTCC",
        "GGAAACCCGGG---CC",
    ]
    assert df["codon_positions"].tolist() == [(0,), (1,), (2,), (3,)]
    assert df["wt_codons"].tolist() == [("AAA",), ("CCC",), ("GGG",), ("TTT",)]
    assert list(zip(df["start"], df["end"])) == [(0, 3), (3, 6), (6, 9), (9, 12)]


def test_two_codon_cards_are_in_coding_order_on_negative_frame():
    with pp.Party():
        pool = pp.deletion_scan_orf(
            "ATGAAACCCTTT",
            deletion_codons=2,
            codon_positions=[1],
            frame=-1,
            mode="sequential",
            cards=CARD_MAP,
        )
        row = pool.generate_library().iloc[0]

    assert row["seq"] == "ATG------TTT"
    assert row["codon_positions"] == (1, 2)
    assert row["wt_codons"] == ("GGG", "TTT")
    assert (row["start"], row["end"]) == (3, 9)


def test_explicit_codon_order_and_slice_control_state_order_and_count():
    with pp.Party():
        reordered = pp.deletion_scan_orf(
            "AAACCCGGGTTT",
            1,
            codon_positions=[2, 0],
            mode="sequential",
            cards={"codon_positions": "position"},
        )
        reordered_df = reordered.generate_library()

        sliced = pp.deletion_scan_orf(
            "AAACCCGGGTTT",
            1,
            codon_positions=slice(None, None, 2),
            mode="sequential",
        )

    assert reordered.num_states == 2
    assert reordered_df["seq"].tolist() == ["AAACCC---TTT", "---CCCGGGTTT"]
    assert reordered_df["position"].tolist() == [(2,), (0,)]
    assert sliced.num_states == 2
    assert sliced.generate_library()["seq"].tolist() == [
        "---CCCGGGTTT",
        "AAACCC---TTT",
    ]


def test_true_deletion_composes_with_upstream_states_and_downstream_translate():
    with pp.Party():
        base = pp.from_seqs(["ATGAAATTT", "ATGCCCTTT"], mode="sequential")
        deleted = base.deletion_scan_orf(
            1,
            deletion_marker=None,
            codon_positions=[1],
            mode="sequential",
            cards={"wt_codons": "deleted_wt"},
        )
        deleted_df = deleted.generate_library()
        translated_df = deleted.translate().generate_library()

    assert deleted.num_states == 2
    assert deleted.seq_length == 6
    assert deleted_df["seq"].tolist() == ["ATGTTT", "ATGTTT"]
    assert deleted_df["deleted_wt"].tolist() == [("AAA",), ("CCC",)]
    assert translated_df["seq"].tolist() == ["MF", "MF"]


def test_different_orf_deletion_widths_can_coexist_in_one_party():
    with pp.Party() as party:
        base = pp.from_seq("AAACCCGGGTTT")
        one_codon = base.deletion_scan_orf(1, mode="sequential")
        two_codons = base.deletion_scan_orf(2, mode="sequential")

        assert one_codon.num_states == 4
        assert two_codons.num_states == 3
        assert party.get_region("_del_len3").seq_length == 3
        assert party.get_region("_del_len6").seq_length == 6
        assert len(one_codon.generate_library()) == 4
        assert len(two_codons.generate_library()) == 3


@pytest.mark.parametrize("marker", ["", "--", "DEL"])
def test_deletion_marker_must_be_one_character(marker):
    with pp.Party(), pytest.raises(ValueError, match="exactly one character"):
        pp.deletion_scan_orf("AAACCC", 1, deletion_marker=marker)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"deletion_codons": 0}, "must be > 0"),
        ({"deletion_codons": 3}, "exceeds the number of complete codons"),
        (
            {"deletion_codons": 1, "codon_positions": [2]},
            "out of range",
        ),
        ({"deletion_codons": 1, "frame": 4}, "frame must be one of"),
    ],
)
def test_invalid_orf_deletion_geometry_is_rejected(kwargs, message):
    with pp.Party(), pytest.raises(ValueError, match=message):
        pp.deletion_scan_orf("AAACCC", **kwargs)


def test_named_plain_region_requires_an_explicit_frame():
    with pp.Party():
        base = pp.from_seq("AAACCC").annotate_region("coding", extent=(0, 6))
        with pytest.raises(ValueError, match="plain Region"):
            base.deletion_scan_orf(1, region="coding")


def test_short_offset_region_reports_zero_complete_codons():
    with pp.Party(), pytest.raises(ValueError, match=r"complete codons \(0\)"):
        pp.deletion_scan_orf("A", 1, frame=3)


def test_unknown_input_span_is_rejected_before_scan_construction():
    with pp.Party():
        variable = pp.from_seqs(["AAA", "AAACCC"], mode="sequential")
        with pytest.raises(ValueError, match="fixed-length"):
            variable.deletion_scan_orf(1)


@pytest.mark.parametrize("seq", ["AAANNN", "AAA---CCC"])
def test_non_acgt_or_gapped_orf_is_rejected_at_generation(seq):
    with pp.Party():
        pool = pp.deletion_scan_orf(
            seq,
            1,
            codon_positions=[0],
            mode="sequential",
        )
        with pytest.raises(ValueError, match="ungapped ACGT"):
            pool.generate_library()


def test_nested_target_annotations_are_rejected_before_region_scan():
    with pp.Party():
        base = pp.from_seq("<orf>AAA<x>CCC</x>GGG</orf>").annotate_orf(
            "orf", frame=1
        )
        pool = base.deletion_scan_orf(
            1,
            region="orf",
            codon_positions=[0],
            mode="sequential",
        )
        with pytest.raises(ValueError, match="nested region tags"):
            pool.generate_library()


def test_generic_region_scan_cards_are_not_exposed_as_orf_cards():
    with pp.Party(), pytest.raises(ValueError, match="position_index"):
        pp.deletion_scan_orf("AAACCC", 1, cards=["position_index"])


def test_universal_state_card_reports_selected_scan_state():
    with pp.Party():
        pool = pp.deletion_scan_orf(
            "AAACCCGGG",
            1,
            mode="sequential",
            cards={"state": "scan_state", "start": "start"},
        )
        df = pool.generate_library()

    assert df["scan_state"].tolist() == [0, 1, 2]
    assert df["start"].tolist() == [0, 3, 6]


def test_public_function_and_mixin_are_both_exposed():
    assert callable(pp.deletion_scan_orf)
    assert hasattr(pp.DnaPool, "deletion_scan_orf")

    with pp.Party():
        functional = pp.deletion_scan_orf(
            "AAACCC", 1, codon_positions=[1], mode="sequential"
        )
        chained = pp.from_seq("AAACCC").deletion_scan_orf(
            1, codon_positions=[1], mode="sequential"
        )

    assert functional.generate_library()["seq"].tolist() == ["AAA---"]
    assert chained.generate_library()["seq"].tolist() == ["AAA---"]


@pytest.mark.parametrize("copy_method", ["copy", "deepcopy"])
def test_copy_operations_preserve_orf_deletion_behavior_and_cards(copy_method):
    with pp.Party():
        original = pp.deletion_scan_orf(
            "AAACCC",
            1,
            mode="sequential",
            cards={"wt_codons": "wt"},
        )
        copied = getattr(original, copy_method)()
        copied_df = copied.generate_library()

    assert copied_df["seq"].tolist() == ["---CCC", "AAA---"]
    assert copied_df["wt"].tolist() == [("AAA",), ("CCC",)]
