"""Focused behavior tests for codon-aware ORF insertion scans."""

import pytest

import poolparty as pp
from poolparty.utils.dna_utils import reverse_complement

SPLICE_CARDS = {
    "codon_slot": "codon_slot",
    "start": "start",
    "end": "end",
}
OVERWRITE_CARDS = {
    "codon_positions": "codon_positions",
    "wt_codons": "wt_codons",
    "start": "start",
    "end": "end",
}


@pytest.mark.parametrize(
    ("frame", "start", "expected"),
    [
        (+1, 0, "TAGAACCCGGGTTTA"),
        (+2, 1, "ATAGACCCGGGTTTA"),
        (+3, 2, "AATAGCCCGGGTTTA"),
        (-1, 12, "AACCCGGGTTTACTA"),
        (-2, 11, "AACCCGGGTTTCTAA"),
        (-3, 10, "AACCCGGGTTCTATA"),
    ],
)
def test_six_frame_hand_derived_splice_slot_zero(frame, start, expected):
    with pp.Party():
        pool = pp.insertion_scan_orf(
            "AACCCGGGTTTA",
            "TAG",
            codon_positions=[0],
            frame=frame,
            mode="sequential",
            cards=SPLICE_CARDS,
        )
        row = pool.generate_library().iloc[0]

    assert row["seq"] == expected
    assert row["codon_slot"] == 0
    assert (row["start"], row["end"]) == (start, start)


@pytest.mark.parametrize(
    ("frame", "start", "wt_codon", "expected"),
    [
        (+1, 0, "AAC", "TAGCCGGGTTTA"),
        (+2, 1, "ACC", "ATAGCGGGTTTA"),
        (+3, 2, "CCC", "AATAGGGGTTTA"),
        (-1, 9, "TAA", "AACCCGGGTCTA"),
        (-2, 8, "AAA", "AACCCGGGCTAA"),
        (-3, 7, "AAC", "AACCCGGCTATA"),
    ],
)
def test_six_frame_hand_derived_overwrite_codon_zero(
    frame, start, wt_codon, expected
):
    with pp.Party():
        pool = pp.insertion_scan_orf(
            "AACCCGGGTTTA",
            "TAG",
            codon_positions=[0],
            frame=frame,
            replace=True,
            mode="sequential",
            cards=OVERWRITE_CARDS,
        )
        row = pool.generate_library().iloc[0]

    assert row["seq"] == expected
    assert row["codon_positions"] == (0,)
    assert row["wt_codons"] == (wt_codon,)
    assert (row["start"], row["end"]) == (start, start + 3)


@pytest.mark.parametrize("frame", [-1, -2, -3])
@pytest.mark.parametrize("replace", [False, True])
@pytest.mark.parametrize("coding_insert", ["TAG", "ATGGAA"])
def test_negative_frame_matches_positive_on_reverse_complement(
    frame, replace, coding_insert
):
    seq = "AACCCGGGTTTAAA"
    cards = OVERWRITE_CARDS if replace else SPLICE_CARDS
    target_codons = (len(seq) - (abs(frame) - 1)) // 3
    insert_codons = len(coding_insert) // 3
    num_slots = (
        target_codons - insert_codons + 1 if replace else target_codons + 1
    )
    kwargs = dict(
        insertion_pool=coding_insert,
        codon_positions=[0, 1, num_slots - 1],
        replace=replace,
        mode="sequential",
        cards=cards,
    )

    with pp.Party():
        negative_df = pp.insertion_scan_orf(
            seq, frame=frame, **kwargs
        ).generate_library()
    with pp.Party():
        positive_df = pp.insertion_scan_orf(
            reverse_complement(seq), frame=-frame, **kwargs
        ).generate_library()

    assert negative_df["seq"].tolist() == [
        reverse_complement(seq) for seq in positive_df["seq"]
    ]
    if replace:
        assert (
            negative_df["codon_positions"].tolist()
            == positive_df["codon_positions"].tolist()
        )
        assert negative_df["wt_codons"].tolist() == positive_df["wt_codons"].tolist()
    else:
        assert negative_df["codon_slot"].tolist() == positive_df["codon_slot"].tolist()
    assert negative_df["start"].tolist() == [
        len(seq) - end for end in positive_df["end"]
    ]
    assert negative_df["end"].tolist() == [
        len(seq) - start for start in positive_df["start"]
    ]


def test_negative_multicodon_overwrite_reverses_the_whole_insert_once():
    with pp.Party():
        pool = pp.insertion_scan_orf(
            "ATGAAACCCTTT",
            "ATGGAA",
            codon_positions=[1],
            frame=-1,
            replace=True,
            mode="sequential",
            cards=OVERWRITE_CARDS,
        )
        row = pool.generate_library().iloc[0]

    assert row["seq"] == "ATGTTCCATTTT"
    assert row["codon_positions"] == (1, 2)
    assert row["wt_codons"] == ("GGG", "TTT")
    assert (row["start"], row["end"]) == (3, 9)


def test_negative_multicodon_insert_mirrors_coding_styles_with_content():
    with pp.Party():
        coding_insert = pp.join(
            [
                pp.from_seq("ATG", style="red"),
                pp.from_seq("GAA", style="blue"),
            ]
        )
        pool = pp.insertion_scan_orf(
            "AAACCC",
            coding_insert,
            codon_positions=[1],
            frame=-1,
            mode="sequential",
        )
        row = pool.generate_library(_include_inline_styles=True).iloc[0]

    assert row["seq"] == "AAATTCCATCCC"
    styles = {
        style: set(positions)
        for style, positions in row["_inline_styles"].style_list
    }
    assert styles == {"blue": {3, 4, 5}, "red": {6, 7, 8}}


@pytest.mark.parametrize("frame", [-1, -2, -3])
def test_negative_insert_works_when_inline_styles_are_disabled(frame):
    with pp.Party():
        pp.toggle_styles(False)
        pool = pp.insertion_scan_orf(
            "AAACCCGGG",
            "TAG",
            codon_positions=[0],
            frame=frame,
            mode="sequential",
        )
        row = pool.generate_library(_include_inline_styles=True).iloc[0]

    assert row["seq"].replace("CTA", "") == "AAACCCGGG"
    assert row["_inline_styles"] is None


@pytest.mark.parametrize(
    ("codon_positions", "expected_slots", "expected_starts"),
    [
        (slice(0, None, 2), [0, 2, 4], [12, 6, 0]),
        ([3, 1], [3, 1], [3, 9]),
    ],
)
def test_negative_splice_cards_preserve_sliced_sparse_and_reordered_positions(
    codon_positions, expected_slots, expected_starts
):
    with pp.Party():
        pool = pp.insertion_scan_orf(
            "AAACCCGGGTTT",
            "TAG",
            codon_positions=codon_positions,
            frame=-1,
            mode="sequential",
            cards=SPLICE_CARDS,
        )
        df = pool.generate_library()

    assert df["codon_slot"].tolist() == expected_slots
    assert df["start"].tolist() == expected_starts
    assert df["end"].tolist() == expected_starts


def test_multistate_insert_pool_forms_product_and_preserves_provenance_and_names():
    with pp.Party():
        inserts = pp.from_seqs(
            ["TAG", "TAA", "TGA"],
            mode="sequential",
            cards={"seq": "coding_insert"},
        )
        pool = pp.insertion_scan_orf(
            "GGGTTT",
            inserts,
            codon_positions=[0, 1],
            frame=-1,
            mode="sequential",
            prefix="variant",
            prefix_position="slot",
            prefix_insert="insert",
            cards={"codon_slot": "codon_slot", "state": "position_state"},
        )
        df = pool.generate_library()

    assert pool.num_states == 6
    assert df["seq"].tolist() == [
        "GGGTTTCTA",
        "GGGCTATTT",
        "GGGTTTTTA",
        "GGGTTATTT",
        "GGGTTTTCA",
        "GGGTCATTT",
    ]
    assert df["coding_insert"].tolist() == ["TAG", "TAG", "TAA", "TAA", "TGA", "TGA"]
    assert df["codon_slot"].tolist() == [0, 1, 0, 1, 0, 1]
    assert df["position_state"].tolist() == [0, 1, 0, 1, 0, 1]
    assert df["name"].tolist() == [
        "variant_0.slot_0.insert_0",
        "variant_3.slot_1.insert_0",
        "variant_1.slot_0.insert_1",
        "variant_4.slot_1.insert_1",
        "variant_2.slot_0.insert_2",
        "variant_5.slot_1.insert_2",
    ]


def test_named_negative_orf_preserves_flanks_orphans_and_translates():
    with pp.Party():
        base = pp.from_seq("GGAATGAAATTTCCC").annotate_orf(
            "orf", extent=(2, 13), frame=-2
        )
        inserted = base.insertion_scan_orf(
            "TAG",
            codon_positions=[1],
            region="orf",
            mode="sequential",
            cards=SPLICE_CARDS,
        )
        inserted_row = inserted.generate_library().iloc[0]
        translated_row = inserted.translate(region="orf").generate_library().iloc[0]

    assert inserted_row["seq"] == "GG<orf>AATGAAACTATTTC</orf>CC"
    assert inserted_row["codon_slot"] == 1
    assert (inserted_row["start"], inserted_row["end"]) == (7, 7)
    assert translated_row["seq"] == "K*FH"


@pytest.mark.parametrize(
    ("replace", "positions", "expected", "starts"),
    [
        (
            False,
            [0, 4],
            ["GGTAGAAACCCGGGTTTCC", "GGAAACCCGGGTTTTAGCC"],
            [0, 12],
        ),
        (
            True,
            [0, 3],
            ["GGTAGCCCGGGTTTCC", "GGAAACCCGGGTAGCC"],
            [0, 9],
        ),
    ],
)
def test_interval_region_supports_first_and_last_slots_or_windows(
    replace, positions, expected, starts
):
    cards = {"start": "start", "end": "end"}
    with pp.Party():
        pool = pp.insertion_scan_orf(
            "GGAAACCCGGGTTTCC",
            "TAG",
            codon_positions=positions,
            region=(2, 14),
            frame=1,
            replace=replace,
            mode="sequential",
            cards=cards,
        )
        df = pool.generate_library()

    assert df["seq"].tolist() == expected
    assert df["start"].tolist() == starts
    expected_ends = [start + (3 if replace else 0) for start in starts]
    assert df["end"].tolist() == expected_ends


def test_different_splice_and_overwrite_widths_can_coexist_in_one_party():
    with pp.Party() as party:
        base = pp.from_seq("AAACCCGGG")
        splice_one = base.insertion_scan_orf("TAG", mode="sequential")
        splice_two = base.insertion_scan_orf("ATGGAA", mode="sequential")
        overwrite_one = base.insertion_scan_orf(
            "TAG", replace=True, mode="sequential"
        )
        overwrite_two = base.insertion_scan_orf(
            "ATGGAA", replace=True, mode="sequential"
        )

        assert party.get_region("_ins").seq_length == 0
        assert party.get_region("_rep_len3").seq_length == 3
        assert party.get_region("_rep_len6").seq_length == 6
        assert len(splice_one.generate_library()) == 4
        assert len(splice_two.generate_library()) == 4
        assert len(overwrite_one.generate_library()) == 3
        assert len(overwrite_two.generate_library()) == 2


@pytest.mark.parametrize(
    ("insert", "message"),
    [
        ("", "at least one complete codon"),
        ("A", "divisible by 3"),
        ("ATGA", "divisible by 3"),
    ],
)
def test_invalid_fixed_insert_lengths_are_rejected(insert, message):
    with pp.Party(), pytest.raises(ValueError, match=message):
        pp.insertion_scan_orf("AAACCC", insert)


def test_unknown_insert_length_is_rejected_before_construction():
    with pp.Party():
        inserts = pp.from_seqs(["TAG", "TAATAG"], mode="sequential")
        with pytest.raises(ValueError, match="defined seq_length"):
            pp.insertion_scan_orf("AAACCC", inserts)


@pytest.mark.parametrize("invalid_insert", ["NNN", "---"])
def test_invalid_insert_state_is_rejected_at_generation(invalid_insert):
    with pp.Party():
        inserts = pp.from_seqs(["TAG", invalid_insert], mode="sequential")
        pool = pp.insertion_scan_orf(
            "AAACCC",
            inserts,
            codon_positions=[1],
            mode="sequential",
        )
        with pytest.raises(ValueError, match="ungapped ACGT"):
            pool.generate_library()


def test_tagged_insert_content_is_rejected_at_generation():
    with pp.Party():
        insert = pp.from_seq("TAG").annotate_region("insert", extent=(0, 3))
        pool = pp.insertion_scan_orf(
            "AAACCC", insert, codon_positions=[1], mode="sequential"
        )
        with pytest.raises(ValueError, match="tagged insertion content"):
            pool.generate_library()


def test_nested_target_annotations_are_rejected_before_region_scan():
    with pp.Party():
        base = pp.from_seq("<orf>AAA<x>CCC</x>GGG</orf>").annotate_orf(
            "orf", frame=1
        )
        pool = base.insertion_scan_orf(
            "TAG", region="orf", codon_positions=[0], mode="sequential"
        )
        with pytest.raises(ValueError, match="nested region tags"):
            pool.generate_library()


def test_overwrite_insert_cannot_exceed_complete_target_codons():
    with pp.Party(), pytest.raises(ValueError, match="exceeds"):
        pp.insertion_scan_orf("AAACCC", "AAACCCGGG", replace=True)


def test_splice_rejects_frame_offset_beyond_target_span():
    with pp.Party(), pytest.raises(ValueError, match="No valid codon positions"):
        pp.insertion_scan_orf("A", "TAG", frame=3)


@pytest.mark.parametrize(
    ("replace", "invalid_card"),
    [(False, "wt_codons"), (True, "codon_slot"), (False, "position_index")],
)
def test_cards_are_specific_to_the_orf_insertion_mode(replace, invalid_card):
    with pp.Party(), pytest.raises(ValueError, match=invalid_card):
        pp.insertion_scan_orf(
            "AAACCC", "TAG", replace=replace, cards=[invalid_card]
        )


def test_public_function_and_mixin_are_both_exposed():
    assert callable(pp.insertion_scan_orf)
    assert hasattr(pp.DnaPool, "insertion_scan_orf")

    with pp.Party():
        functional = pp.insertion_scan_orf(
            "AAACCC", "TAG", codon_positions=[1], mode="sequential"
        )
        chained = pp.from_seq("AAACCC").insertion_scan_orf(
            "TAG", codon_positions=[1], mode="sequential"
        )

    assert functional.generate_library()["seq"].tolist() == ["AAATAGCCC"]
    assert chained.generate_library()["seq"].tolist() == ["AAATAGCCC"]


@pytest.mark.parametrize("copy_method", ["copy", "deepcopy"])
def test_copy_operations_preserve_insertion_behavior_and_cards(copy_method):
    with pp.Party():
        original = pp.insertion_scan_orf(
            "AAACCC",
            "TAG",
            mode="sequential",
            cards={"codon_slot": "slot"},
        )
        copied = getattr(original, copy_method)()
        copied_df = copied.generate_library()

    assert copied_df["seq"].tolist() == [
        "TAGAAACCC",
        "AAATAGCCC",
        "AAACCCTAG",
    ]
    assert copied_df["slot"].tolist() == [0, 1, 2]
