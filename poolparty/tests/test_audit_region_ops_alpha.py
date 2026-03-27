"""Audit tests for region operations (alpha pass)."""

import xml.etree.ElementTree as ET

import pytest

import poolparty as pp
from poolparty.region_ops import strip_all_tags, validate_single_region


def _assert_xml_well_formed(seq: str) -> None:
    ET.fromstring(f"<root>{seq}</root>")


def _assert_exhaustion(pool, *, seed: int = 7) -> None:
    df = pool.generate_library(num_cycles=1, seed=seed)
    assert len(df) == pool.num_states


def _assert_clean_len(seq: str, expected: int) -> None:
    assert len(strip_all_tags(seq)) == expected


def test_region_scan_core_invariants_i2_i3_i5_and_xml() -> None:
    with pp.Party():
        pool = pp.region_scan(
            "AACCGG",
            tag_name="scan",
            positions=[0, 2, 4],
            region_length=2,
            mode="sequential",
            cards={"start": "start", "end": "end", "region_seq": "region_seq"},
        )
        df1 = pool.generate_library(num_cycles=1, seed=13)
        df2 = pool.generate_library(num_cycles=1, seed=13)

    assert len(df1) == pool.num_states == 3
    assert df1["seq"].tolist() == df2["seq"].tolist()

    for row in df1.itertuples():
        _assert_xml_well_formed(row.seq)
        parsed = validate_single_region(row.seq, "scan")
        card_region = validate_single_region(row.region_seq, "scan")
        clean = strip_all_tags(row.seq)
        assert parsed.content == clean[row.start : row.end]
        assert parsed.content == card_region.content
        _assert_clean_len(row.seq, 6)


def test_region_scan_i6_region_only_modification_named_region() -> None:
    with pp.Party():
        base = "AAA<target>CCCC</target>TTT"
        pool = pp.region_scan(
            base,
            tag_name="ins",
            region="target",
            positions=[1],
            region_length=2,
            mode="sequential",
        )
        df = pool.generate_library(num_cycles=1)

    seq = df["seq"].iloc[0]
    assert seq.startswith("AAA")
    assert seq.endswith("TTT")
    target = validate_single_region(seq, "target")
    assert "<ins>" in target.content and "</ins>" in target.content
    _assert_xml_well_formed(seq)


def test_region_scan_adversarial_known_interval_and_cycling() -> None:
    with pp.Party():
        pool = pp.region_scan(
            "GGGATGCTT",
            tag_name="m",
            region=[1, 3],
            region_length=1,
            mode="sequential",
            num_states=5,
            cards={"start": "start"},
        )
        df = pool.generate_library(num_cycles=1)

    assert len(df) == 5
    assert set(df["start"]).issubset({0, 1})
    for seq in df["seq"]:
        _assert_xml_well_formed(seq)


def test_region_scan_adversarial_random_named_region_and_composed_state() -> None:
    with pp.Party():
        parent = pp.from_seqs(
            ["AAA<r>CCCC</r>TT", "GGG<r>TTTT</r>AA"],
            mode="sequential",
        )
        pool = pp.region_scan(
            parent,
            tag_name="m",
            region="r",
            region_length=2,
            mode="random",
            num_states=3,
        )
        df1 = pool.generate_library(num_cycles=1, seed=22)
        df2 = pool.generate_library(num_cycles=1, seed=22)

    assert len(df1) == pool.num_states == 6
    assert df1["seq"].tolist() == df2["seq"].tolist()
    for seq in df1["seq"]:
        _assert_xml_well_formed(seq)
        target = validate_single_region(seq, "r")
        assert "<m>" in target.content and "</m>" in target.content


def test_region_scan_adversarial_very_short_parent() -> None:
    with pp.Party():
        pool = pp.region_scan("AC", tag_name="m", region_length=2, mode="sequential")
        df = pool.generate_library(num_cycles=1)

    assert len(df) == pool.num_states == 1
    _assert_xml_well_formed(df["seq"].iloc[0])


def test_region_multiscan_core_invariants_i2_i3_and_xml() -> None:
    with pp.Party():
        pool = pp.region_multiscan(
            "AACCGGTT",
            tag_names=["m1", "m2"],
            num_insertions=2,
            region_length=[1, 2],
            insertion_mode="ordered",
            mode="sequential",
            cards={"starts": "starts", "ends": "ends", "names": "names"},
        )
        df = pool.generate_library(num_cycles=1)

    assert len(df) == pool.num_states
    for row in df.itertuples():
        _assert_xml_well_formed(row.seq)
        assert len(row.starts) == len(row.ends) == len(row.names) == 2
        assert row.starts[0] <= row.starts[1]


def test_region_multiscan_adversarial_interval_region_spacing() -> None:
    with pp.Party():
        pool = pp.region_multiscan(
            "TTTAAACCCGGG",
            tag_names=["x", "y"],
            num_insertions=2,
            region=[2, 10],
            region_length=[1, 1],
            min_spacing=1,
            mode="sequential",
        )
        df = pool.generate_library(num_cycles=1)

    assert len(df) == pool.num_states
    for seq in df["seq"]:
        _assert_xml_well_formed(seq)


def test_region_multiscan_adversarial_random_named_region_composed_state() -> None:
    with pp.Party():
        parent = pp.from_seqs(
            ["AA<core>CCCC</core>TT", "GG<core>TTTT</core>CC"],
            mode="sequential",
        )
        pool = pp.region_multiscan(
            parent,
            tag_names=["a", "b"],
            num_insertions=2,
            region="core",
            region_length=0,
            mode="random",
            num_states=4,
        )
        df1 = pool.generate_library(num_cycles=1, seed=41)
        df2 = pool.generate_library(num_cycles=1, seed=41)

    assert len(df1) == pool.num_states == 8
    assert df1["seq"].tolist() == df2["seq"].tolist()


def test_region_multiscan_adversarial_short_parent() -> None:
    with pp.Party():
        pool = pp.region_multiscan(
            "AC",
            tag_names=["u", "v"],
            num_insertions=2,
            region_length=0,
            mode="sequential",
            num_states=3,
        )
        df = pool.generate_library(num_cycles=1)

    assert len(df) == 3
    for seq in df["seq"]:
        _assert_xml_well_formed(seq)


def test_replace_region_contract_tracing_c1_c2_and_i8() -> None:
    with pp.Party():
        bg = pp.stack([pp.from_seq("AA<ins/>TT"), pp.from_seq("GG<ins/>CC")])
        content = pp.from_seqs(["X", "Y", "Z"], mode="sequential")
        replaced = pp.replace_region(bg, content, "ins")
        df = replaced.generate_library(num_cycles=1)

    assert replaced.seq_length == 5
    assert len(df) == replaced.num_states == 6
    expected = {
        "AAXTT",
        "AAYTT",
        "AAZTT",
        "GGXCC",
        "GGYCC",
        "GGZCC",
    }
    assert set(df["seq"]) == expected
    _assert_exhaustion(replaced)


def test_replace_region_adversarial_sync_keep_tags_and_xml() -> None:
    with pp.Party():
        bg = pp.from_seqs(["AA<bc/>TT", "GG<bc/>CC"], mode="sequential")
        content = pp.from_seqs(["X", "Y"], mode="sequential")
        replaced = pp.replace_region(bg, content, "bc", sync=True, keep_tags=True)
        df = replaced.generate_library(num_cycles=1)

    assert replaced.num_states == 2
    assert len(df) == 2
    for seq in df["seq"]:
        assert "<bc>" in seq and "</bc>" in seq
        _assert_xml_well_formed(seq)


def test_replace_region_adversarial_rc() -> None:
    with pp.Party():
        replaced = pp.replace_region("A<r/>T", "AG", "r", rc=True)
        df = replaced.generate_library(num_cycles=1)

    assert df["seq"].iloc[0] == "ACTT"


def test_apply_at_region_adversarial_diagonals() -> None:
    with pp.Party():
        base = pp.from_seq("AAA<target>ATGC</target>TTT")
        rc_result = pp.apply_at_region(base, "target", pp.rc, remove_tags=True)
        keep_tag_result = pp.apply_at_region(base, "target", pp.rc, remove_tags=False)
        composed = pp.apply_at_region(
            pp.from_seqs(["AA<target>ACGT</target>TT", "GG<target>TTAA</target>CC"], mode="sequential"),
            "target",
            lambda p: pp.mutagenize(p, num_mutations=1, mode="sequential"),
            remove_tags=True,
        )

        rc_df = rc_result.generate_library(num_cycles=1)
        keep_df = keep_tag_result.generate_library(num_cycles=1)
        composed_df = composed.generate_library(num_cycles=1)

    assert rc_df["seq"].iloc[0] == "AAAGCATTTT"
    assert "<target>" in keep_df["seq"].iloc[0] and "</target>" in keep_df["seq"].iloc[0]
    assert len(composed_df) == composed.num_states
    for seq in keep_df["seq"]:
        _assert_xml_well_formed(seq)


def test_region_ops_mixin_runtime_forwarding_non_bug_paths() -> None:
    with pp.Party():
        pool = pp.from_seq("AAAACCCCTTTT")
        annotated = pool.annotate_region(
            "mark",
            extent=(2, 8),
            style="blue",
            iter_order=1,
            prefix="ann",
        )
        inserted = pool.insert_tags("ins", start=4, stop=6, iter_order=2, prefix="ins")
        removed = inserted.remove_tags("ins", keep_content=True, iter_order=3, prefix="rm")
        replaced = inserted.replace_region("GG", "ins", sync=False, keep_tags=False)
        applied = inserted.apply_at_region("ins", pp.rc, remove_tags=True, iter_order=4, prefix="ap")

        for candidate in (annotated, inserted, removed, replaced, applied):
            _assert_exhaustion(candidate)


def test_scan_ops_mixin_runtime_forwarding_region_backed_consumer() -> None:
    with pp.Party():
        pool = pp.from_seq("AACCGGTT")
        scanned = pool.deletion_scan(
            deletion_length=2,
            positions=[1, 3],
            mode="sequential",
            cards={"start": "start", "end": "end"},
        )
        df = scanned.generate_library(num_cycles=1)

    assert len(df) == scanned.num_states
    assert {"start", "end"}.issubset(df.columns)


def test_region_scan_and_region_multiscan_are_top_level_factories() -> None:
    with pp.Party():
        pool = pp.from_seq("ACGT")
        assert not hasattr(pool, "region_scan")
        assert not hasattr(pool, "region_multiscan")
        _assert_exhaustion(pp.region_scan(pool, tag_name="r", mode="sequential"))
        _assert_exhaustion(pp.region_multiscan(pool, tag_names=["r"], num_insertions=1, mode="sequential"))


def test_annotate_region_extent_none_should_cover_variable_length_sequences() -> None:
    with pp.Party():
        variable = pp.from_seqs(["AAAA", "AAAAAA"], mode="sequential")
        annotated = pp.annotate_region(variable, "r")
        seqs = annotated.generate_library(num_cycles=1)["seq"].tolist()
    assert seqs == ["<r>AAAA</r>", "<r>AAAAAA</r>"]


def test_region_scan_should_declare_seq_length_for_fixed_inputs() -> None:
    with pp.Party():
        pool = pp.region_scan("AACCGG", tag_name="m", mode="sequential")
    assert pool.seq_length == 6


def test_region_multiscan_should_declare_seq_length_for_fixed_inputs() -> None:
    with pp.Party():
        pool = pp.region_multiscan(
            "AACCGG",
            tag_names=["m1", "m2"],
            num_insertions=2,
            mode="sequential",
        )
    assert pool.seq_length == 6


def test_region_scan_sequential_should_reject_unknown_geometry() -> None:
    with pp.Party():
        variable = pp.from_seqs(["AA", "AAAA"], mode="sequential")
        with pytest.raises(ValueError):
            pp.region_scan(variable, tag_name="m", mode="sequential")


def test_region_multiscan_sequential_should_reject_unknown_geometry() -> None:
    with pp.Party():
        variable = pp.from_seqs(["AAA", "AAAAA"], mode="sequential")
        with pytest.raises(ValueError):
            pp.region_multiscan(variable, tag_names=["m1", "m2"], num_insertions=2, mode="sequential")


def test_apply_at_region_mixin_should_forward_rc() -> None:
    with pp.Party():
        pool = pp.from_seq("AA<r>CC</r>TT")
        via_mixin = pool.apply_at_region("r", pp.rc, rc=True)
        via_factory = pp.apply_at_region(pool, "r", pp.rc, rc=True)
        assert via_mixin.generate_library(num_cycles=1)["seq"].tolist() == via_factory.generate_library(
            num_cycles=1
        )["seq"].tolist()


def test_replace_region_mixin_should_forward_rc() -> None:
    with pp.Party():
        pool = pp.from_seq("AA<r/>TT")
        content = pp.from_seq("AG")
        via_mixin = pool.replace_region(content, "r", rc=True)
        via_factory = pp.replace_region(pool, content, "r", rc=True)
        assert via_mixin.generate_library(num_cycles=1)["seq"].tolist() == via_factory.generate_library(
            num_cycles=1
        )["seq"].tolist()


def test_clear_annotation_should_run_and_clear_content() -> None:
    with pp.Party():
        out = pp.clear_annotation("A-C<r>g</r>t")
        seq = out.generate_library(num_cycles=1)["seq"].iloc[0]
    assert seq == "ACGT"
