"""Audit tests for scan and multiscan operations (alpha pass).

Scope:
- scan_ops: deletion_scan, insertion_scan, replacement_scan, shuffle_scan,
  mutagenize_scan, subseq_scan
- multiscan_ops: deletion_multiscan, insertion_multiscan, replacement_multiscan

Methodology: follows operation_audit.mdc Steps 2-6 with runtime evidence.
Step 7 findings are documented in dev/audit/scan_multiscan_ops_alpha.md.
"""

import inspect
import warnings
import xml.etree.ElementTree as ET

import pytest

import poolparty as pp
from poolparty.region_ops import strip_all_tags


def _bio_len(seq: str) -> int:
    return len(strip_all_tags(seq))


def _assert_i1_i2(pool, *, seed: int = 7):
    df = pool.generate_library(num_cycles=1, seed=seed)
    assert len(df) == pool.num_states
    if pool.seq_length is not None:
        for seq in df["seq"]:
            assert _bio_len(seq) == pool.seq_length
    return df


def _assert_random_determinism(build_fn):
    with pp.Party():
        pool = build_fn()
        seqs1 = pool.generate_library(num_cycles=1, seed=42)["seq"].tolist()
        seqs2 = pool.generate_library(num_cycles=1, seed=42)["seq"].tolist()
    assert seqs1 == seqs2


class TestStep2RuntimeForwarding:
    """Step 2: mixin forwarding checks at runtime."""

    def test_scan_and_multiscan_mixins_forward_runtime_parameters(self):
        with pp.Party():
            base = pp.from_seq("AA<core>CCGG</core>TT")
            plain = pp.from_seq("AACCGGTT")

            deletion = base.deletion_scan(
                deletion_length=1,
                deletion_marker="-",
                positions=[0, 1],
                region="core",
                prefix="del",
                mode="sequential",
                num_states=2,
                style="gray",
                iter_order=1,
                cards={"start": "del_start", "end": "del_end"},
            )
            insertion = base.insertion_scan(
                ins_pool="X",
                positions=[0, 1],
                region="core",
                replace=False,
                style="blue",
                prefix="ins",
                prefix_position="pos",
                prefix_insert="site",
                mode="sequential",
                num_states=2,
                iter_order=1,
                cards={"start": "ins_start", "end": "ins_end"},
            )
            replacement = base.replacement_scan(
                ins_pool="X",
                positions=[0, 1],
                region="core",
                style="purple",
                prefix="rep",
                prefix_position="pos",
                prefix_insert="site",
                mode="sequential",
                num_states=2,
                iter_order=1,
                cards={"start": "rep_start", "end": "rep_end"},
            )
            shuf = base.shuffle_scan(
                shuffle_length=2,
                positions=[0, 1],
                region="core",
                shuffle_type="mono",
                shuffles_per_position=2,
                prefix="shf",
                prefix_position="pos",
                prefix_shuffle="perm",
                mode="sequential",
                num_states=2,
                style="green",
                iter_order=1,
                cards=(
                    {"start": "shf_start"},
                    {"permutation": "shf_perm"},
                ),
            )
            mut = base.mutagenize_scan(
                mutagenize_length=2,
                num_mutations=1,
                positions=[0, 1],
                region="core",
                prefix=("scan", "mut"),
                mode=("sequential", "random"),
                num_states=(2, 2),
                style="red",
                iter_order=(1, -1),
                cards=(
                    {"start": "mut_start"},
                    {"positions": "mut_positions"},
                ),
            )
            subseq = base.subseq_scan(
                subseq_length=2,
                positions=[0, 1],
                region="core",
                prefix="sub",
                mode="sequential",
                num_states=2,
                iter_order=1,
                cards={"start": "sub_start", "end": "sub_end"},
            )
            del_ms = plain.deletion_multiscan(
                deletion_length=1,
                num_deletions=2,
                deletion_marker="-",
                positions=[0, 2, 4],
                region=None,
                names=["d0", "d1"],
                min_spacing=0,
                max_spacing=None,
                prefix="dms",
                mode="sequential",
                num_states=3,
                style="gray",
                iter_order=1,
                cards={"starts": "dms_starts"},
            )
            ins_ms = plain.insertion_multiscan(
                num_insertions=2,
                insertion_pools=pp.from_seq("A"),
                positions=[0, 2, 4],
                region=None,
                names=["i0", "i1"],
                replace=False,
                insertion_mode="ordered",
                min_spacing=0,
                max_spacing=None,
                prefix="ims",
                mode="sequential",
                num_states=3,
                iter_order=1,
                cards={"starts": "ims_starts"},
            )
            rep_ms = plain.replacement_multiscan(
                num_replacements=2,
                replacement_pools=[pp.from_seq("A"), pp.from_seq("T")],
                positions=[[0, 2, 4], [0, 2, 4]],
                region=None,
                names=["r0", "r1"],
                insertion_mode="ordered",
                min_spacing=0,
                max_spacing=None,
                prefix="rms",
                mode="sequential",
                num_states=3,
                iter_order=1,
                cards={"starts": "rms_starts"},
            )

            for candidate in (
                deletion,
                insertion,
                replacement,
                shuf,
                mut,
                subseq,
                del_ms,
                ins_ms,
                rep_ms,
            ):
                df = _assert_i1_i2(candidate)
                assert len(df.columns) > 1

            assert {"del_start", "del_end"}.issubset(deletion.generate_library().columns)
            assert {"ins_start", "ins_end"}.issubset(insertion.generate_library().columns)
            assert {"rep_start", "rep_end"}.issubset(replacement.generate_library().columns)
            assert {"shf_start", "shf_perm"}.issubset(shuf.generate_library().columns)
            assert {"mut_start", "mut_positions"}.issubset(mut.generate_library().columns)
            assert {"sub_start", "sub_end"}.issubset(subseq.generate_library().columns)
            assert {"dms_starts"}.issubset(del_ms.generate_library().columns)
            assert {"ims_starts"}.issubset(ins_ms.generate_library().columns)
            assert {"rms_starts"}.issubset(rep_ms.generate_library().columns)


class TestStep3CoreInvariants:
    """Step 3: I1/I2 for all ops + trigger minimums I3/I5/I6."""

    def test_i1_i2_for_all_audited_ops(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            ops = {
                "deletion_scan": pp.deletion_scan(base, deletion_length=2, positions=[1, 3], mode="sequential"),
                "insertion_scan": pp.insertion_scan(base, "TT", positions=[1, 3], mode="sequential"),
                "replacement_scan": pp.replacement_scan(base, "TT", positions=[1, 3], mode="sequential"),
                "shuffle_scan": pp.shuffle_scan(base, shuffle_length=2, positions=[1, 3], mode="sequential"),
                "mutagenize_scan": pp.mutagenize_scan(
                    base,
                    mutagenize_length=2,
                    num_mutations=1,
                    positions=[1, 3],
                    mode=("sequential", "sequential"),
                ),
                "subseq_scan": pp.subseq_scan(base, subseq_length=3, positions=[1, 3], mode="sequential"),
                "deletion_multiscan": pp.deletion_multiscan(
                    base,
                    deletion_length=1,
                    num_deletions=2,
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
                "insertion_multiscan": pp.insertion_multiscan(
                    base,
                    num_insertions=2,
                    insertion_pools=pp.from_seq("A"),
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
                "replacement_multiscan": pp.replacement_multiscan(
                    base,
                    num_replacements=2,
                    replacement_pools=[pp.from_seq("A"), pp.from_seq("T")],
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
            }
            expected_len = {
                "deletion_scan": 8,
                "insertion_scan": 10,
                "replacement_scan": 8,
                "shuffle_scan": 8,
                "mutagenize_scan": 8,
                "subseq_scan": 3,
                "deletion_multiscan": 8,
                "insertion_multiscan": 10,
                "replacement_multiscan": 8,
            }

            for op_name, op_pool in ops.items():
                df = op_pool.generate_library(num_cycles=1, seed=11)
                assert len(df) == op_pool.num_states
                for seq in df["seq"]:
                    assert _bio_len(seq) == expected_len[op_name]

    def test_i5_random_determinism_for_all_ops(self):
        _assert_random_determinism(
            lambda: pp.deletion_scan(pp.from_seq("AACCGGTT"), deletion_length=2, mode="random", num_states=5)
        )
        _assert_random_determinism(
            lambda: pp.insertion_scan(pp.from_seq("AACCGGTT"), "TT", mode="random", num_states=5)
        )
        _assert_random_determinism(
            lambda: pp.replacement_scan(pp.from_seq("AACCGGTT"), "TT", mode="random", num_states=5)
        )
        _assert_random_determinism(
            lambda: pp.mutagenize_scan(
                pp.from_seq("AACCGGTT"),
                mutagenize_length=2,
                num_mutations=1,
                mode=("random", "random"),
                num_states=(3, 2),
            )
        )
        _assert_random_determinism(
            lambda: pp.subseq_scan(pp.from_seq("AACCGGTT"), subseq_length=3, mode="random", num_states=5)
        )
        _assert_random_determinism(
            lambda: pp.deletion_multiscan(
                pp.from_seq("AACCGGTT"),
                deletion_length=1,
                num_deletions=2,
                mode="random",
                num_states=5,
            )
        )
        _assert_random_determinism(
            lambda: pp.insertion_multiscan(
                pp.from_seq("AACCGGTT"),
                num_insertions=2,
                insertion_pools=pp.from_seq("A"),
                mode="random",
                num_states=5,
            )
        )
        _assert_random_determinism(
            lambda: pp.replacement_multiscan(
                pp.from_seq("AACCGGTT"),
                num_replacements=2,
                replacement_pools=[pp.from_seq("A"), pp.from_seq("T")],
                mode="random",
                num_states=5,
            )
        )

    def test_i6_region_isolation_examples(self):
        with pp.Party():
            base = pp.from_seq("AA<core>CCGGAA</core>TT")
            candidates = [
                pp.deletion_scan(base, deletion_length=1, region="core", mode="sequential"),
                pp.insertion_scan(base, "X", region="core", mode="sequential"),
                pp.replacement_scan(base, "X", region="core", mode="sequential"),
                pp.shuffle_scan(base, shuffle_length=2, region="core", mode="sequential"),
                pp.mutagenize_scan(
                    base,
                    mutagenize_length=2,
                    num_mutations=1,
                    region="core",
                    mode=("sequential", "sequential"),
                ),
                pp.deletion_multiscan(
                    base,
                    deletion_length=1,
                    num_deletions=2,
                    region="core",
                    mode="sequential",
                ),
                pp.insertion_multiscan(
                    base,
                    num_insertions=2,
                    insertion_pools=pp.from_seq("A"),
                    region="core",
                    mode="sequential",
                ),
                pp.replacement_multiscan(
                    base,
                    num_replacements=2,
                    replacement_pools=pp.from_seq("A"),
                    region="core",
                    mode="sequential",
                ),
            ]

            for candidate in candidates:
                df = candidate.generate_library(num_cycles=1, seed=17)
                for seq in df["seq"]:
                    clean = strip_all_tags(seq)
                    assert clean.startswith("AA")
                    assert clean.endswith("TT")

            subseq = pp.subseq_scan(base, subseq_length=2, region="core", mode="sequential")
            sub_df = subseq.generate_library(num_cycles=1, seed=17)
            for seq in sub_df["seq"]:
                assert _bio_len(seq) == 2


class TestStep4HighRiskAdversarial:
    """Step 4: >=3 diagonal adversarial combinations per high-risk op."""

    def test_insertion_scan_diagonal_1_named_region_multi_state_insert(self):
        with pp.Party():
            base = pp.from_seq("AA<core>CCCC</core>TT")
            ins = pp.from_seqs(["G", "T"], mode="sequential")
            out = pp.insertion_scan(
                base,
                ins,
                region="core",
                positions=[0, 2],
                mode="sequential",
                num_states=3,
            )
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states == 6

    def test_insertion_scan_diagonal_2_interval_region_replace_mode(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            out = pp.replacement_scan(
                base,
                "TT",
                region=[2, 6],
                positions=[0, 2],
                mode="sequential",
            )
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states == 2
            assert all(_bio_len(s) == 8 for s in df["seq"])

    def test_insertion_scan_diagonal_3_compositional_stress(self):
        with pp.Party():
            parent = pp.from_seqs(["AACCGG", "TTGGCC"], mode="sequential")
            mid = pp.insertion_scan(parent, "AA", positions=[1, 3], mode="sequential")
            out = pp.flip(mid, mode="sequential", num_states=2)
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states == 8

    def test_mutagenize_scan_diagonal_1_tuple_controls_and_cards(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            out = pp.mutagenize_scan(
                base,
                mutagenize_length=2,
                num_mutations=1,
                positions=[1, 3],
                mode=("random", "sequential"),
                num_states=(3, None),
                cards=({"start": "start"}, {"positions": "mut_pos"}),
            )
            df = out.generate_library(num_cycles=1, seed=3)
            assert len(df) == out.num_states
            assert {"start", "mut_pos"}.issubset(df.columns)

    def test_mutagenize_scan_diagonal_2_named_region_variable_parent(self):
        """Variable-length parent + named region + sequential: region_scan now
        rejects unknown geometry when region 'r' is not registered on the Party
        (from_seqs doesn't register embedded tags as regions)."""
        with pp.Party():
            parent = pp.from_seqs(["AA<r>CC</r>T", "GG<r>TT</r>AA"], mode="sequential")
            with pytest.raises(ValueError, match="known scan geometry"):
                pp.mutagenize_scan(
                    parent,
                    mutagenize_length=2,
                    num_mutations=1,
                    region="r",
                    mode=("sequential", "sequential"),
                )

    def test_mutagenize_scan_diagonal_3_compositional_stress(self):
        with pp.Party():
            parent = pp.from_seqs(["AACCGGTT", "TTGGCCAA"], mode="sequential")
            mid = pp.mutagenize_scan(
                parent,
                mutagenize_length=2,
                mutation_rate=0.5,
                positions=[1, 3],
                mode=("sequential", "random"),
                num_states=(2, 3),
            )
            out = pp.flip(mid, mode="sequential", num_states=2)
            df = out.generate_library(num_cycles=1, seed=5)
            assert len(df) == out.num_states == 24

    def test_deletion_multiscan_diagonal_1_spacing_constraints(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            out = pp.deletion_multiscan(
                base,
                deletion_length=1,
                num_deletions=2,
                positions=[0, 2, 4, 6],
                min_spacing=1,
                mode="sequential",
                cards={"starts": "starts"},
            )
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states
            for starts in df["starts"]:
                if len(starts) == 2:
                    assert starts[1] - starts[0] >= 1

    def test_deletion_multiscan_diagonal_2_random_determinism_named_region(self):
        with pp.Party():
            base = pp.from_seq("AA<core>CCGGAA</core>TT")
            out = pp.deletion_multiscan(
                base,
                deletion_length=1,
                num_deletions=2,
                region="core",
                min_spacing=1,
                max_spacing=2,
                mode="random",
                num_states=4,
            )
            df1 = out.generate_library(num_cycles=1, seed=12)
            df2 = out.generate_library(num_cycles=1, seed=12)
            assert df1["seq"].tolist() == df2["seq"].tolist()

    def test_deletion_multiscan_diagonal_3_compositional_stress(self):
        with pp.Party():
            parent = pp.from_seqs(["AACCGGTT", "TTGGCCAA"], mode="sequential")
            mid = pp.deletion_multiscan(
                parent,
                deletion_length=1,
                num_deletions=2,
                positions=[0, 2, 4],
                mode="sequential",
            )
            out = pp.flip(mid, mode="sequential", num_states=2)
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states

    def test_insertion_multiscan_diagonal_1_unordered_distinct_pools(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            out = pp.replacement_multiscan(
                base,
                num_replacements=2,
                replacement_pools=[pp.from_seq("A"), pp.from_seq("TT")],
                positions=[[0, 2, 4], [0, 1, 2]],
                insertion_mode="unordered",
                mode="sequential",
            )
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states

    def test_insertion_multiscan_diagonal_2_single_pool_deepcopy(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            ins = pp.from_seqs(["A", "T"], mode="sequential")
            out = pp.insertion_multiscan(
                base,
                num_insertions=2,
                insertion_pools=ins,
                positions=[0, 2, 4],
                mode="sequential",
            )
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states
            assert df["seq"].nunique() > 1

    def test_insertion_multiscan_diagonal_3_compositional_stress(self):
        with pp.Party():
            parent = pp.from_seqs(["AACCGGTT", "TTGGCCAA"], mode="sequential")
            mid = pp.insertion_multiscan(
                parent,
                num_insertions=2,
                insertion_pools=[pp.from_seq("A"), pp.from_seq("T")],
                positions=[0, 2, 4],
                mode="sequential",
            )
            out = pp.flip(mid, mode="sequential", num_states=2)
            df = out.generate_library(num_cycles=1)
            assert len(df) == out.num_states


class TestStep5ContractTracing:
    """Step 5: C1/C2/C3 tracing for highest-risk ops."""

    def test_mutagenize_scan_c1_state_to_output_window_mapping(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            out = pp.mutagenize_scan(
                base,
                mutagenize_length=2,
                num_mutations=1,
                positions=[0, 2, 4],
                mode=("sequential", "sequential"),
                cards=({"start": "start"}, {"positions": "mut_positions"}),
            )
            df = out.generate_library(num_cycles=1)

        for row in df.itertuples():
            clean = strip_all_tags(row.seq)
            diffs = [i for i, (a, b) in enumerate(zip(clean, "AACCGGTT")) if a != b]
            assert all(row.start <= i < row.start + 2 for i in diffs)

    def test_insertion_multiscan_c2_state_space_matches_region_multiscan_with_single_state_content(self):
        with pp.Party():
            parent = pp.from_seqs(["AACCGGTT", "TTGGCCAA"], mode="sequential")
            region_only = pp.region_multiscan(
                parent,
                tag_names=["r0", "r1"],
                num_insertions=2,
                positions=[0, 2, 4],
                region_length=0,
                insertion_mode="ordered",
                mode="sequential",
            )
            wrapped = pp.insertion_multiscan(
                parent,
                num_insertions=2,
                insertion_pools=pp.from_seq("A"),
                positions=[0, 2, 4],
                names=["r0", "r1"],
                mode="sequential",
            )
            df = wrapped.generate_library(num_cycles=1)

        assert wrapped.num_states == region_only.num_states
        assert len(df) == wrapped.num_states
        seqs = df["seq"].tolist()
        assert any(s.startswith("AA") for s in seqs)
        assert any(s.startswith("TT") for s in seqs)


class TestStep6ApiConsistencyAndFindings:
    """Step 6: API consistency plus strict repro tests for confirmed findings."""

    def test_wrapper_equivalence_replacement_scan(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTT")
            via_wrapper = pp.replacement_scan(bg, "TT", positions=[1, 3], mode="sequential")
            via_base = pp.insertion_scan(bg, "TT", positions=[1, 3], replace=True, mode="sequential")
            assert via_wrapper.generate_library(seed=1)["seq"].tolist() == via_base.generate_library(seed=1)[
                "seq"
            ].tolist()

    def test_wrapper_equivalence_replacement_multiscan(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTT")
            via_wrapper = pp.replacement_multiscan(
                bg,
                num_replacements=2,
                replacement_pools=[pp.from_seq("A"), pp.from_seq("T")],
                positions=[0, 2, 4],
                mode="sequential",
            )
            via_base = pp.insertion_multiscan(
                bg,
                num_insertions=2,
                insertion_pools=[pp.from_seq("A"), pp.from_seq("T")],
                positions=[0, 2, 4],
                replace=True,
                mode="sequential",
            )
            assert via_wrapper.generate_library(seed=1)["seq"].tolist() == via_base.generate_library(seed=1)[
                "seq"
            ].tolist()

    def test_mode_validation_fixed_rejected(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTT")
            with pytest.raises(ValueError, match="mode must be"):
                pp.deletion_scan(bg, deletion_length=1, mode="fixed")
            with pytest.raises(ValueError, match="mode must be"):
                pp.deletion_multiscan(bg, deletion_length=1, num_deletions=2, mode="fixed")

    def test_num_states_cycling_and_clipping(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTT")
            clipped = pp.insertion_scan(bg, "A", mode="sequential", num_states=3)
            assert len(clipped.generate_library(num_cycles=1)) == 3

            cycled = pp.insertion_scan(bg, "A", mode="sequential", num_states=20)
            df = cycled.generate_library(num_cycles=1)
            assert len(df) == 20

    def test_shuffle_scan_random_seed_init_state_replay(self):
        """init_state=0 replays identical output for same seed on re-generation."""
        with pp.Party():
            out = pp.shuffle_scan("AACCGGTT", shuffle_length=2, mode="random", num_states=5)
            seqs1 = out.generate_library(num_cycles=1, seed=42, init_state=0)["seq"].tolist()
            seqs2 = out.generate_library(num_cycles=1, seed=42, init_state=0)["seq"].tolist()
        assert seqs1 == seqs2

    def test_bug_deletion_multiscan_empty_string_marker_should_not_insert_dash(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTT")
            out = pp.deletion_multiscan(
                bg,
                deletion_length=2,
                num_deletions=1,
                deletion_marker="",
                positions=[2],
                mode="sequential",
            )
            seq = out.generate_library(num_cycles=1)["seq"].iloc[0]
        assert "-" not in strip_all_tags(seq)

    def test_deletion_scan_accepts_factory_name(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTT")
            out = pp.deletion_scan(bg, deletion_length=2, positions=[1], mode="sequential", _factory_name="audit")
            _assert_i1_i2(out)

    def test_subseq_scan_accepts_factory_name(self):
        with pp.Party():
            bg = pp.from_seq("AACCGGTT")
            out = pp.subseq_scan(bg, subseq_length=2, positions=[1], mode="sequential", _factory_name="audit")
            _assert_i1_i2(out)

    def test_bug_insertion_scan_should_preserve_known_seq_length(self):
        with pp.Party():
            out = pp.insertion_scan("AACCGGTT", "TT", positions=[1], mode="sequential")
        assert out.seq_length == 10

    def test_bug_deletion_multiscan_should_preserve_known_seq_length(self):
        with pp.Party():
            out = pp.deletion_multiscan("AACCGGTT", deletion_length=1, num_deletions=2, positions=[0, 2], mode="sequential")
        assert out.seq_length == 8

    def test_bug_mutagenize_scan_sequential_unknown_geometry_should_raise(self):
        with pp.Party():
            variable = pp.from_seqs(["AAAC", "AAACC"], mode="sequential")
            with pytest.raises(ValueError):
                pp.mutagenize_scan(
                    variable,
                    mutagenize_length=2,
                    num_mutations=1,
                    mode=("sequential", "sequential"),
                )

    def test_signature_inventory_snapshot(self):
        scan_params = inspect.signature(pp.insertion_scan).parameters
        multi_params = inspect.signature(pp.insertion_multiscan).parameters
        mixin_params = inspect.signature(pp.from_seq("ACGT").insertion_scan).parameters
        assert "cards" in scan_params and "cards" in multi_params and "cards" in mixin_params


class TestStep2SmokeAndMixinPresence:
    """Updated Step 2 requirements: smoke each factory + verify pool methods exist."""

    def test_factory_smoke_construct_and_generate(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            factories = [
                lambda: pp.deletion_scan(base, deletion_length=1, positions=[0], mode="sequential"),
                lambda: pp.insertion_scan(base, "A", positions=[0], mode="sequential"),
                lambda: pp.replacement_scan(base, "A", positions=[0], mode="sequential"),
                lambda: pp.shuffle_scan(base, shuffle_length=2, positions=[0], mode="sequential"),
                lambda: pp.mutagenize_scan(
                    base,
                    mutagenize_length=2,
                    num_mutations=1,
                    positions=[0],
                    mode=("sequential", "sequential"),
                ),
                lambda: pp.subseq_scan(base, subseq_length=2, positions=[0], mode="sequential"),
                lambda: pp.deletion_multiscan(
                    base,
                    deletion_length=1,
                    num_deletions=2,
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
                lambda: pp.insertion_multiscan(
                    base,
                    num_insertions=2,
                    insertion_pools=pp.from_seq("A"),
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
                lambda: pp.replacement_multiscan(
                    base,
                    num_replacements=2,
                    replacement_pools=[pp.from_seq("A"), pp.from_seq("T")],
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
            ]
            for build in factories:
                out = build()
                df = out.generate_library(num_cycles=1, seed=3)
                assert len(df) == out.num_states

    def test_pool_has_methods_for_all_factories(self):
        with pp.Party():
            pool = pp.from_seq("AACCGGTT")
            for method in [
                "deletion_scan",
                "insertion_scan",
                "replacement_scan",
                "shuffle_scan",
                "mutagenize_scan",
                "subseq_scan",
                "deletion_multiscan",
                "insertion_multiscan",
                "replacement_multiscan",
            ]:
                assert hasattr(pool, method)


class TestStep3AdditionalContractDimensions:
    """Updated Step 3 dimensions: nested tags, pool type, I4/I8/I10, representative coverage."""

    def test_nested_tag_behavior_empirical_and_xml_well_formed(self):
        with pp.Party():
            base = pp.from_seq("AA<outer>CC<inner>GG</inner>TT</outer>AA")
            out = pp.insertion_scan(base, "X", region="outer", positions=[1], mode="sequential")
            seq = out.generate_library(num_cycles=1)["seq"].iloc[0]
        ET.fromstring(f"<root>{seq}</root>")
        assert "<outer>" in seq and "</outer>" in seq
        assert "<inner>" in seq and "</inner>" in seq

    def test_pool_type_preservation_for_unary_scan_consumers(self):
        with pp.Party():
            base = pp.from_seq("AACCGGTT")
            candidates = [
                pp.deletion_scan(base, deletion_length=1, positions=[0], mode="sequential"),
                pp.insertion_scan(base, "A", positions=[0], mode="sequential"),
                pp.replacement_scan(base, "A", positions=[0], mode="sequential"),
                pp.shuffle_scan(base, shuffle_length=2, positions=[0], mode="sequential"),
                pp.mutagenize_scan(
                    base,
                    mutagenize_length=2,
                    num_mutations=1,
                    positions=[0],
                    mode=("sequential", "sequential"),
                ),
                pp.subseq_scan(base, subseq_length=2, positions=[0], mode="sequential"),
                pp.deletion_multiscan(
                    base,
                    deletion_length=1,
                    num_deletions=2,
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
                pp.insertion_multiscan(
                    base,
                    num_insertions=2,
                    insertion_pools=pp.from_seq("A"),
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
                pp.replacement_multiscan(
                    base,
                    num_replacements=2,
                    replacement_pools=[pp.from_seq("A"), pp.from_seq("T")],
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
            ]
            for out in candidates:
                assert type(out) is type(base)

    def test_i8_replacement_length_algebra_replacement_scan(self):
        with pp.Party():
            out = pp.replacement_scan("AACCGGTT", "TT", positions=[1], mode="sequential")
        assert out.seq_length == 8

    def test_i10_state_space_immutability_representatives(self):
        with pp.Party():
            reps = [
                pp.insertion_scan(pp.from_seq("AACCGGTT"), "A", positions=[0, 2], mode="sequential"),
                pp.mutagenize_scan(
                    pp.from_seq("AACCGGTT"),
                    mutagenize_length=2,
                    num_mutations=1,
                    positions=[0, 2],
                    mode=("sequential", "sequential"),
                ),
                pp.insertion_multiscan(
                    pp.from_seq("AACCGGTT"),
                    num_insertions=2,
                    insertion_pools=pp.from_seq("A"),
                    positions=[0, 2, 4],
                    mode="sequential",
                ),
            ]
            for out in reps:
                ns_before = out.num_states
                sv_before = out.operation.state._num_values
                out.generate_library(num_cycles=1, seed=4)
                assert out.num_states == ns_before
                assert out.operation.state._num_values == sv_before

    def test_c3_region_roundtrip_style_and_downstream_filter_contract(self):
        with pp.Party():
            base = pp.from_seq("AA<outer>CC<inner>GG</inner>TT</outer>AA")
            out = pp.insertion_scan(
                base,
                "X",
                region="outer",
                positions=[1],
                mode="sequential",
                style="red",
            )
            df = out.generate_library(num_cycles=1, _include_inline_styles=True)
            seq = df["seq"].iloc[0]
            inline_style = df["_inline_styles"].iloc[0]
            # C3: style metadata survives round-trip and aligns to output length.
            assert len(inline_style) == len(seq)
            # C3: downstream filter on clean content remains functional.
            filtered = out.filter(lambda s: len(s) == len(strip_all_tags(seq)), cards=["passed"])
            filt_df = filtered.generate_library(num_cycles=1)
            card_col = [c for c in filt_df.columns if c.endswith(".passed")][0]
            assert bool(filt_df[card_col].iloc[0]) is True


class TestStep4AssumptionInversionHighRisk:
    """Updated Step 4: explicit assumption inversion per high-risk op."""

    # insertion_scan assumptions
    def test_insertion_scan_assumption_parent_seq_length_known_when_region_none(self):
        with pp.Party():
            variable = pp.from_seqs(["AA", "AAA"], mode="sequential")
            with pytest.raises(ValueError, match="defined seq_length"):
                pp.insertion_scan(variable, "A", mode="sequential")

    def test_insertion_scan_assumption_insert_seq_length_known(self):
        with pp.Party():
            variable_insert = pp.from_seqs(["A", "TT"], mode="sequential")
            with pytest.raises(ValueError, match="ins_pool must have a defined seq_length"):
                pp.insertion_scan("AACCGGTT", variable_insert, mode="sequential")

    def test_insertion_scan_assumption_named_region_exists(self):
        with pp.Party():
            out = pp.insertion_scan("AACCGGTT", "A", region="missing", positions=[0], mode="sequential")
            with pytest.raises(ValueError, match="Region 'missing' not found"):
                out.generate_library(num_cycles=1)

    # mutagenize_scan assumptions
    def test_mutagenize_scan_assumption_exactly_one_mutation_parameter(self):
        with pp.Party():
            with pytest.raises(ValueError, match="must be provided"):
                pp.mutagenize_scan("AACCGGTT", mutagenize_length=2, mode=("sequential", "sequential"))
            with pytest.raises(ValueError, match="Only one of"):
                pp.mutagenize_scan(
                    "AACCGGTT",
                    mutagenize_length=2,
                    num_mutations=1,
                    mutation_rate=0.1,
                    mode=("sequential", "sequential"),
                )

    def test_mutagenize_scan_assumption_named_region_exists(self):
        with pp.Party():
            out = pp.mutagenize_scan(
                "AACCGGTT",
                mutagenize_length=2,
                num_mutations=1,
                region="missing",
                mode=("sequential", "sequential"),
            )
            with pytest.raises(ValueError, match="Region 'missing' not found"):
                out.generate_library(num_cycles=1)

    # deletion_multiscan assumptions
    def test_deletion_multiscan_assumption_fit_non_overlapping_deletions(self):
        with pp.Party():
            with pytest.raises(ValueError, match="Cannot fit"):
                pp.deletion_multiscan("AACCGGTT", deletion_length=3, num_deletions=3, mode="sequential")

    def test_deletion_multiscan_assumption_names_length_matches_num_deletions(self):
        with pp.Party():
            with pytest.raises(ValueError, match="len\\(names\\)"):
                pp.deletion_multiscan(
                    "AACCGGTT",
                    deletion_length=1,
                    num_deletions=2,
                    names=["only_one"],
                    mode="sequential",
                )

    # insertion_multiscan assumptions
    def test_insertion_multiscan_assumption_pool_count_matches_num_insertions(self):
        with pp.Party():
            with pytest.raises(ValueError, match="insertion_pools length"):
                pp.insertion_multiscan(
                    "AACCGGTT",
                    num_insertions=2,
                    insertion_pools=[pp.from_seq("A")],
                    mode="sequential",
                )

    def test_insertion_multiscan_assumption_replacement_pools_have_known_length(self):
        with pp.Party():
            with pytest.raises(ValueError, match="must have a defined seq_length"):
                pp.insertion_multiscan(
                    "AACCGGTT",
                    num_insertions=2,
                    insertion_pools=[pp.from_seq("A"), pp.from_seqs(["A", "TT"], mode="sequential")],
                    replace=True,
                    mode="sequential",
                )

    def test_insertion_multiscan_assumption_per_insert_positions_shape_clear_error(self):
        with pp.Party():
            with pytest.raises(ValueError, match="per-insert positions has 1 sublists"):
                pp.insertion_multiscan(
                    "AACCGGTT",
                    num_insertions=2,
                    insertion_pools=pp.from_seq("A"),
                    positions=[[0, 1, 2]],
                    mode="sequential",
                )

