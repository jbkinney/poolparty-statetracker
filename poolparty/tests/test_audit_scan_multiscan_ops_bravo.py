"""Audit tests for scan and multiscan operations (bravo).

Covers:
  Scan ops: deletion_scan, insertion_scan, replacement_scan,
            shuffle_scan, mutagenize_scan, subseq_scan
  Multiscan ops: deletion_multiscan, insertion_multiscan, replacement_multiscan

Follows operation_audit.mdc Steps 1-7.
"""

import inspect
import warnings

import pytest

import poolparty as pp
from poolparty.region_ops import strip_all_tags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bio_len(seq_str: str) -> int:
    """Biological (nontag) length of a sequence string."""
    return len(strip_all_tags(seq_str))


# ===================================================================
# Step 2: Smoke tests — minimal-input construct + generate_library
# ===================================================================

class TestSmokeAllFactories:
    """Smoke test every factory with minimal valid inputs.

    Any crash here is a Critical finding (completely non-functional op).
    """

    def test_smoke_deletion_scan(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.deletion_scan(pool, deletion_length=1)
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_smoke_insertion_scan(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.insertion_scan(pool, pp.from_seq("N"))
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_smoke_replacement_scan(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.replacement_scan(pool, pp.from_seq("N"))
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_smoke_shuffle_scan(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.shuffle_scan(pool, shuffle_length=2)
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_smoke_mutagenize_scan(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.mutagenize_scan(
                pool, mutagenize_length=2, num_mutations=1,
                mode=("random", "random"),
            )
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_smoke_subseq_scan(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.subseq_scan(pool, subseq_length=2)
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_smoke_deletion_multiscan(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_multiscan(pool, deletion_length=1, num_deletions=2)
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_smoke_insertion_multiscan(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.insertion_multiscan(
                pool, num_insertions=2, insertion_pools=pp.from_seq("N")
            )
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0

    def test_smoke_replacement_multiscan(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.replacement_multiscan(
                pool, num_replacements=2, replacement_pools=pp.from_seq("N")
            )
            df = result.generate_library(num_cycles=1)
            assert len(df) > 0


# ===================================================================
# Step 2: hasattr mixin existence checks
# ===================================================================

class TestMixinExistence:
    """Verify each scan/multiscan factory has a corresponding pool method."""

    def test_all_scan_ops_have_pool_methods(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            for method_name in [
                "deletion_scan", "insertion_scan", "replacement_scan",
                "shuffle_scan", "mutagenize_scan", "subseq_scan",
                "deletion_multiscan", "insertion_multiscan", "replacement_multiscan",
            ]:
                assert hasattr(pool, method_name), (
                    f"Pool missing mixin method: {method_name}"
                )
                assert callable(getattr(pool, method_name))


# ===================================================================
# I1 + I2 baseline for ALL 9 ops
# ===================================================================

class TestScanOpsBaseline:
    """I1 (output length) and I2 (state exhaustion) for every scan/multiscan op."""

    # -- deletion_scan --

    def test_deletion_scan_I1_I2_with_marker(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(pool, deletion_length=2, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    def test_deletion_scan_I1_I2_no_marker(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=2, deletion_marker=None, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 6

    # -- insertion_scan --

    def test_insertion_scan_I1_I2(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 10

    # -- replacement_scan --

    def test_replacement_scan_I1_I2(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.replacement_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    # -- shuffle_scan --

    def test_shuffle_scan_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.shuffle_scan(pool, shuffle_length=3, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    # -- mutagenize_scan --

    def test_mutagenize_scan_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool,
                    mutagenize_length=3,
                    num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    # -- subseq_scan --

    def test_subseq_scan_I1_I2(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.subseq_scan(pool, subseq_length=4, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 4
            assert result.seq_length == 4

    # -- deletion_multiscan --

    def test_deletion_multiscan_I1_I2_with_marker(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            result = pp.deletion_multiscan(
                pool, deletion_length=2, num_deletions=2, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 12

    def test_deletion_multiscan_I1_I2_no_marker(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            result = pp.deletion_multiscan(
                pool,
                deletion_length=2,
                num_deletions=2,
                deletion_marker=None,
                mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    # -- insertion_multiscan --

    def test_insertion_multiscan_I1_I2(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 12

    # -- replacement_multiscan --

    def test_replacement_multiscan_I1_I2(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGT")
            rep = pp.from_seq("NN")
            result = pp.replacement_multiscan(
                bg, num_replacements=2, replacement_pools=rep, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 12


# ===================================================================
# I5 — Determinism for ALL ops (random mode)
# ===================================================================

class TestScanOpsDeterminism:
    """I5: random mode with seed produces identical output twice."""

    def _check_determinism(self, build_fn):
        """Run build_fn twice with same seed, assert identical output."""
        with pp.Party() as _:
            pool = build_fn()
            df1 = pool.generate_library(seed=42)
        with pp.Party() as _:
            pool = build_fn()
            df2 = pool.generate_library(seed=42)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_deletion_scan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGT")
            return pp.deletion_scan(bg, deletion_length=2, mode="random", num_states=5)
        self._check_determinism(build)

    def test_insertion_scan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            return pp.insertion_scan(bg, ins, mode="random", num_states=5)
        self._check_determinism(build)

    def test_replacement_scan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            return pp.replacement_scan(bg, ins, mode="random", num_states=5)
        self._check_determinism(build)

    def test_shuffle_scan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGT")
            return pp.shuffle_scan(bg, shuffle_length=3, mode="random", num_states=5)
        self._check_determinism(build)

    def test_mutagenize_scan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGT")
            return pp.mutagenize_scan(
                bg, mutagenize_length=3, num_mutations=1,
                mode=("random", "random"), num_states=(3, 3),
            )
        self._check_determinism(build)

    def test_subseq_scan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGT")
            return pp.subseq_scan(bg, subseq_length=4, mode="random", num_states=5)
        self._check_determinism(build)

    def test_deletion_multiscan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGTACGT")
            return pp.deletion_multiscan(
                bg, deletion_length=2, num_deletions=2, mode="random", num_states=5
            )
        self._check_determinism(build)

    def test_insertion_multiscan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            return pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="random", num_states=5
            )
        self._check_determinism(build)

    def test_replacement_multiscan_determinism(self):
        def build():
            bg = pp.from_seq("ACGTACGTACGT")
            rep = pp.from_seq("NN")
            return pp.replacement_multiscan(
                bg, num_replacements=2, replacement_pools=rep, mode="random", num_states=5
            )
        self._check_determinism(build)


# ===================================================================
# I6 — Region-only modification for ALL ops that support region=
# ===================================================================

class TestScanOpsRegionIsolation:
    """I6: when region= is set, prefix/suffix outside region are preserved."""

    BG_SEQ = "AA<core>CCGG</core>TT"

    def test_deletion_scan_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq(self.BG_SEQ)
            result = pp.deletion_scan(
                pool, deletion_length=1, region="core", mode="sequential"
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_insertion_scan_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq(self.BG_SEQ)
            ins = pp.from_seq("X")
            result = pp.insertion_scan(pool, ins, region="core", mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_replacement_scan_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGG</core>TT")
            ins = pp.from_seq("X")
            result = pp.replacement_scan(pool, ins, region="core", mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_shuffle_scan_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGG</core>TT")
            result = pp.shuffle_scan(
                pool, shuffle_length=2, region="core", mode="sequential"
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_mutagenize_scan_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGG</core>TT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool,
                    mutagenize_length=2,
                    num_mutations=1,
                    region="core",
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_subseq_scan_with_named_region(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGG</core>TT")
            result = pp.subseq_scan(pool, subseq_length=2, region="core", mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 2

    def test_subseq_scan_with_interval_region(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.subseq_scan(
                pool, subseq_length=2, region=[2, 6], mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 2

    def test_deletion_multiscan_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGGAATT</core>TT")
            result = pp.deletion_multiscan(
                pool,
                deletion_length=1,
                num_deletions=2,
                region="core",
                mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_insertion_multiscan_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGGAATT</core>TT")
            ins = pp.from_seq("X")
            result = pp.insertion_multiscan(
                pool,
                num_insertions=2,
                insertion_pools=ins,
                region="core",
                mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_replacement_multiscan_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGGAATT</core>TT")
            rep = pp.from_seq("X")
            result = pp.replacement_multiscan(
                pool,
                num_replacements=2,
                replacement_pools=rep,
                region="core",
                mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")


# ===================================================================
# I3 — Card-sequence agreement for ops exposing cards
# ===================================================================

class TestScanOpsCards:
    """I3: card claims match actual sequence output."""

    def test_deletion_scan_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=2, mode="sequential",
                cards=["start", "end"],
            )
            df = result.generate_library()
            for _, row in df.iterrows():
                start_col = [c for c in df.columns if c.endswith(".start")]
                end_col = [c for c in df.columns if c.endswith(".end")]
                if start_col and end_col:
                    start = row[start_col[0]]
                    end = row[end_col[0]]
                    assert end - start == 2

    def test_insertion_scan_cards(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.insertion_scan(
                bg, ins, mode="sequential",
                cards=["start", "end"],
            )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".start")]
            assert len(start_cols) >= 1

    def test_shuffle_scan_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.shuffle_scan(
                pool, shuffle_length=3, mode="sequential",
                cards=(["start", "end"], None),
            )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".start")]
            assert len(start_cols) >= 1

    def test_mutagenize_scan_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool,
                    mutagenize_length=3,
                    num_mutations=1,
                    mode=("sequential", "sequential"),
                    cards=(["start", "end"], ["positions"]),
                )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".start")]
            pos_cols = [c for c in df.columns if c.endswith(".positions")]
            assert len(start_cols) >= 1
            assert len(pos_cols) >= 1

    def test_subseq_scan_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.subseq_scan(
                pool, subseq_length=4, mode="sequential",
                cards=["start", "end"],
            )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".start")]
            assert len(start_cols) >= 1

    def test_deletion_multiscan_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            result = pp.deletion_multiscan(
                pool, deletion_length=2, num_deletions=2,
                mode="sequential", cards=["starts", "ends"],
            )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".starts")]
            assert len(start_cols) >= 1

    def test_insertion_multiscan_cards(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("N")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins,
                mode="sequential", cards=["starts", "ends"],
            )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".starts")]
            assert len(start_cols) >= 1

    def test_replacement_multiscan_cards(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGT")
            rep = pp.from_seq("NN")
            result = pp.replacement_multiscan(
                bg, num_replacements=2, replacement_pools=rep,
                mode="sequential", cards=["starts", "ends"],
            )
            df = result.generate_library()
            start_cols = [c for c in df.columns if c.endswith(".starts")]
            assert len(start_cols) >= 1


# ===================================================================
# I10 — State-space immutability during compute
# ===================================================================

class TestScanOpsStateImmutability:
    """I10: num_states and state._num_values unchanged after generate_library."""

    def _check_immutability(self, pool):
        ns_before = pool.num_states
        sv_before = pool.operation.state._num_values
        pool.generate_library(seed=1)
        assert pool.num_states == ns_before
        assert pool.operation.state._num_values == sv_before

    def test_deletion_scan_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(pool, deletion_length=2, mode="sequential")
            self._check_immutability(result)

    def test_insertion_scan_state_immutability(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            self._check_immutability(result)

    def test_mutagenize_scan_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=3, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            self._check_immutability(result)

    def test_insertion_multiscan_state_immutability(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("N")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="sequential"
            )
            self._check_immutability(result)


# ===================================================================
# Nested-tag empirical tests (Step 3 contract dimension)
# ===================================================================

class TestNestedTagBehavior:
    """Verify tag behavior with nested-tag inputs.

    Uses AA<outer>CC<inner>GG</inner>TT</outer>AA as the canonical
    nested-tag input.
    """

    NESTED_SEQ = "AA<outer>CC<inner>GG</inner>TT</outer>AA"

    @pytest.mark.xfail(
        reason="Scan tag insertion into nested-tag seq produces crossing XML tags",
        raises=ValueError,
        strict=True,
    )
    def test_deletion_scan_preserves_nested_tags(self):
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            result = pp.deletion_scan(
                pool, deletion_length=1, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == bio_len(self.NESTED_SEQ)

    def test_insertion_scan_preserves_nested_tags(self):
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            ins = pp.from_seq("X")
            result = pp.insertion_scan(pool, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == bio_len(self.NESTED_SEQ) + 1

    @pytest.mark.xfail(
        reason="Scan tag insertion into nested-tag seq produces crossing XML tags",
        raises=ValueError,
        strict=True,
    )
    def test_shuffle_scan_preserves_nested_tags(self):
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            result = pp.shuffle_scan(
                pool, shuffle_length=2, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == bio_len(self.NESTED_SEQ)

    @pytest.mark.xfail(
        reason="Scan tag insertion into nested-tag seq produces crossing XML tags",
        raises=ValueError,
        strict=True,
    )
    def test_mutagenize_scan_preserves_nested_tags(self):
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=2, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == bio_len(self.NESTED_SEQ)

    def test_insertion_scan_region_inside_nested(self):
        """Insertion scan constrained to inner region of nested tags."""
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            ins = pp.from_seq("X")
            result = pp.insertion_scan(
                pool, ins, region="inner", mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("AA")

    @pytest.mark.xfail(
        reason="Scan tag insertion into nested-tag seq produces crossing XML tags",
        raises=ValueError,
        strict=True,
    )
    def test_deletion_multiscan_preserves_nested_tags(self):
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            result = pp.deletion_multiscan(
                pool, deletion_length=1, num_deletions=2, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    @pytest.mark.xfail(
        reason="Scan tag insertion into nested-tag seq produces crossing XML tags",
        raises=ValueError,
        strict=True,
    )
    def test_replacement_scan_preserves_nested_tags(self):
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            ins = pp.from_seq("X")
            result = pp.replacement_scan(pool, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states

    @pytest.mark.xfail(
        reason="Scan tag insertion into nested-tag seq produces crossing XML tags",
        raises=ValueError,
        strict=True,
    )
    def test_subseq_scan_with_nested_tags(self):
        """subseq_scan extracts correct-length subseqs from nested-tag input."""
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            result = pp.subseq_scan(pool, subseq_length=3, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 3

    def test_insertion_multiscan_preserves_nested_tags(self):
        """insertion_multiscan succeeds with nested tags (unlike 7 other ops)."""
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            ins = pp.from_seq("A")
            result = pp.insertion_multiscan(
                pool, num_insertions=2, insertion_pools=ins, mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert len(df) > 0

    @pytest.mark.xfail(
        reason="Scan tag insertion into nested-tag seq produces crossing XML tags",
        raises=ValueError,
        strict=True,
    )
    def test_replacement_multiscan_preserves_nested_tags(self):
        with pp.Party():
            pool = pp.from_seq(self.NESTED_SEQ)
            rep = pp.from_seq("A")
            result = pp.replacement_multiscan(
                pool, num_replacements=2, replacement_pools=rep, mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states


class TestPoolTypePreservation:
    """Verify pool type preservation for non-source unary ops.

    type(output_pool) should match type(input_pool) for scan/multiscan ops.
    """

    def test_deletion_scan_preserves_pool_type(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.deletion_scan(pool, deletion_length=1, mode="sequential")
            assert type(result) == type(pool)

    def test_insertion_scan_preserves_pool_type(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            ins = pp.from_seq("N")
            result = pp.insertion_scan(pool, ins, mode="sequential")
            assert type(result) == type(pool)

    def test_insertion_multiscan_preserves_pool_type(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("N")
            result = pp.insertion_multiscan(
                pool, num_insertions=2, insertion_pools=ins, mode="sequential"
            )
            assert type(result) == type(pool)


# ===================================================================
# Group A representative: insertion_scan — full I1–I10 + domain
# ===================================================================

class TestInsertionScanFull:
    """Full invariant suite for insertion_scan (Group A representative)."""

    def test_I1_insert_mode_output_length(self):
        """Output length = bg + ins for insertion mode."""
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("NN")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 6

    def test_I1_replace_mode_output_length(self):
        """Output length = bg for replacement mode."""
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("NN")
            result = pp.replacement_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 4

    def test_I2_sequential_exhaustion(self):
        """Sequential mode enumerates all valid positions."""
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("X")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 5

    def test_I2_replacement_sequential_exhaustion(self):
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("X")
            result = pp.replacement_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 4

    def test_I7_composition_cartesian_product(self):
        """Chained sequential ops produce correct Cartesian product."""
        with pp.Party():
            bg = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            ins = pp.from_seq("X")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 2 * 5

    def test_I8_replacement_length_algebra(self):
        """seq_length = parent_len - region_len + insert_len for replacement."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NNN")
            result = pp.replacement_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    def test_D1_position_enumeration_covers_all(self):
        """Sequential mode visits every valid position exactly once."""
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("X")
            result = pp.insertion_scan(
                bg, ins, mode="sequential",
                cards=["start"],
            )
            df = result.generate_library()
            start_col = [c for c in df.columns if c.endswith(".start")][0]
            positions_seen = sorted(df[start_col].tolist())
            assert positions_seen == [0, 1, 2, 3, 4]

    def test_D2_insert_at_correct_position(self):
        """Insert content appears at the declared position in output."""
        with pp.Party():
            bg = pp.from_seq("AAAA")
            ins = pp.from_seq("X")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            seqs = [strip_all_tags(s) for s in df["seq"]]
            assert "XAAAA" in seqs
            assert "AXAAA" in seqs
            assert "AAXAA" in seqs
            assert "AAAXА" in seqs or "AAAXA" in seqs
            assert "AAAAX" in seqs

    def test_insertion_scan_with_multiple_insert_states(self):
        """Insert pool with multiple states creates Cartesian product."""
        with pp.Party():
            bg = pp.from_seq("AAAA")
            ins = pp.from_seqs(["X", "Y"], mode="sequential")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 5 * 2


# ===================================================================
# Group B representative: mutagenize_scan — full suite
# ===================================================================

class TestMutagenizeScanFull:
    """Full invariant suite for mutagenize_scan (Group B representative)."""

    def test_I1_output_length_preserved(self):
        """Mutagenize_scan preserves sequence length."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=3, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    def test_I2_sequential_exhaustion_both_dims(self):
        """Both scan and mutagenize in sequential mode: correct total states."""
        with pp.Party():
            pool = pp.from_seq("ACGTAC")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=3, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_I7_composition_with_parent(self):
        """Chained from_seqs -> mutagenize_scan produces Cartesian product."""
        with pp.Party():
            bg = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    bg, mutagenize_length=2, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states % 2 == 0

    def test_D1_position_enumeration(self):
        """Sequential scan positions cover all valid windows."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=2, num_mutations=1,
                    mode=("sequential", "sequential"),
                    cards=(["start"], ["positions"]),
                )
            df = result.generate_library()
            start_col = [c for c in df.columns if c.endswith(".start")][0]
            positions_seen = sorted(set(df[start_col].tolist()))
            assert positions_seen == [0, 1, 2]

    def test_D3_mutations_within_window_only(self):
        """Mutations only occur within the mutagenize window, not outside."""
        with pp.Party():
            pool = pp.from_seq("AAAA")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=2, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert len(clean) == 4
                diffs = [i for i in range(4) if clean[i] != "A"]
                assert len(diffs) <= 1

    def test_tuple_mode_independent_control(self):
        """Tuple mode controls scan and mutagenize independently."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.mutagenize_scan(
                pool, mutagenize_length=2, num_mutations=1,
                mode=("sequential", "random"), num_states=(None, 3),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_scalar_mode_broadcast_warns(self):
        """Scalar mode emits broadcast warning."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pp.mutagenize_scan(
                    pool, mutagenize_length=2, num_mutations=1,
                    mode="sequential",
                )
            broadcast_warns = [x for x in w if "broadcast" in str(x.message).lower()]
            assert len(broadcast_warns) >= 1

    def test_scalar_num_states_broadcast_warns(self):
        """Scalar num_states emits broadcast warning."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pp.mutagenize_scan(
                    pool, mutagenize_length=2, num_mutations=1,
                    mode=("random", "random"), num_states=3,
                )
            broadcast_warns = [x for x in w if "broadcast" in str(x.message).lower()]
            assert len(broadcast_warns) >= 1


# ===================================================================
# Group D representative: insertion_multiscan — full suite
# ===================================================================

class TestInsertionMultiscanFull:
    """Full invariant suite for insertion_multiscan (Group D representative)."""

    def test_I1_output_length_insert(self):
        """Output length = bg + sum(ins_lengths) for insert mode."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="sequential"
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 12

    def test_I1_output_length_replace(self):
        """Output length preserved for replacement mode."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGT")
            rep = pp.from_seq("NN")
            result = pp.replacement_multiscan(
                bg, num_replacements=2, replacement_pools=rep, mode="sequential"
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 12

    def test_I2_sequential_exhaustion(self):
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("N")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_I7_composition(self):
        """Chained from_seqs -> insertion_multiscan produces Cartesian product."""
        with pp.Party():
            bg = pp.from_seqs(["ACGTACGT", "TGCATGCA"], mode="sequential")
            ins = pp.from_seq("N")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states % 2 == 0

    def test_D1_non_overlapping_insertions(self):
        """Inserted regions do not overlap in sequential output."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="sequential",
                cards=["starts", "ends"],
            )
            df = result.generate_library()
            starts_col = [c for c in df.columns if c.endswith(".starts")]
            ends_col = [c for c in df.columns if c.endswith(".ends")]
            if starts_col and ends_col:
                for _, row in df.iterrows():
                    starts = row[starts_col[0]]
                    ends = row[ends_col[0]]
                    if len(starts) == 2:
                        assert ends[0] <= starts[1]

    def test_D2_spacing_constraints_respected(self):
        """Min/max spacing constraints are respected."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGT")
            ins = pp.from_seq("N")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins,
                min_spacing=2, mode="sequential",
                cards=["starts"],
            )
            df = result.generate_library()
            starts_col = [c for c in df.columns if c.endswith(".starts")]
            if starts_col:
                for _, row in df.iterrows():
                    starts = row[starts_col[0]]
                    if len(starts) == 2:
                        gap = starts[1] - starts[0]
                        assert gap >= 2

    def test_deepcopy_single_pool_independent_states(self):
        """Single pool input is deepcopied; copies are independent."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seqs(["X", "Y"], mode="sequential")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert len(df) > 0

    def test_mixed_length_replacement_pools(self):
        """Different-length replacement pools work correctly."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGT")
            rep1 = pp.from_seq("XX")
            rep2 = pp.from_seq("YYY")
            result = pp.replacement_multiscan(
                bg, num_replacements=2, replacement_pools=[rep1, rep2],
                mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 12


# ===================================================================
# Mixin runtime forwarding tests
# ===================================================================

class TestMixinForwarding:
    """Verify mixin methods don't raise TypeError for all supported params."""

    def test_deletion_scan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pool.deletion_scan(
                deletion_length=2, deletion_marker="-",
                positions=[0, 1, 2], region=None,
                prefix="del", mode="sequential",
                num_states=3, style="red",
                iter_order=1, cards=["start"],
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_insertion_scan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pool.insertion_scan(
                ins_pool=ins, positions=[0, 1, 2],
                region=None, replace=False,
                style="blue", prefix="ins",
                prefix_position="pos", prefix_insert="content",
                mode="sequential", num_states=3,
                iter_order=1, cards=["start"],
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_replacement_scan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pool.replacement_scan(
                ins_pool=ins, positions=[0, 1, 2],
                region=None, style="blue",
                prefix="rep", prefix_position="pos",
                prefix_insert="content", mode="sequential",
                num_states=3, iter_order=1,
                cards=["start"],
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_shuffle_scan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pool.shuffle_scan(
                shuffle_length=3, positions=[0, 1, 2],
                region=None, shuffle_type="mono",
                shuffles_per_position=1,
                prefix="shuf", prefix_position="pos",
                prefix_shuffle="var",
                mode="sequential", num_states=3,
                style="green", iter_order=1,
                cards=(["start"], None),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_mutagenize_scan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pool.mutagenize_scan(
                mutagenize_length=3, num_mutations=1,
                positions=[0, 1, 2], region=None,
                prefix=("scan", "mut"),
                mode=("sequential", "random"),
                num_states=(3, 2), style="red",
                iter_order=(1, -1),
                cards=(["start"], None),
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_subseq_scan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pool.subseq_scan(
                subseq_length=4, positions=[0, 1, 2],
                region=None, prefix="sub",
                mode="sequential", num_states=3,
                iter_order=1, cards=["start"],
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_deletion_multiscan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            result = pool.deletion_multiscan(
                deletion_length=2, num_deletions=2,
                deletion_marker="-", positions=None,
                region=None, names=["d0", "d1"],
                min_spacing=0, max_spacing=None,
                prefix="del", mode="sequential",
                num_states=None, style="red",
                iter_order=1, cards=["starts"],
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_insertion_multiscan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("N")
            result = pool.insertion_multiscan(
                num_insertions=2, insertion_pools=ins,
                positions=None, region=None,
                names=["i0", "i1"], replace=False,
                insertion_mode="ordered",
                min_spacing=0, max_spacing=None,
                prefix="ins", mode="sequential",
                num_states=None, iter_order=1,
                cards=["starts"],
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_replacement_multiscan_mixin(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            rep = pp.from_seq("NN")
            result = pool.replacement_multiscan(
                num_replacements=2, replacement_pools=rep,
                positions=None, region=None,
                names=["r0", "r1"],
                insertion_mode="ordered",
                min_spacing=0, max_spacing=None,
                prefix="rep", mode="sequential",
                num_states=None, iter_order=1,
                cards=["starts"],
            )
            df = result.generate_library()
            assert len(df) == result.num_states


# ===================================================================
# API consistency checks (Step 6)
# ===================================================================

class TestAPIConsistency:
    """API consistency checks across scan/multiscan ops."""

    def test_fixed_mode_rejected_by_region_scan(self):
        """Scan ops delegate to region_scan which rejects mode='fixed'."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="mode must be"):
                pp.deletion_scan(pool, deletion_length=1, mode="fixed")

    def test_fixed_mode_rejected_by_region_multiscan(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            with pytest.raises(ValueError, match="mode must be"):
                pp.deletion_multiscan(
                    pool, deletion_length=1, num_deletions=2, mode="fixed"
                )

    def test_num_states_cycling(self):
        """num_states > natural cycles correctly."""
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("X")
            natural = 5
            result = pp.insertion_scan(
                bg, ins, mode="sequential", num_states=natural * 2
            )
            df = result.generate_library()
            assert len(df) == natural * 2
            first_half = [strip_all_tags(s) for s in df["seq"].iloc[:natural]]
            second_half = [strip_all_tags(s) for s in df["seq"].iloc[natural:]]
            assert first_half == second_half

    def test_num_states_clipping(self):
        """num_states < natural clips output."""
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("X")
            result = pp.insertion_scan(bg, ins, mode="sequential", num_states=3)
            df = result.generate_library()
            assert len(df) == 3

    def test_replacement_scan_wraps_insertion_scan(self):
        """replacement_scan produces same output as insertion_scan(replace=True)."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            r1 = pp.replacement_scan(bg, ins, mode="sequential")
            df1 = r1.generate_library(seed=1)
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            r2 = pp.insertion_scan(bg, ins, replace=True, mode="sequential")
            df2 = r2.generate_library(seed=1)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_replacement_multiscan_wraps_insertion_multiscan(self):
        """replacement_multiscan produces same as insertion_multiscan(replace=True)."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGT")
            rep = pp.from_seq("NN")
            r1 = pp.replacement_multiscan(
                bg, num_replacements=2, replacement_pools=rep, mode="sequential"
            )
            df1 = r1.generate_library(seed=1)
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGT")
            rep = pp.from_seq("NN")
            r2 = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=rep,
                replace=True, mode="sequential",
            )
            df2 = r2.generate_library(seed=1)
        assert list(df1["seq"]) == list(df2["seq"])

    def test_insertion_multiscan_has_style(self):
        """insertion_multiscan has style param like deletion_multiscan and insertion_scan."""
        ins_ms_params = inspect.signature(pp.insertion_multiscan).parameters
        del_ms_params = inspect.signature(pp.deletion_multiscan).parameters
        ins_s_params = inspect.signature(pp.insertion_scan).parameters

        assert "style" in ins_ms_params
        assert "style" in del_ms_params
        assert "style" in ins_s_params

    def test_replacement_multiscan_has_style(self):
        """replacement_multiscan has style param."""
        rep_ms_params = inspect.signature(pp.replacement_multiscan).parameters
        assert "style" in rep_ms_params

    def test_deletion_scan_has_factory_name(self):
        """deletion_scan now has _factory_name param (F9 fixed)."""
        del_params = inspect.signature(pp.deletion_scan).parameters
        ins_params = inspect.signature(pp.insertion_scan).parameters
        assert "_factory_name" in del_params
        assert "_factory_name" in ins_params

    def test_subseq_scan_has_factory_name(self):
        """subseq_scan now has _factory_name param (F10 fixed)."""
        sub_params = inspect.signature(pp.subseq_scan).parameters
        assert "_factory_name" in sub_params

    def test_all_scan_ops_have_beartype(self):
        """All scan/multiscan factory functions have @beartype."""
        for fn in [
            pp.deletion_scan, pp.insertion_scan, pp.replacement_scan,
            pp.shuffle_scan, pp.mutagenize_scan, pp.subseq_scan,
            pp.deletion_multiscan, pp.insertion_multiscan, pp.replacement_multiscan,
        ]:
            assert hasattr(fn, "__wrapped__"), f"{fn.__name__} missing @beartype"


# ===================================================================
# Adversarial: insertion_scan (HIGH risk)
# ===================================================================

class TestInsertionScanAdversarial:
    """Adversarial patterns for insertion_scan."""

    def test_short_seq_interval_region_sequential(self):
        """Short sequence (3 bp) + interval region + sequential + cycling."""
        with pp.Party():
            bg = pp.from_seq("ACG")
            ins = pp.from_seq("X")
            result = pp.insertion_scan(
                bg, ins, region=[1, 2], mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean[0] == "A"
                assert clean[-1] == "G"

    def test_variable_length_parent_rejects(self):
        """Variable-length parent (seq_length=None) raises ValueError."""
        with pp.Party():
            bg = pp.from_seqs(["ACG", "TGCA"], mode="sequential")
            ins = pp.from_seq("X")
            with pytest.raises(ValueError, match="seq_length"):
                pp.insertion_scan(bg, ins, mode="sequential")

    def test_chained_sequential_cartesian(self):
        """from_seqs(sequential) -> insertion_scan(sequential) = Cartesian product."""
        with pp.Party():
            bg = pp.from_seqs(["AAAA", "CCCC"], mode="sequential")
            ins = pp.from_seq("X")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 2 * 5
            seqs = [strip_all_tags(s) for s in df["seq"]]
            a_seqs = [s for s in seqs if "A" in s]
            c_seqs = [s for s in seqs if "C" in s]
            assert len(a_seqs) == 5
            assert len(c_seqs) == 5

    def test_insert_pool_with_states_region_combination(self):
        """Multi-state insert + region constraint."""
        with pp.Party():
            bg = pp.from_seq("AA<core>CCGG</core>TT")
            ins = pp.from_seqs(["X", "Y"], mode="sequential")
            result = pp.insertion_scan(bg, ins, region="core", mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")


# ===================================================================
# Adversarial: mutagenize_scan (HIGH risk)
# ===================================================================

class TestMutagenizeScanAdversarial:
    """Adversarial patterns for mutagenize_scan."""

    def test_mixed_mode_with_region(self):
        """Sequential scan + random mutagenize + region constraint."""
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGG</core>TT")
            result = pp.mutagenize_scan(
                pool, mutagenize_length=2, num_mutations=1,
                region="core",
                mode=("sequential", "random"), num_states=(None, 3),
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_scalar_mode_broadcast_correct_states(self):
        """Scalar mode broadcast produces correct total state count."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=2, num_mutations=1,
                    mode="sequential",
                )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_min_mutagenize_length_1bp(self):
        """Mutagenize window of 1 bp with 1 mutation covers all positions."""
        with pp.Party():
            pool = pp.from_seq("AAAA")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=1, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert len(clean) == 4
                diffs = sum(1 for i in range(4) if clean[i] != "A")
                assert diffs <= 1

    def test_mutation_rate_instead_of_num_mutations(self):
        """mutation_rate works with mutagenize_scan."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.mutagenize_scan(
                pool, mutagenize_length=4, mutation_rate=0.5,
                mode=("sequential", "random"), num_states=(None, 3),
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 8


# ===================================================================
# Adversarial: insertion_multiscan (HIGH risk)
# ===================================================================

class TestInsertionMultiscanAdversarial:
    """Adversarial patterns for insertion_multiscan."""

    def test_single_pool_deepcopy_independent(self):
        """Single pool is deepcopied; copies iterate independently."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seqs(["X", "Y"], mode="sequential")
            result = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            seqs = [strip_all_tags(s) for s in df["seq"]]
            assert len(set(seqs)) > 1

    def test_mixed_length_pools_spacing_sequential(self):
        """Different-length replacement pools with spacing in sequential mode."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGTACGT")
            rep1 = pp.from_seq("XX")
            rep2 = pp.from_seq("YYY")
            result = pp.replacement_multiscan(
                bg, num_replacements=2,
                replacement_pools=[rep1, rep2],
                min_spacing=1,
                mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert len(clean) == 16

    def test_tight_packing_max_insertions(self):
        """Maximum insertions that fit: tight packing with sequential exhaustion."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("X")
            result = pp.insertion_multiscan(
                bg, num_insertions=3, insertion_pools=ins,
                mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 11

    def test_replacement_multiscan_region_constraint(self):
        """Replacement multiscan with named region constraint."""
        with pp.Party():
            bg = pp.from_seq("AA<core>CCGGAATT</core>TT")
            rep = pp.from_seq("X")
            result = pp.replacement_multiscan(
                bg, num_replacements=2,
                replacement_pools=rep,
                region="core",
                mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")


# ===================================================================
# Domain-specific: shuffle_scan
# ===================================================================

class TestShuffleScanDomain:
    """Domain-specific invariants for shuffle_scan."""

    def test_shuffle_preserves_composition(self):
        """Shuffled region preserves nucleotide composition."""
        with pp.Party():
            pool = pp.from_seq("AACCGGTT")
            result = pp.shuffle_scan(
                pool, shuffle_length=4, mode="sequential",
                shuffles_per_position=1,
            )
            df = result.generate_library()
            parent_seq = "AACCGGTT"
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert sorted(clean) == sorted(parent_seq)

    def test_dinuc_shuffle_type(self):
        """Dinucleotide shuffle type is accepted."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.shuffle_scan(
                pool, shuffle_length=4,
                shuffle_type="dinuc",
                mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_shuffles_per_position_multiplies_states(self):
        """shuffles_per_position > 1 multiplies total states."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            r1 = pp.shuffle_scan(pool, shuffle_length=3, mode="sequential",
                                  shuffles_per_position=1)
            n1 = r1.num_states
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            r2 = pp.shuffle_scan(pool, shuffle_length=3, mode="sequential",
                                  shuffles_per_position=3)
            n2 = r2.num_states
        assert n2 == n1 * 3


# ===================================================================
# Domain-specific: subseq_scan
# ===================================================================

class TestSubseqScanDomain:
    """Domain-specific invariants for subseq_scan."""

    def test_extracted_subseqs_match_parent(self):
        """Extracted subsequences are genuine substrings of parent."""
        with pp.Party():
            parent = "ACGTACGT"
            pool = pp.from_seq(parent)
            result = pp.subseq_scan(pool, subseq_length=3, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean in parent

    def test_all_windows_covered(self):
        """Sequential mode covers all windows of the given length."""
        with pp.Party():
            parent = "ACGTAC"
            pool = pp.from_seq(parent)
            result = pp.subseq_scan(pool, subseq_length=3, mode="sequential")
            df = result.generate_library()
            expected_subseqs = {parent[i:i+3] for i in range(4)}
            actual_subseqs = {strip_all_tags(s) for s in df["seq"]}
            assert actual_subseqs == expected_subseqs


# ===================================================================
# Domain-specific: deletion_scan
# ===================================================================

class TestDeletionScanDomain:
    """Domain-specific invariants for deletion_scan."""

    def test_marker_content_matches_deletion_length(self):
        """Marker characters appear with correct count."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=3, deletion_marker="*",
                mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.count("*") == 3

    def test_no_marker_reduces_length(self):
        """deletion_marker=None removes content, reducing length."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=3, deletion_marker=None,
                mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 5


# ===================================================================
# Assumption inversion (Step 4, mandatory for HIGH-risk ops)
# ===================================================================

class TestInsertionScanAssumptionInversion:
    """Identify implicit assumptions in insertion_scan and violate them.

    Assumptions from reading the code:
    A1. "Parent sequence has at least region_length nontag characters"
    A2. "Marker name _ins/_rep is unique in the Party"
    A3. "ins_pool is a simple single-op pool"
    """

    def test_A1_scan_length_equals_seq_length(self):
        """Boundary: scanning window equals entire sequence length.

        The region_scan's region_length = 0 (insert mode), so the scan
        window covers all positions. Should work — 1 position if insert,
        bg_length+1 positions total.
        """
        with pp.Party():
            bg = pp.from_seq("AC")
            ins = pp.from_seq("X")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 3
            for seq in df["seq"]:
                assert bio_len(seq) == 3

    def test_A1_replacement_length_equals_seq_length(self):
        """Boundary: replacement length equals entire sequence.

        replacement_scan with ins_pool of same length as bg replaces
        entire content. Only 1 valid position.
        """
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("XXXX")
            result = pp.replacement_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 1
            clean = strip_all_tags(df["seq"].iloc[0])
            assert clean == "XXXX"

    def test_A2_double_insertion_scan_same_party(self):
        """Two insertion_scans in same Party reuse marker name _ins.

        Code uses hardcoded _ins — if both scans are on the same chain,
        second region_scan may encounter the first's tags.
        Result: clear ValueError at construction time.
        """
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins1 = pp.from_seq("X")
            r1 = pp.insertion_scan(bg, ins1, mode="sequential", num_states=2)
            ins2 = pp.from_seq("Y")
            r2 = pp.insertion_scan(r1, ins2, mode="sequential", num_states=2)
            df = r2.generate_library()
            assert len(df) == r2.num_states

    def test_A3_chained_ins_pool(self):
        """ins_pool is itself a chained pool (from_seqs → mutagenize).

        Code captures ins_pool.state — verify this works for complex pools.
        """
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            base = pp.from_seqs(["AA", "CC"], mode="sequential")
            ins = pp.mutagenize(base, num_mutations=1, mode="random", num_states=1)
            result = pp.insertion_scan(bg, ins, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 10


class TestMutagenizeScanAssumptionInversion:
    """Identify implicit assumptions in mutagenize_scan and violate them.

    Assumptions:
    A1. "mutagenize_length <= seq_length" (no explicit check)
    A2. "num_mutations <= mutagenize_length" (delegated to mutagenize)
    A3. "marker name _mut is unique in the Party"
    """

    def test_A1_mutagenize_length_equals_seq_length(self):
        """Boundary: mutagenize window covers entire sequence.

        region_scan should produce exactly 1 position (the whole seq).
        """
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=4, num_mutations=1,
                    mode=("sequential", "sequential"),
                )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 4

    def test_A2_num_mutations_equals_mutagenize_length(self):
        """Boundary: mutate every position in the window.

        All positions within the window should differ from original.
        """
        with pp.Party():
            pool = pp.from_seq("AAAA")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = pp.mutagenize_scan(
                    pool, mutagenize_length=2, num_mutations=2,
                    mode=("sequential", "random"), num_states=(None, 3),
                )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 4

    def test_A3_double_mutagenize_scan_same_party(self):
        """Two mutagenize_scans in same Party both use marker _mut.

        Second scan operates on output of first, which already had _mut
        tags processed. Should either error clearly or produce correct output.
        """
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r1 = pp.mutagenize_scan(
                    pool, mutagenize_length=2, num_mutations=1,
                    mode=("sequential", "random"), num_states=(2, 1),
                )
                r2 = pp.mutagenize_scan(
                    r1, mutagenize_length=2, num_mutations=1,
                    mode=("sequential", "random"), num_states=(2, 1),
                )
            df = r2.generate_library()
            assert len(df) == r2.num_states
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert len(clean) > 0


class TestInsertionMultiscanAssumptionInversion:
    """Identify implicit assumptions in insertion_multiscan and violate them.

    Assumptions:
    A1. "There are enough valid positions for num_insertions given spacing"
    A2. "num_insertions=1 edge case" (minimum)
    A3. "Internal region names _ins_0, _ins_1 don't collide with existing regions"
    """

    def test_A1_barely_enough_room(self):
        """Tight constraint: just barely enough room for all insertions.

        8 bp seq with 3 insertions of length 1 each, replace mode.
        """
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            rep = pp.from_seq("X")
            result = pp.replacement_multiscan(
                bg, num_replacements=3, replacement_pools=rep,
                mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states >= 1
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    def test_A2_single_insertion(self):
        """Edge case: num_insertions=1 (degenerates to single-insert behavior)."""
        with pp.Party():
            bg = pp.from_seq("ACGT")
            ins = pp.from_seq("X")
            result = pp.insertion_multiscan(
                bg, num_insertions=1, insertion_pools=ins, mode="sequential"
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 5
            for seq in df["seq"]:
                assert bio_len(seq) == 5

    @pytest.mark.xfail(
        reason=(
            "Chained insertion_multiscan fails: first multiscan output has "
            "seq_length=None (insert mode changes length per combo), "
            "so second multiscan rejects it without a region constraint. "
            "Clear error, not silent wrong output."
        ),
        raises=ValueError,
        strict=True,
    )
    def test_A3_chained_multiscan(self):
        """Two insertion_multiscans chained — internal names _ins_0, _ins_1
        from first scan may conflict with second.

        Result: clear ValueError at construction time.
        """
        with pp.Party():
            bg = pp.from_seq("ACGTACGTACGTACGT")
            ins1 = pp.from_seq("X")
            r1 = pp.insertion_multiscan(
                bg, num_insertions=2, insertion_pools=ins1,
                mode="sequential", num_states=2,
            )
            ins2 = pp.from_seq("Y")
            r2 = pp.insertion_multiscan(
                r1, num_insertions=2, insertion_pools=ins2,
                mode="sequential", num_states=2,
            )
            df = r2.generate_library()
            assert len(df) == r2.num_states


# ===================================================================
# Medium-risk full invariant suites (Step 3, RECOMMENDED)
# ===================================================================

class TestDeletionScanFullSuite:
    """Full invariant suite for deletion_scan (MEDIUM risk)."""

    def test_I1_with_marker_preserves_length(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=2, deletion_marker="-",
                mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    def test_I1_no_marker_reduces_length(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=2, deletion_marker=None,
                mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 6

    def test_I2_exhaustion(self):
        with pp.Party():
            pool = pp.from_seq("ACGTAC")
            result = pp.deletion_scan(
                pool, deletion_length=2, mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 5

    def test_I3_card_start_end(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=2, mode="sequential",
                cards=["start", "end"],
            )
            df = result.generate_library()
            start_col = [c for c in df.columns if c.endswith(".start")][0]
            end_col = [c for c in df.columns if c.endswith(".end")][0]
            for _, row in df.iterrows():
                assert row[end_col] - row[start_col] == 2

    def test_I5_determinism(self):
        for _ in range(2):
            with pp.Party():
                pool = pp.from_seq("ACGTACGT")
                result = pp.deletion_scan(
                    pool, deletion_length=2, mode="random", num_states=5,
                )
                df = result.generate_library(seed=99)
                seqs = list(df["seq"])
            if _ == 0:
                first = seqs
            else:
                assert seqs == first

    def test_I6_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGGAA</core>TT")
            result = pp.deletion_scan(
                pool, deletion_length=1, region="core", mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_I7_composition_cartesian(self):
        with pp.Party():
            bg = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            result = pp.deletion_scan(
                bg, deletion_length=1, mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 2 * 4

    def test_I10_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(pool, deletion_length=2, mode="sequential")
            ns_before = result.num_states
            sv_before = result.operation.state._num_values
            result.generate_library(seed=1)
            assert result.num_states == ns_before
            assert result.operation.state._num_values == sv_before

    def test_D_position_enumeration_all_windows(self):
        """Every valid deletion window is covered exactly once."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            result = pp.deletion_scan(
                pool, deletion_length=1, mode="sequential",
                cards=["start"],
            )
            df = result.generate_library()
            start_col = [c for c in df.columns if c.endswith(".start")][0]
            positions = sorted(df[start_col].tolist())
            assert positions == [0, 1, 2, 3]


class TestShuffleScanFullSuite:
    """Full invariant suite for shuffle_scan (MEDIUM risk)."""

    def test_I1_preserves_length(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.shuffle_scan(pool, shuffle_length=3, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    def test_I2_exhaustion(self):
        with pp.Party():
            pool = pp.from_seq("ACGTAC")
            result = pp.shuffle_scan(pool, shuffle_length=2, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_I3_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.shuffle_scan(
                pool, shuffle_length=3, mode="sequential",
                cards=(["start", "end"], None),
            )
            df = result.generate_library()
            start_col = [c for c in df.columns if c.endswith(".start")][0]
            end_col = [c for c in df.columns if c.endswith(".end")][0]
            for _, row in df.iterrows():
                assert row[end_col] - row[start_col] == 3

    def test_I5_determinism(self):
        for _ in range(2):
            with pp.Party():
                pool = pp.from_seq("ACGTACGT")
                result = pp.shuffle_scan(
                    pool, shuffle_length=3, mode="random", num_states=5,
                )
                df = result.generate_library(seed=42)
                seqs = list(df["seq"])
            if _ == 0:
                first = seqs
            else:
                assert seqs == first

    def test_I6_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGGAA</core>TT")
            result = pp.shuffle_scan(
                pool, shuffle_length=2, region="core", mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_I7_composition(self):
        with pp.Party():
            bg = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            result = pp.shuffle_scan(bg, shuffle_length=2, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states % 2 == 0

    def test_I10_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.shuffle_scan(pool, shuffle_length=3, mode="sequential")
            ns_before = result.num_states
            sv_before = result.operation.state._num_values
            result.generate_library(seed=1)
            assert result.num_states == ns_before
            assert result.operation.state._num_values == sv_before

    def test_D_composition_preserved(self):
        """Shuffled region has same nucleotide composition as original."""
        with pp.Party():
            parent = "AACCGGTT"
            pool = pp.from_seq(parent)
            result = pp.shuffle_scan(
                pool, shuffle_length=4, mode="sequential",
                shuffles_per_position=1,
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert sorted(strip_all_tags(seq)) == sorted(parent)


class TestSubseqScanFullSuite:
    """Full invariant suite for subseq_scan (MEDIUM risk)."""

    def test_I1_output_length(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.subseq_scan(pool, subseq_length=3, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 3

    def test_I2_exhaustion(self):
        with pp.Party():
            pool = pp.from_seq("ACGTAC")
            result = pp.subseq_scan(pool, subseq_length=3, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 4

    def test_I3_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.subseq_scan(
                pool, subseq_length=3, mode="sequential",
                cards=["start", "end"],
            )
            df = result.generate_library()
            start_col = [c for c in df.columns if c.endswith(".start")][0]
            end_col = [c for c in df.columns if c.endswith(".end")][0]
            for _, row in df.iterrows():
                assert row[end_col] - row[start_col] == 3

    def test_I5_determinism(self):
        for _ in range(2):
            with pp.Party():
                pool = pp.from_seq("ACGTACGT")
                result = pp.subseq_scan(
                    pool, subseq_length=3, mode="random", num_states=5,
                )
                df = result.generate_library(seed=42)
                seqs = list(df["seq"])
            if _ == 0:
                first = seqs
            else:
                assert seqs == first

    def test_I6_region_named(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGGAA</core>TT")
            result = pp.subseq_scan(
                pool, subseq_length=2, region="core", mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 2

    def test_I6_region_interval(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.subseq_scan(
                pool, subseq_length=2, region=[1, 6], mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            for seq in df["seq"]:
                assert bio_len(seq) == 2

    def test_I7_composition(self):
        with pp.Party():
            bg = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            result = pp.subseq_scan(bg, subseq_length=2, mode="sequential")
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states == 2 * 3

    def test_I10_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.subseq_scan(pool, subseq_length=3, mode="sequential")
            ns_before = result.num_states
            sv_before = result.operation.state._num_values
            result.generate_library(seed=1)
            assert result.num_states == ns_before
            assert result.operation.state._num_values == sv_before

    def test_D_subseqs_are_genuine_substrings(self):
        with pp.Party():
            parent = "ACGTACGT"
            pool = pp.from_seq(parent)
            result = pp.subseq_scan(pool, subseq_length=3, mode="sequential")
            df = result.generate_library()
            for seq in df["seq"]:
                assert strip_all_tags(seq) in parent

    def test_D_all_windows_enumerated(self):
        with pp.Party():
            parent = "ACGTAC"
            pool = pp.from_seq(parent)
            result = pp.subseq_scan(pool, subseq_length=3, mode="sequential")
            df = result.generate_library()
            expected = {parent[i:i+3] for i in range(4)}
            actual = {strip_all_tags(s) for s in df["seq"]}
            assert actual == expected


class TestDeletionMultiscanFullSuite:
    """Full invariant suite for deletion_multiscan (MEDIUM risk)."""

    def test_I1_with_marker(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            result = pp.deletion_multiscan(
                pool, deletion_length=2, num_deletions=2, mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 12

    def test_I1_no_marker(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            result = pp.deletion_multiscan(
                pool, deletion_length=2, num_deletions=2,
                deletion_marker=None, mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                assert bio_len(seq) == 8

    def test_I2_exhaustion(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_multiscan(
                pool, deletion_length=1, num_deletions=2, mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states

    def test_I3_cards(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            result = pp.deletion_multiscan(
                pool, deletion_length=2, num_deletions=2,
                mode="sequential", cards=["starts", "ends"],
            )
            df = result.generate_library()
            starts_col = [c for c in df.columns if c.endswith(".starts")]
            assert len(starts_col) >= 1

    def test_I5_determinism(self):
        for _ in range(2):
            with pp.Party():
                pool = pp.from_seq("ACGTACGTACGT")
                result = pp.deletion_multiscan(
                    pool, deletion_length=2, num_deletions=2,
                    mode="random", num_states=5,
                )
                df = result.generate_library(seed=42)
                seqs = list(df["seq"])
            if _ == 0:
                first = seqs
            else:
                assert seqs == first

    def test_I6_region_isolation(self):
        with pp.Party():
            pool = pp.from_seq("AA<core>CCGGAATTCC</core>TT")
            result = pp.deletion_multiscan(
                pool, deletion_length=1, num_deletions=2,
                region="core", mode="sequential",
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert clean.startswith("AA")
                assert clean.endswith("TT")

    def test_I7_composition(self):
        with pp.Party():
            bg = pp.from_seqs(["ACGTACGT", "TGCATGCA"], mode="sequential")
            result = pp.deletion_multiscan(
                bg, deletion_length=1, num_deletions=2, mode="sequential",
            )
            df = result.generate_library()
            assert len(df) == result.num_states
            assert result.num_states % 2 == 0

    def test_I10_state_immutability(self):
        with pp.Party():
            pool = pp.from_seq("ACGTACGTACGT")
            result = pp.deletion_multiscan(
                pool, deletion_length=2, num_deletions=2, mode="sequential",
            )
            ns_before = result.num_states
            sv_before = result.operation.state._num_values
            result.generate_library(seed=1)
            assert result.num_states == ns_before
            assert result.operation.state._num_values == sv_before


# ===================================================================
# Alpha cross-check: confirmed discrepancies
# ===================================================================

class TestAlphaCrossCheck:
    """Tests for findings from the alpha audit that bravo initially missed.

    Each test is strict xfail documenting a confirmed bug with runtime evidence.
    """

    # --- deletion_multiscan empty-string marker (silent wrong output) ---

    def test_bug_deletion_multiscan_empty_string_marker(self):
        """deletion_marker='' should produce no marker, but produces '-'."""
        with pp.Party():
            pool = pp.from_seq("AACCGGTT")
            result = pp.deletion_multiscan(
                pool, deletion_length=2, num_deletions=2,
                deletion_marker="", mode="sequential", num_states=3,
            )
            df = result.generate_library()
            for seq in df["seq"]:
                clean = strip_all_tags(seq)
                assert "-" not in clean, (
                    f"Empty-string marker produced dash: {clean!r}"
                )

    # --- mutagenize_scan sequential + variable-length parent ---

    def test_bug_mutagenize_scan_sequential_variable_length_silent(self):
        """Sequential mode with seq_length=None should raise ValueError."""
        with pp.Party():
            bg = pp.from_seqs(["ACG", "ACGT"], mode="sequential")
            assert bg.seq_length is None
            with pytest.raises(ValueError, match="sequential"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pp.mutagenize_scan(
                        bg, mutagenize_length=2, num_mutations=1,
                        region=None, mode=("sequential", "sequential"),
                    )

    # --- shuffle_scan re-generation: init_state=0 replay semantics ---

    def test_shuffle_scan_regeneration_init_state_replay(self):
        """init_state=0 replays the same output on re-generation (same seed, same pool)."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.shuffle_scan(
                pool, shuffle_length=3, mode="random", num_states=5,
            )
            df1 = result.generate_library(seed=42, init_state=0)
            df2 = result.generate_library(seed=42, init_state=0)
            seqs1 = [strip_all_tags(s) for s in df1["seq"]]
            seqs2 = [strip_all_tags(s) for s in df2["seq"]]
            assert seqs1 == seqs2, (
                f"init_state=0 replay not deterministic:\n  gen1={seqs1}\n  gen2={seqs2}"
            )

    # --- shuffle_scan fresh-context determinism ---

    def test_shuffle_scan_fresh_context_determinism(self):
        """Fresh pp.init() contexts with identical params and same seed produce identical output."""
        pp.init()
        a = pp.shuffle_scan("AACCGGTT", shuffle_length=2, mode="random", num_states=5)
        s1 = [strip_all_tags(s) for s in a.generate_library(num_cycles=1, seed=42)["seq"]]

        pp.init()
        b = pp.shuffle_scan("AACCGGTT", shuffle_length=2, mode="random", num_states=5)
        s2 = [strip_all_tags(s) for s in b.generate_library(num_cycles=1, seed=42)["seq"]]

        assert s1 == s2, (
            f"Fresh contexts differ:\n  ctx1={s1}\n  ctx2={s2}"
        )

    # --- insertion_multiscan per-insert positions shape validation ---

    def test_insertion_multiscan_positions_shape_raises_valueerror(self):
        """Wrong-shape per-insert positions should raise ValueError."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("X")
            with pytest.raises(ValueError, match="per-insert positions has 2 sublists"):
                pp.insertion_multiscan(
                    bg, num_insertions=3, insertion_pools=ins,
                    positions=[[0, 1], [3, 4]],
                    mode="sequential",
                )

    # --- seq_length propagation (FIXED by region_scan/region_multiscan patch) ---

    def test_bug_seq_length_loss_replacement_scan(self):
        """replacement_scan should preserve seq_length = parent length."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.replacement_scan(bg, ins, mode="sequential")
            assert result.seq_length == 8

    def test_bug_seq_length_loss_deletion_scan_marker(self):
        """deletion_scan with marker: seq_length should = parent length."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=2, deletion_marker="-", mode="sequential",
            )
            assert result.seq_length == 8

    def test_bug_seq_length_loss_deletion_scan_no_marker(self):
        """deletion_scan without marker: seq_length = parent - deletion."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_scan(
                pool, deletion_length=2, deletion_marker=None, mode="sequential",
            )
            assert result.seq_length == 6

    def test_bug_seq_length_loss_deletion_multiscan_marker(self):
        """deletion_multiscan with marker: seq_length should = parent length."""
        with pp.Party():
            pool = pp.from_seq("ACGTACGT")
            result = pp.deletion_multiscan(
                pool, deletion_length=2, num_deletions=2,
                deletion_marker="-", mode="sequential",
            )
            assert result.seq_length == 8

    def test_bug_seq_length_loss_insertion_scan(self):
        """insertion_scan: seq_length should be bg + ins."""
        with pp.Party():
            bg = pp.from_seq("ACGTACGT")
            ins = pp.from_seq("NN")
            result = pp.insertion_scan(bg, ins, mode="sequential")
            assert result.seq_length == 10
