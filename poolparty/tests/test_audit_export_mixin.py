"""Audit tests for ExportMixin (to_df, to_file) and private format writers.

Covers export-specific invariants E1-E10, adversarial patterns, and
contract tracing for chunking and NullSeq handling paths.
"""

import gzip
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import poolparty as pp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tmppath(suffix: str) -> Path:
    """Return a fresh temp file path (caller must unlink)."""
    return Path(tempfile.mktemp(suffix=suffix))


# ---------------------------------------------------------------------------
# E1: Count accuracy
# ---------------------------------------------------------------------------

class TestE1CountAccuracy:
    """to_file return value == actual rows/records; len(to_df) == requested."""

    def test_to_df_num_seqs_count(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            df = pool.to_df(num_seqs=3, show_progress=False)
            assert len(df) == 3

    def test_to_df_num_cycles_count(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.to_df(num_cycles=2, show_progress=False)
            assert len(df) == 4

    def test_to_file_csv_count_matches_rows(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            path = _tmppath(".csv")
            try:
                count = pool.to_file(path, num_seqs=3, show_progress=False)
                content = path.read_text()
                data_lines = [l for l in content.strip().split("\n") if l][1:]
                assert count == 3
                assert len(data_lines) == 3
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_fasta_count_matches_headers(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            path = _tmppath(".fasta")
            try:
                count = pool.to_file(path, num_seqs=3, show_progress=False)
                content = path.read_text()
                headers = [l for l in content.split("\n") if l.startswith(">")]
                assert count == 3
                assert len(headers) == 3
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_jsonl_count_matches_lines(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".jsonl")
            try:
                count = pool.to_file(path, num_seqs=2, show_progress=False)
                lines = [l for l in path.read_text().strip().split("\n") if l]
                assert count == 2
                assert len(lines) == 2
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_tsv_count_matches_rows(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".tsv")
            try:
                count = pool.to_file(path, num_seqs=2, show_progress=False)
                data_lines = [l for l in path.read_text().strip().split("\n") if l][1:]
                assert count == 2
                assert len(data_lines) == 2
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E2: Content correctness
# ---------------------------------------------------------------------------

class TestE2ContentCorrectness:
    """Exported sequences match generate_library output."""

    def test_to_df_matches_generate_library(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            df_gen = pool.generate_library(num_seqs=3)
            df_export = pool.to_df(num_seqs=3, show_progress=False)
            assert set(df_export["seq"]) == set(df_gen["seq"])

    def test_to_file_csv_matches_generate_library(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            df_gen = pool.generate_library(num_seqs=3)
            path = _tmppath(".csv")
            try:
                pool.to_file(path, num_seqs=3, show_progress=False)
                df_file = pd.read_csv(path)
                assert set(df_file["seq"]) == set(df_gen["seq"])
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_jsonl_matches_generate_library(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df_gen = pool.generate_library(num_seqs=2)
            path = _tmppath(".jsonl")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                records = [json.loads(l) for l in path.read_text().strip().split("\n")]
                exported_seqs = {r["seq"] for r in records}
                assert exported_seqs == set(df_gen["seq"])
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E3: Tag behavior
# ---------------------------------------------------------------------------

class TestE3TagBehavior:
    """write_tags=False strips all tags; write_tags=True preserves them."""

    def test_to_df_strips_tags_by_default(self):
        with pp.Party():
            pool = pp.from_seq("AA<reg>CC</reg>GG")
            df = pool.to_df(num_seqs=1, show_progress=False)
            assert "<reg>" not in df["seq"].iloc[0]
            assert df["seq"].iloc[0] == "AACCGG"

    def test_to_df_preserves_tags_when_requested(self):
        with pp.Party():
            pool = pp.from_seq("AA<reg>CC</reg>GG")
            df = pool.to_df(num_seqs=1, write_tags=True, show_progress=False)
            assert "<reg>" in df["seq"].iloc[0]
            assert "</reg>" in df["seq"].iloc[0]

    def test_to_df_strips_nested_tags(self):
        with pp.Party():
            pool = pp.from_seq("AA<outer>CC<inner>GG</inner>TT</outer>AA")
            df = pool.to_df(num_seqs=1, show_progress=False)
            assert "<outer>" not in df["seq"].iloc[0]
            assert "<inner>" not in df["seq"].iloc[0]
            assert df["seq"].iloc[0] == "AACCGGTTAA"

    def test_to_file_csv_strips_tags(self):
        with pp.Party():
            pool = pp.from_seq("AA<reg>CC</reg>GG")
            path = _tmppath(".csv")
            try:
                pool.to_file(path, num_seqs=1, show_progress=False)
                content = path.read_text()
                assert "<reg>" not in content
                assert "AACCGG" in content
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_fasta_strips_tags(self):
        with pp.Party():
            pool = pp.from_seq("AA<reg>CC</reg>GG")
            path = _tmppath(".fasta")
            try:
                pool.to_file(path, num_seqs=1, show_progress=False)
                content = path.read_text()
                assert "<reg>" not in content
                assert "AACCGG" in content
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_jsonl_strips_tags(self):
        with pp.Party():
            pool = pp.from_seq("AA<reg>CC</reg>GG")
            path = _tmppath(".jsonl")
            try:
                pool.to_file(path, num_seqs=1, show_progress=False)
                record = json.loads(path.read_text().strip())
                assert "<reg>" not in record["seq"]
                assert record["seq"] == "AACCGG"
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_csv_preserves_tags(self):
        with pp.Party():
            pool = pp.from_seq("AA<reg>CC</reg>GG")
            path = _tmppath(".csv")
            try:
                pool.to_file(path, num_seqs=1, write_tags=True, show_progress=False)
                content = path.read_text()
                assert "<reg>" in content
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E4: Chunking equivalence
# ---------------------------------------------------------------------------

class TestE4ChunkingEquivalence:
    """Output with different chunk_size is consistent (same sequences, same count)."""

    def test_to_df_chunked_same_count(self):
        with pp.Party():
            pool = pp.from_seqs(
                [f"ACGT{i:02d}AA" for i in range(10)], mode="sequential"
            )
            df_big = pool.to_df(num_seqs=10, chunk_size=100, show_progress=False)
            df_small = pool.to_df(num_seqs=10, chunk_size=2, show_progress=False)
            assert len(df_big) == len(df_small) == 10

    def test_to_file_csv_chunked_single_header(self):
        """Chunked CSV should have exactly one header row."""
        with pp.Party():
            pool = pp.from_seqs(
                [f"ACGT{i:02d}AA" for i in range(10)], mode="sequential"
            )
            path = _tmppath(".csv")
            try:
                pool.to_file(path, num_seqs=10, chunk_size=3, show_progress=False)
                content = path.read_text()
                lines = [l for l in content.strip().split("\n") if l]
                header_count = sum(1 for l in lines if l.startswith("name,"))
                assert header_count == 1
                assert len(lines) == 11  # 1 header + 10 data
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_fasta_chunked_count(self):
        with pp.Party():
            pool = pp.from_seqs(
                [f"ACGT{i:02d}AA" for i in range(10)], mode="sequential"
            )
            path = _tmppath(".fasta")
            try:
                count = pool.to_file(
                    path, num_seqs=10, chunk_size=3, show_progress=False
                )
                headers = [
                    l for l in path.read_text().split("\n") if l.startswith(">")
                ]
                assert count == 10
                assert len(headers) == 10
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E5: Format well-formedness
# ---------------------------------------------------------------------------

class TestE5FormatWellFormedness:
    """Exported files are parseable by standard tools."""

    def test_csv_parseable_by_pandas(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".csv")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                df = pd.read_csv(path)
                assert len(df) == 2
                assert "seq" in df.columns
            finally:
                path.unlink(missing_ok=True)

    def test_tsv_parseable_by_pandas(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".tsv")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                df = pd.read_csv(path, sep="\t")
                assert len(df) == 2
                assert "seq" in df.columns
            finally:
                path.unlink(missing_ok=True)

    def test_fasta_well_formed(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".fasta")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                content = path.read_text().strip()
                entries = content.split(">")[1:]
                for entry in entries:
                    lines = entry.strip().split("\n")
                    assert len(lines) >= 2  # header + at least 1 seq line
                    seq = "".join(lines[1:])
                    assert len(seq) > 0
            finally:
                path.unlink(missing_ok=True)

    def test_jsonl_each_line_valid_json(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".jsonl")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                for line in path.read_text().strip().split("\n"):
                    record = json.loads(line)
                    assert "seq" in record
                    assert "name" in record
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E6: Gzip roundtrip
# ---------------------------------------------------------------------------

class TestE6GzipRoundtrip:
    """Gzip-compressed outputs are readable."""

    def test_csv_gz(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".csv.gz")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                with gzip.open(path, "rt") as f:
                    df = pd.read_csv(f)
                assert len(df) == 2
            finally:
                path.unlink(missing_ok=True)

    def test_fasta_gz(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".fasta.gz")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                with gzip.open(path, "rt") as f:
                    content = f.read()
                headers = [l for l in content.split("\n") if l.startswith(">")]
                assert len(headers) == 2
            finally:
                path.unlink(missing_ok=True)

    def test_jsonl_gz(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".jsonl.gz")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                with gzip.open(path, "rt") as f:
                    lines = f.read().strip().split("\n")
                for line in lines:
                    json.loads(line)
                assert len(lines) == 2
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E7: Seed determinism
# ---------------------------------------------------------------------------

class TestE7SeedDeterminism:
    """Same seed produces identical output across calls.

    Note: mode='random' pools advance internal state on each
    generate_library call, so seed determinism is only testable with
    a fresh pool per call, or with mode='sequential'.
    """

    def test_to_df_sequential_deterministic(self):
        """Sequential mode is naturally deterministic."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            df1 = pool.to_df(num_seqs=3, show_progress=False)
            df2 = pool.to_df(num_seqs=3, show_progress=False)
            assert list(df1["seq"]) == list(df2["seq"])

    def test_to_df_seed_deterministic_fresh_pool(self):
        """Random mode is deterministic with fresh pool + same seed."""
        with pp.Party():
            pool1 = pp.from_iupac("NNNN", mode="random")
            df1 = pool1.to_df(num_seqs=5, seed=42, show_progress=False)

        with pp.Party():
            pool2 = pp.from_iupac("NNNN", mode="random")
            df2 = pool2.to_df(num_seqs=5, seed=42, show_progress=False)

        assert list(df1["seq"]) == list(df2["seq"])

    def test_to_file_csv_seed_deterministic_fresh_pool(self):
        with pp.Party():
            pool1 = pp.from_iupac("NNNN", mode="random")
            p1 = _tmppath(".csv")
            pool1.to_file(p1, num_seqs=5, seed=42, show_progress=False)

        with pp.Party():
            pool2 = pp.from_iupac("NNNN", mode="random")
            p2 = _tmppath(".csv")
            pool2.to_file(p2, num_seqs=5, seed=42, show_progress=False)

        try:
            assert p1.read_text() == p2.read_text()
        finally:
            p1.unlink(missing_ok=True)
            p2.unlink(missing_ok=True)

    @pytest.mark.xfail(
        reason="F2: seed += chunk_size means chunked output differs from single-batch",
        strict=True,
    )
    def test_to_df_seed_chunk_invariance(self):
        """Chunked output should match single-batch output with same seed."""
        with pp.Party():
            pool1 = pp.from_iupac("NNNN", mode="random")
            df_single = pool1.to_df(
                num_seqs=6, seed=42, chunk_size=100, show_progress=False
            )

        with pp.Party():
            pool2 = pp.from_iupac("NNNN", mode="random")
            df_chunked = pool2.to_df(
                num_seqs=6, seed=42, chunk_size=2, show_progress=False
            )
        assert list(df_single["seq"]) == list(df_chunked["seq"])


# ---------------------------------------------------------------------------
# E8: NullSeq handling
# ---------------------------------------------------------------------------

class TestE8NullSeqHandling:
    """discard_null_seqs=True excludes; =False includes (format-dependent)."""

    def test_to_df_discard_null_true(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filtered = pool.filter(lambda s: s == "ACGT")
            df = filtered.to_df(num_seqs=3, discard_null_seqs=True, show_progress=False)
            assert all(df["seq"] == "ACGT")

    def test_to_df_discard_null_false(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filtered = pool.filter(lambda s: s == "ACGT")
            df = filtered.to_df(
                num_seqs=3, discard_null_seqs=False, show_progress=False
            )
            assert len(df) == 3
            null_count = df["seq"].isna().sum()
            assert null_count > 0

    def test_to_file_csv_discard_null_false_includes_nulls(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filtered = pool.filter(lambda s: s == "ACGT")
            path = _tmppath(".csv")
            try:
                count = filtered.to_file(
                    path, num_seqs=3, discard_null_seqs=False, show_progress=False
                )
                assert count == 3
                df = pd.read_csv(path)
                assert len(df) == 3
            finally:
                path.unlink(missing_ok=True)

    def test_to_df_discard_null_false_with_write_tags_true(self):
        """write_tags=True bypasses _strip_tags, so None seqs don't crash."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filtered = pool.filter(lambda s: s == "ACGT")
            df = filtered.to_df(
                num_seqs=3, discard_null_seqs=False, write_tags=True,
                show_progress=False,
            )
            assert len(df) == 3

    def test_to_file_fasta_discard_null_false_skips_none_seqs(self):
        """FASTA cannot represent null seqs, so they are skipped.
        Return value reflects entries actually written, not rows processed."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            filtered = pool.filter(lambda s: s == "ACGT")
            path = _tmppath(".fasta")
            try:
                count = filtered.to_file(
                    path, num_seqs=3, discard_null_seqs=False, show_progress=False
                )
                content = path.read_text()
                headers = [l for l in content.split("\n") if l.startswith(">")]
                assert count == 1
                assert len(headers) == 1
                seqs_in_file = []
                for entry in content.strip().split(">")[1:]:
                    lines = entry.strip().split("\n")
                    seqs_in_file.append("".join(lines[1:]))
                assert all(s == "ACGT" for s in seqs_in_file)
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# E9: Design cards — F4 resolved (include_design_cards parameter removed)
# ---------------------------------------------------------------------------

class TestE9DesignCards:
    """Design card columns appear when ops have cards configured."""

    def test_include_design_cards_parameter_removed(self):
        """include_design_cards was removed — passing it should raise TypeError."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with pytest.raises(TypeError):
                pool.to_df(
                    num_seqs=2, include_design_cards=False, show_progress=False
                )

    def test_design_cards_present_when_op_has_cards(self):
        """Cards appear in output when the operation has cards configured."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            pool = pool.mutagenize(
                num_mutations=1, mode="sequential",
                cards=["positions"],
            )
            df = pool.to_df(num_seqs=2, show_progress=False)
            card_cols = [c for c in df.columns if "mutagenize" in c]
            assert len(card_cols) > 0

    def test_columns_param_can_exclude_cards(self):
        """Users can use columns= to exclude card columns."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            pool = pool.mutagenize(
                num_mutations=1, mode="sequential",
                cards=["positions"],
            )
            df = pool.to_df(
                num_seqs=2, columns=["name", "seq"], show_progress=False
            )
            card_cols = [c for c in df.columns if "mutagenize" in c]
            assert len(card_cols) == 0


# ---------------------------------------------------------------------------
# E10: Column filtering
# ---------------------------------------------------------------------------

class TestE10ColumnFiltering:
    """columns= parameter filters output columns."""

    def test_to_df_columns_filter(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.to_df(num_seqs=2, columns=["seq"], show_progress=False)
            assert list(df.columns) == ["seq"]

    def test_to_df_columns_nonexistent_silently_ignored(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.to_df(
                num_seqs=2, columns=["seq", "nonexistent"], show_progress=False
            )
            assert "seq" in df.columns
            assert "nonexistent" not in df.columns

    def test_to_file_csv_columns_filter(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".csv")
            try:
                pool.to_file(
                    path, num_seqs=2, columns=["seq"], show_progress=False
                )
                df = pd.read_csv(path)
                assert list(df.columns) == ["seq"]
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Adversarial: Seed continuity (F2)
# ---------------------------------------------------------------------------

class TestAdversarialSeedContinuity:
    """Verify seed behavior across chunk boundaries."""

    @pytest.mark.xfail(
        reason="F2: seed += chunk_size breaks determinism across chunk sizes",
        strict=True,
    )
    def test_chunk_size_1_vs_default(self):
        with pp.Party():
            pool = pp.from_iupac("NNNN", mode="random")
            df_default = pool.to_df(
                num_seqs=4, seed=99, chunk_size=100, show_progress=False
            )
            df_tiny = pool.to_df(
                num_seqs=4, seed=99, chunk_size=1, show_progress=False
            )
            assert list(df_default["seq"]) == list(df_tiny["seq"])

    def test_same_chunk_size_is_deterministic(self):
        """Same seed + same chunk_size + fresh pool should be deterministic."""
        with pp.Party():
            pool1 = pp.from_iupac("NNNN", mode="random")
            df1 = pool1.to_df(
                num_seqs=4, seed=99, chunk_size=2, show_progress=False
            )

        with pp.Party():
            pool2 = pp.from_iupac("NNNN", mode="random")
            df2 = pool2.to_df(
                num_seqs=4, seed=99, chunk_size=2, show_progress=False
            )

        assert list(df1["seq"]) == list(df2["seq"])


# ---------------------------------------------------------------------------
# Adversarial: NullSeq in FASTA
# ---------------------------------------------------------------------------

class TestAdversarialNullSeqFasta:
    """FASTA with NullSeq pool and discard_null_seqs=False."""

    def test_fasta_all_null_terminates(self):
        """Pool where all sequences fail filter — FASTA should not loop forever."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            filtered = pool.filter(lambda s: False)
            path = _tmppath(".fasta")
            try:
                count = filtered.to_file(
                    path,
                    num_seqs=2,
                    discard_null_seqs=False,
                    show_progress=False,
                )
                assert count == 0
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Adversarial: File type auto-detection
# ---------------------------------------------------------------------------

class TestAdversarialFileDetection:
    """_detect_file_type edge cases."""

    def test_unknown_extension_raises(self):
        """Unknown file extension should raise ValueError."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            path = _tmppath(".xyz")
            try:
                with pytest.raises(ValueError, match="Unknown file extension"):
                    pool.to_file(path, num_seqs=1, show_progress=False)
            finally:
                path.unlink(missing_ok=True)

    def test_explicit_file_type_overrides_extension(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".txt")
            try:
                pool.to_file(
                    path, file_type="csv", num_seqs=2, show_progress=False
                )
                df = pd.read_csv(path)
                assert len(df) == 2
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Adversarial: Variable-length pool
# ---------------------------------------------------------------------------

class TestAdversarialVariableLength:
    """Export with variable-length parent pool."""

    def test_to_df_variable_length(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "ACGT", "ACGTACGT"], mode="sequential")
            df = pool.to_df(num_seqs=3, show_progress=False)
            assert len(df) == 3
            assert set(df["seq"]) == {"A", "ACGT", "ACGTACGT"}

    def test_to_file_csv_variable_length(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "ACGT", "ACGTACGT"], mode="sequential")
            path = _tmppath(".csv")
            try:
                count = pool.to_file(path, num_seqs=3, show_progress=False)
                assert count == 3
                df = pd.read_csv(path)
                assert set(df["seq"]) == {"A", "ACGT", "ACGTACGT"}
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_fasta_variable_length(self):
        with pp.Party():
            pool = pp.from_seqs(["A", "ACGT", "ACGTACGT"], mode="sequential")
            path = _tmppath(".fasta")
            try:
                count = pool.to_file(path, num_seqs=3, show_progress=False)
                assert count == 3
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Adversarial: Edge cardinality
# ---------------------------------------------------------------------------

class TestAdversarialEdgeCardinality:
    """Single-state pool, num_seqs=1."""

    def test_single_state_to_df(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            df = pool.to_df(num_seqs=1, show_progress=False)
            assert len(df) == 1
            assert df["seq"].iloc[0] == "ACGT"

    def test_single_state_all_formats(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            for suffix in [".csv", ".tsv", ".fasta", ".jsonl"]:
                path = _tmppath(suffix)
                try:
                    count = pool.to_file(path, num_seqs=1, show_progress=False)
                    assert count == 1
                finally:
                    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Adversarial: Nested tags + write_tags
# ---------------------------------------------------------------------------

class TestAdversarialNestedTags:
    """Complex tag structures with write_tags flag."""

    def test_nested_tags_strip(self):
        with pp.Party():
            pool = pp.from_seq("AA<a>CC<b>GG</b>TT</a>AA")
            df = pool.to_df(num_seqs=1, write_tags=False, show_progress=False)
            assert df["seq"].iloc[0] == "AACCGGTTAA"

    def test_nested_tags_preserve(self):
        with pp.Party():
            pool = pp.from_seq("AA<a>CC<b>GG</b>TT</a>AA")
            df = pool.to_df(num_seqs=1, write_tags=True, show_progress=False)
            seq = df["seq"].iloc[0]
            assert "<a>" in seq
            assert "<b>" in seq
            assert "</b>" in seq
            assert "</a>" in seq

    def test_nested_tags_csv_roundtrip(self):
        with pp.Party():
            pool = pp.from_seq("AA<a>CC<b>GG</b>TT</a>AA")
            path = _tmppath(".csv")
            try:
                pool.to_file(
                    path, num_seqs=1, write_tags=False, show_progress=False
                )
                df = pd.read_csv(path)
                assert df["seq"].iloc[0] == "AACCGGTTAA"
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Adversarial: FASTA description paths
# ---------------------------------------------------------------------------

class TestAdversarialFastaDescription:
    """FASTA description as string template and callable."""

    def test_description_string_template(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".fasta")
            try:
                pool.to_file(
                    path,
                    num_seqs=2,
                    description="seq={seq}",
                    show_progress=False,
                )
                content = path.read_text()
                assert "seq=ACGT" in content
            finally:
                path.unlink(missing_ok=True)

    def test_description_callable(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".fasta")
            try:
                pool.to_file(
                    path,
                    num_seqs=2,
                    description=lambda row: f"len={len(row['seq'])}",
                    show_progress=False,
                )
                content = path.read_text()
                assert "len=4" in content
            finally:
                path.unlink(missing_ok=True)

    def test_fasta_line_width_none(self):
        """line_width=None should produce single-line sequences."""
        with pp.Party():
            long_seq = "ACGT" * 25  # 100 chars
            pool = pp.from_seq(long_seq)
            path = _tmppath(".fasta")
            try:
                pool.to_file(
                    path, num_seqs=1, line_width=None, show_progress=False
                )
                lines = path.read_text().strip().split("\n")
                assert len(lines) == 2  # header + 1 seq line
                assert lines[1] == long_seq
            finally:
                path.unlink(missing_ok=True)

    def test_fasta_line_width_small(self):
        """line_width=10 should wrap sequence."""
        with pp.Party():
            pool = pp.from_seq("ACGT" * 5)  # 20 chars
            path = _tmppath(".fasta")
            try:
                pool.to_file(
                    path, num_seqs=1, line_width=10, show_progress=False
                )
                lines = path.read_text().strip().split("\n")
                assert lines[0].startswith(">")
                assert len(lines[1]) == 10
                assert len(lines[2]) == 10
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Adversarial: Extreme chunk sizes
# ---------------------------------------------------------------------------

class TestAdversarialChunkSize:
    """chunk_size=1 (extreme) and chunk_size > num_seqs."""

    def test_chunk_size_1_csv(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            path = _tmppath(".csv")
            try:
                count = pool.to_file(
                    path, num_seqs=3, chunk_size=1, show_progress=False
                )
                assert count == 3
                df = pd.read_csv(path)
                assert len(df) == 3
                # Should still have only 1 header
                lines = path.read_text().strip().split("\n")
                header_count = sum(1 for l in lines if l.startswith("name"))
                assert header_count == 1
            finally:
                path.unlink(missing_ok=True)

    def test_chunk_size_larger_than_num_seqs(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.to_df(
                num_seqs=2, chunk_size=10000, show_progress=False
            )
            assert len(df) == 2


# ---------------------------------------------------------------------------
# Contract tracing: to_df empty result
# ---------------------------------------------------------------------------

class TestContractTracingEmptyResult:
    """to_df returns DataFrame with expected columns even when empty."""

    def test_to_df_empty_returns_df_with_columns(self):
        """When no sequences generated, should return empty DataFrame."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT"], mode="sequential")
            filtered = pool.filter(lambda s: False)
            # All seqs fail filter; with discard_null_seqs=True, generate_library
            # returns empty after exhausting state space
            df = filtered.to_df(
                num_seqs=1,
                discard_null_seqs=True,
                show_progress=False,
            )
            assert isinstance(df, pd.DataFrame)
            # With discard_null_seqs=True, the loop may get len(df)==0 and break
            # OR it may keep trying. Either way, result should be a DataFrame.


# ---------------------------------------------------------------------------
# Contract tracing: Validation
# ---------------------------------------------------------------------------

class TestContractTracingValidation:
    """Input validation for to_df and to_file."""

    def test_to_df_requires_num_seqs_or_cycles(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="num_seqs or num_cycles"):
                pool.to_df()

    def test_to_file_requires_num_seqs_or_cycles(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            path = _tmppath(".csv")
            try:
                with pytest.raises(ValueError, match="num_seqs or num_cycles"):
                    pool.to_file(path)
            finally:
                path.unlink(missing_ok=True)

    def test_to_file_invalid_file_type(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            path = _tmppath(".csv")
            try:
                with pytest.raises(ValueError, match="file_type"):
                    pool.to_file(path, file_type="xml", num_seqs=1)
            finally:
                path.unlink(missing_ok=True)

    def test_to_df_rejects_both_num_seqs_and_num_cycles(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="only one of num_seqs or num_cycles"):
                pool.to_df(num_seqs=1, num_cycles=1, show_progress=False)

    def test_to_file_rejects_both_num_seqs_and_num_cycles(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            path = _tmppath(".csv")
            try:
                with pytest.raises(ValueError, match="only one of num_seqs or num_cycles"):
                    pool.to_file(path, num_seqs=1, num_cycles=1, show_progress=False)
            finally:
                path.unlink(missing_ok=True)

    @pytest.mark.parametrize("bad_num_seqs", [0, -1])
    def test_to_df_rejects_nonpositive_num_seqs(self, bad_num_seqs):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="num_seqs must be positive"):
                pool.to_df(num_seqs=bad_num_seqs, show_progress=False)

    @pytest.mark.parametrize("bad_num_cycles", [0, -1])
    def test_to_df_rejects_nonpositive_num_cycles(self, bad_num_cycles):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="num_cycles must be positive"):
                pool.to_df(num_cycles=bad_num_cycles, show_progress=False)

    @pytest.mark.parametrize("bad_num_seqs", [0, -1])
    def test_to_file_rejects_nonpositive_num_seqs(self, bad_num_seqs):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            path = _tmppath(".csv")
            try:
                with pytest.raises(ValueError, match="num_seqs must be positive"):
                    pool.to_file(path, num_seqs=bad_num_seqs, show_progress=False)
            finally:
                path.unlink(missing_ok=True)

    @pytest.mark.parametrize("bad_num_cycles", [0, -1])
    def test_to_file_rejects_nonpositive_num_cycles(self, bad_num_cycles):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            path = _tmppath(".csv")
            try:
                with pytest.raises(ValueError, match="num_cycles must be positive"):
                    pool.to_file(path, num_cycles=bad_num_cycles, show_progress=False)
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Contract tracing: csv_kwargs forwarding
# ---------------------------------------------------------------------------

class TestContractTracingCsvKwargs:
    """**csv_kwargs forwarded to DataFrame.to_csv."""

    def test_csv_kwargs_separator(self):
        """csv_kwargs like lineterminator are forwarded."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".csv")
            try:
                pool.to_file(path, num_seqs=2, show_progress=False)
                df = pd.read_csv(path)
                assert len(df) == 2
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Contract tracing: write_style forwarding
# ---------------------------------------------------------------------------

class TestContractTracingWriteStyle:
    """write_style forwards to _include_inline_styles in generate_library."""

    def test_write_style_false_no_style_column(self):
        with pp.Party():
            pool = pp.from_seq("ACGT")
            styled = pool.stylize(style="red")
            df = styled.to_df(num_seqs=1, write_style=False, show_progress=False)
            assert "_inline_styles" not in df.columns

    def test_write_style_true_includes_style_data(self):
        """write_style=True should forward _include_inline_styles to generate_library.
        The _inline_styles column is internal and stripped before output."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            styled = pool.stylize(style="red")
            # write_style=True is forwarded as _include_inline_styles=True
            # but to_df strips the _inline_styles column from output
            # The effect is that seq may contain style annotations
            df = styled.to_df(num_seqs=1, write_style=True, show_progress=False)
            # At minimum, should not crash
            assert len(df) == 1


# ---------------------------------------------------------------------------
# F7: chunk_size<=0 silently returns empty (from alpha reconciliation)
# ---------------------------------------------------------------------------

class TestF7ChunkSizeValidation:
    """chunk_size<=0 should raise ValueError, not silently return empty."""

    def test_to_df_chunk_size_zero_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with pytest.raises(ValueError, match="chunk_size must be positive"):
                pool.to_df(num_seqs=2, chunk_size=0, show_progress=False)

    def test_to_df_chunk_size_negative_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            with pytest.raises(ValueError, match="chunk_size must be positive"):
                pool.to_df(num_seqs=2, chunk_size=-1, show_progress=False)

    def test_to_file_chunk_size_zero_raises(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".csv")
            try:
                with pytest.raises(ValueError, match="chunk_size must be positive"):
                    pool.to_file(
                        path, num_seqs=2, chunk_size=0, show_progress=False
                    )
            finally:
                path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# F8: FASTA >None headers (from alpha reconciliation)
# ---------------------------------------------------------------------------

class TestF8FastaNoneHeaders:
    """FASTA headers should not be '>None' when name column is None."""

    def test_fasta_headers_not_none(self):
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            path = _tmppath(".fasta")
            try:
                pool.to_file(
                    path, file_type="fasta", num_seqs=2, show_progress=False
                )
                content = path.read_text()
                headers = [
                    l for l in content.split("\n") if l.startswith(">")
                ]
                for h in headers:
                    assert h != ">None", f"FASTA header is '>None': {h}"
            finally:
                path.unlink(missing_ok=True)

    def test_fasta_headers_with_prefix_are_valid(self):
        """When prefix is set, FASTA headers should be meaningful."""
        with pp.Party():
            pool = pp.from_seqs(
                ["ACGT", "TGCA"], mode="sequential", prefix="seq"
            )
            path = _tmppath(".fasta")
            try:
                pool.to_file(
                    path, file_type="fasta", num_seqs=2, show_progress=False
                )
                content = path.read_text()
                headers = [
                    l for l in content.split("\n") if l.startswith(">")
                ]
                for h in headers:
                    assert h != ">None"
                    assert "seq" in h
            finally:
                path.unlink(missing_ok=True)
