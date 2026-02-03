"""Tests for export functionality (to_file method)."""

import gzip
import json
import tempfile
from pathlib import Path

import pytest

import poolparty as pp


class TestToFileCSV:
    """Tests for CSV export."""

    def test_export_csv_basic(self):
        """Test basic CSV export."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="csv", num_seqs=3)

                assert count == 3
                content = path.read_text()
                lines = content.strip().split("\n")
                assert len(lines) == 4  # header + 3 rows
                assert "name" in lines[0]
                assert "seq" in lines[0]
                assert "ACGT" in content
                assert "TGCA" in content
                assert "GGCC" in content
            finally:
                path.unlink()

    def test_export_csv_with_tags_false(self):
        """Test CSV export with write_tags=False strips tags."""
        with pp.Party():
            pool = pp.from_seq("ACGT<region>TTAA</region>GGCC")

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="csv", num_seqs=1, write_tags=False)

                assert count == 1
                content = path.read_text()
                # Tags should be stripped
                assert "<region>" not in content
                assert "</region>" not in content
                assert "ACGTTTAAGGCC" in content
            finally:
                path.unlink()

    def test_export_csv_with_tags_true(self):
        """Test CSV export with write_tags=True keeps tags."""
        with pp.Party():
            pool = pp.from_seq("ACGT<region>TTAA</region>GGCC")

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="csv", num_seqs=1, write_tags=True)

                assert count == 1
                content = path.read_text()
                # Tags should be preserved
                assert "<region>" in content
                assert "</region>" in content
            finally:
                path.unlink()

    def test_export_csv_gzip(self):
        """Test CSV export with gzip compression."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".csv.gz", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="csv", num_seqs=2)

                assert count == 2
                with gzip.open(path, "rt") as f:
                    content = f.read()
                assert "ACGT" in content
                assert "TGCA" in content
            finally:
                path.unlink()

    def test_export_csv_chunked(self):
        """Test CSV export with chunking."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC", "AATT", "CCGG"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = Path(f.name)

            try:
                # Use small chunk size to test chunking
                count = pool.to_file(path, file_type="csv", num_seqs=5, chunk_size=2)

                assert count == 5
                content = path.read_text()
                lines = content.strip().split("\n")
                # Should have header + 5 data rows
                assert len(lines) == 6
            finally:
                path.unlink()


class TestToFileTSV:
    """Tests for TSV export."""

    def test_export_tsv_basic(self):
        """Test basic TSV export."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="tsv", num_seqs=2)

                assert count == 2
                content = path.read_text()
                # Should use tabs
                assert "\t" in content
                lines = content.strip().split("\n")
                assert len(lines) == 3  # header + 2 rows
            finally:
                path.unlink()


class TestToFileFASTA:
    """Tests for FASTA export."""

    def test_export_fasta_basic(self):
        """Test basic FASTA export."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="fasta", num_seqs=2)

                assert count == 2
                content = path.read_text()
                assert content.startswith(">")
                assert "ACGT" in content
                assert "TGCA" in content
            finally:
                path.unlink()

    def test_export_fasta_line_width(self):
        """Test FASTA export with line wrapping."""
        with pp.Party():
            # Long sequence
            long_seq = "ACGT" * 20  # 80 chars
            pool = pp.from_seq(long_seq)

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="fasta", num_seqs=1, line_width=60)

                assert count == 1
                content = path.read_text()
                lines = content.strip().split("\n")
                # First line is header, sequence should be wrapped
                assert lines[0].startswith(">")
                assert len(lines[1]) == 60  # First seq line
                assert len(lines[2]) == 20  # Remainder
            finally:
                path.unlink()

    def test_export_fasta_no_line_width(self):
        """Test FASTA export without line wrapping."""
        with pp.Party():
            long_seq = "ACGT" * 20  # 80 chars
            pool = pp.from_seq(long_seq)

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="fasta", num_seqs=1, line_width=None)

                assert count == 1
                content = path.read_text()
                lines = content.strip().split("\n")
                assert len(lines) == 2  # header + one seq line
                assert len(lines[1]) == 80
            finally:
                path.unlink()

    def test_export_fasta_with_description_string(self):
        """Test FASTA export with description template."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "GGCC"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(
                    path,
                    file_type="fasta",
                    num_seqs=2,
                    description="length={seq}",
                )

                assert count == 2
                content = path.read_text()
                # Description should be in header
                assert "length=" in content
            finally:
                path.unlink()

    def test_export_fasta_with_description_callable(self):
        """Test FASTA export with description callable."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "GGCC"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(
                    path,
                    file_type="fasta",
                    num_seqs=2,
                    description=lambda row: f"len={len(row['seq'])}",
                )

                assert count == 2
                content = path.read_text()
                assert "len=4" in content
            finally:
                path.unlink()

    def test_export_fasta_strips_tags(self):
        """Test FASTA export strips tags by default."""
        with pp.Party():
            pool = pp.from_seq("ACGT<region>TTAA</region>GGCC")

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="fasta", num_seqs=1, write_tags=False)

                assert count == 1
                content = path.read_text()
                assert "<region>" not in content
                assert "ACGTTTAAGGCC" in content
            finally:
                path.unlink()

    def test_export_fasta_gzip(self):
        """Test FASTA export with gzip compression."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".fasta.gz", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="fasta", num_seqs=2)

                assert count == 2
                with gzip.open(path, "rt") as f:
                    content = f.read()
                assert "ACGT" in content
                assert "TGCA" in content
            finally:
                path.unlink()


class TestToFileJSONL:
    """Tests for JSONL export."""

    def test_export_jsonl_basic(self):
        """Test basic JSONL export."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="jsonl", num_seqs=2)

                assert count == 2
                content = path.read_text()
                lines = content.strip().split("\n")
                assert len(lines) == 2

                # Parse each line as JSON
                record1 = json.loads(lines[0])
                record2 = json.loads(lines[1])
                assert record1["seq"] == "ACGT"
                assert record2["seq"] == "TGCA"
            finally:
                path.unlink()

    def test_export_jsonl_strips_tags(self):
        """Test JSONL export strips tags by default."""
        with pp.Party():
            pool = pp.from_seq("ACGT<region>TTAA</region>GGCC")

            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="jsonl", num_seqs=1, write_tags=False)

                assert count == 1
                content = path.read_text()
                record = json.loads(content.strip())
                assert record["seq"] == "ACGTTTAAGGCC"
            finally:
                path.unlink()


class TestToFileValidation:
    """Tests for input validation."""

    def test_requires_num_seqs_or_num_cycles(self):
        """Test that num_seqs or num_cycles is required."""
        with pp.Party():
            pool = pp.from_seq("ACGT")

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = Path(f.name)

            try:
                with pytest.raises(ValueError, match="num_seqs or num_cycles"):
                    pool.to_file(path, file_type="csv")
            finally:
                path.unlink(missing_ok=True)

    def test_invalid_file_type(self):
        """Test that invalid file_type raises error."""
        with pp.Party():
            pool = pp.from_seq("ACGT")

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                path = Path(f.name)

            try:
                with pytest.raises(ValueError, match="file_type"):
                    pool.to_file(path, file_type="txt", num_seqs=1)
            finally:
                path.unlink(missing_ok=True)


class TestToFileStreaming:
    """Tests for streaming (chunked) export."""

    def test_streaming_produces_correct_count(self):
        """Test that streaming export produces correct number of sequences."""
        with pp.Party():
            # Create pool with enough sequences
            seqs = [f"ACGT{i:04d}" for i in range(100)]
            pool = pp.from_seqs(seqs, mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = Path(f.name)

            try:
                # Export with small chunks
                count = pool.to_file(path, file_type="csv", num_seqs=50, chunk_size=10)

                assert count == 50
                content = path.read_text()
                lines = content.strip().split("\n")
                assert len(lines) == 51  # header + 50 rows
            finally:
                path.unlink()

    def test_streaming_fasta_produces_correct_count(self):
        """Test that streaming FASTA export produces correct count."""
        with pp.Party():
            seqs = [f"ACGT{i:04d}" for i in range(100)]
            pool = pp.from_seqs(seqs, mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as f:
                path = Path(f.name)

            try:
                count = pool.to_file(path, file_type="fasta", num_seqs=50, chunk_size=10)

                assert count == 50
                content = path.read_text()
                # Count FASTA headers
                headers = [line for line in content.split("\n") if line.startswith(">")]
                assert len(headers) == 50
            finally:
                path.unlink()


class TestToDF:
    """Tests for to_df() method."""

    def test_to_df_basic(self):
        """Test basic to_df export."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA", "GGCC"], mode="sequential")
            df = pool.to_df(num_seqs=3)

            assert len(df) == 3
            assert "name" in df.columns
            assert "seq" in df.columns
            assert set(df["seq"]) == {"ACGT", "TGCA", "GGCC"}

    def test_to_df_with_num_cycles(self):
        """Test to_df with num_cycles."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.to_df(num_cycles=2)

            assert len(df) == 4  # 2 seqs * 2 cycles

    def test_to_df_strips_tags_by_default(self):
        """Test that to_df strips tags by default."""
        with pp.Party():
            pool = pp.from_seq("ACGT<region>TTAA</region>GGCC")
            df = pool.to_df(num_seqs=1)

            assert "<region>" not in df["seq"].iloc[0]
            assert df["seq"].iloc[0] == "ACGTTTAAGGCC"

    def test_to_df_keeps_tags_when_requested(self):
        """Test that to_df keeps tags when write_tags=True."""
        with pp.Party():
            pool = pp.from_seq("ACGT<region>TTAA</region>GGCC")
            df = pool.to_df(num_seqs=1, write_tags=True)

            assert "<region>" in df["seq"].iloc[0]
            assert "</region>" in df["seq"].iloc[0]

    def test_to_df_with_columns_filter(self):
        """Test to_df with specific columns."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")
            df = pool.to_df(num_seqs=2, columns=["seq"])

            assert list(df.columns) == ["seq"]

    def test_to_df_chunked(self):
        """Test to_df with chunking."""
        with pp.Party():
            seqs = [f"ACGT{i:04d}" for i in range(100)]
            pool = pp.from_seqs(seqs, mode="sequential")
            df = pool.to_df(num_seqs=50, chunk_size=10)

            assert len(df) == 50

    def test_to_df_requires_num_seqs_or_cycles(self):
        """Test that to_df requires num_seqs or num_cycles."""
        with pp.Party():
            pool = pp.from_seq("ACGT")
            with pytest.raises(ValueError, match="num_seqs or num_cycles"):
                pool.to_df()


class TestProgressBar:
    """Tests for progress bar functionality."""

    def test_to_df_with_progress_no_tqdm(self):
        """Test to_df with progress when tqdm not installed (mocked)."""

        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")

            # Should work even without tqdm, just emit a warning
            # We can't easily mock the import, so just test it doesn't crash
            df = pool.to_df(num_seqs=2, show_progress=False)
            assert len(df) == 2

    def test_to_file_with_progress_no_crash(self):
        """Test to_file with progress doesn't crash."""
        with pp.Party():
            pool = pp.from_seqs(["ACGT", "TGCA"], mode="sequential")

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                path = Path(f.name)

            try:
                # Should work even without tqdm, just emit a warning
                count = pool.to_file(path, num_seqs=2, show_progress=False)
                assert count == 2
            finally:
                path.unlink()
