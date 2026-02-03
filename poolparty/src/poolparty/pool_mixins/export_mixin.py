"""Export mixin for Pool class - provides file export methods."""

import gzip
import json
import re
from pathlib import Path

import pandas as pd

from ..types import Callable, Integral, Literal, Optional, Union

# Regex to strip region tags from sequences
_TAG_PATTERN = re.compile(r"</?[^>]+>")


def _strip_tags(seq: str) -> str:
    """Remove XML-like region tags from a sequence string."""
    return _TAG_PATTERN.sub("", seq)


def _open_file(path: Path, mode: str):
    """Open file, with gzip support based on extension."""
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def _get_progress_bar(iterable, total, desc, show_progress):
    """Get a progress bar wrapper if requested and tqdm is available."""
    if not show_progress:
        return iterable

    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc, unit="seq")
    except ImportError:
        import warnings

        warnings.warn(
            "tqdm not installed. Install with 'pip install tqdm' for progress bars.",
            stacklevel=3,
        )
        return iterable


class ExportMixin:
    """Mixin providing file export methods for Pool.

    Supports streaming export to avoid loading entire libraries into memory.
    """

    def to_df(
        self,
        num_seqs: Optional[Integral] = None,
        num_cycles: Optional[Integral] = None,
        chunk_size: Integral = 10000,
        write_tags: bool = False,
        write_style: bool = False,
        seed: Optional[Integral] = None,
        discard_null_seqs: bool = True,
        include_design_cards: bool = False,
        columns: Optional[list[str]] = None,
        show_progress: bool = False,
    ) -> pd.DataFrame:
        """Generate library as a pandas DataFrame with optional streaming.

        Generates sequences in chunks and concatenates them into a single
        DataFrame. For very large libraries, consider using to_file() instead
        to avoid memory issues.

        Parameters
        ----------
        num_seqs : int, optional
            Number of sequences to generate. Required if num_cycles not specified.
        num_cycles : int, optional
            Number of complete cycles through state space.
        chunk_size : int, default 10000
            Number of sequences to generate per chunk. Larger values may be
            faster but use more memory during generation.
        write_tags : bool, default False
            If True, include region tags (e.g., <region>...</region>) in output.
            If False (default), strip all tags from sequences.
        write_style : bool, default False
            If True, include inline style annotations in sequences.
            If False (default), output plain sequences without styles.
        seed : int, optional
            Random seed for reproducibility.
        discard_null_seqs : bool, default True
            If True, skip sequences that were filtered out (NullSeq).
        include_design_cards : bool, default False
            Include design card columns in output.
        columns : list[str], optional
            Specific columns to include. Default is all columns.
        show_progress : bool, default False
            If True, show a tqdm progress bar during generation.
            Requires tqdm to be installed.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the generated library.

        Examples
        --------
        >>> df = pool.to_df(num_seqs=1000)
        >>> df = pool.to_df(num_cycles=2, include_design_cards=True)
        >>> df = pool.to_df(num_seqs=100000, show_progress=True)
        """
        # Validate arguments
        if num_seqs is None and num_cycles is None:
            raise ValueError("Either num_seqs or num_cycles must be specified")

        # Determine target count
        if num_seqs is not None:
            target_count = int(num_seqs)
        else:
            target_count = int(num_cycles) * self.state.num_values

        chunks = []
        generated = 0

        # Create progress tracking
        if show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=target_count, desc="Generating", unit="seq")
            except ImportError:
                import warnings

                warnings.warn(
                    "tqdm not installed. Install with 'pip install tqdm' for progress bars.",
                    stacklevel=2,
                )
                pbar = None
        else:
            pbar = None

        try:
            while generated < target_count:
                remaining = target_count - generated
                this_chunk = min(int(chunk_size), remaining)

                df = self.generate_library(
                    num_seqs=this_chunk,
                    seed=seed,
                    discard_null_seqs=discard_null_seqs,
                    report_design_cards=include_design_cards,
                    _include_inline_styles=write_style,
                )

                if len(df) == 0:
                    break

                # Strip tags if requested
                if not write_tags and "seq" in df.columns:
                    df = df.copy()
                    df["seq"] = df["seq"].apply(_strip_tags)

                # Filter columns if specified
                if columns is not None:
                    available = [c for c in columns if c in df.columns]
                    df = df[available]

                chunks.append(df)
                generated += len(df)

                if pbar is not None:
                    pbar.update(len(df))

                # Increment seed for next chunk to get different sequences
                if seed is not None:
                    seed += this_chunk
        finally:
            if pbar is not None:
                pbar.close()

        if not chunks:
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=columns if columns else ["name", "seq"])

        return pd.concat(chunks, ignore_index=True)

    def to_file(
        self,
        path: Union[str, Path],
        file_type: Literal["csv", "fasta", "tsv", "jsonl"] = "csv",
        num_seqs: Optional[Integral] = None,
        num_cycles: Optional[Integral] = None,
        chunk_size: Integral = 10000,
        write_tags: bool = False,
        write_style: bool = False,
        seed: Optional[Integral] = None,
        discard_null_seqs: bool = True,
        # CSV/TSV options
        include_design_cards: bool = False,
        columns: Optional[list[str]] = None,
        # FASTA options
        line_width: Optional[Integral] = 60,
        description: Optional[Union[str, Callable]] = None,
        # Progress bar
        show_progress: bool = False,
        # Additional CSV options
        **csv_kwargs,
    ) -> int:
        """Export library to file with streaming.

        Generates sequences in chunks and writes them incrementally to avoid
        loading the entire library into memory.

        Parameters
        ----------
        path : str or Path
            Output file path. Supports .gz extension for gzip compression.
        file_type : {'csv', 'fasta', 'tsv', 'jsonl'}, default 'csv'
            Output format:
            - 'csv': Comma-separated values
            - 'tsv': Tab-separated values
            - 'fasta': FASTA format (sequence name and sequence)
            - 'jsonl': JSON Lines (one JSON object per line)
        num_seqs : int, optional
            Number of sequences to export. Required if num_cycles not specified.
        num_cycles : int, optional
            Number of complete cycles through state space.
        chunk_size : int, default 10000
            Number of sequences to generate per chunk. Larger values use more
            memory but may be faster.
        write_tags : bool, default False
            If True, include region tags (e.g., <region>...</region>) in output.
            If False (default), strip all tags from sequences.
        write_style : bool, default False
            If True, include inline style annotations in sequences.
            If False (default), output plain sequences without styles.
        seed : int, optional
            Random seed for reproducibility.
        discard_null_seqs : bool, default True
            If True, skip sequences that were filtered out (NullSeq).
        include_design_cards : bool, default False
            For CSV/TSV/JSONL: include design card columns in output.
        columns : list[str], optional
            For CSV/TSV: specific columns to include. Default is ['name', 'seq'].
        line_width : int, optional, default 60
            For FASTA: wrap sequences at this width. None for no wrapping.
        description : str or callable, optional
            For FASTA: additional description after sequence name.
            If str, used as format template with row dict (e.g., "GC={gc:.2f}").
            If callable, called with row dict, should return string.
        show_progress : bool, default False
            If True, show a tqdm progress bar during export.
            Requires tqdm to be installed.
        **csv_kwargs
            Additional arguments passed to DataFrame.to_csv() for CSV/TSV format.

        Returns
        -------
        int
            Number of sequences written to file.

        Examples
        --------
        >>> pool.to_file("library.csv", num_seqs=100000)
        100000

        >>> pool.to_file("library.fasta", file_type="fasta", num_seqs=10000)
        10000

        >>> pool.to_file(
        ...     "library.csv.gz",
        ...     num_seqs=1000000,
        ...     chunk_size=50000,
        ...     include_design_cards=True,
        ... )
        1000000

        >>> # FASTA with custom description
        >>> pool.to_file(
        ...     "library.fasta",
        ...     file_type="fasta",
        ...     num_seqs=1000,
        ...     description=lambda row: f"length={len(row['seq'])}",
        ... )
        1000

        >>> # With progress bar
        >>> pool.to_file("library.csv", num_seqs=100000, show_progress=True)
        100000
        """
        path = Path(path)

        # Validate arguments
        if num_seqs is None and num_cycles is None:
            raise ValueError("Either num_seqs or num_cycles must be specified")

        if file_type not in ("csv", "fasta", "tsv", "jsonl"):
            raise ValueError(
                f"file_type must be 'csv', 'fasta', 'tsv', or 'jsonl', got {file_type!r}"
            )

        # Determine target count
        if num_seqs is not None:
            target_count = num_seqs
        else:
            target_count = num_cycles * self.state.num_values

        # Dispatch to format-specific writer
        if file_type == "csv":
            return self._export_csv(
                path=path,
                target_count=target_count,
                chunk_size=chunk_size,
                write_tags=write_tags,
                write_style=write_style,
                seed=seed,
                discard_null_seqs=discard_null_seqs,
                include_design_cards=include_design_cards,
                columns=columns,
                sep=",",
                show_progress=show_progress,
                **csv_kwargs,
            )
        elif file_type == "tsv":
            return self._export_csv(
                path=path,
                target_count=target_count,
                chunk_size=chunk_size,
                write_tags=write_tags,
                write_style=write_style,
                seed=seed,
                discard_null_seqs=discard_null_seqs,
                include_design_cards=include_design_cards,
                columns=columns,
                sep="\t",
                show_progress=show_progress,
                **csv_kwargs,
            )
        elif file_type == "fasta":
            return self._export_fasta(
                path=path,
                target_count=target_count,
                chunk_size=chunk_size,
                write_tags=write_tags,
                write_style=write_style,
                seed=seed,
                discard_null_seqs=discard_null_seqs,
                line_width=line_width,
                description=description,
                show_progress=show_progress,
            )
        else:  # jsonl
            return self._export_jsonl(
                path=path,
                target_count=target_count,
                chunk_size=chunk_size,
                write_tags=write_tags,
                write_style=write_style,
                seed=seed,
                discard_null_seqs=discard_null_seqs,
                include_design_cards=include_design_cards,
                show_progress=show_progress,
            )

    def _export_csv(
        self,
        path: Path,
        target_count: int,
        chunk_size: int,
        write_tags: bool,
        write_style: bool,
        seed: Optional[int],
        discard_null_seqs: bool,
        include_design_cards: bool,
        columns: Optional[list[str]],
        sep: str,
        show_progress: bool = False,
        **csv_kwargs,
    ) -> int:
        """Export to CSV/TSV format."""
        written = 0
        first_chunk = True

        # Create progress bar if requested
        if show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=target_count, desc="Exporting", unit="seq")
            except ImportError:
                import warnings

                warnings.warn(
                    "tqdm not installed. Install with 'pip install tqdm' for progress bars.",
                    stacklevel=3,
                )
                pbar = None
        else:
            pbar = None

        try:
            while written < target_count:
                remaining = target_count - written
                this_chunk = min(chunk_size, remaining)

                df = self.generate_library(
                    num_seqs=this_chunk,
                    seed=seed,
                    discard_null_seqs=discard_null_seqs,
                    report_design_cards=include_design_cards,
                    _include_inline_styles=write_style,
                )

                if len(df) == 0:
                    break

                # Strip tags if requested
                if not write_tags and "seq" in df.columns:
                    df = df.copy()
                    df["seq"] = df["seq"].apply(_strip_tags)

                # Filter columns if specified
                if columns is not None:
                    available = [c for c in columns if c in df.columns]
                    df = df[available]

                # Write to file
                with _open_file(path, "w" if first_chunk else "a") as f:
                    df.to_csv(
                        f,
                        sep=sep,
                        index=False,
                        header=first_chunk,
                        **csv_kwargs,
                    )

                written += len(df)
                first_chunk = False

                if pbar is not None:
                    pbar.update(len(df))

                # Increment seed for next chunk to get different sequences
                if seed is not None:
                    seed += this_chunk
        finally:
            if pbar is not None:
                pbar.close()

        return written

    def _export_fasta(
        self,
        path: Path,
        target_count: int,
        chunk_size: int,
        write_tags: bool,
        write_style: bool,
        seed: Optional[int],
        discard_null_seqs: bool,
        line_width: Optional[int],
        description: Optional[Union[str, Callable]],
        show_progress: bool = False,
    ) -> int:
        """Export to FASTA format."""
        written = 0
        first_chunk = True

        # Create progress bar if requested
        if show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=target_count, desc="Exporting", unit="seq")
            except ImportError:
                import warnings

                warnings.warn(
                    "tqdm not installed. Install with 'pip install tqdm' for progress bars.",
                    stacklevel=3,
                )
                pbar = None
        else:
            pbar = None

        try:
            while written < target_count:
                remaining = target_count - written
                this_chunk = min(chunk_size, remaining)

                df = self.generate_library(
                    num_seqs=this_chunk,
                    seed=seed,
                    discard_null_seqs=discard_null_seqs,
                    report_design_cards=False,
                    _include_inline_styles=write_style,
                )

                if len(df) == 0:
                    break

                chunk_written = 0
                # Write FASTA entries
                with _open_file(path, "w" if first_chunk else "a") as f:
                    for _, row in df.iterrows():
                        name = row.get("name", f"seq_{written}")
                        seq = row.get("seq", "")

                        if seq is None:
                            continue

                        # Strip tags if requested
                        if not write_tags:
                            seq = _strip_tags(seq)

                        # Build header line
                        header = f">{name}"
                        if description is not None:
                            if callable(description):
                                desc_str = description(row.to_dict())
                            else:
                                desc_str = description.format(**row.to_dict())
                            header = f"{header} {desc_str}"

                        f.write(header + "\n")

                        # Write sequence with optional line wrapping
                        if line_width is not None and line_width > 0:
                            for i in range(0, len(seq), line_width):
                                f.write(seq[i : i + line_width] + "\n")
                        else:
                            f.write(seq + "\n")

                        written += 1
                        chunk_written += 1

                first_chunk = False

                if pbar is not None:
                    pbar.update(chunk_written)

                # Increment seed for next chunk
                if seed is not None:
                    seed += this_chunk
        finally:
            if pbar is not None:
                pbar.close()

        return written

    def _export_jsonl(
        self,
        path: Path,
        target_count: int,
        chunk_size: int,
        write_tags: bool,
        write_style: bool,
        seed: Optional[int],
        discard_null_seqs: bool,
        include_design_cards: bool,
        show_progress: bool = False,
    ) -> int:
        """Export to JSON Lines format."""
        written = 0
        first_chunk = True

        # Create progress bar if requested
        if show_progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(total=target_count, desc="Exporting", unit="seq")
            except ImportError:
                import warnings

                warnings.warn(
                    "tqdm not installed. Install with 'pip install tqdm' for progress bars.",
                    stacklevel=3,
                )
                pbar = None
        else:
            pbar = None

        try:
            while written < target_count:
                remaining = target_count - written
                this_chunk = min(chunk_size, remaining)

                df = self.generate_library(
                    num_seqs=this_chunk,
                    seed=seed,
                    discard_null_seqs=discard_null_seqs,
                    report_design_cards=include_design_cards,
                    _include_inline_styles=write_style,
                )

                if len(df) == 0:
                    break

                chunk_written = 0
                # Write JSONL entries
                with _open_file(path, "w" if first_chunk else "a") as f:
                    for _, row in df.iterrows():
                        record = row.to_dict()

                        # Strip tags if requested
                        if not write_tags and "seq" in record and record["seq"] is not None:
                            record["seq"] = _strip_tags(record["seq"])

                        # Convert numpy types to Python types for JSON serialization
                        for key, value in record.items():
                            if hasattr(value, "item"):  # numpy scalar
                                record[key] = value.item()
                            elif value is None or (
                                hasattr(value, "__len__") and len(str(value)) == 0
                            ):
                                record[key] = None

                        f.write(json.dumps(record) + "\n")
                        written += 1
                        chunk_written += 1

                first_chunk = False

                if pbar is not None:
                    pbar.update(chunk_written)

                # Increment seed for next chunk
                if seed is not None:
                    seed += this_chunk
        finally:
            if pbar is not None:
                pbar.close()

        return written
