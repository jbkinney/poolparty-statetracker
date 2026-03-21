"""Helper class for region-based operation logic."""

from dataclasses import dataclass, field

from ..types import Optional, RegionType, Seq, SeqStyle, Sequence, StyleList, Union, beartype
from .dna_seq import DnaSeq
from .parsing_utils import (
    ParsedRegion,
    build_region_tags,
    find_all_regions,
    nontag_pos_to_literal_pos,
    validate_single_region_from_list,
)


@beartype
@dataclass
class RegionContext:
    """Encapsulates region extraction, reassembly, and tag handling.

    This class manages the complex logic of extracting a region from a sequence,
    splitting styles, and reassembling results with proper tag handling.
    """

    prefix: str
    region_content: str
    suffix: str
    region_start: int
    region_end: int
    region_name: Optional[str]  # None if region is [start, stop] interval
    strand: Optional[str]  # '+' or '-' for named regions
    remove_tags: bool
    _original_seq: str  # Original parent sequence (for parsing)

    # Pre-split style parts (populated by split_parent_styles)
    _prefix_styles: StyleList = field(default_factory=list)
    _suffix_styles: StyleList = field(default_factory=list)
    _prefix_seq_style: Optional[SeqStyle] = field(default=None)
    _suffix_seq_style: Optional[SeqStyle] = field(default=None)

    # Cached parsed regions (to avoid re-parsing in reassemble methods)
    _parsed_regions: Sequence[ParsedRegion] = field(default_factory=tuple)
    _region_obj: Optional[ParsedRegion] = field(default=None)

    @classmethod
    def from_sequence(
        cls,
        seq_obj: Union[Seq, str],
        region: RegionType,
        remove_tags: bool = False,
    ) -> "RegionContext":
        """Create RegionContext from a Seq/string and region specification.

        Parameters
        ----------
        seq_obj : Union[Seq, str]
            The Seq object or string containing the region.
        region : RegionType
            Region specification: region name (str) or [start, stop] interval.
        remove_tags : bool, default=False
            Whether to remove region tags during reassembly.

        Returns
        -------
        RegionContext
            Context object containing extracted region parts.
        """
        seq = seq_obj.string if isinstance(seq_obj, Seq) else seq_obj

        if isinstance(region, str):
            # Named region - parse ONCE and reuse
            parsed_regions = find_all_regions(seq)
            region_obj = validate_single_region_from_list(parsed_regions, region, seq)

            # Extract parts directly from region_obj (no second parse needed)
            content = region_obj.content
            region_start = region_obj.content_start
            region_end = region_obj.content_end

            return cls(
                prefix=seq[: region_obj.start],  # Everything before opening tag
                region_content=content,
                suffix=seq[region_obj.end :],  # Everything after closing tag
                region_start=region_start,
                region_end=region_end,
                region_name=region,
                strand=None,  # Strand no longer stored in tags
                remove_tags=remove_tags,
                _original_seq=seq,
                _parsed_regions=tuple(parsed_regions),
                _region_obj=region_obj,
            )
        else:
            # Interval region [start, stop] — molecular (nontag) positions
            # Use last-char+1 for stop to exclude trailing tags from region content,
            # keeping them in the suffix (preserves tag structure for operations that strip tags)
            start, stop = int(region[0]), int(region[1])
            lit_start = nontag_pos_to_literal_pos(seq, start)
            if start < stop:
                lit_stop = nontag_pos_to_literal_pos(seq, stop - 1) + 1
            else:
                lit_stop = lit_start
            return cls(
                prefix=seq[:lit_start],
                region_content=seq[lit_start:lit_stop],
                suffix=seq[lit_stop:],
                region_start=lit_start,
                region_end=lit_stop,
                region_name=None,
                strand=None,
                remove_tags=remove_tags,
                _original_seq=seq,
            )

    def split_parent_seq(self, parent: Seq) -> tuple[Seq, Seq, Seq]:
        """Split parent Seq into prefix, region, suffix Seq objects.

        Parameters
        ----------
        parent : Seq
            Parent Seq to split.

        Returns
        -------
        tuple[Seq, Seq, Seq]
            Prefix, region, and suffix as Seq objects with appropriate slicing.
        """
        if self.region_name is None:
            # Interval region - simple slicing
            prefix = parent[: self.region_start]
            region = parent[self.region_start : self.region_end]
            suffix = parent[self.region_end :]
            return prefix, region, suffix
        else:
            # Named region - use cached region_obj (no re-parsing needed)
            region_obj = self._region_obj

            # Prefix: everything before opening tag
            prefix = parent[: region_obj.start]

            # Region: extract content between tags
            region_string = self.region_content
            region_style = (
                parent.style[self.region_start : self.region_end]
                if parent.style is not None
                else None
            )
            region = DnaSeq(region_string, region_style)

            # Suffix: everything after closing tag
            suffix = parent[region_obj.end :]

            return prefix, region, suffix

    def reassemble_seq_string(self, output_string: str) -> str:
        """Legacy method: Reassemble output string with proper tag handling.

        Parameters
        ----------
        output_string : str
            The sequence output string from compute().

        Returns
        -------
        str
            Reassembled full sequence string with tags as appropriate.
        """
        if self.region_name is None:
            # Interval region - simple concatenation
            return self.prefix + output_string + self.suffix

        # Named region - use cached region_obj for clean parts
        region_obj = self._region_obj
        clean_prefix = self._original_seq[: region_obj.start]
        clean_suffix = self._original_seq[region_obj.end :]

        if self.remove_tags:
            # Remove tags
            return clean_prefix + output_string + clean_suffix
        else:
            # Keep tags - rebuild with new content
            wrapped = build_region_tags(self.region_name, output_string)
            return clean_prefix + wrapped + clean_suffix

    def split_parent_styles(
        self,
        parent_styles: list[SeqStyle | None] | None,
    ) -> SeqStyle | None:
        """Split first parent style into prefix/region/suffix parts.

        Parameters
        ----------
        parent_styles : list[SeqStyle | None] | None
            Input styles from parent pools.

        Returns
        -------
        SeqStyle | None
            Region style for compute input (0-indexed for region content), or None if suppressed.
        """
        if parent_styles and len(parent_styles) > 0:
            parent_style = parent_styles[0]
            if parent_style is None:
                # Styles suppressed
                return None
            prefix_style, region_style, suffix_style = parent_style.split(
                [self.region_start, self.region_end]
            )
            # Store raw StyleLists for reassembly
            self._prefix_styles = prefix_style.style_list
            self._suffix_styles = suffix_style.style_list
            # Store SeqStyle objects for tag slicing
            self._prefix_seq_style = prefix_style
            self._suffix_seq_style = suffix_style
            return region_style
        else:
            # No parent styles
            from .style_utils import styles_suppressed

            return None if styles_suppressed() else SeqStyle.empty(len(self.region_content))

    def reassemble_seq(self, prefix: Seq, output: Seq, suffix: Seq) -> Seq:
        """Reassemble Seq from prefix, output, and suffix with proper tag handling.

        Parameters
        ----------
        prefix : Seq
            Prefix Seq.
        output : Seq
            Output Seq from compute().
        suffix : Seq
            Suffix Seq.

        Returns
        -------
        Seq
            Reassembled Seq with tags handled appropriately.
        """
        if self.region_name is None:
            # Interval region - sliced parts are tag-free, use fast join
            return Seq._join_fast([prefix, output, suffix])

        # Named region - use cached region_obj (no re-parsing needed)
        region_obj = self._region_obj

        if self.remove_tags:
            # Remove tags - join clean parts (result is tag-free, use fast join)
            clean_prefix_seq = prefix[: region_obj.start]
            clean_suffix_seq = suffix

            return Seq._join_fast([clean_prefix_seq, output, clean_suffix_seq])
        else:
            # Keep tags - wrap output with region tags
            wrapped_string = build_region_tags(
                self.region_name,
                output.string,
            )

            # Calculate tag lengths
            test_tag = build_region_tags(self.region_name, "X")
            opening_tag_len = test_tag.index(">") + 1
            closing_tag_len = len(wrapped_string) - opening_tag_len - len(output.string)

            # Create style for wrapped output (empty styles for tags)
            if output.style is None:
                wrapped_style = None
            else:
                wrapped_style = SeqStyle.join(
                    [
                        SeqStyle.empty(opening_tag_len),
                        output.style,
                        SeqStyle.empty(closing_tag_len),
                    ]
                )

            wrapped_seq = DnaSeq(wrapped_string, wrapped_style)

            # Use cached region_obj for clean prefix
            clean_prefix_seq = prefix[: region_obj.start]

            # Result has tags, use regular join to parse them
            return DnaSeq.join([clean_prefix_seq, wrapped_seq, suffix])

    def reassemble_style(
        self,
        output_style: SeqStyle | None,
        output_seq: str,
    ) -> SeqStyle | None:
        """Reassemble output style with proper position adjustments.

        Parameters
        ----------
        output_style : SeqStyle | None
            The style output from compute() (0-indexed for region), or None if suppressed.
        output_seq : str
            The sequence output from compute() (needed for length calculations).

        Returns
        -------
        SeqStyle | None
            Reassembled full style with correct global positions, or None if suppressed.
        """
        if output_style is None:
            return None

        if self.region_name is None:
            # Interval region - rebuild prefix/suffix with actual lengths
            prefix_seq_style = SeqStyle.from_style_list(self._prefix_styles, len(self.prefix))
            suffix_seq_style = SeqStyle.from_style_list(self._suffix_styles, len(self.suffix))
            return SeqStyle.join([prefix_seq_style, output_style, suffix_seq_style])

        # Named region - use cached region_obj (no re-parsing needed)
        region_obj = self._region_obj

        # Calculate clean lengths from cached data
        # clean_prefix is everything before the opening tag
        clean_prefix_len = region_obj.start
        # clean_suffix is everything after the closing tag
        clean_suffix_len = len(self._original_seq) - region_obj.end

        if self.remove_tags:
            # When tags removed: slice prefix/suffix to exclude tag positions
            opening_tag_len = self.region_start - region_obj.start
            closing_tag_len = region_obj.end - self.region_end

            # Slice to exclude tag positions (or create empty if no parent styles)
            if self._prefix_seq_style is not None:
                prefix_seq_style = self._prefix_seq_style[: region_obj.start]
            else:
                prefix_seq_style = SeqStyle.empty(clean_prefix_len)

            if self._suffix_seq_style is not None:
                suffix_seq_style = self._suffix_seq_style[closing_tag_len:]
            else:
                suffix_seq_style = SeqStyle.empty(clean_suffix_len)

            return SeqStyle.join([prefix_seq_style, output_style, suffix_seq_style])
        else:
            # When tags kept: add empty styles for tag positions
            test_tag = build_region_tags(self.region_name, "X")
            opening_tag_len = test_tag.index(">") + 1
            new_wrapped = build_region_tags(self.region_name, output_seq)
            closing_tag_len = len(new_wrapped) - opening_tag_len - len(output_seq)

            # Rebuild prefix/suffix with clean lengths
            prefix_seq_style = SeqStyle.from_style_list(self._prefix_styles, clean_prefix_len)
            suffix_seq_style = SeqStyle.from_style_list(self._suffix_styles, clean_suffix_len)

            return SeqStyle.join(
                [
                    prefix_seq_style,
                    SeqStyle.empty(opening_tag_len),
                    output_style,
                    SeqStyle.empty(closing_tag_len),
                    suffix_seq_style,
                ]
            )
