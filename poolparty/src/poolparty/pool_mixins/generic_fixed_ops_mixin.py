"""Generic fixed operation mixins for Pool class - operations that work on any sequence type."""

from numbers import Integral
from typing import Literal

from typing_extensions import Self

from ..types import Optional, Pool_type, Real, RegionType


class GenericFixedOpsMixin:
    """Mixin providing generic fixed operation methods for Pool."""

    def slice_seq(
        self,
        region: RegionType = None,
        start: Optional[Integral] = None,
        stop: Optional[Integral] = None,
        step: Optional[Integral] = None,
        keep_context: bool = False,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        style: Optional[str] = None,
    ) -> Self:
        """Extract a subsequence based on region and/or Python-style slice parameters.

        Parameters
        ----------
        region : RegionType, default=None
            Region to slice from. Can be:
            - str: Name of an annotated region (e.g., 'orf')
            - Sequence[int]: [start, stop] interval in the sequence
            - None: Use the full sequence
        start : Optional[Integral], default=None
            Start position for slicing (0-indexed, Python-style).
        stop : Optional[Integral], default=None
            Stop position for slicing (exclusive, Python-style).
        step : Optional[Integral], default=None
            Step for slicing (Python-style).
        keep_context : bool, default=False
            If True, reassemble the sliced content back into the original sequence
            context (prefix + sliced_content + suffix).
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        style : Optional[str], default=None
            Style to apply to the resulting sliced sequences (e.g., 'red', 'blue bold').

        Returns
        -------
        Pool
            A Pool containing sliced sequences.
        """
        from ..fixed_ops.slice_seq import slice_seq

        return slice_seq(
            pool=self,
            region=region,
            start=start,
            stop=stop,
            step=step,
            keep_context=keep_context,
            iter_order=iter_order,
            prefix=prefix,
            style=style,
        )

    def add_prefix(
        self,
        prefix: str,
        iter_order: Optional[Real] = None,
    ) -> Self:
        """Add a prefix to sequence names without modifying the sequences.

        Parameters
        ----------
        prefix : str
            Prefix to add to sequence names.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.

        Returns
        -------
        Pool
            A Pool with the same sequences but updated names.
        """
        from ..fixed_ops.add_prefix import add_prefix

        return add_prefix(
            pool=self,
            prefix=prefix,
            iter_order=iter_order,
        )

    def swapcase(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        style: Optional[str] = None,
    ) -> Self:
        """Swap case of sequence characters, preserving marker tags.

        Parameters
        ----------
        region : RegionType, default=None
            Region to apply transformation to. Can be marker name (str),
            [start, stop], or None.
        remove_tags : Optional[bool], default=None
            If True and region is a marker name, remove marker tags from output.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        style : Optional[str], default=None
            Style to apply to the resulting sequences (e.g., 'red', 'blue bold').

        Returns
        -------
        Pool
            A Pool containing case-swapped sequences.
        """
        from ..fixed_ops.swapcase import swapcase

        return swapcase(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            prefix=prefix,
            style=style,
        )

    def upper(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        style: Optional[str] = None,
    ) -> Self:
        """Convert sequence characters to uppercase, preserving marker tags.

        Parameters
        ----------
        region : RegionType, default=None
            Region to apply transformation to. Can be marker name (str),
            [start, stop], or None.
        remove_tags : Optional[bool], default=None
            If True and region is a marker name, remove marker tags from output.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        style : Optional[str], default=None
            Style to apply to the resulting sequences (e.g., 'red', 'blue bold').

        Returns
        -------
        Pool
            A Pool containing uppercase sequences.
        """
        from ..fixed_ops.upper import upper

        return upper(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            prefix=prefix,
            style=style,
        )

    def lower(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        style: Optional[str] = None,
    ) -> Self:
        """Convert sequence characters to lowercase, preserving marker tags.

        Parameters
        ----------
        region : RegionType, default=None
            Region to apply transformation to. Can be marker name (str),
            [start, stop], or None.
        remove_tags : Optional[bool], default=None
            If True and region is a marker name, remove marker tags from output.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.
        style : Optional[str], default=None
            Style to apply to the resulting sequences (e.g., 'red', 'blue bold').

        Returns
        -------
        Pool
            A Pool containing lowercase sequences.
        """
        from ..fixed_ops.lower import lower

        return lower(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            prefix=prefix,
            style=style,
        )

    def clear_gaps(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Remove all gap/non-molecular characters from sequences.

        Parameters
        ----------
        region : RegionType, default=None
            Region to apply transformation to. Can be marker name (str),
            [start, stop], or None.
        remove_tags : Optional[bool], default=None
            If True and region is a marker name, remove marker tags from output.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool containing only molecular alphabet characters.
        """
        from ..fixed_ops.clear_gaps import clear_gaps

        return clear_gaps(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            prefix=prefix,
        )

    def clear_annotation(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Remove all annotations and uppercase sequences.

        Parameters
        ----------
        region : RegionType, default=None
            Region to apply transformation to. Can be marker name (str),
            [start, stop], or None.
        remove_tags : Optional[bool], default=None
            If True and region is a marker name, remove marker tags from output.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool with cleared annotations and uppercase sequences.
        """
        from ..fixed_ops.clear_annotation import clear_annotation

        return clear_annotation(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            prefix=prefix,
        )

    def stylize(
        self,
        region: RegionType = None,
        *,
        style: str,
        which: Literal["all", "upper", "lower", "gap", "tags", "contents"] = "contents",
        regex: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Apply inline styling to sequences without modifying them.

        Parameters
        ----------
        region : RegionType, default=None
            Region to restrict styling. Can be marker name or [start, stop].
            If None, styles the entire sequence.
        style : str
            Style spec string (e.g., 'red bold', 'lower cyan').
        which : WhichType, default='contents'
            Pattern selector: 'all', 'upper', 'lower', 'gap', 'tags', 'contents'.
        regex : Optional[str], default=None
            Custom regex pattern. If specified, overrides ``which``.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool with inline styling attached to sequences.
        """
        from ..fixed_ops.stylize import stylize

        return stylize(
            pool=self,
            region=region,
            style=style,
            which=which,
            regex=regex,
            iter_order=iter_order,
            prefix=prefix,
        )
