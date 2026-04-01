"""Region operation mixins for Pool class."""

from typing_extensions import Self

from ..types import Callable, Optional, Pool_type, Real, Union


class RegionOpsMixin:
    """Mixin providing region operation methods for Pool."""

    def annotate_region(
        self,
        region_name: str,
        extent: Optional[tuple[int, int]] = None,
        style: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Annotate a region in sequences, optionally applying a style.

        Parameters
        ----------
        region_name : str
            Name for the region.
        extent : Optional[tuple[int, int]], default=None
            Start and stop positions (0-indexed, stop exclusive) for the region.
            If None and region doesn't exist, uses the entire sequence.
            Must be None if region already exists.
        style : Optional[str], default=None
            Style to apply to the region (e.g., 'red', 'bold blue').
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            Pool with region annotated and optionally styled.
        """
        from ..region_ops.annotate_region import annotate_region

        return annotate_region(
            self,
            region_name,
            extent=extent,
            style=style,
            iter_order=iter_order,
            prefix=prefix,
        )

    def apply_at_region(
        self,
        region_name: str,
        transform_fn: Callable,
        rc: bool = False,
        remove_tags: bool = True,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Apply a transformation to the content of a region.

        Parameters
        ----------
        region_name : str
            Name of the region whose content to transform.
        transform_fn : Callable[[Pool], Pool]
            Function that takes a Pool and returns a transformed Pool.
        rc : bool, default=False
            If True, reverse-complement content before transform and
            reverse-complement result back before insertion.
        remove_tags : bool, default=True
            If True, region tags are removed from the result.
            If False, region tags are preserved around the transformed content.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool with the region content transformed.
        """
        from ..region_ops.apply_at_region import apply_at_region

        return apply_at_region(
            self,
            region_name,
            transform_fn,
            rc=rc,
            remove_tags=remove_tags,
            iter_order=iter_order,
            prefix=prefix,
        )

    def extract_region(
        self,
        region_name: str,
        rc: bool = False,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Extract content from a named region as a new Pool.

        Parameters
        ----------
        region_name : str
            Name of the region to extract content from.
        rc : bool, default=False
            If True, reverse-complement the extracted content.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool yielding the content inside the region.
        """
        from ..region_ops.extract_region import extract_region

        return extract_region(
            self,
            region_name,
            rc=rc,
            iter_order=iter_order,
            prefix=prefix,
        )

    def insert_tags(
        self,
        region_name: str,
        start: int,
        stop: Optional[int] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Insert XML-style region tags at a fixed position in sequences.

        Parameters
        ----------
        region_name : str
            Name for the region (e.g., 'region', 'orf', 'insert').
        start : int
            Start position (0-based) for the region.
        stop : Optional[int], default=None
            End position (exclusive). If None, creates a zero-length region
            at start.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool yielding sequences with the region tags inserted.
        """
        from ..region_ops.insert_tags import insert_tags

        return insert_tags(
            self,
            region_name,
            start,
            stop,
            iter_order=iter_order,
            prefix=prefix,
        )

    def remove_tags(
        self,
        region_name: str,
        keep_content: bool = True,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Remove region tags from sequences.

        Parameters
        ----------
        region_name : str
            Name of the region to remove.
        keep_content : bool, default=True
            If True, keep the content inside the region (just remove tags).
            If False, remove both the region tags and their content.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool yielding sequences with the region tags removed.
        """
        from ..region_ops.remove_tags import remove_tags

        return remove_tags(
            self,
            region_name,
            keep_content,
            iter_order=iter_order,
            prefix=prefix,
        )

    def replace_region(
        self,
        content_pool: Union[Pool_type, str],
        region_name: str,
        rc: bool = False,
        sync: bool = False,
        keep_tags: bool = False,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
    ) -> Self:
        """Replace a region with content from another Pool.

        Parameters
        ----------
        content_pool : Pool or str
            Pool or sequence string to insert at the region position.
        region_name : str
            Name of the region to replace.
        rc : bool, default=False
            If True, reverse-complement the content before insertion.
        sync : bool, default=False
            If True, synchronize self and content_pool so they iterate
            in lock-step (1:1 pairing) instead of a Cartesian product.
        keep_tags : bool, default=False
            If True, preserve the region's XML tags around the new content.
        iter_order : Optional[Real], default=None
            Iteration order priority for the Operation.
        prefix : Optional[str], default=None
            Prefix for sequence names in the resulting Pool.

        Returns
        -------
        Pool
            A Pool with the region replaced by content_pool sequences.
        """
        from ..region_ops.replace_region import replace_region

        return replace_region(
            self,
            content_pool,
            region_name,
            rc=rc,
            sync=sync,
            keep_tags=keep_tags,
            iter_order=iter_order,
            prefix=prefix,
        )

    def clear_tags(self, **kwargs) -> Self:
        """Remove all region tags from sequences, keeping content.

        Returns
        -------
        Pool
            A Pool yielding sequences with all region tags stripped.
        """
        from ..fixed_ops.fixed import fixed_operation
        from ..utils.parsing_utils import strip_all_tags

        result = fixed_operation(
            parent_pools=[self],
            seq_from_seqs_fn=lambda seqs: strip_all_tags(seqs[0]),
            seq_length_from_pool_lengths_fn=lambda lengths: lengths[0],
            **kwargs,
        )
        result._regions = set()
        return result
