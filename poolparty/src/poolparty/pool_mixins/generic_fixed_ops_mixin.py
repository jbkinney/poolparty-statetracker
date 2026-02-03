"""Generic fixed operation mixins for Pool class - operations that work on any sequence type."""

from numbers import Integral
from typing import Literal

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
    ) -> Pool_type:
        """Slice sequences. See slice_seq() for details."""
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
    ) -> Pool_type:
        """Add a prefix to sequence names without modifying sequences."""
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
    ) -> Pool_type:
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
    ) -> Pool_type:
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
    ) -> Pool_type:
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
    ) -> Pool_type:
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
    ) -> Pool_type:
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
    ) -> Pool_type:
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
