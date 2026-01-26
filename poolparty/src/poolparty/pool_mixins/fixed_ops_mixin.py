"""Fixed operation mixins for Pool class."""
from ..types import Pool_type, RegionType, Optional, Real, beartype
from typing import Literal


class FixedOpsMixin:
    """Mixin providing fixed operation methods for Pool."""
    
    def rc(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from ..fixed_ops.rc import rc
        return rc(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            style=style,
        )
    
    def swapcase(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from ..fixed_ops.swapcase import swapcase
        return swapcase(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            style=style,
        )
    
    def upper(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from ..fixed_ops.upper import upper
        return upper(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            style=style,
        )
    
    def lower(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from ..fixed_ops.lower import lower
        return lower(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
            style=style,
        )
    
    def clear_gaps(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..fixed_ops.clear_gaps import clear_gaps
        return clear_gaps(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
        )
    
    def clear_annotation(
        self,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..fixed_ops.clear_annotation import clear_annotation
        return clear_annotation(
            pool=self,
            region=region,
            remove_tags=remove_tags,
            iter_order=iter_order,
        )
    
    def stylize(
        self,
        region: RegionType = None,
        *,
        style: str,
        which: Literal['all', 'upper', 'lower', 'gap', 'tags', 'contents'] = 'contents',
        regex: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..fixed_ops.style import stylize
        return stylize(
            pool=self,
            region=region,
            style=style,
            which=which,
            regex=regex,
            iter_order=iter_order,
        )
