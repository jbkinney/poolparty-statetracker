"""Region operation mixins for Pool class."""
from ..types import Pool_type, Optional, Real, Callable, Union, beartype


class RegionOpsMixin:
    """Mixin providing region operation methods for Pool."""
    
    def apply_at_region(
        self,
        region_name: str,
        transform_fn: Callable,
        remove_tags: bool = True,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..region_ops.apply_at_region import apply_at_region
        return apply_at_region(
            self,
            region_name,
            transform_fn,
            remove_tags=remove_tags,
            iter_order=iter_order,
        )
    
    def insert_tags(
        self,
        region_name: str,
        start: int,
        stop: Optional[int] = None,
        strand: str = '+',
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..region_ops.insert_tags import insert_tags
        return insert_tags(
            self,
            region_name,
            start,
            stop,
            strand,
            iter_order=iter_order,
        )
    
    def remove_tags(
        self,
        region_name: str,
        keep_content: bool = True,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..region_ops.remove_tags import remove_tags
        return remove_tags(
            self,
            region_name,
            keep_content,
            iter_order=iter_order,
        )
    
    def replace_region(
        self,
        content_pool: Union[Pool_type, str],
        region_name: str,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from ..region_ops.replace_region import replace_region
        return replace_region(
            self,
            content_pool,
            region_name,
            iter_order=iter_order,
        )
    
    def clear_tags(self, **kwargs) -> Pool_type:
        """Remove all region tags from sequences, keeping content."""
        from ..fixed_ops.fixed import fixed_operation
        from ..utils.parsing_utils import strip_all_tags
        
        result = fixed_operation(
            parent_pools=[self],
            seq_from_seqs_fn=lambda seqs: strip_all_tags(seqs[0]),
            seq_length_from_pool_lengths_fn=lambda lengths: None,
            **kwargs,
        )
        result._regions = set()
        return result
