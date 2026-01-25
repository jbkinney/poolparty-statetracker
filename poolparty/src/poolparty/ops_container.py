"""OpsContainer class housing convenience methods for Pool."""
from .types import Pool_type, Union, Optional, Real, Callable, Integral, Sequence, beartype


@beartype
class OpsContainer:
    """Container for Pool convenience methods that wrap Operation factory functions."""
    
    def __init__(self, pool: Pool_type) -> None:
        """Initialize with a reference to the parent Pool."""
        self.pool = pool
    
    def mutagenize(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .base_ops.mutagenize import mutagenize
        return mutagenize(pool=self.pool, region=region, **kwargs)
    
    def mutagenize_scan(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .scan_ops.mutagenize_scan import mutagenize_scan
        return mutagenize_scan(pool=self.pool, region=region, **kwargs)
    
    def deletion_scan(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .scan_ops.deletion_scan import deletion_scan
        return deletion_scan(pool=self.pool, region=region, **kwargs)
    
    def insertion_scan(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .scan_ops.insertion_scan import insertion_scan
        return insertion_scan(pool=self.pool, region=region, **kwargs)
    
    def replacement_scan(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .scan_ops.insertion_scan import replacement_scan
        return replacement_scan(pool=self.pool, region=region, **kwargs)
    
    def shuffle_scan(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .scan_ops.shuffle_scan import shuffle_scan
        return shuffle_scan(pool=self.pool, region=region, **kwargs)
    
    def shuffle_seq(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .base_ops.shuffle_seq import shuffle_seq
        return shuffle_seq(pool=self.pool, region=region, **kwargs)
    
    def insert_from_iupac(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .base_ops.from_iupac import from_iupac
        return from_iupac(pool=self.pool, region=region, **kwargs)
    
    def insert_from_motif(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .base_ops.from_motif import from_motif
        return from_motif(pool=self.pool, region=region, **kwargs)
    
    def insert_kmers(
        self,
        region: Optional[str] = None,
        style_kmers: Optional[str] = None,
        style_background: Optional[str] = None,
        **kwargs,
    ) -> Pool_type:
        from .base_ops.get_kmers import get_kmers
        from .fixed_ops.stylize import stylize
        
        # Map style_kmers to style parameter for get_kmers
        if style_kmers is not None:
            kwargs['style'] = style_kmers
        
        # Apply style_background to pool before kmer insertion
        pool = self.pool
        if style_background is not None:
            pool = stylize(pool, style=style_background)
        
        return get_kmers(pool=pool, region=region, **kwargs)
    
    #########################################################################
    # Fixed operation convenience methods
    #########################################################################
    
    def rc(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .fixed_ops.rc import rc
        return rc(pool=self.pool, region=region, **kwargs)
    
    def swapcase(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .fixed_ops.swapcase import swapcase
        return swapcase(pool=self.pool, region=region, **kwargs)
    
    def upper(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .fixed_ops.upper import upper
        return upper(pool=self.pool, region=region, **kwargs)
    
    def lower(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .fixed_ops.lower import lower
        return lower(pool=self.pool, region=region, **kwargs)
    
    def clear_gaps(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .fixed_ops.clear_gaps import clear_gaps
        return clear_gaps(pool=self.pool, region=region, **kwargs)
    
    def clear_annotation(self, region: Optional[str] = None, **kwargs) -> Pool_type:
        from .fixed_ops.clear_annotation import clear_annotation
        return clear_annotation(pool=self.pool, region=region, **kwargs)
    
    def stylize(self, region=None, *, style: str, **kwargs) -> Pool_type:
        from .fixed_ops.stylize import stylize
        return stylize(pool=self.pool, region=region, style=style, **kwargs)
    
    #########################################################################
    # State operation convenience methods
    #########################################################################
    
    def repeat_states(self, times: Integral, **kwargs) -> Pool_type:
        from .state_ops.repeat import repeat
        return repeat(pool=self.pool, times=times, **kwargs)
    
    def sample_states(self, **kwargs) -> Pool_type:
        from .state_ops.state_sample import state_sample
        return state_sample(pool=self.pool, **kwargs)
    
    def shuffle_states(self, **kwargs) -> Pool_type:
        from .state_ops.state_shuffle import state_shuffle
        return state_shuffle(pool=self.pool, **kwargs)
    
    def slice_states(self, key: Union[Integral, slice], **kwargs) -> Pool_type:
        from .state_ops.state_slice import state_slice
        return state_slice(pool=self.pool, key=key, **kwargs)
    
    #########################################################################
    # Marker management methods
    #########################################################################
    
    def apply_at_marker(self, marker_name: str, transform_fn: Callable, remove_tags: Optional[bool] = None, **kwargs) -> Pool_type:
        from .marker_ops.apply_at_marker import apply_at_marker
        return apply_at_marker(self.pool, marker_name, transform_fn, remove_marker=remove_tags, **kwargs)
    
    def insert_marker(
        self,
        marker_name: str,
        start: int,
        stop: Optional[int] = None,
        strand: str = '+',
        **kwargs,
    ) -> Pool_type:
        """Insert an XML-style marker at a fixed position in sequences."""
        from .marker_ops.insert_marker import insert_marker
        return insert_marker(self.pool, marker_name, start, stop, strand, **kwargs)
    
    def remove_marker(
        self,
        marker_name: str,
        keep_content: bool = True,
        **kwargs,
    ) -> Pool_type:
        """Remove a marker from sequences."""
        from .marker_ops.remove_marker import remove_marker
        return remove_marker(self.pool, marker_name, keep_content, **kwargs)
    
    def replace_marker_content(
        self,
        content_pool: Union[Pool_type, str],
        marker_name: str,
        **kwargs,
    ) -> Pool_type:
        """Replace a marker region with content from another Pool."""
        from .marker_ops.replace_marker_content import replace_marker_content
        return replace_marker_content(self.pool, content_pool, marker_name, **kwargs)
    
    def clear_markers(self, **kwargs) -> Pool_type:
        """Remove all marker tags from sequences, keeping content."""
        from .fixed_ops.fixed import fixed_operation
        from .marker_ops.parsing import strip_all_markers
        
        result = fixed_operation(
            parent_pools=[self.pool],
            seq_from_seqs_fn=lambda seqs: strip_all_markers(seqs[0]),
            seq_length_from_pool_lengths_fn=lambda lengths: None,
            **kwargs,
        )
        result._markers = set()
        return result

