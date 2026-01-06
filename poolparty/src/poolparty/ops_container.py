"""OpsContainer class housing convenience methods for Pool."""
from .types import Pool_type, Union, Optional, Real, Callable, Integral, Sequence, beartype


@beartype
class OpsContainer:
    """Container for Pool convenience methods that wrap Operation factory functions."""
    
    def __init__(self, pool: Pool_type) -> None:
        """Initialize with a reference to the parent Pool."""
        self.pool = pool
    
    def apply_at_marker(
        self,
        marker_name: str,
        transform_fn: Callable,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply a transformation to the content of a marked region."""
        from .marker_ops.apply_at_marker import apply_at_marker
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        return apply_at_marker(self.pool, marker_name, transform_fn, remove_marker=remove_marker, **kwargs)
    
    def mutagenize(
        self,
        region: str,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply mutagenize() to a marked region."""
        from .base_ops.mutagenize import mutagenize
        return mutagenize(self.pool, region=region, remove_marker=remove_marker, **kwargs)
    
    def deletion_scan(
        self,
        region: str,
        deletion_length: Integral,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply deletion_scan() to a marked region."""
        from .scan_ops.deletion_scan import deletion_scan
        return self.apply_at_marker(
            region,
            lambda p: deletion_scan(p, deletion_length, **kwargs),
            remove_marker=remove_marker,
        )
    
    def insertion_scan(
        self,
        region: str,
        ins_pool: Union[Pool_type, str],
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply insertion_scan() to a marked region."""
        from .scan_ops.insertion_scan import insertion_scan
        return self.apply_at_marker(
            region,
            lambda p: insertion_scan(p, ins_pool, **kwargs),
            remove_marker=remove_marker,
        )
    
    def replacement_scan(
        self,
        region: str,
        ins_pool: Union[Pool_type, str],
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply replacement_scan() to a marked region."""
        from .scan_ops.replacement_scan import replacement_scan
        return self.apply_at_marker(
            region,
            lambda p: replacement_scan(p, ins_pool, **kwargs),
            remove_marker=remove_marker,
        )
    
    def mutagenize_scan(
        self,
        region: str,
        mutagenize_length: Integral,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply mutagenize_scan() to a marked region."""
        from .scan_ops.mutagenize_scan import mutagenize_scan
        return self.apply_at_marker(
            region,
            lambda p: mutagenize_scan(p, mutagenize_length, **kwargs),
            remove_marker=remove_marker,
        )
    
    def shuffle_scan(
        self,
        region: str,
        shuffle_length: Integral,
        shuffles_per_position: Integral = 1,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply shuffle_scan() to a marked region."""
        from .scan_ops.shuffle_scan import shuffle_scan
        return self.apply_at_marker(
            region,
            lambda p: shuffle_scan(p, shuffle_length=shuffle_length, shuffles_per_position=shuffles_per_position, **kwargs),
            remove_marker=remove_marker,
        )
    
    def shuffle_seq(
        self,
        region: str,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply shuffle_seq() to a marked region."""
        from .base_ops.shuffle_seq import shuffle_seq
        return shuffle_seq(self.pool, region=region, remove_marker=remove_marker, **kwargs)
    
    def insert_from_iupac(
        self,
        region: str,
        iupac_seq: str,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Replace marker content with IUPAC-generated sequences."""
        from .base_ops.from_iupac_motif import from_iupac_motif
        from .marker_ops.replace_marker_content import replace_marker_content
        from .marker_ops.apply_at_marker import _replace_keeping_marker
        content = from_iupac_motif(iupac_seq, **kwargs)
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        if remove_marker:
            return replace_marker_content(self.pool, content, region)
        else:
            return _replace_keeping_marker(self.pool, content, region)
    
    def insert_from_motif(
        self,
        region: str,
        prob_df,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Replace marker content with probability-sampled sequences."""
        from .base_ops.from_prob_motif import from_prob_motif
        from .marker_ops.replace_marker_content import replace_marker_content
        from .marker_ops.apply_at_marker import _replace_keeping_marker
        content = from_prob_motif(prob_df, **kwargs)
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        if remove_marker:
            return replace_marker_content(self.pool, content, region)
        else:
            return _replace_keeping_marker(self.pool, content, region)
    
    def insert_kmers(
        self,
        region: str,
        length: int,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Replace marker content with k-mer sequences."""
        from .base_ops.get_kmers import get_kmers
        from .marker_ops.replace_marker_content import replace_marker_content
        from .marker_ops.apply_at_marker import _replace_keeping_marker
        content = get_kmers(length, **kwargs)
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        if remove_marker:
            return replace_marker_content(self.pool, content, region)
        else:
            return _replace_keeping_marker(self.pool, content, region)
    
    #########################################################################
    # Fixed operation convenience methods
    #########################################################################
    
    def reverse_complement(
        self,
        region: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply reverse_complement() to the pool or a marked region."""
        from .fixed_ops.reverse_complement import reverse_complement
        return reverse_complement(self.pool, region=region, remove_marker=remove_marker, **kwargs)
    
    def swapcase(
        self,
        region: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply swapcase() to the pool or a marked region."""
        from .fixed_ops.swapcase import swapcase
        return swapcase(self.pool, region=region, remove_marker=remove_marker, **kwargs)
    
    def upper(
        self,
        region: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply upper() to the pool or a marked region."""
        from .fixed_ops.upper import upper
        return upper(self.pool, region=region, remove_marker=remove_marker, **kwargs)
    
    def lower(
        self,
        region: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply lower() to the pool or a marked region."""
        from .fixed_ops.lower import lower
        return lower(self.pool, region=region, remove_marker=remove_marker, **kwargs)
    
    def clear_gap_chars(
        self,
        region: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply clear_gap_chars() to the pool or a marked region."""
        from .fixed_ops.clear_gap_chars import clear_gap_chars
        return clear_gap_chars(self.pool, region=region, remove_marker=remove_marker, **kwargs)
    
    #########################################################################
    # State operation convenience methods
    #########################################################################
    
    def repeat_states(
        self,
        times: int,
        seq_name_prefix: Optional[str] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Repeat the Pool's states a specified number of times."""
        from .state_ops.repeat import repeat
        return repeat(
            self.pool,
            times,
            seq_name_prefix=seq_name_prefix,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
    def sample_states(
        self,
        num_states: Optional[Integral] = None,
        sampled_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
        seq_name_prefix: Optional[str] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Sample states from the Pool. Wrapper for state_sample()."""
        from .state_ops.state_sample import state_sample
        return state_sample(
            self.pool,
            num_states=num_states,
            sampled_states=sampled_states,
            seed=seed,
            with_replacement=with_replacement,
            seq_name_prefix=seq_name_prefix,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
    def shuffle_states(
        self,
        seed: Optional[Integral] = None,
        permutation: Optional[Sequence[Integral]] = None,
        seq_name_prefix: Optional[str] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Shuffle (permute) the Pool's states. Wrapper for state_shuffle()."""
        from .state_ops.state_shuffle import state_shuffle
        return state_shuffle(
            self.pool,
            seed=seed,
            permutation=permutation,
            seq_name_prefix=seq_name_prefix,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
    def slice_states(
        self,
        key: Union[Integral, slice],
        seq_name_prefix: Optional[str] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Slice the Pool's states. Wrapper for state_slice()."""
        from .state_ops.state_slice import state_slice
        return state_slice(
            self.pool,
            key,
            seq_name_prefix=seq_name_prefix,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
    #########################################################################
    # Marker management methods
    #########################################################################
    
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
