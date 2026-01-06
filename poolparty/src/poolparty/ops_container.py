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
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Apply a transformation to the content of a marked region.
        
        This is a thin wrapper around poolparty.apply_at_marker().
        See that function for full documentation of parameters.
        """
        from .marker_ops.apply_at_marker import apply_at_marker
        # Resolve None to party default
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        return apply_at_marker(
            self.pool, marker_name, transform_fn,
            remove_marker=remove_marker, name=name, iter_order=iter_order,
        )
    
    def mutagenize(
        self,
        marker_name: str,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply mutagenize() to a marked region.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker whose content to mutagenize.
        remove_marker : bool, default=True
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the mutagenized content.
        **kwargs
            Arguments passed to mutagenize() (e.g., num_mutations,
            mutation_rate, mark_changes, mode, num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with the marker region mutagenized.
        """
        from .base_ops.mutagenize import mutagenize
        from .marker_ops.remove_marker import remove_marker as remove_marker_op
        
        # Resolve None to party default
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        
        # Use region parameter to mutagenize only the marker content
        result = mutagenize(self.pool, region=marker_name, **kwargs)
        
        # Remove marker tags if requested
        if remove_marker:
            result = remove_marker_op(result, marker_name, keep_content=True)
        
        return result
    
    def deletion_scan(
        self,
        marker_name: str,
        deletion_length: Integral,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply deletion_scan() to a marked region.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker whose content to scan.
        deletion_length : Integral
            Number of characters to delete at each position.
        remove_marker : Optional[bool], default=None
            If None, uses party default.
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the scanned content.
        **kwargs
            Arguments passed to deletion_scan() (e.g., deletion_marker,
            spacer_str, positions, mode, num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with deletion scan applied to the marker region.
        """
        from .scan_ops.deletion_scan import deletion_scan
        return self.apply_at_marker(
            marker_name,
            lambda p: deletion_scan(p, deletion_length, **kwargs),
            remove_marker=remove_marker,
        )
    
    def insertion_scan(
        self,
        marker_name: str,
        ins_pool: Union[Pool_type, str],
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply insertion_scan() to a marked region.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker whose content to scan.
        ins_pool : Pool or str
            The insert Pool or sequence string to be inserted.
        remove_marker : bool, default=True
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the scanned content.
        **kwargs
            Arguments passed to insertion_scan() (e.g., positions,
            spacer_str, mode, num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with insertion scan applied to the marker region.
        """
        from .scan_ops.insertion_scan import insertion_scan
        return self.apply_at_marker(
            marker_name,
            lambda p: insertion_scan(p, ins_pool, **kwargs),
            remove_marker=remove_marker,
        )
    
    def replacement_scan(
        self,
        marker_name: str,
        ins_pool: Union[Pool_type, str],
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply replacement_scan() to a marked region.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker whose content to scan.
        ins_pool : Pool or str
            The insert Pool or sequence string to replace segments.
        remove_marker : bool, default=True
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the scanned content.
        **kwargs
            Arguments passed to replacement_scan() (e.g., positions,
            spacer_str, mark_changes, mode, num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with replacement scan applied to the marker region.
        """
        from .scan_ops.replacement_scan import replacement_scan
        return self.apply_at_marker(
            marker_name,
            lambda p: replacement_scan(p, ins_pool, **kwargs),
            remove_marker=remove_marker,
        )
    
    def mutagenize_scan(
        self,
        marker_name: str,
        mutagenize_length: Integral,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply mutagenize_scan() to a marked region.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker whose content to scan.
        mutagenize_length : Integral
            Length of the region to mutagenize at each position.
        remove_marker : bool, default=True
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the scanned content.
        **kwargs
            Arguments passed to mutagenize_scan() (e.g., num_mutations,
            mutation_rate, positions, spacer_str, mark_changes, mode).
        
        Returns
        -------
        Pool
            A Pool with mutagenize scan applied to the marker region.
        """
        from .scan_ops.mutagenize_scan import mutagenize_scan
        return self.apply_at_marker(
            marker_name,
            lambda p: mutagenize_scan(p, mutagenize_length, **kwargs),
            remove_marker=remove_marker,
        )
    
    def shuffle_scan(
        self,
        marker_name: str,
        shuffle_length: Integral,
        shuffles_per_position: Integral = 1,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply shuffle_scan() to a marked region.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker whose content to scan.
        shuffle_length : Integral
            Length of the region to shuffle at each position.
        shuffles_per_position : Integral, default=1
            Number of shuffles to perform at each position.
        remove_marker : Optional[bool], default=None
            If None, uses party default.
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the scanned content.
        **kwargs
            Arguments passed to shuffle_scan() (e.g., positions,
            spacer_str, mark_changes, mode, num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with shuffle scan applied to the marker region.
        """
        from .scan_ops.shuffle_scan import shuffle_scan
        return self.apply_at_marker(
            marker_name,
            lambda p: shuffle_scan(p, shuffle_length=shuffle_length, shuffles_per_position=shuffles_per_position, **kwargs),
            remove_marker=remove_marker,
        )
    
    def shuffle_seq(
        self,
        marker_name: str,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Apply shuffle_seq() to a marked region.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker whose content to shuffle.
        remove_marker : Optional[bool], default=None
            If None, uses party default.
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the shuffled content.
        **kwargs
            Arguments passed to shuffle_seq() (e.g., mode, num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with the marker content shuffled.
        """
        from .base_ops.shuffle_seq import shuffle_seq
        from .marker_ops.remove_marker import remove_marker as remove_marker_op
        
        # Resolve None to party default
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        
        # Use region parameter to shuffle only the marker content
        result = shuffle_seq(self.pool, region=marker_name, **kwargs)
        
        # Remove marker tags if requested
        if remove_marker:
            result = remove_marker_op(result, marker_name, keep_content=True)
        
        return result
    
    def insert_from_iupac(
        self,
        marker_name: str,
        iupac_seq: str,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Replace marker content with IUPAC-generated sequences.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker to replace.
        iupac_seq : str
            IUPAC sequence string (e.g., 'RN' for purine + any base).
        remove_marker : Optional[bool], default=None
            If None, uses party default.
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the inserted content.
        **kwargs
            Arguments passed to from_iupac_motif() (e.g., mark_changes,
            mode, num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with marker replaced by IUPAC-generated sequences.
        """
        from .base_ops.from_iupac_motif import from_iupac_motif
        from .marker_ops.replace_marker_content import replace_marker_content
        from .marker_ops.apply_at_marker import _replace_keeping_marker
        content = from_iupac_motif(iupac_seq, **kwargs)
        # Resolve None to party default
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        if remove_marker:
            return replace_marker_content(self.pool, content, marker_name)
        else:
            return _replace_keeping_marker(self.pool, content, marker_name)
    
    def insert_from_motif(
        self,
        marker_name: str,
        prob_df,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Replace marker content with probability-sampled sequences.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker to replace.
        prob_df : pd.DataFrame
            DataFrame with probability values for each position.
        remove_marker : Optional[bool], default=None
            If None, uses party default.
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the inserted content.
        **kwargs
            Arguments passed to from_prob_motif() (e.g., mode,
            num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with marker replaced by probability-sampled sequences.
        """
        from .base_ops.from_prob_motif import from_prob_motif
        from .marker_ops.replace_marker_content import replace_marker_content
        from .marker_ops.apply_at_marker import _replace_keeping_marker
        content = from_prob_motif(prob_df, **kwargs)
        # Resolve None to party default
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        if remove_marker:
            return replace_marker_content(self.pool, content, marker_name)
        else:
            return _replace_keeping_marker(self.pool, content, marker_name)
    
    def insert_kmers(
        self,
        marker_name: str,
        length: int,
        remove_marker: Optional[bool] = None,
        **kwargs,
    ) -> Pool_type:
        """Replace marker content with k-mer sequences.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker to replace.
        length : int
            Length of k-mers to generate.
        remove_marker : Optional[bool], default=None
            If None, uses party default.
            If True, marker tags are removed from the result.
            If False, marker tags are preserved around the inserted content.
        **kwargs
            Arguments passed to get_kmers() (e.g., mode,
            num_hybrid_states).
        
        Returns
        -------
        Pool
            A Pool with marker replaced by k-mer sequences.
        """
        from .base_ops.get_kmers import get_kmers
        from .marker_ops.replace_marker_content import replace_marker_content
        from .marker_ops.apply_at_marker import _replace_keeping_marker
        content = get_kmers(length, **kwargs)
        # Resolve None to party default
        if remove_marker is None:
            remove_marker = self.pool._party.get_default('remove_marker', True)
        if remove_marker:
            return replace_marker_content(self.pool, content, marker_name)
        else:
            return _replace_keeping_marker(self.pool, content, marker_name)
    
    #########################################################################
    # Fixed operation convenience methods
    #########################################################################
    
    def reverse_complement(
        self,
        marker_name: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Create a Pool containing the reverse complement of sequences.
        
        Parameters
        ----------
        marker_name : Optional[str], default=None
            If provided, apply only to the marker region.
        remove_marker : Optional[bool], default=None
            If True and marker_name is provided, remove marker tags from output.
        
        Returns
        -------
        Pool
            A Pool containing reverse-complemented sequences.
        """
        from .fixed_ops.reverse_complement import reverse_complement
        return reverse_complement(
            self.pool,
            region=marker_name,
            remove_marker=remove_marker,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
    def swapcase(
        self,
        marker_name: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Create a Pool containing case-swapped sequences.
        
        Parameters
        ----------
        marker_name : Optional[str], default=None
            If provided, apply only to the marker region.
        remove_marker : Optional[bool], default=None
            If True and marker_name is provided, remove marker tags from output.
        
        Returns
        -------
        Pool
            A Pool containing case-swapped sequences.
        """
        from .fixed_ops.swapcase import swapcase
        return swapcase(
            self.pool,
            region=marker_name,
            remove_marker=remove_marker,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
    def upper(
        self,
        marker_name: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Create a Pool containing uppercase sequences.
        
        Parameters
        ----------
        marker_name : Optional[str], default=None
            If provided, apply only to the marker region.
        remove_marker : Optional[bool], default=None
            If True and marker_name is provided, remove marker tags from output.
        
        Returns
        -------
        Pool
            A Pool containing uppercase sequences.
        """
        from .fixed_ops.upper import upper
        return upper(
            self.pool,
            region=marker_name,
            remove_marker=remove_marker,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
    def lower(
        self,
        marker_name: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Create a Pool containing lowercase sequences.
        
        Parameters
        ----------
        marker_name : Optional[str], default=None
            If provided, apply only to the marker region.
        remove_marker : Optional[bool], default=None
            If True and marker_name is provided, remove marker tags from output.
        
        Returns
        -------
        Pool
            A Pool containing lowercase sequences.
        """
        from .fixed_ops.lower import lower
        return lower(
            self.pool,
            region=marker_name,
            remove_marker=remove_marker,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
    def clear_gap_chars(
        self,
        marker_name: Optional[str] = None,
        remove_marker: Optional[bool] = None,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Create a Pool with all gap/non-molecular characters removed.
        
        Removes everything not in the alphabet's all_chars (gaps, spaces, etc.)
        while preserving marker tags.
        
        Parameters
        ----------
        marker_name : Optional[str], default=None
            If provided, apply only to the marker region.
        remove_marker : Optional[bool], default=None
            If True and marker_name is provided, remove marker tags from output.
        
        Returns
        -------
        Pool
            A Pool containing only molecular alphabet characters (markers preserved).
        """
        from .fixed_ops.clear_gap_chars import clear_gap_chars
        return clear_gap_chars(
            self.pool,
            region=marker_name,
            remove_marker=remove_marker,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
    
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
        """Repeat the Pool's states a specified number of times.
        
        This is a thin wrapper around poolparty.repeat().
        See that function for full documentation of parameters.
        
        Parameters
        ----------
        times : int
            The number of times to repeat the pool's states.
        seq_name_prefix : Optional[str], default=None
            Prefix for auto-generated sequence names (e.g., 'v' produces 'v0', 'v1', ...).
        name : Optional[str], default=None
            Name to assign to the resulting Pool.
        op_name : Optional[str], default=None
            Name to assign to the internal operation.
        iter_order : Optional[Real], default=None
            Iteration order priority for the resulting Pool.
        op_iter_order : Optional[Real], default=None
            Iteration order priority for the internal operation.
        
        Returns
        -------
        Pool
            A new Pool with `times` as many states as the input pool.
        """
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
        """Insert an XML-style marker at a fixed position in sequences.
        
        Parameters
        ----------
        marker_name : str
            Name for the marker (e.g., 'region', 'orf', 'insert').
        start : int
            Start position (0-based) for the marker.
        stop : Optional[int], default=None
            End position (exclusive). If None, creates a zero-length marker.
        strand : str, default='+'
            Strand annotation ('+' or '-').
        **kwargs
            Arguments passed to insert_marker() (e.g., name, op_name,
            iter_order, op_iter_order).
        
        Returns
        -------
        Pool
            A Pool with the marker inserted.
        """
        from .marker_ops.insert_marker import insert_marker
        return insert_marker(self.pool, marker_name, start, stop, strand, **kwargs)
    
    def remove_marker(
        self,
        marker_name: str,
        keep_content: bool = True,
        **kwargs,
    ) -> Pool_type:
        """Remove a marker from sequences.
        
        Parameters
        ----------
        marker_name : str
            Name of the marker to remove.
        keep_content : bool, default=True
            If True, keep the content inside the marker (just remove tags).
            If False, remove both the marker tags and their content.
        **kwargs
            Arguments passed to remove_marker() (e.g., name, op_name,
            iter_order, op_iter_order).
        
        Returns
        -------
        Pool
            A Pool with the marker removed.
        """
        from .marker_ops.remove_marker import remove_marker
        return remove_marker(self.pool, marker_name, keep_content, **kwargs)
    
    def replace_marker_content(
        self,
        content_pool: Union[Pool_type, str],
        marker_name: str,
        **kwargs,
    ) -> Pool_type:
        """Replace a marker region with content from another Pool.
        
        Parameters
        ----------
        content_pool : Pool or str
            Pool or sequence string to insert at the marker position.
        marker_name : str
            Name of the marker to replace.
        **kwargs
            Arguments passed to replace_marker_content() (e.g., name,
            op_name, iter_order, op_iter_order).
        
        Returns
        -------
        Pool
            A Pool with the marker replaced by content_pool sequences.
        """
        from .marker_ops.replace_marker_content import replace_marker_content
        return replace_marker_content(self.pool, content_pool, marker_name, **kwargs)
    
    def clear_markers(
        self,
        name: Optional[str] = None,
        op_name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        op_iter_order: Optional[Real] = None,
    ) -> Pool_type:
        """Remove all marker tags from sequences, keeping content.
        
        Parameters
        ----------
        name : Optional[str], default=None
            Name for the resulting Pool.
        op_name : Optional[str], default=None
            Name for the underlying Operation.
        iter_order : Optional[Real], default=None
            Iteration order priority for the resulting Pool.
        op_iter_order : Optional[Real], default=None
            Iteration order priority for the underlying Operation.
        
        Returns
        -------
        Pool
            A Pool with all marker tags removed (content preserved).
        """
        from .fixed_ops.fixed import fixed_operation
        from .marker_ops.parsing import strip_all_markers
        
        def seq_from_seqs_fn(seqs: list[str]) -> str:
            return strip_all_markers(seqs[0])
        
        result = fixed_operation(
            parent_pools=[self.pool],
            seq_from_seqs_fn=seq_from_seqs_fn,
            seq_length_from_pool_lengths_fn=lambda lengths: None,
            name=name,
            op_name=op_name,
            iter_order=iter_order,
            op_iter_order=op_iter_order,
        )
        # Clear all markers from the pool's marker set
        result._markers = set()
        return result
