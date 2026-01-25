"""Pool class for poolparty."""
import statetracker as st
from .types import Pool_type, Operation_type, Union, Optional, Real, Callable, Integral, Sequence, beartype, ModeType, RegionType, PositionsType
from typing import Literal
from .marker import Marker
import pandas as pd


@beartype
class Pool:
    """A node in the computation DAG."""
    
    def __init__(
        self,
        operation: Operation_type,
        name: Optional[str] = None,
        state: Optional[st.State] = None,
        iter_order: Optional[Real] = None,
        markers: Optional[set[Marker]] = None,
    ) -> None:
        """Initialize Pool and build its state."""
        from .party import get_active_party
        from .marker import Marker
        
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "Pools must be created inside a Party context. "
                "Use: with pp.Party() as party: ..."
            )
        self._party = party
        self._id = party._get_next_pool_id()
        self.operation = operation
        if state is not None:
            self.state = state
        else:
            self.state: st.State | None = operation.build_pool_counter(
                operation.parent_pools
            )
        if iter_order is not None and self.state is not None:
            self.state.iter_order = iter_order
        self._name: str = ""
        self.name = name if name is not None else f'pool[{self._id}]'
        
        # Track markers: inherit from parents if not explicitly provided
        if markers is not None:
            self._markers: set[Marker] = set(markers)
        else:
            # Inherit markers from all parent pools
            self._markers = set()
            for parent in operation.parent_pools:
                if hasattr(parent, '_markers'):
                    self._markers.update(parent._markers)
        
        # Register pool with party after name is set
        party._register_pool(self)
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this pool."""
        if self.state is None:
            return 0
        return self.state.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order for this pool."""
        if self.state is not None:
            self.state.iter_order = value
    
    @property
    def name(self) -> str:
        """Name of this pool."""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set pool name and update state name.
        
        Validates name uniqueness with the Party before accepting.
        
        Raises:
            ValueError: If the name is already used by another pool.
        """
        # Validate name with party (excludes self for renaming case)
        self._party._validate_pool_name(value, self)
        old_name = self._name
        self._name = value
        # When pool.state is the same as operation.state (source operations),
        # preserve operation state name if operation has explicit name (not default)
        # Otherwise, use pool state name
        if self.state is not None:
            if self.state is self.operation.state:
                # Check if operation has explicit name (not default like "op[0]:from_seqs")
                op_name = self.operation.name
                is_default_op_name = op_name.startswith('op[') and ']:' in op_name
                if not is_default_op_name:
                    # Operation has explicit name, preserve it
                    # State name should already be set to operation name
                    pass
                else:
                    # Operation has default name, use pool name
                    self.state.name = f"{value}.state"
            else:
                # Different states, set pool state name normally
                self.state.name = f"{value}.state"
        # Update party's name tracking if this is a rename (not initial set)
        if old_name:
            self._party._update_pool_name(self, old_name, value)
    
    @property
    def num_states(self) -> int | None:
        """Number of states for this pool."""
        if self.state is None:
            return None
        return self.state.num_values
    
    @property
    def parents(self) -> list:
        """Get parent pools from the operation."""
        return self.operation.parent_pools
    
    @property
    def seq_length(self) -> Optional[int]:
        """Sequence length (None if variable)."""
        return self.operation.seq_length
    
    @property
    def markers(self) -> set[Marker]:
        """Set of Marker objects present in this pool's sequences."""
        return self._markers
    
    def has_marker(self, name: str) -> bool:
        """Check if a marker with the given name is present in this pool."""
        return any(m.name == name for m in self._markers)
    
    def add_marker(self, marker: Marker) -> None:
        """Add a marker to this pool's marker set."""
        self._markers.add(marker)
    
    def _untrack_marker(self, name: str) -> None:
        """Remove a marker from this pool's marker set by name."""
        self._markers = {m for m in self._markers if m.name != name}
    
    #########################################################################
    # Counter-based operators
    #########################################################################
    
    def __add__(self, other: Pool_type) -> Pool_type:
        """Stack two pools (union of states via sum_counters)."""
        from .state_ops.stack import stack
        return stack([self, other])
    
    def __mul__(self, n: int) -> Pool_type:
        """Repeat this pool n times (repeat states)."""
        from .state_ops.repeat import repeat
        return repeat(self, n)
    
    def __rmul__(self, n: int) -> Pool_type:
        """Repeat this pool n times (repeat states)."""
        return self.__mul__(n)
    
    def __getitem__(self, key: Union[int, slice]) -> Pool_type:
        """Slice this pool's states (not sequences)."""
        from .state_ops.state_slice import state_slice
        return state_slice(self, key)
    
    def __repr__(self) -> str:
        num_states_str = "None" if self.num_states is None else str(self.num_states)
        return f"Pool(id={self._id}, name={self.name!r}, op={self.operation.name!r}, num_states={num_states_str})"
    
    def named(self, name: str, op_name: Optional[str] = None) -> Pool_type:
        """Set the name of this pool and its operation, return self for chaining."""
        self.name = name
        #self.operation.name = op_name if op_name is not None else name + '.op'
        return self
    
    def clear_seq_names(self) -> Pool_type:
        """Clear sequence names for this pool, returning None for all names."""
        self.operation._block_seq_names = True
        return self
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this pool's state.
        
        Lower values iterate faster (come first in product states).
        """
        if self.state is None:
            return 0
        return self.state.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order on this pool's state."""
        if self.state is not None:
            self.state.iter_order = value
    
    def copy(self, name: Optional[str] = None) -> Pool_type:
        """Create a copy of this pool with a copied operation.
        
        The copied operation references the same parent_pools, so the copy
        represents a parallel branch in the computation graph that shares
        the same upstream DAG.
        
        Must be called within an active Party context.
        
        Args:
            name: Optional name for the copied pool. If None, uses
                self.name + '.copy' as the default.
        
        Returns:
            A new Pool backed by a copied Operation.
        """
        new_op = self.operation.copy()
        new_pool = Pool(operation=new_op)
        if name is not None:
            new_pool.name = name
        else:
            new_pool.name = self.name + '.copy'
        return new_pool
    
    def deepcopy(self, name: Optional[str] = None) -> Pool_type:
        """Create a deep copy of this pool, recursively copying the entire upstream DAG.
        
        Unlike copy(), this creates independent copies of all upstream pools
        and operations, resulting in a fully independent computation DAG.
        
        Must be called within an active Party context.
        
        Args:
            name: Optional name for the copied pool. If None, uses
                self.name + '.copy' as the default.
        
        Returns:
            A new Pool backed by a recursively copied Operation.
        """
        new_op = self.operation.deepcopy()
        new_pool = Pool(operation=new_op)
        if name is not None:
            new_pool.name = name
        else:
            new_pool.name = self.name + '.copy'
        return new_pool
    
    #########################################################################
    # Generation
    #########################################################################
    
    def generate_library(
        self,
        num_cycles: int = 1,
        num_seqs: Optional[int] = None,
        seed: Optional[int] = None,
        init_state: Optional[int] = None,
        seqs_only: bool = False,
        report_design_cards: bool = False,
        aux_pools: Sequence[Pool_type] = (),
        report_seq: bool = True,
        report_pool_seqs: bool = True,
        report_pool_states: bool = True,
        report_op_states: bool = True,
        report_op_keys: bool = True,
        pools_to_report: Union[str, Sequence[Pool_type]] = 'all',
        organize_columns_by: Literal['pool', 'type'] = 'type',
    ) -> Union[pd.DataFrame, list[str]]:
        from .generate_library import generate_library
        return generate_library(
            pool=self,
            num_cycles=num_cycles,
            num_seqs=num_seqs,
            seed=seed,
            init_state=init_state,
            seqs_only=seqs_only,
            report_design_cards=report_design_cards,
            aux_pools=aux_pools,
            report_seq=report_seq,
            report_pool_seqs=report_pool_seqs,
            report_pool_states=report_pool_states,
            report_op_states=report_op_states,
            report_op_keys=report_op_keys,
            pools_to_report=pools_to_report,
            organize_columns_by=organize_columns_by,
        )
    
    def print_library(
        self,
        num_seqs: Optional[Integral] = None,
        num_cycles: Optional[Integral] = None,
        show_header: bool = True,
        show_state: bool = True,
        show_name: bool = True,
        show_seq: bool = True,
        pad_names: bool = True,
        seed: Optional[Integral] = None,
    ) -> Pool_type:
        """Print preview sequences from this pool; returns self for chaining.
        
        Args:
            num_seqs: Number of sequences to generate.
            num_cycles: Number of complete iterations through all states.
            show_header: Whether to show the pool header line.
            show_state: Whether to show the state column.
            show_name: Whether to show the name column.
            show_seq: Whether to show the seq column.
            pad_names: Whether to pad names to align sequences.
            seed: Random seed for reproducibility.
        """
        # Build kwargs for generate_library, only including num_cycles when needed
        gen_kwargs = {
            'seqs_only': False,
            'report_design_cards': True,
            'init_state': 0,
            'seed': seed,
        }
        if num_seqs is not None:
            gen_kwargs['num_seqs'] = num_seqs
        else:
            gen_kwargs['num_cycles'] = num_cycles if num_cycles is not None else 1
        df = self.generate_library(**gen_kwargs)
        has_name = show_name and 'name' in df.columns and df['name'].notna().any()
        max_name_len = df['name'].str.len().max() if has_name and pad_names else 0
        
        if show_header:
            num_states_str = "None" if self.num_states is None else str(self.num_states)
            print(f"{self.name}: seq_length={self.seq_length}, num_states={num_states_str}")
            # Build header columns
            header_parts = []
            if show_state:
                header_parts.append("state")
            if has_name:
                header_parts.append(f"{'name':<{max_name_len}}" if pad_names else "name")
            if show_seq:
                header_parts.append("seq")
            if header_parts:
                print("  ".join(header_parts))
        
        state_col = f"{self.name}.state"
        for _, row in df.iterrows():
            # Build row columns
            row_parts = []
            if show_state:
                row_parts.append(f"{row[state_col]:5d}")
            if has_name:
                if pad_names:
                    row_parts.append(f"{row['name']:<{max_name_len}}")
                else:
                    row_parts.append(f"{row['name']}")
            if show_seq:
                seq = row['seq']
                from .utils.style_utils import apply_inline_styles
                # Get per-sequence inline styles (from operation style parameters)
                inline_styles = row.get('_inline_styles', [])
                # Apply inline styles
                seq = apply_inline_styles(seq, inline_styles)
                row_parts.append(seq)
            print("  ".join(row_parts))
        print('')
        return self # For chaining
    
    #########################################################################
    # Tree visualization
    #########################################################################
    
    def print_dag(self, style: str = 'clean', show_pools: bool = True) -> Pool_type:
        """Print the ASCII tree visualization rooted at this pool."""
        from .text_viz import print_pool_tree
        print_pool_tree(self, style=style, show_pools=show_pools)
        return self # For chaining
    
    #########################################################################
    # Base operations
    #########################################################################
    
    def mutagenize(
        self,
        region: RegionType = None,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        allowed_chars: Optional[str] = None,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .base_ops.mutagenize import mutagenize
        return mutagenize(
            pool=self,
            region=region,
            num_mutations=num_mutations,
            mutation_rate=mutation_rate,
            allowed_chars=allowed_chars,
            style=style,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )
    
    def shuffle_seq(
        self,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from .base_ops.shuffle_seq import shuffle_seq
        return shuffle_seq(
            pool=self,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
        )
    
    def insert_from_iupac(
        self,
        iupac_seq: str,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from .base_ops.from_iupac import from_iupac
        return from_iupac(
            iupac_seq=iupac_seq,
            bg_pool=self,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
        )
    
    def insert_from_motif(
        self,
        prob_df: pd.DataFrame,
        region: RegionType = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[int] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from .base_ops.from_motif import from_motif
        return from_motif(
            prob_df=prob_df,
            bg_pool=self,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
            style=style,
        )
    
    def insert_kmers(
        self,
        length: Integral,
        region: RegionType = None,
        style: Optional[str] = None,
        case: Literal['lower', 'upper'] = 'upper',
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .base_ops.get_kmers import get_kmers
        return get_kmers(
            length=length,
            pool=self,
            region=region,
            style=style,
            case=case,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )
    
    #########################################################################
    # Scan operations
    #########################################################################
    
    def mutagenize_scan(
        self,
        mutagenize_length: Integral,
        num_mutations: Optional[Integral] = None,
        mutation_rate: Optional[Real] = None,
        positions: PositionsType = None,
        region: RegionType = None,
        prefix: Optional[Union[str, Sequence[str]]] = None,
        mode: Union[ModeType, tuple[ModeType, ModeType]] = 'random',
        num_states: Optional[Union[Integral, Sequence[Integral]]] = None,
        iter_order: Optional[Union[Real, Sequence[Real]]] = None,
    ) -> Pool_type:
        from .scan_ops.mutagenize_scan import mutagenize_scan
        return mutagenize_scan(
            pool=self,
            mutagenize_length=mutagenize_length,
            num_mutations=num_mutations,
            mutation_rate=mutation_rate,
            positions=positions,
            region=region,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )
    
    def deletion_scan(
        self,
        deletion_length: Integral,
        deletion_marker: Optional[str] = '-',
        region: RegionType = None,
        positions: PositionsType = None,
        prefix: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[Integral] = None,
        style: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .scan_ops.deletion_scan import deletion_scan
        return deletion_scan(
            pool=self,
            deletion_length=deletion_length,
            deletion_marker=deletion_marker,
            region=region,
            positions=positions,
            prefix=prefix,
            mode=mode,
            num_states=num_states,
            style=style,
            iter_order=iter_order,
        )
    
    def insertion_scan(
        self,
        ins_pool: Union[Pool_type, str],
        positions: PositionsType = None,
        region: RegionType = None,
        replace: bool = False,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        prefix_position: Optional[str] = None,
        prefix_insert: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .scan_ops.insertion_scan import insertion_scan
        return insertion_scan(
            pool=self,
            ins_pool=ins_pool,
            positions=positions,
            region=region,
            replace=replace,
            style=style,
            prefix=prefix,
            prefix_position=prefix_position,
            prefix_insert=prefix_insert,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )
    
    def replacement_scan(
        self,
        ins_pool: Union[Pool_type, str],
        positions: PositionsType = None,
        region: RegionType = None,
        style: Optional[str] = None,
        prefix: Optional[str] = None,
        prefix_position: Optional[str] = None,
        prefix_insert: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[Integral] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .scan_ops.insertion_scan import replacement_scan
        return replacement_scan(
            pool=self,
            ins_pool=ins_pool,
            positions=positions,
            region=region,
            style=style,
            prefix=prefix,
            prefix_position=prefix_position,
            prefix_insert=prefix_insert,
            mode=mode,
            num_states=num_states,
            iter_order=iter_order,
        )
    
    def shuffle_scan(
        self,
        shuffle_length: Integral,
        positions: PositionsType = None,
        region: RegionType = None,
        shuffles_per_position: Integral = 1,
        prefix: Optional[str] = None,
        prefix_position: Optional[str] = None,
        prefix_shuffle: Optional[str] = None,
        mode: ModeType = 'random',
        num_states: Optional[Integral] = None,
        style_shuffle: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .scan_ops.shuffle_scan import shuffle_scan
        return shuffle_scan(
            pool=self,
            shuffle_length=shuffle_length,
            positions=positions,
            region=region,
            shuffles_per_position=shuffles_per_position,
            prefix=prefix,
            prefix_position=prefix_position,
            prefix_shuffle=prefix_shuffle,
            mode=mode,
            num_states=num_states,
            style_shuffle=style_shuffle,
            iter_order=iter_order,
        )
    
    #########################################################################
    # Fixed operations
    #########################################################################
    
    def rc(
        self,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from .fixed_ops.rc import rc
        return rc(
            pool=self,
            region=region,
            remove_marker=remove_marker,
            iter_order=iter_order,
            style=style,
        )
    
    def swapcase(
        self,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from .fixed_ops.swapcase import swapcase
        return swapcase(
            pool=self,
            region=region,
            remove_marker=remove_marker,
            iter_order=iter_order,
            style=style,
        )
    
    def upper(
        self,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from .fixed_ops.upper import upper
        return upper(
            pool=self,
            region=region,
            remove_marker=remove_marker,
            iter_order=iter_order,
            style=style,
        )
    
    def lower(
        self,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        iter_order: Optional[Real] = None,
        style: Optional[str] = None,
    ) -> Pool_type:
        from .fixed_ops.lower import lower
        return lower(
            pool=self,
            region=region,
            remove_marker=remove_marker,
            iter_order=iter_order,
            style=style,
        )
    
    def clear_gaps(
        self,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .fixed_ops.clear_gaps import clear_gaps
        return clear_gaps(
            pool=self,
            region=region,
            remove_marker=remove_marker,
            iter_order=iter_order,
        )
    
    def clear_annotation(
        self,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .fixed_ops.clear_annotation import clear_annotation
        return clear_annotation(
            pool=self,
            region=region,
            remove_marker=remove_marker,
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
        from .fixed_ops.style import stylize
        return stylize(
            pool=self,
            region=region,
            style=style,
            which=which,
            regex=regex,
            iter_order=iter_order,
        )
    
    #########################################################################
    # State operations
    #########################################################################
    
    def repeat_states(
        self,
        times: Integral,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .state_ops.repeat import repeat
        return repeat(
            pool=self,
            times=times,
            prefix=prefix,
            iter_order=iter_order,
        )
    
    def sample_states(
        self,
        num_values: Optional[Integral] = None,
        sampled_states: Optional[Sequence[Integral]] = None,
        seed: Optional[Integral] = None,
        with_replacement: bool = True,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .state_ops.state_sample import state_sample
        return state_sample(
            pool=self,
            num_values=num_values,
            sampled_states=sampled_states,
            seed=seed,
            with_replacement=with_replacement,
            prefix=prefix,
            iter_order=iter_order,
        )
    
    def shuffle_states(
        self,
        seed: Optional[Integral] = None,
        permutation: Optional[Sequence[Integral]] = None,
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .state_ops.state_shuffle import state_shuffle
        return state_shuffle(
            pool=self,
            seed=seed,
            permutation=permutation,
            prefix=prefix,
            iter_order=iter_order,
        )
    
    def slice_states(
        self,
        key: Union[Integral, slice],
        prefix: Optional[str] = None,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .state_ops.state_slice import state_slice
        return state_slice(
            pool=self,
            key=key,
            prefix=prefix,
            iter_order=iter_order,
        )
    
    #########################################################################
    # Marker operations
    #########################################################################
    
    def apply_at_marker(
        self,
        marker_name: str,
        transform_fn: Callable,
        remove_marker: bool = True,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .marker_ops.apply_at_marker import apply_at_marker
        return apply_at_marker(
            self,
            marker_name,
            transform_fn,
            remove_marker=remove_marker,
            iter_order=iter_order,
        )
    
    def insert_marker(
        self,
        marker_name: str,
        start: int,
        stop: Optional[int] = None,
        strand: str = '+',
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .marker_ops.insert_marker import insert_marker
        return insert_marker(
            self,
            marker_name,
            start,
            stop,
            strand,
            iter_order=iter_order,
        )
    
    def remove_marker(
        self,
        marker_name: str,
        keep_content: bool = True,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .marker_ops.remove_marker import remove_marker
        return remove_marker(
            self,
            marker_name,
            keep_content,
            iter_order=iter_order,
        )
    
    def replace_marker_content(
        self,
        content_pool: Union[Pool_type, str],
        marker_name: str,
        iter_order: Optional[Real] = None,
    ) -> Pool_type:
        from .marker_ops.replace_marker_content import replace_marker_content
        return replace_marker_content(
            self,
            content_pool,
            marker_name,
            iter_order=iter_order,
        )
    
    def clear_markers(self, **kwargs) -> Pool_type:
        """Remove all marker tags from sequences, keeping content."""
        from .fixed_ops.fixed import fixed_operation
        from .marker_ops.parsing import strip_all_markers
        
        result = fixed_operation(
            parent_pools=[self],
            seq_from_seqs_fn=lambda seqs: strip_all_markers(seqs[0]),
            seq_length_from_pool_lengths_fn=lambda lengths: None,
            **kwargs,
        )
        result._markers = set()
        return result