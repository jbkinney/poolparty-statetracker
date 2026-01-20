"""Operation base class for poolparty."""
from numbers import Real
import statetracker as st
from .types import Pool_type, Sequence, ModeType, Optional, RegionType, beartype
from . import dna
import numpy as np


@beartype
class Operation:
    """Base class for all operations."""
    design_card_keys: Sequence[str] = []
    num_outputs: int = 1
    max_num_sequential_states: int = 1_000_000
    factory_name: str = "op"
    
    @classmethod
    def validate_num_values(cls, num_values: int | float, mode: ModeType) -> int | float:
        """Validate num_values against max_num_sequential_states."""
        if num_values != np.inf and num_values < 1:
            raise ValueError(f"num_values must be >= 1 or np.inf, got {num_values}")
        if num_values > cls.max_num_sequential_states:
            if mode == 'sequential':
                raise ValueError(
                    f"Number of values ({num_values}) exceeds "
                    f"max_num_sequential_states ({cls.max_num_sequential_states}). "
                    f"Use mode='random' or mode='hybrid' instead."
                )
            return np.inf
        return num_values
    
    def __init__(
        self,
        parent_pools: Sequence[Pool_type],
        num_values: int = 1,
        mode: ModeType = 'fixed',
        seq_length: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        seq_name_prefix: Optional[str] = None,
        region: RegionType = None,
        remove_marker: Optional[bool] = None,
        spacer_str: str = '',
    ) -> None:
        """Initialize Operation."""
        from .party import get_active_party
        party = get_active_party()
        if party is None:
            raise RuntimeError(
                "Operations must be created inside a Party context. "
                "Use: with pp.Party() as party: ..."
            )
        self._party = party
        self.parent_pools = list(parent_pools)
        self.mode = mode
        self._id = party._get_next_op_id()
        # Set _name directly during init (state doesn't exist yet)
        self._name = name if name is not None else f'op[{self._id}]:{self.factory_name}'
        self._seq_length = seq_length
        if mode == 'random':
            num_values = 1
        validated_num_values = self.validate_num_values(num_values, mode)
        self.state = st.State(num_values=validated_num_values, name=f"{self._name}.state", iter_order=iter_order)
        self.rng: np.random.Generator | None = None
        self.num_values = validated_num_values
        # Sequence naming attributes
        self.name_prefix: Optional[str] = seq_name_prefix
        self.clear_parent_names: bool = False
        self._block_seq_names: bool = False
        
        # Region handling
        self._region = region
        self._validate_region(region)
        if region is not None and len(self.parent_pools) == 0:
            raise ValueError("region requires at least one parent pool")
        # Resolve remove_marker from party default if None
        if remove_marker is None:
            self._remove_marker = party.get_default('remove_marker', True)
        else:
            self._remove_marker = remove_marker
        
        # Spacer string for region operations
        self._spacer_str = spacer_str
        
        # Register operation with party after name is set
        party._register_operation(self)
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this operation."""
        return self.state.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order for this operation."""
        self.state.iter_order = value
    
    @property
    def seq_length(self) -> Optional[int]:
        """Sequence length produced by this operation (None if variable)."""
        return self._seq_length
    
    def _get_effective_seq_length(self, seq: str) -> int:
        """Get effective sequence length (DNA characters only, excluding markers)."""
        return dna.get_seq_length(seq)
    
    def _get_length_without_markers(self, seq: str) -> int:
        """Get sequence length excluding only marker tags (includes all other chars)."""
        return dna.get_length_without_markers(seq)
    
    def _get_nonmarker_positions(self, seq: str) -> list[int]:
        """Get raw string positions of all chars excluding marker interiors."""
        return dna.get_nonmarker_positions(seq)
    
    def _get_molecular_positions(self, seq: str) -> list[int]:
        """Get raw string positions of valid DNA characters, excluding marker interiors."""
        return dna.get_molecular_positions(seq)
    
    def _resolve_region(self, seq: str, region: RegionType) -> tuple[int, int] | None:
        """Resolve region to (start, stop) interval, or None if no region specified.
        
        Parameters
        ----------
        seq : str
            The sequence containing potential markers.
        region : RegionType
            Region specification: marker name (str), [start, stop] interval, or None.
        
        Returns
        -------
        tuple[int, int] | None
            (start, stop) interval in raw string positions, or None if region is None.
        """
        if region is None:
            return None
        
        if isinstance(region, str):
            # Marker name - look up in sequence
            from .marker_ops.parsing import validate_single_marker
            marker = validate_single_marker(seq, region)
            return (marker.content_start, marker.content_end)
        else:
            # Explicit [start, stop] interval
            return (int(region[0]), int(region[1]))
    
    def _extract_region_parts(self, seq: str, region: RegionType) -> tuple[str, str, str]:
        """Extract (prefix, region_content, suffix) from sequence based on region.
        
        Parameters
        ----------
        seq : str
            The sequence to split.
        region : RegionType
            Region specification: marker name (str), [start, stop] interval, or None.
        
        Returns
        -------
        tuple[str, str, str]
            (prefix, region_content, suffix) where prefix + region_content + suffix == seq.
            If region is None, returns ('', seq, '').
        """
        bounds = self._resolve_region(seq, region)
        if bounds is None:
            return ('', seq, '')
        start, stop = bounds
        return (seq[:start], seq[start:stop], seq[stop:])
    
    @staticmethod
    def _validate_region(region: RegionType) -> None:
        """Validate region parameter format.
        
        Raises ValueError if region is invalid.
        """
        if region is not None and not isinstance(region, str):
            if len(region) != 2:
                raise ValueError(f"region interval must be [start, stop], got {region}")
            if region[0] < 0:
                raise ValueError(f"region start must be >= 0, got {region[0]}")
            if region[1] < region[0]:
                raise ValueError(f"region stop must be >= start, got [{region[0]}, {region[1]}]")
    
    @property
    def id(self) -> int:
        """Unique ID for this operation."""
        return self._id
    
    @property
    def name(self) -> str:
        """Name of this operation."""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set operation name and update counter name.
        
        Validates name uniqueness with the Party before accepting.
        
        Raises:
            ValueError: If the name is already used by another operation.
        """
        # Validate name with party (excludes self for renaming case)
        self._party._validate_op_name(value, self)
        old_name = self._name
        self._name = value
        # Update state name if state exists
        if hasattr(self, 'state'):
            self.state.name = f"{value}.state"
        # Update party's name tracking if this is a rename (not initial set)
        if old_name:
            self._party._update_op_name(self, old_name, value)
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this operation's state."""
        return self.state.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order on this operation's state."""
        self.state.iter_order = value
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool_type],
    ) -> st.State:
        """Build the output Pool's state from parent pool states, sorted by iteration_order."""
        parent_states = [p.state for p in parent_pools]
        if not parent_states:
            # Source operation: pool state IS the operation state
            return st.passthrough(self.state)
        else: 
            return st.ordered_product(states=parent_states + [self.state])

    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Compute design card containing all design decisions.
        
        Returns a dictionary with the design decisions (matching design_card_keys).
        Does NOT include output sequences.
        """
        raise NotImplementedError("Subclasses must implement compute_design_card()")
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Compute output sequences from design card.
        
        Returns a dictionary with seq_0, seq_1, ... keys.
        """
        raise NotImplementedError("Subclasses must implement compute_seq_from_card()")
    
    def wrapped_compute_design_card(
        self,
        parent_seqs: list[str],
        rng: np.random.Generator | None = None,
    ) -> dict:
        """Compute design card with automatic region handling.
        
        If region is specified, extracts region content from parent_seqs[0]
        before calling compute_design_card.
        """
        if self._region is not None:
            _, region_content, _ = self._extract_region_parts(parent_seqs[0], self._region)
            modified_seqs = [region_content] + parent_seqs[1:]
            return self.compute_design_card(modified_seqs, rng)
        return self.compute_design_card(parent_seqs, rng)
    
    def wrapped_compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Compute sequences with automatic region handling, spacer insertion, and marker removal.
        
        If region is specified:
        1. Extracts region content from parent_seqs[0]
        2. Calls compute_seq_from_card with modified sequences
        3. Wraps result with spacer_str if specified
        4. Reassembles prefix + result + suffix
        5. Removes marker tags if remove_marker=True and region is a marker name
        """
        if self._region is None:
            return self.compute_seq_from_card(parent_seqs, card)
        
        # Extract region parts from parent_seqs[0]
        prefix, region_content, suffix = self._extract_region_parts(
            parent_seqs[0], self._region
        )
        
        # Call subclass with region content as first sequence
        modified_seqs = [region_content] + parent_seqs[1:]
        result = self.compute_seq_from_card(modified_seqs, card)
        
        # Reassemble each output sequence
        reassembled = {}
        for key, seq in result.items():
            if key.startswith('seq_'):
                # Apply spacer_str if specified
                if self._spacer_str:
                    seq = self._spacer_str + seq + self._spacer_str
                
                if isinstance(self._region, str):
                    # Region is a marker name - get clean prefix/suffix
                    from .marker_ops.parsing import parse_marker, build_marker_tag
                    clean_prefix, _, clean_suffix, strand = parse_marker(
                        parent_seqs[0], self._region
                    )
                    if self._remove_marker:
                        # Remove marker tags
                        reassembled[key] = clean_prefix + seq + clean_suffix
                    else:
                        # Keep marker - rebuild with new content
                        wrapped = build_marker_tag(self._region, seq, strand=strand)
                        reassembled[key] = clean_prefix + wrapped + clean_suffix
                else:
                    # Region is [start, stop] interval - just reassemble
                    reassembled[key] = prefix + seq + suffix
            else:
                reassembled[key] = seq
        
        return reassembled
    
    def compute_seq_names(
        self,
        parent_names: list[Optional[str]],
        card: dict,
    ) -> dict:
        """Compute output sequence names from parent names and design card.
        
        Returns a dictionary with name_0, name_1, ... keys matching num_outputs.
        """
        # Block all names if _block_seq_names is set
        if self._block_seq_names:
            return {f'name_{i}': None for i in range(self.num_outputs)}
        
        # Apply clear_parent_names if set
        if self.clear_parent_names:
            parent_names = [None] * len(parent_names)
        
        # Combine all non-None parent names
        non_none_names = [n for n in parent_names if n is not None]
        parent_name = '.'.join(non_none_names) if non_none_names else None
        
        # If no name_prefix, pass through parent name
        if self.name_prefix is None:
            if self.num_outputs == 1:
                return {'name_0': parent_name}
            return {f'name_{i}': parent_name for i in range(self.num_outputs)}
        
        # Build name(s) with prefix
        value = self.state.value
        if value is None:
            # Inactive state - return None for all outputs
            return {f'name_{i}': None for i in range(self.num_outputs)}
        
        if self.num_outputs == 1:
            op_name = f'{self.name_prefix}{value}'
            full_name = f'{parent_name}.{op_name}' if parent_name else op_name
            return {'name_0': full_name}
        else:
            # Multi-output: use f'{prefix}{value}({i})' format
            result = {}
            for i in range(self.num_outputs):
                op_name = f'{self.name_prefix}{value}({i})'
                full_name = f'{parent_name}.{op_name}' if parent_name else op_name
                result[f'name_{i}'] = full_name
            return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._id}, name={self.name!r}, mode={self.mode!r}, num_values={self.num_values})"
    
    def _get_copy_params(self) -> dict:
        """Return the parameters needed to create a copy of this operation.
        
        Subclasses must override this to return a dict of kwargs for __init__.
        The 'name' key should be set to None so the copy gets a new auto-name.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _get_copy_params()"
        )
    
    def copy(self, name: Optional[str] = None) -> "Operation":
        """Create a copy of this operation with a new ID.
        
        The copy references the same parent_pools but has its own Counter.
        Must be called within an active Party context.
        
        Args:
            name: Optional name for the copied operation. If None, uses
                self.name + '.copy' as the default.
        
        Returns:
            A new Operation of the same type with the same parameters.
        """
        init_params = self._get_copy_params()
        if name is not None:
            init_params['name'] = name
        else:
            init_params['name'] = self.name + '.copy'
        return self.__class__(**init_params)
    
    def deepcopy(self, name: Optional[str] = None) -> "Operation":
        """Create a deep copy of this operation, recursively copying all parent pools.
        
        Unlike copy(), this creates independent copies of all upstream pools,
        resulting in a fully independent computation DAG.
        
        Must be called within an active Party context.
        
        Args:
            name: Optional name for the copied operation. If None, uses
                self.name + '.copy' as the default.
        
        Returns:
            A new Operation with recursively copied parent pools.
        """
        # Recursively deepcopy all parent pools
        new_parent_pools = [p.deepcopy() for p in self.parent_pools]
        
        # Get copy params and substitute parent pools
        init_params = self._get_copy_params()
        
        if 'parent_pool' in init_params and new_parent_pools:
            init_params['parent_pool'] = new_parent_pools[0]
        elif 'parent_pools' in init_params:
            init_params['parent_pools'] = new_parent_pools
        
        # Handle 'bg_pool' parameter (used by several operations)
        if 'bg_pool' in init_params and new_parent_pools:
            init_params['bg_pool'] = new_parent_pools[0]
        
        # Handle 'pool' parameter (used by mutagenize and other operations)
        if 'pool' in init_params and new_parent_pools:
            init_params['pool'] = new_parent_pools[0]
        
        # Handle 'content_pool' parameter (used by ReplaceMarkerContentOp)
        if 'content_pool' in init_params and len(new_parent_pools) > 1:
            init_params['content_pool'] = new_parent_pools[1]
        
        if name is not None:
            init_params['name'] = name
        else:
            init_params['name'] = self.name + '.copy'
        
        return self.__class__(**init_params)
    
    def print_dag(self, style: str = 'clean') -> None:
        """Print the ASCII tree visualization rooted at this operation.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from .text_viz import print_operation_tree
        print_operation_tree(self, style=style)