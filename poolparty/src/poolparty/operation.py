"""Operation base class for poolparty."""
from numbers import Real
import statetracker as st
from .types import Pool_type, Sequence, ModeType, Optional, RegionType, beartype, StyleList
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
    def validate_num_values(cls, num_values: int | float | None, mode: ModeType) -> int | float | None:
        """Validate num_values against max_num_sequential_states."""
        if num_values is None:
            return None
        if num_values != np.inf and num_values < 1:
            raise ValueError(f"num_values must be >= 1, np.inf, or None, got {num_values}")
        if num_values > cls.max_num_sequential_states:
            if mode == 'sequential':
                raise ValueError(
                    f"Number of values ({num_values}) exceeds "
                    f"max_num_sequential_states ({cls.max_num_sequential_states}). "
                    f"Use mode='random' instead."
                )
            return np.inf
        return num_values
    
    def __init__(
        self,
        parent_pools: Sequence[Pool_type],
        num_values: int | None = 1,
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
        validated_num_values = self.validate_num_values(num_values, mode)
        
        # Track whether this operation's state is synced to parent states
        self._random_synced_to_parents = False
        
        if validated_num_values is not None and mode != 'fixed':
            # Non-fixed ops with explicit num_values - create state
            self.state = st.State(num_values=validated_num_values, name=f"{self._name}.state", iter_order=iter_order)
        elif mode == 'random' and parent_pools:
            # Random mode with no explicit num_states - check for parent states
            parent_states = [p.state for p in parent_pools if p.state is not None]
            if parent_states:
                # Create state synced to parent states
                if len(parent_states) == 1:
                    # Single parent - create synced state
                    self.state = st.synced_to(parent_states[0], name=f"{self._name}.state")
                else:
                    # Multiple parents - create product state
                    self.state = st.ordered_product(states=parent_states)
                    self.state.name = f"{self._name}.state"
                if iter_order is not None:
                    self.state.iter_order = iter_order
                self._random_synced_to_parents = True
                validated_num_values = self.state.num_values
            else:
                # All parents are stateless - remain stateless
                self.state = None
        else:
            # No parents or not random mode - state is None
            self.state = None
        
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
        if self.state is None:
            return 0
        return self.state.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order for this operation."""
        if self.state is not None:
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
    
    def _compute_style_position_offset(
        self,
        parent_seq: str,
        prefix: str,
    ) -> int:
        """Compute the offset to add to style positions for reassembly.
        
        This method calculates how many characters appear before the region
        content in the final output sequence. It handles:
        - Prefix length (characters before the region)
        - Opening marker tag length (when region is a marker and remove_marker=False)
        - Spacer string offset
        
        Parameters
        ----------
        parent_seq : str
            The original parent sequence (before region extraction).
        prefix : str
            The prefix extracted from the parent sequence (seq[:region_start]).
        
        Returns
        -------
        int
            The total offset to add to style positions.
        """
        if self._region is None:
            return 0
        
        if isinstance(self._region, str):
            # Region is a marker name
            from .marker_ops.parsing import parse_marker, build_marker_tag
            clean_prefix, _, _, strand = parse_marker(parent_seq, self._region)
            prefix_len = len(clean_prefix)
            
            if not self._remove_marker:
                # Account for opening tag if we're keeping the marker
                # Build a marker tag with dummy content to get the opening tag format
                # (empty content creates self-closing tags which have different format)
                test_tag = build_marker_tag(self._region, 'X', strand=strand)
                # test_tag is like '<name>X</name>' - opening tag ends at first '>'
                opening_tag_len = test_tag.index('>') + 1
                prefix_len += opening_tag_len
        else:
            # Region is [start, stop] interval
            prefix_len = len(prefix)
        
        # Account for spacer_str
        spacer_offset = len(self._spacer_str) if self._spacer_str else 0
        
        return prefix_len + spacer_offset
    
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
        # Update state name if state exists and is not None
        if hasattr(self, 'state') and self.state is not None:
            self.state.name = f"{value}.state"
        # Update party's name tracking if this is a rename (not initial set)
        if old_name:
            self._party._update_op_name(self, old_name, value)
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this operation's state."""
        if self.state is None:
            return 0
        return self.state.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order on this operation's state."""
        if self.state is not None:
            self.state.iter_order = value
    
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool_type],
    ) -> st.State | None:
        """Build the output Pool's state from parent pool states."""
        # Collect parent states
        parent_states = [p.state for p in parent_pools if p.state is not None]
        
        if self.mode == 'fixed':
            # Fixed operations: derive pool state from parents only (op.state is None)
            if not parent_states:
                # No parents with state - create new state with this op's num_values
                return st.State(num_values=self.num_values or 1)
            elif len(parent_states) == 1:
                # Single parent - create synced state (don't share object to avoid name conflicts)
                return st.synced_to(parent_states[0])
            else:
                # Multiple parents - product
                return st.ordered_product(states=parent_states)
        else:
            # Non-fixed: include op.state in the product
            if self._random_synced_to_parents and self.state is not None:
                return st.synced_to(self.state)
            
            op_states = [self.state] if self.state is not None else []
            all_states = parent_states + op_states
            
            if not all_states:
                # Fully random DAG - no states
                return None
            elif len(all_states) == 1:
                # Single state - synced
                return st.synced_to(all_states[0])
            else:
                # Multiple states - product
                return st.ordered_product(states=all_states)

    
    def compute(
        self,
        parent_seqs: list[str],
        rng: np.random.Generator | None = None,
        parent_styles: list[StyleList] | None = None,
    ) -> dict:
        """Compute design card, output sequences, and output styles together.
        
        Parameters
        ----------
        parent_seqs : list[str]
            Input sequences from parent pools.
        rng : np.random.Generator | None
            Random number generator (for random mode operations).
        parent_styles : list[StyleList] | None
            Input styles from parent pools. Each element is a list of
            (style_spec, positions) tuples for the corresponding parent sequence.
        
        Returns a dictionary containing:
        - Design card keys (matching design_card_keys)
        - Output sequence keys (seq_0, seq_1, ...)
        - Output style keys (style_0, style_1, ...) - list of (spec, positions) tuples
        """
        raise NotImplementedError("Subclasses must implement compute()")
    
    def wrapped_compute(
        self,
        parent_seqs: list[str],
        rng: np.random.Generator | None = None,
        parent_styles: list[StyleList] | None = None,
    ) -> dict:
        """Compute with automatic region handling, spacer insertion, and marker removal.
        
        If region is specified:
        1. Extracts region content from parent_seqs[0]
        2. Adjusts input style positions to be region-relative
        3. Calls compute with modified sequences and styles
        4. Wraps result sequences with spacer_str if specified
        5. Reassembles prefix + result + suffix
        6. Adjusts output style positions to account for prefix
        7. Removes marker tags if remove_marker=True and region is a marker name
        """
        if self._region is None:
            return self.compute(parent_seqs, rng, parent_styles)
        
        # Extract region parts from parent_seqs[0]
        prefix, region_content, suffix = self._extract_region_parts(
            parent_seqs[0], self._region
        )
        
        # Get region bounds for position adjustment
        bounds = self._resolve_region(parent_seqs[0], self._region)
        region_start = bounds[0] if bounds else 0
        region_end = region_start + len(region_content)
        
        # Adjust parent styles for first parent (shift positions by -region_start)
        # Also preserve styles for positions outside the region
        modified_styles = None
        preserved_prefix_styles: StyleList = []  # Styles for positions before region
        preserved_suffix_styles: StyleList = []  # Styles for positions after region
        if parent_styles:
            modified_styles = list(parent_styles)  # Copy the list
            if modified_styles and len(modified_styles) > 0:
                # Adjust first parent's styles to be region-relative
                first_parent_styles = modified_styles[0]
                adjusted_first_styles: StyleList = []
                for spec, positions in first_parent_styles:
                    # Styles before region - preserve unchanged
                    prefix_mask = positions < region_start
                    if np.any(prefix_mask):
                        preserved_prefix_styles.append((spec, positions[prefix_mask]))
                    
                    # Styles within region - shift to region-relative coords
                    region_mask = (positions >= region_start) & (positions < region_end)
                    if np.any(region_mask):
                        adjusted_positions = positions[region_mask] - region_start
                        adjusted_first_styles.append((spec, adjusted_positions))
                    
                    # Styles after region - preserve (will adjust later if length changes)
                    suffix_mask = positions >= region_end
                    if np.any(suffix_mask):
                        preserved_suffix_styles.append((spec, positions[suffix_mask]))
                
                modified_styles[0] = adjusted_first_styles
        
        # Call subclass with region content as first sequence
        modified_seqs = [region_content] + parent_seqs[1:]
        result = self.compute(modified_seqs, rng, modified_styles)
        
        # Helper to identify sequence output keys (seq_0, seq_1, etc.)
        def is_seq_output(key: str) -> bool:
            return key.startswith('seq_') and len(key) > 4 and key[4:].isdigit()
        
        # Helper to identify style output keys (style_0, style_1, etc.)
        def is_style_output(key: str) -> bool:
            return key.startswith('style_') and len(key) > 6 and key[6:].isdigit()
        
        # Compute total offset for style positions using centralized helper
        total_offset = self._compute_style_position_offset(parent_seqs[0], prefix)
        
        # Parse marker info for sequence reassembly (if region is a marker)
        if isinstance(self._region, str):
            from .marker_ops.parsing import parse_marker, build_marker_tag
            clean_prefix, _, clean_suffix, strand = parse_marker(
                parent_seqs[0], self._region
            )
        
        # Reassemble each output sequence and style
        reassembled = {}
        for key, value in result.items():
            if is_seq_output(key):
                seq = value
                # Apply spacer_str if specified
                if self._spacer_str:
                    seq = self._spacer_str + seq + self._spacer_str
                
                if isinstance(self._region, str):
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
            elif is_style_output(key):
                # Adjust style positions to account for prefix and spacer
                styles = value
                adjusted_styles: StyleList = []
                
                # First add preserved prefix styles (positions unchanged)
                adjusted_styles.extend(preserved_prefix_styles)
                
                # Add styles from compute result (shifted by total_offset)
                for spec, positions in styles:
                    adjusted_positions = positions + total_offset
                    adjusted_styles.append((spec, adjusted_positions))
                
                # Add preserved suffix styles, adjusting for any length change
                # in the region content (relevant for ops that change length)
                if preserved_suffix_styles:
                    # Get the output sequence to determine new region length
                    seq_key = 'seq_' + key.split('_')[1]
                    output_seq = result.get(seq_key, '')
                    new_region_len = len(output_seq)
                    if self._spacer_str:
                        new_region_len += 2 * len(self._spacer_str)
                    
                    # Calculate length delta accounting for marker removal
                    if isinstance(self._region, str) and self._remove_marker:
                        # When removing marker, the old region included marker tags
                        # Find the full marker span in the original sequence
                        from .marker_ops.parsing import validate_single_marker
                        marker_info = validate_single_marker(parent_seqs[0], self._region)
                        old_region_len_with_tags = marker_info.end - marker_info.start
                        length_delta = new_region_len - old_region_len_with_tags
                    else:
                        old_region_len = len(region_content)
                        length_delta = new_region_len - old_region_len
                    
                    for spec, positions in preserved_suffix_styles:
                        adjusted_positions = positions + length_delta
                        adjusted_styles.append((spec, adjusted_positions))
                
                reassembled[key] = adjusted_styles
            else:
                # Keep design card keys as-is
                reassembled[key] = value
        
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
        if self.state is None:
            # Stateless operation - return None for all outputs
            return {f'name_{i}': None for i in range(self.num_outputs)}
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
        num_values_str = "None" if self.num_values is None else str(self.num_values)
        return f"{self.__class__.__name__}(id={self._id}, name={self.name!r}, mode={self.mode!r}, num_values={num_values_str})"
    
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