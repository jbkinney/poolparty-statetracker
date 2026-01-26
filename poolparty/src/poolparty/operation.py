"""Operation base class for poolparty."""
from numbers import Real
import statetracker as st
from .types import Pool_type, Sequence, ModeType, Optional, RegionType, beartype, StyleList
from .utils import dna_utils
import numpy as np


@beartype
class Operation:
    """Base class for all operations."""
    design_card_keys: Sequence[str] = []
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
        prefix: Optional[str] = None,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
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
        self.name_prefix: Optional[str] = prefix
        self.clear_parent_names: bool = False
        self._block_seq_names: bool = False
        
        # Region handling
        self._region = region
        self._validate_region(region)
        if region is not None and len(self.parent_pools) == 0:
            raise ValueError("region requires at least one parent pool")
        # Resolve remove_tags from party default if None
        if remove_tags is None:
            self._remove_tags = party.get_default('remove_tags', True)
        else:
            self._remove_tags = remove_tags
        
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
        return dna_utils.get_seq_length(seq)
    
    def _get_length_without_tags(self, seq: str) -> int:
        """Get sequence length excluding only region tags (includes all other chars)."""
        return dna_utils.get_length_without_tags(seq)
    
    def _get_nontag_positions(self, seq: str) -> list[int]:
        """Get raw string positions of all chars excluding tag interiors."""
        return dna_utils.get_nontag_positions(seq)
    
    def _get_molecular_positions(self, seq: str) -> list[int]:
        """Get raw string positions of valid DNA characters, excluding marker interiors."""
        return dna_utils.get_molecular_positions(seq)
    
    def _resolve_region(self, seq: str, region: RegionType) -> tuple[int, int] | None:
        """Resolve region to (start, stop) interval, or None if no region specified.
        
        Parameters
        ----------
        seq : str
            The sequence containing potential regions.
        region : RegionType
            Region specification: region name (str), [start, stop] interval, or None.
        
        Returns
        -------
        tuple[int, int] | None
            (start, stop) interval in raw string positions, or None if region is None.
        """
        if region is None:
            return None
        
        if isinstance(region, str):
            # Region name - look up in sequence
            from .region_ops.parsing import validate_single_region
            region_obj = validate_single_region(seq, region)
            return (region_obj.content_start, region_obj.content_end)
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
            Region specification: region name (str), [start, stop] interval, or None.
        
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
        - Opening tag length (when region is a region name and remove_tags=False)
        
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
            # Region is a region name
            from .region_ops.parsing import parse_region, build_region_tags
            clean_prefix, _, _, strand = parse_region(parent_seq, self._region)
            prefix_len = len(clean_prefix)
            
            if not self._remove_tags:
                # Account for opening tag if we're keeping the tags
                # Build a region tag with dummy content to get the opening tag format
                # (empty content creates self-closing tags which have different format)
                test_tag = build_region_tags(self._region, 'X', strand=strand)
                # test_tag is like '<name>X</name>' - opening tag ends at first '>'
                opening_tag_len = test_tag.index('>') + 1
                prefix_len += opening_tag_len
        else:
            # Region is [start, stop] interval
            prefix_len = len(prefix)
        
        return prefix_len
    
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
        - 'seq': output sequence string
        - 'style': output style list - list of (spec, positions) tuples
        """
        raise NotImplementedError("Subclasses must implement compute()")
    
    def wrapped_compute(
        self,
        parent_seqs: list[str],
        rng: np.random.Generator | None = None,
        parent_styles: list[StyleList] | None = None,
    ) -> dict:
        """Compute with automatic region handling and tag removal.
        
        If region is specified:
        1. Extracts region content from parent_seqs[0]
        2. Adjusts input style positions to be region-relative
        3. Calls compute with modified sequences and styles
        4. Reassembles prefix + result + suffix
        5. Adjusts output style positions to account for prefix
        6. Removes region tags if remove_tags=True and region is a region name
        """
        from .utils.style_utils import split_styles_by_region, reassemble_styles
        
        if self._region is None:
            return self.compute(parent_seqs, rng, parent_styles)
        
        # Extract region parts from parent_seqs[0]
        prefix, region_content, suffix = self._extract_region_parts(
            parent_seqs[0], self._region
        )
        
        # Get region bounds for position adjustment
        bounds = self._resolve_region(parent_seqs[0], self._region)
        region_start = bounds[0] if bounds else 0
        region_end = bounds[1] if bounds else len(parent_seqs[0])
        
        # Split styles for first parent using utility function
        modified_styles = None
        prefix_styles: StyleList = []
        suffix_styles: StyleList = []
        if parent_styles:
            modified_styles = list(parent_styles)  # Copy the list
            if modified_styles and len(modified_styles) > 0:
                prefix_styles, region_styles, suffix_styles = split_styles_by_region(
                    modified_styles[0], region_start, region_end
                )
                modified_styles[0] = region_styles
        
        # Call subclass with region content as first sequence
        modified_seqs = [region_content] + parent_seqs[1:]
        result = self.compute(modified_seqs, rng, modified_styles)
        
        # Helper to identify sequence output key
        def is_seq_output(key: str) -> bool:
            return key == 'seq'
        
        # Helper to identify style output key
        def is_style_output(key: str) -> bool:
            return key == 'style'
        
        # Parse region info for sequence reassembly (if region is a region name)
        if isinstance(self._region, str):
            from .region_ops.parsing import parse_region, build_region_tags
            clean_prefix, _, clean_suffix, strand = parse_region(
                parent_seqs[0], self._region
            )
        
        # Reassemble each output sequence and style
        reassembled = {}
        for key, value in result.items():
            if is_seq_output(key):
                seq = value
                
                if isinstance(self._region, str):
                    if self._remove_tags:
                        # Remove tags
                        reassembled[key] = clean_prefix + seq + clean_suffix
                    else:
                        # Keep tags - rebuild with new content
                        wrapped = build_region_tags(self._region, seq, strand=strand)
                        reassembled[key] = clean_prefix + wrapped + clean_suffix
                else:
                    # Region is [start, stop] interval - just reassemble
                    reassembled[key] = prefix + seq + suffix
            elif is_style_output(key):
                # Get the output sequence to determine new region length
                output_seq = result.get('seq', '')
                
                # Calculate prefix length for style reassembly
                # For regions with remove_tags=False, region styles need prefix + opening tag
                # For regions with remove_tags=True, region styles just need prefix
                # For intervals, region styles just need prefix
                # Suffix styles: for regions, use region.end (where suffix starts in original)
                #                 for intervals, use region_end (where suffix starts)
                if isinstance(self._region, str):
                    from .region_ops.parsing import validate_single_region, build_region_tags
                    region_info = validate_single_region(parent_seqs[0], self._region)
                    clean_prefix_len = len(clean_prefix)
                    
                    if self._remove_tags:
                        # Region styles shifted by clean_prefix only
                        region_prefix_len = clean_prefix_len
                        # Suffix styles: suffix starts at region.content_end in original
                        suffix_start_pos = region_info.content_end
                    else:
                        # Region styles shifted by clean_prefix + opening tag
                        test_tag = build_region_tags(self._region, 'X', strand=strand)
                        opening_tag_len = test_tag.index('>') + 1
                        region_prefix_len = clean_prefix_len + opening_tag_len
                        # Suffix styles: suffix starts at region.end in original
                        suffix_start_pos = region_info.end
                else:
                    # Region is [start, stop] interval
                    region_prefix_len = len(prefix)
                    suffix_start_pos = region_end
                
                # Calculate old and new region lengths accounting for tag changes
                if isinstance(self._region, str):
                    if self._remove_tags:
                        # When removing tags: compare new content length vs old tag span
                        # (suffix positions need to shift by the removed tag length)
                        old_region_len = region_info.end - region_info.start
                        new_region_len = len(output_seq)
                    else:
                        # When keeping tags: compare new tag span vs old tag span
                        old_region_len = region_info.end - region_info.start
                        new_wrapped = build_region_tags(self._region, output_seq, strand=strand)
                        new_region_len = len(new_wrapped)
                else:
                    # Region is [start, stop] interval
                    old_region_len = len(region_content)
                    new_region_len = len(output_seq)
                
                # Reassemble styles manually to use different prefix lengths for region vs suffix
                region_styles = value
                result_styles: StyleList = []
                
                # Add prefix styles unchanged
                result_styles.extend(prefix_styles)
                
                # Shift region styles by region_prefix_len
                from .utils.style_utils import shift_style_positions
                shifted_region = shift_style_positions(region_styles, region_prefix_len)
                result_styles.extend(shifted_region)
                
                # Shift suffix styles: they start at suffix_start_pos in original,
                # and need to move to suffix_start_pos + length_delta in new sequence
                length_delta = new_region_len - old_region_len
                suffix_offset = length_delta
                shifted_suffix = shift_style_positions(suffix_styles, suffix_offset)
                result_styles.extend(shifted_suffix)
                
                reassembled[key] = result_styles
            else:
                # Keep design card keys as-is
                reassembled[key] = value
        
        return reassembled
    
    def compute_seq_names(
        self,
        parent_names: list[Optional[str]],
        card: dict,
    ) -> Optional[str]:
        """Compute output sequence name from parent names and design card.
        
        Returns a string name or None.
        """
        # Block name if _block_seq_names is set
        if self._block_seq_names:
            return None
        
        # Apply clear_parent_names if set
        if self.clear_parent_names:
            parent_names = [None] * len(parent_names)
        
        # Combine all non-None parent names
        non_none_names = [n for n in parent_names if n is not None]
        parent_name = '.'.join(non_none_names) if non_none_names else None
        
        # If no name_prefix, pass through parent name
        if self.name_prefix is None:
            return parent_name
        
        # Build name with prefix
        if self.state is None:
            # Stateless operation - return None
            return None
        value = self.state.value
        if value is None:
            # Inactive state - return None
            return None
        
        op_name = f'{self.name_prefix}_{value}'
        full_name = f'{parent_name}.{op_name}' if parent_name else op_name
        return full_name
    
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
        
        # Handle 'content_pool' parameter (used by ReplaceRegionOp)
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