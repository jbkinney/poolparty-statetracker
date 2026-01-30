"""Operation base class for poolparty."""
from numbers import Real
import statetracker as st
from .types import Pool_type, Sequence, ModeType, Optional, RegionType, beartype, SeqStyle, Seq
from .utils import dna_utils
import numpy as np


class Operation:
    """Base class for all operations."""
    design_card_keys: Sequence[str] = []
    max_num_sequential_states: int = 1_000_000
    factory_name: str = "op"
    
    @classmethod
    def validate_num_states(cls, num_states: int | float | None, mode: ModeType) -> int | float | None:
        """Validate num_states against max_num_sequential_states."""
        if num_states is None:
            return None
        if num_states != np.inf and num_states < 1:
            raise ValueError(f"num_states must be >= 1, np.inf, or None, got {num_states}")
        if num_states > cls.max_num_sequential_states:
            if mode == 'sequential':
                raise ValueError(
                    f"Number of states ({num_states}) exceeds "
                    f"max_num_sequential_states ({cls.max_num_sequential_states}). "
                    f"Use mode='random' instead."
                )
            return np.inf
        return num_states
    
    def __init__(
        self,
        parent_pools: Sequence[Pool_type],
        num_states: int | None = 1,
        mode: ModeType = 'fixed',
        seq_length: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        _natural_num_states: Optional[int] = None,
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
        validated_num_states = self.validate_num_states(num_states, mode)
        
        # Store natural num_states for cycling in sequential mode
        # If not provided, defaults to the effective num_states
        self._natural_num_states = _natural_num_states
        
        if validated_num_states is not None and mode != 'fixed':
            # Non-fixed ops with explicit num_states - create state
            self.state = st.State(num_values=validated_num_states, name=f"{self._name}.state", iter_order=iter_order)
        else:
            # No explicit num_states or fixed mode - no state
            self.state = None
        
        self.rng: np.random.Generator | None = None
        self.num_states = validated_num_states
        # Set natural_num_states: use provided value, else effective num_states
        if self._natural_num_states is None:
            self._natural_num_states = validated_num_states
        # Sequence naming attributes
        self.prefix: Optional[str] = prefix
        
        # Region handling
        self._region = region
        self._validate_region(region)
        if region is not None and len(self.parent_pools) == 0:
            raise ValueError("region requires at least one parent pool")
        # Resolve remove_tags from party config if None
        if remove_tags is None:
            # Check legacy defaults first for backwards compatibility
            legacy_default = party.get_default('remove_tags', None)
            if legacy_default is not None:
                self._remove_tags = legacy_default
            else:
                self._remove_tags = party._config.remove_tags
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
    
    @property
    def natural_num_states(self) -> Optional[int]:
        """Natural number of states (computed from operation, before user override)."""
        return self._natural_num_states
    
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
                # No parents with state - create new state with this op's num_states
                return st.State(num_values=self.num_states or 1)
            elif len(parent_states) == 1:
                # Single parent - create synced state (don't share object to avoid name conflicts)
                return st.synced_to(parent_states[0])
            else:
                # Multiple parents - product
                return st.ordered_product(states=parent_states)
        else:
            # Non-fixed: include op.state in the product
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
        parents: list[Seq],
        rng: np.random.Generator | None = None,
    ) -> tuple[Seq, dict]:
        """Compute output Seq and design card with automatic region handling.
        
        This is the public entry point for operations. It handles region
        extraction/reassembly automatically, then delegates to _compute_core().
        
        Parameters
        ----------
        parents : list[Seq]
            Input Seq objects from parent pools.
        rng : np.random.Generator | None
            Random number generator (for random mode operations).
        
        Returns
        -------
        tuple[Seq, dict]
            Output Seq (with string and style) and design card dict.
        
        If region is specified:
        1. Extracts region from parents[0] as a Seq
        2. Calls _compute_core with modified parent list
        3. Reassembles prefix + result + suffix using Seq.join
        4. Removes region tags if remove_tags=True and region is a region name
        """
        if self._region is None:
            output_seq, card = self._compute_core(parents, rng)
        else:
            # Create context from first parent sequence
            from .utils.region_context import RegionContext
            ctx = RegionContext.from_sequence(
                parents[0], self._region, self._remove_tags
            )
            
            # Split first parent into prefix, region, suffix
            prefix_seq, region_seq, suffix_seq = ctx.split_parent_seq(parents[0])
            
            # Prepare modified parents list (region as first element)
            modified_parents = [region_seq] + parents[1:]
            
            # Call subclass _compute_core
            output_seq, card = self._compute_core(modified_parents, rng)
            
            # Reassemble with prefix and suffix
            output_seq = ctx.reassemble_seq(prefix_seq, output_seq, suffix_seq)
        
        # Filter design card based on config
        card = self._filter_design_card(card)
        
        return output_seq, card
    
    def _filter_design_card(self, card: dict) -> dict:
        """Filter design card to only include enabled keys based on config."""
        config = self._party._config
        if config is None or config.suppress_cards:
            return {}
        enabled = config.get_enabled_keys(self.factory_name, list(self.design_card_keys))
        return {k: v for k, v in card.items() if k in enabled}
    
    def _compute_core(
        self,
        parents: list[Seq],
        rng: np.random.Generator | None = None,
    ) -> tuple[Seq, dict]:
        """Compute output Seq and design card (core implementation).
        
        Subclasses must implement this method. It receives the actual sequences
        to operate on (which may be region-extracted if region was specified).
        
        Parameters
        ----------
        parents : list[Seq]
            Input Seq objects from parent pools. When region is specified,
            parents[0] contains only the region content.
        rng : np.random.Generator | None
            Random number generator (for random mode operations).
        
        Returns
        -------
        tuple[Seq, dict]
            Output Seq (with string and style) and design card dict.
        """
        raise NotImplementedError("Subclasses must implement _compute_core()")
    
    def compute_name_contributions(self, global_state: Optional[int] = None) -> list[str]:
        """Compute this operation's contributions to the final sequence name.
        
        Returns list of name elements in the order they should appear.
        Default: [prefix_state.value] when active, [] otherwise.
        For stateless random operations, uses global_state if provided.
        
        Parameters
        ----------
        global_state : Optional[int]
            The global row index, used for stateless random operations.
        
        Returns
        -------
        list[str]
            List of name elements, or empty list if no contribution.
        """
        if self.prefix is None:
            return []
        if self.state is not None and self.state.value is not None:
            # Has state: use state value
            return [f'{self.prefix}_{self.state.value}']
        elif self.mode == 'random' and global_state is not None:
            # Stateless random: use global_state
            return [f'{self.prefix}_{global_state}']
        else:
            return []
    
    def __repr__(self) -> str:
        num_states_str = "None" if self.num_states is None else str(self.num_states)
        return f"{self.__class__.__name__}(id={self._id}, name={self.name!r}, mode={self.mode!r}, num_states={num_states_str})"
    
    def _get_copy_params(self) -> dict:
        """Auto-generate copy params from __init__ signature using conventions.
        
        Subclasses can override for custom behavior.
        """
        import inspect
        sig = inspect.signature(self.__class__.__init__)
        params = {}
        
        for param_name, param_spec in sig.parameters.items():
            if param_name == 'self':
                continue
            value = self._resolve_param(param_name, param_spec)
            params[param_name] = value
        
        # Always override name to None for fresh auto-naming
        params['name'] = None
        return params
    
    def _resolve_param(self, param_name: str, param_spec=None):
        """Resolve parameter value using naming conventions."""
        import inspect
        
        # Special cases that don't follow standard patterns
        if param_name == 'name':
            return None
        elif param_name in ('pool', 'parent_pool'):
            return self.parent_pools[0] if self.parent_pools else None
        elif param_name == 'content_pool':
            return self.parent_pools[1] if len(self.parent_pools) > 1 else None
        elif param_name == 'num_states':
            # Only preserve for random mode with explicit values > 1
            if self.mode == 'random' and self.num_states is not None and self.num_states > 1:
                return self.num_states
            return None
        
        # Standard convention: try _param_name, then param_name
        for attr_name in (f'_{param_name}', param_name):
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                # Auto-copy mutable objects with .copy() method
                if hasattr(value, 'copy') and callable(value.copy):
                    return value.copy()
                return value
        
        # Couldn't resolve - use default from signature if available
        if param_spec is not None and param_spec.default is not inspect.Parameter.empty:
            return param_spec.default
        
        # Last resort - return None
        return None
    
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
        
        # Handle 'parent_pool' parameter (used by several operations)
        if 'parent_pool' in init_params and new_parent_pools:
            init_params['parent_pool'] = new_parent_pools[0]
        
        # Handle 'pool' parameter (used by mutagenize and other operations)
        if 'pool' in init_params and new_parent_pools:
            init_params['pool'] = new_parent_pools[0]
        
        # Handle 'content_pool' parameter (used by ReplaceRegionOp)
        if 'content_pool' in init_params and len(new_parent_pools) > 1:
            init_params['content_pool'] = new_parent_pools[1]
        
        init_params['name'] = name
        
        return self.__class__(**init_params)
    
    def print_dag(self, style: str = 'clean') -> None:
        """Print the ASCII tree visualization rooted at this operation.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from .text_viz import print_operation_tree
        print_operation_tree(self, style=style)