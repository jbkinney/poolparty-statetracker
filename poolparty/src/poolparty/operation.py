"""Operation base class for poolparty."""
from numbers import Real
import statecounter as sc
from .types import Pool_type, Sequence, ModeType, Optional, beartype
import numpy as np


class Operation:
    """Base class for all operations."""
    design_card_keys: Sequence[str] = []
    num_outputs: int = 1
    max_num_sequential_states: int = 100_000
    factory_name: str = "op"
    
    @classmethod
    def validate_num_states(cls, num_states: int | float, mode: ModeType) -> int | float:
        """Validate num_states against max_num_sequential_states."""
        if num_states != np.inf and num_states < 1:
            raise ValueError(f"num_states must be >= 1 or np.inf, got {num_states}")
        if num_states > cls.max_num_sequential_states:
            if mode == 'sequential':
                raise ValueError(
                    f"Number of states ({num_states}) exceeds "
                    f"max_num_sequential_states ({cls.max_num_sequential_states}). "
                    f"Use mode='random' or mode='hybrid' instead."
                )
            return np.inf
        return num_states
    
    @beartype
    def __init__(
        self,
        parent_pools: Sequence[Pool_type],
        num_states: int = 1,
        mode: ModeType = 'fixed',
        seq_length: Optional[int] = None,
        name: Optional[str] = None,
        iter_order: Real = 0,
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
        # Set _name directly during init (counter doesn't exist yet)
        self._name = name if name is not None else f'op[{self._id}]:{self.factory_name}'
        self._seq_length = seq_length
        if mode == 'random':
            num_states = 1
        self.counter = sc.Counter(num_states=num_states, name=f"{self._name}.state", iter_order=iter_order)
        self.rng: np.random.Generator | None = None
        self.num_states = num_states
        # Register operation with party after name is set
        party._register_operation(self)
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this operation."""
        return self.counter.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order for this operation."""
        self.counter.iter_order = value
    
    @property
    def seq_length(self) -> Optional[int]:
        """Sequence length produced by this operation (None if variable)."""
        return self._seq_length
    
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
        # Update counter name if counter exists
        if hasattr(self, 'counter'):
            self.counter.name = f"{value}.state"
        # Update party's name tracking if this is a rename (not initial set)
        if old_name:
            self._party._update_op_name(self, old_name, value)
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this operation's counter."""
        return self.counter.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order on this operation's counter."""
        self.counter.iter_order = value
    
    @beartype
    def build_pool_counter(
        self,
        parent_pools: Sequence[Pool_type],
    ) -> sc.Counter:
        """Build the output Pool's counter from parent pool counters, sorted by iteration_order."""
        parent_counters = [p.counter for p in parent_pools]
        product_counter = sc.ordered_product(parent_counters + [self.counter])
        return product_counter
    
    @beartype
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
    
    @beartype
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Compute output sequences from design card.
        
        Returns a dictionary with seq_0, seq_1, ... keys.
        """
        raise NotImplementedError("Subclasses must implement compute_seq_from_card()")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self._id}, name={self.name!r}, mode={self.mode!r}, num_states={self.num_states})"
    
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
        
        if name is not None:
            init_params['name'] = name
        else:
            init_params['name'] = self.name + '.copy'
        
        return self.__class__(**init_params)
    
    def print_tree(self, style: str = 'clean') -> None:
        """Print the ASCII tree visualization rooted at this operation.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from .text_viz import print_operation_tree
        print_operation_tree(self, style=style)