"""Pool class for poolparty."""
import statecounter as sc
from .types import Pool_type, Operation_type, Union, Optional, Real, Integral, beartype
from .marker import Marker
from .ops_container import OpsContainer
import pandas as pd


@beartype
class Pool:
    """A node in the computation DAG."""
    
    def __init__(
        self,
        operation: Operation_type,
        output_index: int = 0,
        name: Optional[str] = None,
        counter: Optional[sc.Counter] = None,
        iter_order: Optional[Real] = None,
        markers: Optional[set[Marker]] = None,
    ) -> None:
        """Initialize Pool and build its counter."""
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
        self.output_index = output_index
        if counter is not None:
            self.counter = counter
        else:
            self.counter: sc.Counter = operation.build_pool_counter(
                operation.parent_pools
            )
        if iter_order is not None:  
            self.counter.iter_order = iter_order
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
        
        # Create ops container for convenience methods
        self.ops = OpsContainer(self)
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this pool."""
        return self.counter.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order for this pool."""
        self.counter.iter_order = value
    
    @property
    def name(self) -> str:
        """Name of this pool."""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set pool name and update counter name.
        
        Validates name uniqueness with the Party before accepting.
        
        Raises:
            ValueError: If the name is already used by another pool.
        """
        # Validate name with party (excludes self for renaming case)
        self._party._validate_pool_name(value, self)
        old_name = self._name
        self._name = value
        # When pool.counter is the same as operation.counter (source operations),
        # preserve operation counter name if operation has explicit name (not default)
        # Otherwise, use pool counter name
        if self.counter is self.operation.counter:
            # Check if operation has explicit name (not default like "op[0]:from_seqs")
            op_name = self.operation.name
            is_default_op_name = op_name.startswith('op[') and ']:' in op_name
            if not is_default_op_name:
                # Operation has explicit name, preserve it
                # Counter name should already be set to operation name
                pass
            else:
                # Operation has default name, use pool name
                self.counter.name = f"{value}.state"
        else:
            # Different counters, set pool counter name normally
            self.counter.name = f"{value}.state"
        # Update party's name tracking if this is a rename (not initial set)
        if old_name:
            self._party._update_pool_name(self, old_name, value)
    
    @property
    def num_states(self) -> int:
        """Number of states for this pool."""
        return self.counter.num_states
    
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
    
    def remove_marker(self, name: str) -> None:
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
        if self.operation.num_outputs > 1:
            return f"Pool(id={self._id}, name={self.name!r}, op={self.operation.name!r}, out={self.output_index}, num_states={self.num_states})"
        return f"Pool(id={self._id}, name={self.name!r}, op={self.operation.name!r}, num_states={self.num_states})"
    
    def named(self, name: str, op_name: Optional[str] = None) -> Pool_type:
        """Set the name of this pool and its operation, return self for chaining."""
        self.name = name
        self.operation.name = op_name if op_name is not None else name + '.op'
        return self
    
    @property
    def iter_order(self) -> Real:
        """Iteration order for this pool's counter.
        
        Lower values iterate faster (come first in product counters).
        """
        return self.counter.iter_order
    
    @iter_order.setter
    def iter_order(self, value: Real) -> None:
        """Set iteration order on this pool's counter."""
        self.counter.iter_order = value
    
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
        new_pool = Pool(operation=new_op, output_index=self.output_index)
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
        new_pool = Pool(operation=new_op, output_index=self.output_index)
        if name is not None:
            new_pool.name = name
        else:
            new_pool.name = self.name + '.copy'
        return new_pool
    
    #########################################################################
    # Generation
    #########################################################################
    
    def generate_library(self, **kwargs) -> Union[pd.DataFrame, list[str]]:
        """Generate sequences from this pool.
        
        This is a thin wrapper around poolparty.generate_library().
        See that function for full documentation of parameters.
        """
        from .generate_library import generate_library
        return generate_library(self, **kwargs)
    
    def print_library(
        self,
        num_seqs: Optional[Integral] = None,
        num_cycles: Optional[Integral] = None,
        show_header: bool = True,
        show_states: bool = True,
    ) -> None:
        """Print preview sequences from this pool."""
        if num_seqs is None and num_cycles is None:
            num_cycles = 1
        df = self.generate_library(
            num_seqs=num_seqs,
            num_cycles=num_cycles,
            seqs_only=False,
            init_state=0,
        )
        if show_header:
            print(f"{self.name}: seq_length={self.seq_length}, num_states={self.num_states}")
            print("state  seq" if show_states else "seq")
        state_col = f"{self.name}.state"
        for _, row in df.iterrows():
            if show_states:
                print(f"{row[state_col]:5d}  {row['seq']}")
            else:
                print(row['seq'])
        print('')
    
    #########################################################################
    # Tree visualization
    #########################################################################
    
    def print_tree(self, style: str = 'clean') -> None:
        """Print the ASCII tree visualization rooted at this pool.
        
        Args:
            style: Display style - 'clean' (default), 'minimal', or 'repr'.
        """
        from .text_viz import print_pool_tree
        print_pool_tree(self, style=style)