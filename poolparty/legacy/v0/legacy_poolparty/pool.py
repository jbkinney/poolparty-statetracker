from typing import Tuple, List, Union, Dict, Any, Optional, Set
import random

__all__ = ['Pool']

# Import after defining to avoid circular import
def _get_default_max_num_states():
    from . import DEFAULT_MAX_NUM_STATES
    return DEFAULT_MAX_NUM_STATES


# =============================================================================
# Pool Class
# =============================================================================

class Pool:
    """A class representing an oligonucleotide sequence with lazy evaluation.
    
    Can be initialized with a list of sequences (seqs parameter) to select from,
    or created through operations on other pools (concatenation, repetition, slicing).
    
    Supports sequence concatenation (+) and repetition (*) operations through
    a computation graph structure. Properly handles cases where the same Pool
    appears multiple times in the computation graph.
    """
    # Class-level counter for auto-assigning iteration_order based on creation order
    _creation_counter = 0
    # Reserved names that cannot be used for pools
    RESERVED_NAMES = {'sequence', 'sequence_id', 'sequence_length'}
    
    # Valid metadata levels
    VALID_METADATA_LEVELS = {'core', 'features', 'complete'}
    
    def __init__(self, 
                 seqs: List[str] | None = None, 
                 parents: Tuple[Union['Pool', str, int],...] = (), 
                 op: str | None = None,
                 max_num_states: int | None = None,
                 mode: str = 'random',
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        
        # Validate and store name
        if name is not None and name in self.RESERVED_NAMES:
            raise ValueError(
                f"Pool name '{name}' is reserved. "
                f"Reserved names: {self.RESERVED_NAMES}"
            )
        self.name = name
        
        # Validate and store metadata level
        if metadata not in self.VALID_METADATA_LEVELS:
            raise ValueError(
                f"metadata must be one of {self.VALID_METADATA_LEVELS}, got '{metadata}'"
            )
        self._metadata_level = metadata

    
        # Validation: exactly one of seqs or parents must be provided
        # Exception: subclasses can pass op without seqs/parents (e.g., KmerPool, SubseqPool)
        if seqs is not None and parents:
            raise ValueError("Cannot provide both 'seqs' and 'parents'")
        if seqs is None and not parents and op is None:
            raise ValueError("Must provide either 'seqs' or 'parents'")
        
        # Validate and store seqs if provided
        if seqs is not None:
            if len(seqs) == 0:
                raise ValueError("seqs must be a non-empty list")
            
            # Validate that all sequences have the same length
            lengths = []
            for s in seqs:
                if isinstance(s, Pool):
                    lengths.append(s.seq_length)
                else:
                    lengths.append(len(s))
            
            if len(set(lengths)) > 1:
                raise ValueError(
                    f"All sequences in seqs must have the same length. "
                    f"Found lengths: {set(lengths)}"
                )
        
        self.seqs = seqs
        self.parents = parents
        self.op = op
        self.internal_random_state = 0
        self.internal_sequential_state = 0
        self.mode = mode
        
        # Auto-assign iteration_order if not provided
        if iteration_order is None:
            self.iteration_order = Pool._creation_counter
            Pool._creation_counter += 1
        else:
            self.iteration_order = iteration_order
        
        # Set max_num_states (use DEFAULT for composite pools or if not provided)
        if max_num_states is None:
            self.max_num_states = _get_default_max_num_states()
        else:
            self.max_num_states = max_num_states
        
        # Collect all unique Pool ancestors (pools with independent state)
        # Now includes self, since every pool has its own internal state
        self.ancestors = self._collect_ancestors()
        
        # Calculate number of internal states (intrinsic to this pool) and number of states
        self.num_internal_states = self._calculate_num_internal_states()
        self.num_states = self._calculate_num_states()
    
    def get_state(self) -> int:
        """Return the appropriate internal state based on current mode.
        
        Returns this pool's own internal state (not the global state across ancestors).
        """
        return self.internal_random_state if self.mode == 'random' else self.internal_sequential_state
        
    def _collect_ancestors(self) -> list:
        """Collect all unique Pool ancestors in the computation graph.
        
        Returns a sorted list of all Pool objects that have independent state,
        including this pool itself. Every pool manages its own internal state.
        
        For pools with parents, also collects all ancestors from parent pools.
        For pools without parents (leaf nodes), returns [self].
        
        The list is sorted by id() to ensure deterministic ordering for state 
        decomposition. This ordering is computed once during initialization.
        """
        # Always include self - every pool has its own internal state
        ancestors_set = {self}
        
        # Also collect ancestors from any Pool parents
        for parent in self.parents:
            if isinstance(parent, Pool):
                # Parent ancestors are already lists, add them to set for uniqueness
                ancestors_set.update(parent.ancestors)
        
        # Sort by id() for deterministic ordering
        return sorted(ancestors_set, key=id)
    
    def _calculate_num_internal_states(self) -> int | float:
        """Calculate number of states intrinsic to this pool (not including ancestors).
        
        For base Pool with seqs: returns len(seqs)
        For composite operations (+, *, slice): returns 1 (no internal variation)
        Subclasses override this to return their specific number of internal states.
        
        Examples:
        - Pool with seqs: len(seqs)
        - Composite pools (concatenation, repetition, slicing): 1
        - KmerPool: len(alphabet) ** length
        - SubseqPool: (L - O - W + S) // S
        - RandomMutationPool: float('inf')
        """
        if self.seqs is not None:
            return len(self.seqs)
        return 1
    
    def _calculate_num_states(self) -> int | float:
        """Calculate total number of distinct states = internal states × ancestor states.
        
        The total states is the product of:
        1. This pool's internal states (num_internal_states)
        2. All ancestor pools' internal states (excluding self)
        
        If any component is infinite, the total is infinite.
        """
        # Start with this pool's internal states
        total = self.num_internal_states
        
        # Short-circuit if already infinite
        if total == float('inf'):
            return float('inf')
        
        # Multiply by each ancestor's internal states (excluding self)
        for ancestor in self.ancestors:
            if ancestor is not self:
                ancestor_states = ancestor.num_internal_states
                if ancestor_states == float('inf'):
                    return float('inf')
                # Multiply (both are finite integers at this point)
                total *= ancestor_states
        
        return total
    
    def _calculate_seq_length(self) -> int:
        """Calculate the sequence length.
        
        For pools with seqs: returns length of first sequence (all same length)
        For composite pools: calculates based on operation
        All pools must generate sequences of a single well-defined length.
        """
        if self.seqs is not None:
            # Pool with list of sequences
            if not self.seqs:
                return 0
            s = self.seqs[0]
            if isinstance(s, Pool):
                return s.seq_length
            else:
                return len(s)
        
        # Composite pool - calculate based on operation
        if not self.parents:
            return 0
        
        match self.op:
            case '+':
                # Concatenation: sum of parent lengths
                total = 0
                for parent in self.parents:
                    if isinstance(parent, Pool):
                        total += parent.seq_length
                    elif isinstance(parent, str):
                        total += len(parent)
                return total
            case '*':
                # Repetition: seq * times
                seq_parent = self.parents[0]
                times = self.parents[1]
                if isinstance(seq_parent, Pool):
                    return seq_parent.seq_length * times
                else:
                    return len(seq_parent) * times
            case 'slice':
                # Slicing: calculate slice length
                seq_parent = self.parents[0]
                slice_obj = self.parents[1]
                # For slicing, compute the slice result
                if isinstance(seq_parent, Pool):
                    # Use a dummy sequence to compute length
                    dummy_seq = 'x' * seq_parent.seq_length
                    return len(dummy_seq[slice_obj])
                else:
                    return len(seq_parent[slice_obj])
            case _:
                # Unknown operation - return 0 as safe default
                return 0
        
    @property
    def seq(self) -> str:
        return self._compute_seq()
    
    def is_sequential_compatible(self) -> bool:
        """Check if this pool is compatible with sequential iteration.
        
        Returns True if the pool has a positive, finite number of states
        that does not exceed max_num_states. Pools with num_states > max_num_states
        are treated as incompatible with sequential iteration and will use random access.
        """
        return 0 < self.num_states <= self.max_num_states
    
    def set_max_num_states(self, value: int) -> None:
        """Set the maximum number of states for finite behavior.
        
        Args:
            value: Maximum number of states (must be a positive integer)
            
        Raises:
            ValueError: If value is not a positive integer
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("max_num_states must be a positive integer")
        self.max_num_states = value
    
    def set_mode(self, mode: str, iteration_order: int = None) -> None:
        """Set the mode for sequence generation.
        
        Args:
            mode: Either 'random' or 'sequential'
            iteration_order: Optional iteration order for sequential mode (default: keep existing)
            
        Raises:
            ValueError: If mode is not 'random' or 'sequential', or if setting to 'sequential'
                       but pool is not sequential-compatible
        """
        if mode not in ('random', 'sequential'):
            raise ValueError(f"mode must be 'random' or 'sequential', got '{mode}'")
        
        if mode == 'sequential' and not self.is_sequential_compatible():
            raise ValueError(
                f"Cannot set mode to 'sequential' for this pool. "
                f"Pool has {self.num_states} states which exceeds max_num_states={self.max_num_states}"
            )
        
        self.mode = mode
        
        if iteration_order is not None:
            self.iteration_order = iteration_order
    
    def set_state(self, 
                  state: int, 
                  seed: int = None,
                  sequential_pools: List['Pool'] = None,
                  random_pools: List['Pool'] = None,
                  complete_states: int = None) -> None:
        """Set the state using mixed-radix for sequential pools and RNG for random pools.
        
        Decomposes the global state index into individual internal states for:
        - Sequential pools: via mixed-radix decomposition
        - Random pools: via seeded RNG for independent random draws
        
        Args:
            state: Global state index
            seed: Optional seed for random pool RNG (used in generate_seqs)
            sequential_pools: Precomputed list of sequential ancestors (for efficiency)
            random_pools: Precomputed list of random ancestors (for efficiency)
            complete_states: Precomputed product of sequential pool states (for efficiency)
        """
        # Compute if not provided (for standalone set_state calls)
        if sequential_pools is None:
            sequential_pools = self._collect_sequential_ancestors()
        
        if random_pools is None:
            # Quick set difference
            random_pools = [a for a in self.ancestors if a not in sequential_pools]
        
        if complete_states is None and sequential_pools:
            complete_states = 1
            for pool in sequential_pools:
                complete_states *= pool.num_internal_states
        
        # Branch A: Sequential pools - mixed-radix decomposition
        if sequential_pools and complete_states:
            base_state = state % complete_states
            remaining = base_state
            for pool in reversed(sequential_pools):
                pool_state = remaining % pool.num_internal_states
                pool.internal_sequential_state = pool_state
                remaining //= pool.num_internal_states
        
        # Branch B: Random pools - RNG-based
        if random_pools:
            # Create deterministic seed: hash tuple if seed provided, else use state directly
            rng_seed = hash((state, seed)) if seed is not None else state
            rng = random.Random(rng_seed)
            for ancestor in random_pools:
                ancestor.internal_random_state = rng.randint(0, 10**9)

    def _compute_seq(self) -> str:
        # Handle pools with seqs
        if self.seqs is not None:
            index = self.get_state() % len(self.seqs)
            og = self.seqs[index]
            return og.seq if isinstance(og, Pool) else og
        
        # Handle composite pools
        match self.op, self.parents:
            case '+', (str(), Pool()):
                return self.parents[0] + self.parents[1].seq
            case '+', (Pool(), str()):
                return self.parents[0].seq + self.parents[1]
            case '+', (Pool(), Pool()):
                return self.parents[0].seq + self.parents[1].seq
            case '*', (Pool(), int()):
                return self.parents[0].seq * self.parents[1]
            case 'slice', (Pool(), slice()):
                return self.parents[0].seq[self.parents[1]]
            case _:
                raise ValueError(f"Invalid op: {self.op}")
    
    def __add__(self, other: 'Pool') -> 'Pool':
        return Pool(parents=(self, other), op='+', mode='random')
    
    def __radd__(self, other: 'Pool') -> 'Pool':
        return Pool(parents=(other, self), op='+', mode='random')
    
    def __mul__(self, other: int) -> 'Pool':
        return Pool(parents=(self, other), op='*', mode='random')
    
    def __rmul__(self, other: int) -> 'Pool':
        return self.__mul__(other)
    
    def __str__(self) -> str:
        return self._compute_seq()

    def __repr__(self) -> str:
        if self.seqs is not None:
            return f"Pool({len(self.seqs)} seqs)"
        return f"Pool(seq={repr(self.seq)})"

    @property
    def seq_length(self) -> int:
        """Return the sequence length. All pools generate sequences of a single well-defined length."""
        return self._calculate_seq_length()
    
    def __getitem__(self, key: Union[int, slice]) -> Union[str, 'Pool']:
        return Pool(None, parents=(self, key), op='slice', mode='random')
        
    def visualize_graph(self, indent=0):
        """Print the computation graph structure."""
        if self.num_states == float('inf'):
            states_info = "(infinite)"
        else:
            states_info = f"({self.num_states} states)"
        unique_ancestors = f"[{len(self.ancestors)} unique]" if self.ancestors else ""
        print("  " * indent + f"{self.op or 'input'}: {str(self)} {states_info} {unique_ancestors}")
        for parent in self.parents:
            if isinstance(parent, Pool):
                parent.visualize_graph(indent + 1)
            else: 
                print("  " * (indent + 1) + f"input: {str(parent)}")
    
    def _build_computation_graph(self):
        """Build a JSON representation of the computation graph using BFS.
        
        Returns:
            Tuple of (graph_dict, node_to_id, id_to_node) where:
            - graph_dict: Dict with 'nodes' list containing node information
            - node_to_id: Dict mapping Python object id to node_id
            - id_to_node: Dict mapping node_id to actual object (Pool or literal)
        """
        from collections import deque
        
        graph = {"nodes": []}
        node_to_id = {}
        id_to_node = {}
        next_id = 0
        
        # BFS queue: (object, parent_node_ids)
        queue = deque([(self, [])])
        visited = set()
        
        while queue:
            obj, parent_ids = queue.popleft()
            obj_id = id(obj)
            
            # Skip if already visited
            if obj_id in visited:
                continue
            visited.add(obj_id)
            
            # Assign node_id
            node_id = next_id
            next_id += 1
            node_to_id[obj_id] = node_id
            id_to_node[node_id] = obj
            
            # Build node info
            if isinstance(obj, Pool):
                node_info = {
                    "node_id": node_id,
                    "type": "Pool",
                    "op": obj.op,
                    "num_states": obj.num_states if obj.num_states != float('inf') else "infinite",
                    "mode": obj.mode,
                    "name": obj.name,
                    "parent_ids": []
                }
                
                # Add parents to queue and collect their IDs
                for parent in obj.parents:
                    parent_id = id(parent)
                    # Add to queue for processing
                    queue.append((parent, [node_id]))
            elif isinstance(obj, str):
                node_info = {
                    "node_id": node_id,
                    "type": "literal",
                    "value_type": "str",
                    "value": obj,
                    "parent_ids": []
                }
            elif isinstance(obj, int):
                node_info = {
                    "node_id": node_id,
                    "type": "literal",
                    "value_type": "int",
                    "value": obj,
                    "parent_ids": []
                }
            elif isinstance(obj, slice):
                node_info = {
                    "node_id": node_id,
                    "type": "literal",
                    "value_type": "slice",
                    "value": f"slice({obj.start}, {obj.stop}, {obj.step})",
                    "parent_ids": []
                }
            else:
                node_info = {
                    "node_id": node_id,
                    "type": "literal",
                    "value_type": type(obj).__name__,
                    "value": str(obj),
                    "parent_ids": []
                }
            
            graph["nodes"].append(node_info)
        
        # Second pass: fill in parent_ids now that all nodes have IDs
        for node_info in graph["nodes"]:
            node_id = node_info["node_id"]
            obj = id_to_node[node_id]
            
            if isinstance(obj, Pool):
                for parent in obj.parents:
                    parent_id = id(parent)
                    if parent_id in node_to_id:
                        node_info["parent_ids"].append(node_to_id[parent_id])
        
        return graph, node_to_id, id_to_node
    
    def _collect_node_sequences(self, node_to_id, id_to_node):
        """Collect the current sequence/value for each node in the computation graph.
        
        Args:
            node_to_id: Dict mapping Python object id to node_id
            id_to_node: Dict mapping node_id to actual object
            
        Returns:
            Dict mapping node_id (as string) to sequence/value
        """
        node_sequences = {}
        
        for node_id, obj in id_to_node.items():
            if isinstance(obj, Pool):
                # Get the current sequence from the Pool
                node_sequences[str(node_id)] = obj.seq
            elif isinstance(obj, slice):
                # For slices, store string representation
                node_sequences[str(node_id)] = f"slice({obj.start}, {obj.stop}, {obj.step})"
            else:
                # For literals (str, int, etc.), store the value
                node_sequences[str(node_id)] = obj
        
        return node_sequences
    
    def _collect_sequential_ancestors(self) -> List['Pool']:
        """Collect all unique ancestors with mode='sequential'.
        
        Returns:
            List of Pool objects with mode='sequential', sorted by iteration_order
        """
        sequential_pools = set()
        
        def _collect_from_pool(pool):
            if pool.mode == 'sequential':
                sequential_pools.add(pool)
            for parent in pool.parents:
                if isinstance(parent, Pool):
                    _collect_from_pool(parent)
        
        _collect_from_pool(self)
        
        # Sort by iteration_order
        return sorted(sequential_pools, key=lambda p: p.iteration_order)
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this pool at the current state.
        
        Base implementation returns universal fields based on metadata level.
        Subclasses override to add pool-specific fields.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end (no value - memory efficient)
            - 'features': core + pool-specific fields (default)
            - 'complete': features + value (full sequence)
        
        Args:
            abs_start: Absolute start position in the final sequence
            abs_end: Absolute end position in the final sequence
            
        Returns:
            Dictionary with metadata fields based on metadata level.
        """
        # Core fields (always included)
        metadata = {
            'index': self.get_state() % self.num_internal_states if self.num_internal_states != float('inf') else self.get_state(),
            'abs_start': abs_start,
            'abs_end': abs_end,
        }
        
        # Add value only for 'complete' level
        if self._metadata_level == 'complete':
            metadata['value'] = self.seq
        
        return metadata
    
    def _build_linear_structure(self) -> List[Dict[str, Any]]:
        """Build flat list of segments in final sequence order.
        
        Traverses the computation graph and creates a segment for each
        pool that contributes to the final sequence. MixedPool children
        are fully expanded.
        
        Returns:
            List of segment dictionaries with keys:
                - pool: The Pool object (or None for literals)
                - literal: String literal (if not a pool)
                - abs_start: Absolute start position
                - abs_end: Absolute end position
                - is_mixed: True if MixedPool
                - is_mixed_child: True if inside a MixedPool
                - mixed_parent: Parent MixedPool (if is_mixed_child)
                - mixed_child_index: Index in parent's pools list
                - is_transformer_parent: True if tracked as transformer parent
        """
        segments = []
        
        # Transformer operations that have a parent pool
        TRANSFORMER_OPS = {
            'k_mutate', 'mutate', 'shuffle', 'subseq',
            'insertion_scan', 'deletion_scan', 'shuffle_scan',
            'k_mutate_orf', 'random_mutate_orf',
            'insertion_scan_orf', 'deletion_scan_orf',
            'spacing_scan',
        }
        
        def add_transformer_chain(transformer, abs_start, abs_end):
            """Recursively add transformer ancestors.
            
            Iterates ALL parent pools (not just first) to support pools
            with multiple Pool inputs like SpacingScanPool and InsertionScanPool.
            Also handles MixedPool children when MixedPool is used as an input.
            """
            def walk_chain(node, child_abs_start, child_abs_end):
                if not hasattr(node, 'parents') or not node.parents:
                    return
                
                # Iterate ALL parents to handle multi-input pools
                for parent in node.parents:
                    if not isinstance(parent, Pool):
                        continue
                    
                    # Skip unnamed parents but continue walking for various composite types
                    if not parent.name:
                        if parent.op in TRANSFORMER_OPS or parent.op == '+':
                            # Recurse to find named ancestors within composite structures
                            walk_chain(parent, None, None)
                        elif parent.op == 'mixed' and hasattr(parent, 'pools'):
                            # MixedPool - walk into each child pool
                            for child in parent.pools:
                                if isinstance(child, Pool):
                                    walk_pool_structure(child)
                        continue
                    
                    # Skip if already tracked as direct segment
                    if any(s.get('pool') is parent and not s.get('is_transformer_parent') for s in segments):
                        continue
                    
                    # Determine coordinates - use None for different-length parents
                    # (e.g., insert pools in SpacingScanPool have different lengths)
                    if child_abs_start is None:
                        parent_abs_start, parent_abs_end = None, None
                    elif node.seq_length == parent.seq_length:
                        parent_abs_start, parent_abs_end = child_abs_start, child_abs_end
                    else:
                        parent_abs_start, parent_abs_end = None, None
                    
                    segments.append({
                        'pool': parent,
                        'literal': None,
                        'is_transformer_parent': True,
                        'abs_start': parent_abs_start,
                        'abs_end': parent_abs_end,
                    })
                    
                    # Continue walking for transformers, composites, and MixedPools
                    if parent.op in TRANSFORMER_OPS or parent.op == '+':
                        walk_chain(parent, parent_abs_start, parent_abs_end)
                    elif parent.op == 'mixed' and hasattr(parent, 'pools'):
                        # MixedPool - walk into each child pool
                        for child in parent.pools:
                            if isinstance(child, Pool):
                                walk_pool_structure(child)
            
            def walk_pool_structure(node):
                """Walk a pool's structure to find named pools."""
                if not isinstance(node, Pool):
                    return
                
                # If named, add as segment
                if node.name:
                    # Skip if already tracked
                    if any(s.get('pool') is node for s in segments):
                        return
                    
                    segments.append({
                        'pool': node,
                        'literal': None,
                        'is_transformer_parent': True,
                        'abs_start': None,  # No position mapping for nested structures
                        'abs_end': None,
                    })
                
                # Continue walking based on op type
                if node.op == '+' and hasattr(node, 'parents'):
                    for parent in node.parents:
                        if isinstance(parent, Pool):
                            walk_pool_structure(parent)
                elif node.op == 'mixed' and hasattr(node, 'pools'):
                    for child in node.pools:
                        if isinstance(child, Pool):
                            walk_pool_structure(child)
                elif node.op in TRANSFORMER_OPS and hasattr(node, 'parents'):
                    for parent in node.parents:
                        if isinstance(parent, Pool):
                            walk_pool_structure(parent)
            
            walk_chain(transformer, abs_start, abs_end)
        
        def traverse_mixed_child(node, base_offset: int, mixed_parent, mixed_child_index: int) -> int:
            """Recursively traverse MixedPool child and add segments."""
            if isinstance(node, str):
                return base_offset + len(node)
            
            if not isinstance(node, Pool):
                return base_offset
            
            if node.op == '+':
                current = base_offset
                for parent in node.parents:
                    current = traverse_mixed_child(parent, current, mixed_parent, mixed_child_index)
                return current
            
            elif node.op == '*':
                seq_node = node.parents[0]
                times = node.parents[1]
                current = base_offset
                for _ in range(times):
                    current = traverse_mixed_child(seq_node, current, mixed_parent, mixed_child_index)
                return current
            
            elif node.op == 'slice':
                segments.append({
                    'pool': node,
                    'literal': None,
                    'abs_start': base_offset,
                    'abs_end': base_offset + node.seq_length,
                    'is_sliced': True,
                    'is_mixed_child': True,
                    'mixed_parent': mixed_parent,
                    'mixed_child_index': mixed_child_index,
                })
                return base_offset + node.seq_length
            
            elif node.op == 'mixed':
                # Nested MixedPool - add as single segment
                segments.append({
                    'pool': node,
                    'literal': None,
                    'abs_start': base_offset,
                    'abs_end': base_offset + node.seq_length,
                    'is_mixed': True,
                    'is_mixed_child': True,
                    'mixed_parent': mixed_parent,
                    'mixed_child_index': mixed_child_index,
                })
                return base_offset + node.seq_length
            
            elif node.op in TRANSFORMER_OPS:
                segments.append({
                    'pool': node,
                    'literal': None,
                    'abs_start': base_offset,
                    'abs_end': base_offset + node.seq_length,
                    'is_mixed_child': True,
                    'mixed_parent': mixed_parent,
                    'mixed_child_index': mixed_child_index,
                })
                # Track transformer parents inside MixedPool children
                add_transformer_chain(node, base_offset, base_offset + node.seq_length)
                return base_offset + node.seq_length
            
            else:
                # Leaf pool
                segments.append({
                    'pool': node,
                    'literal': None,
                    'abs_start': base_offset,
                    'abs_end': base_offset + node.seq_length,
                    'is_mixed_child': True,
                    'mixed_parent': mixed_parent,
                    'mixed_child_index': mixed_child_index,
                })
                return base_offset + node.seq_length
        
        def traverse(node, offset: int) -> int:
            """Traverse tree and return end offset."""
            if isinstance(node, str):
                segments.append({
                    'pool': None,
                    'literal': node,
                    'abs_start': offset,
                    'abs_end': offset + len(node),
                })
                return offset + len(node)
            
            if not isinstance(node, Pool):
                return offset
            
            if node.op == '+':
                current = offset
                for parent in node.parents:
                    current = traverse(parent, current)
                return current
            
            elif node.op == '*':
                seq_node = node.parents[0]
                times = node.parents[1]
                current = offset
                for _ in range(times):
                    current = traverse(seq_node, current)
                return current
            
            elif node.op == 'slice':
                segments.append({
                    'pool': node,
                    'literal': None,
                    'abs_start': offset,
                    'abs_end': offset + node.seq_length,
                    'is_sliced': True,
                })
                return offset + node.seq_length
            
            elif node.op == 'mixed':
                # MixedPool: add as segment
                segments.append({
                    'pool': node,
                    'literal': None,
                    'abs_start': offset,
                    'abs_end': offset + node.seq_length,
                    'is_mixed': True,
                })
                
                # FULL EXPANSION: traverse each child
                for child_index, child in enumerate(node.pools):
                    traverse_mixed_child(child, offset, node, child_index)
                
                return offset + node.seq_length
            
            elif node.op in TRANSFORMER_OPS:
                segments.append({
                    'pool': node,
                    'literal': None,
                    'abs_start': offset,
                    'abs_end': offset + node.seq_length,
                })
                add_transformer_chain(node, offset, offset + node.seq_length)
                return offset + node.seq_length
            
            else:
                # Leaf pool
                segments.append({
                    'pool': node,
                    'literal': None,
                    'abs_start': offset,
                    'abs_end': offset + node.seq_length,
                })
                return offset + node.seq_length
        
        traverse(self, 0)
        return segments
    
    def _assign_occurrences(self, segments: List[Dict]) -> List[Dict]:
        """Assign occurrence indices to pools appearing multiple times.
        
        Args:
            segments: List of segment dictionaries from _build_linear_structure()
            
        Returns:
            Same list with added 'key_prefix' and 'occurrence' fields
        """
        # Count occurrences per pool object
        pool_counts: Dict[int, int] = {}
        for seg in segments:
            pool = seg.get('pool')
            if pool is not None and pool.name:  # Falsy check for name
                pool_id = id(pool)
                pool_counts[pool_id] = pool_counts.get(pool_id, 0) + 1
        
        # Assign indices
        pool_current_idx: Dict[int, int] = {}
        for seg in segments:
            pool = seg.get('pool')
            if pool is not None and pool.name:  # Falsy check for name
                pool_id = id(pool)
                count = pool_counts[pool_id]
                
                if count > 1:
                    idx = pool_current_idx.get(pool_id, 0) + 1
                    pool_current_idx[pool_id] = idx
                    seg['occurrence'] = idx
                    seg['key_prefix'] = f"{pool.name}[{idx}]"
                else:
                    seg['occurrence'] = None
                    seg['key_prefix'] = pool.name
            else:
                seg['key_prefix'] = None
                seg['occurrence'] = None
        
        return segments
    
    def _build_design_card_keys(self, tracked_segments: List[Dict]) -> List[str]:
        """Build column names for design cards.
        
        Keys are built based on each pool's metadata level:
        - 'core': index, abs_start, abs_end only
        - 'features': core + pool-specific fields
        - 'complete': features + value
        
        Also stores 'metadata_key_suffixes' in each segment for efficient
        null-filling of unselected MixedPool children (avoids calling get_metadata).
        
        Args:
            tracked_segments: List of segments to track
            
        Returns:
            List of column names in order
        """
        keys = ['sequence_id', 'sequence_length']
        
        for seg in tracked_segments:
            prefix = seg.get('key_prefix')
            if not prefix:
                continue
            
            pool = seg['pool']
            metadata_level = getattr(pool, '_metadata_level', 'features')
            
            # Track suffixes for this segment (used for null-filling unselected children)
            suffixes = []
            
            # Core fields (always included)
            keys.append(f'{prefix}_index')
            keys.append(f'{prefix}_abs_start')
            keys.append(f'{prefix}_abs_end')
            suffixes.extend(['index', 'abs_start', 'abs_end'])
            
            # Value field only for 'complete' level
            if metadata_level == 'complete':
                keys.append(f'{prefix}_value')
                suffixes.append('value')
            
            # Pool-specific fields for 'features' and 'complete' levels
            if metadata_level in ('features', 'complete'):
                if pool.op == 'mixed':
                    keys.append(f'{prefix}_selected')
                    keys.append(f'{prefix}_selected_name')
                    suffixes.extend(['selected', 'selected_name'])
                elif pool.op == 'insertion_scan':
                    keys.append(f'{prefix}_pos')
                    keys.append(f'{prefix}_pos_abs')
                    keys.append(f'{prefix}_insert')
                    suffixes.extend(['pos', 'pos_abs', 'insert'])
                elif pool.op == 'deletion_scan':
                    keys.append(f'{prefix}_pos')
                    keys.append(f'{prefix}_pos_abs')
                    keys.append(f'{prefix}_del_len')
                    suffixes.extend(['pos', 'pos_abs', 'del_len'])
                elif pool.op == 'subseq':
                    keys.append(f'{prefix}_pos')
                    keys.append(f'{prefix}_width')
                    suffixes.extend(['pos', 'width'])
                elif pool.op == 'shuffle_scan':
                    keys.append(f'{prefix}_pos')
                    keys.append(f'{prefix}_pos_abs')
                    keys.append(f'{prefix}_window_size')
                    suffixes.extend(['pos', 'pos_abs', 'window_size'])
                elif pool.op == 'k_mutate':
                    keys.append(f'{prefix}_mut_pos')
                    keys.append(f'{prefix}_mut_pos_abs')
                    keys.append(f'{prefix}_mut_from')
                    keys.append(f'{prefix}_mut_to')
                    suffixes.extend(['mut_pos', 'mut_pos_abs', 'mut_from', 'mut_to'])
                elif pool.op == 'mutate':  # RandomMutationPool
                    keys.append(f'{prefix}_mut_count')
                    keys.append(f'{prefix}_mut_pos')
                    keys.append(f'{prefix}_mut_pos_abs')
                    keys.append(f'{prefix}_mut_from')
                    keys.append(f'{prefix}_mut_to')
                    suffixes.extend(['mut_count', 'mut_pos', 'mut_pos_abs', 'mut_from', 'mut_to'])
                elif pool.op == 'insertion_scan_orf':
                    keys.append(f'{prefix}_codon_pos')
                    keys.append(f'{prefix}_codon_pos_abs')
                    keys.append(f'{prefix}_insert')
                    keys.append(f'{prefix}_insert_aa')
                    suffixes.extend(['codon_pos', 'codon_pos_abs', 'insert', 'insert_aa'])
                elif pool.op == 'deletion_scan_orf':
                    keys.append(f'{prefix}_codon_pos')
                    keys.append(f'{prefix}_codon_pos_abs')
                    keys.append(f'{prefix}_del_codons')
                    keys.append(f'{prefix}_del_aa')
                    suffixes.extend(['codon_pos', 'codon_pos_abs', 'del_codons', 'del_aa'])
                elif pool.op == 'k_mutate_orf':
                    keys.append(f'{prefix}_codon_pos')
                    keys.append(f'{prefix}_codon_pos_abs')
                    keys.append(f'{prefix}_codon_from')
                    keys.append(f'{prefix}_codon_to')
                    keys.append(f'{prefix}_aa_from')
                    keys.append(f'{prefix}_aa_to')
                    suffixes.extend(['codon_pos', 'codon_pos_abs', 'codon_from', 'codon_to', 'aa_from', 'aa_to'])
                elif pool.op == 'random_mutate_orf':
                    keys.append(f'{prefix}_mut_count')
                    keys.append(f'{prefix}_codon_pos')
                    keys.append(f'{prefix}_codon_pos_abs')
                    keys.append(f'{prefix}_codon_from')
                    keys.append(f'{prefix}_codon_to')
                    keys.append(f'{prefix}_aa_from')
                    keys.append(f'{prefix}_aa_to')
                    suffixes.extend(['mut_count', 'codon_pos', 'codon_pos_abs', 'codon_from', 'codon_to', 'aa_from', 'aa_to'])
                elif pool.op == 'motif':
                    keys.append(f'{prefix}_orientation')
                    suffixes.append('orientation')
                elif pool.op == 'shuffle':
                    keys.append(f'{prefix}_start')
                    keys.append(f'{prefix}_end')
                    keys.append(f'{prefix}_shuffle_mode')
                    suffixes.extend(['start', 'end', 'shuffle_mode'])
                elif pool.op == 'spacing_scan':
                    # Per-insert keys
                    insert_names = getattr(pool, '_insert_names', [])
                    for name in insert_names:
                        keys.append(f'{prefix}_{name}_dist')
                        keys.append(f'{prefix}_{name}_pos_start')
                        keys.append(f'{prefix}_{name}_pos_end')
                        keys.append(f'{prefix}_{name}_abs_pos_start')
                        keys.append(f'{prefix}_{name}_abs_pos_end')
                        suffixes.extend([f'{name}_dist', f'{name}_pos_start', f'{name}_pos_end', 
                                        f'{name}_abs_pos_start', f'{name}_abs_pos_end'])
                    # Pairwise spacing keys
                    for i in range(len(insert_names)):
                        for j in range(i + 1, len(insert_names)):
                            keys.append(f'{prefix}_spacing_{insert_names[i]}_{insert_names[j]}')
                            suffixes.append(f'spacing_{insert_names[i]}_{insert_names[j]}')
            
            # Store suffixes in segment for efficient null-filling
            seg['metadata_key_suffixes'] = suffixes
        
        return keys
    
    def generate_seqs(self, 
                      num_seqs: int = None,
                      num_complete_iterations: int = None,
                      max_num_states: int = None,
                      return_computation_graph: bool = False,
                      return_design_cards: bool = False,
                      track_pools: Optional[List[str]] = None,
                      seed: int = None):
        """Generate a list of sequences from this Pool.
        
        The method automatically collects all ancestor pools with mode='sequential' 
        and iterates through them combinatorially. Pools with mode='random' are 
        sampled using deterministic randomness controlled by the seed parameter.
        
        Args:
            num_seqs: Number of sequences to generate. When no sequential pools exist,
                generates random sequences. When sequential pools exist, loops through
                all combinatorial states, wrapping around if num_seqs exceeds the total
                number of states. Cannot be used together with num_complete_iterations.
            num_complete_iterations: Number of times to iterate through all combinations
                of sequential pools. Only used when sequential pools exist. Cannot be
                used together with num_seqs.
            max_num_states: Maximum number of states allowed for sequential pools
                (default: self.max_num_states)
            return_computation_graph: If True, returns a dictionary with the generated
                sequences, the computation graph structure, and sequences for each node.
                If False (default), returns just the list of generated sequences.
            return_design_cards: If True, returns structured metadata (design cards)
                alongside sequences. Design cards contain per-sequence metadata for
                all tracked pools.
            track_pools: List of pool names to track in design cards. If None (default),
                tracks all named pools. If empty list, tracks no pools.
            seed: Optional seed for deterministic random generation. Controls randomness
                for all pools with mode='random' in the computation graph.
        
        Returns:
            If return_computation_graph=False and return_design_cards=False (default):
                List of generated sequences
            If return_computation_graph=True or return_design_cards=True:
                Dictionary with keys:
                    - 'sequences': List of generated sequences
                    - 'graph': (if return_computation_graph) JSON computation graph
                    - 'node_sequences': (if return_computation_graph) Dict of node sequences
                    - 'design_cards': (if return_design_cards) DesignCards object
            
        Raises:
            ValueError: If both num_seqs and num_complete_iterations are provided,
                if neither is provided when needed, if any sequential pool
                exceeds max_num_states, or if duplicate pool names are detected
        """
        # Use pool's max_num_states if not specified
        if max_num_states is None:
            max_num_states = self.max_num_states
        
        # Validate parameter combinations
        if num_seqs is not None and num_complete_iterations is not None:
            raise ValueError("Cannot specify both num_seqs and num_complete_iterations")
        
        if num_seqs is None and num_complete_iterations is None:
            raise ValueError("Must specify either num_seqs or num_complete_iterations")
        
        # Design cards setup (one-time)
        design_cards = None
        tracked_segments = None
        mixed_segment_index = None
        
        if return_design_cards:
            from .design_cards import DesignCards
            
            # Build linear structure
            segments = self._build_linear_structure()
            segments = self._assign_occurrences(segments)
            
            # Validate unique names
            validate_unique_names(segments)
            
            # Filter to tracked segments
            if track_pools is None:
                # Track all named pools
                tracked_segments = [s for s in segments if s.get('key_prefix')]
            elif len(track_pools) == 0:
                # Track no pools
                tracked_segments = []
            else:
                # Track specific pools
                tracked_segments = [
                    s for s in segments 
                    if s.get('pool') and s['pool'].name in track_pools
                ]
            
            # Build keys and initialize DesignCards
            keys = self._build_design_card_keys(tracked_segments)
            design_cards = DesignCards(keys)
            
            # Build MixedPool segment index for efficient lookup
            # Use tracked_segments to match what's used in design card collection
            mixed_segment_index = MixedPoolSegmentIndex(tracked_segments)
        
        # Precompute optimization values once before loop
        # These are passed to set_state() to avoid redundant calculations
        sequential_pools = self._collect_sequential_ancestors()
        random_pools = [a for a in self.ancestors if a not in sequential_pools]
        
        complete_states = None
        if sequential_pools:
            # Validate sequential pools have finite internal states
            # Note: We check num_internal_states, NOT num_states, because a sequential
            # pool can have random-mode ancestors with infinite states. What matters
            # is that the pool's OWN internal states are finite and enumerable.
            for pool in sequential_pools:
                if pool.num_internal_states == float('inf'):
                    raise ValueError(f"Sequential pool {pool} has infinite internal states")
                if pool.num_internal_states <= 0:
                    raise ValueError(f"Sequential pool {pool} must have positive internal states")
                if pool.num_internal_states > max_num_states:
                    raise ValueError(
                        f"Sequential pool {pool} has {pool.num_internal_states} internal states, "
                        f"which exceeds max_num_states={max_num_states}"
                    )
            
            # Calculate total combinatorial states from sequential pools
            complete_states = 1
            for pool in sequential_pools:
                complete_states *= pool.num_internal_states
        
        # Branch 1: Generate num_seqs sequences
        # States progress linearly: 0, 1, 2, ..., num_seqs-1
        # Sequential pools wrap via modulo in set_state()
        if num_seqs is not None:
            result = []
            
            # Build computation graph if needed (done once before loop)
            if return_computation_graph:
                graph, node_to_id, id_to_node = self._build_computation_graph()
                all_node_sequences = {str(node_id): [] for node_id in id_to_node.keys()}
            
            # Generate sequences by iterating state from 0 to num_seqs-1
            for i in range(num_seqs):
                # Set state with precomputed values (avoids redundant calculations)
                self.set_state(i, 
                              seed=seed,
                              sequential_pools=sequential_pools,
                              random_pools=random_pools,
                              complete_states=complete_states)
                seq = self.seq
                result.append(seq)
                
                # Collect design card metadata for this iteration
                if return_design_cards:
                    row = _collect_design_card_row(
                        i, seq, tracked_segments, mixed_segment_index
                    )
                    design_cards.append_row(row)
                
                # Collect node sequences for this iteration
                if return_computation_graph:
                    node_seqs = self._collect_node_sequences(node_to_id, id_to_node)
                    for node_id_str, node_seq in node_seqs.items():
                        all_node_sequences[node_id_str].append(node_seq)
            
            # Build return value
            if return_computation_graph or return_design_cards:
                ret = {"sequences": result}
                if return_computation_graph:
                    ret["graph"] = graph
                    ret["node_sequences"] = all_node_sequences
                if return_design_cards:
                    ret["design_cards"] = design_cards
                return ret
            return result
        
        # Branch 2: Generate num_complete_iterations × complete_states sequences
        # Iterates through all sequential combinations multiple times
        # Useful for replicates of full combinatorial designs
        result = []
        
        # Build computation graph if needed (done once before loop)
        if return_computation_graph:
            graph, node_to_id, id_to_node = self._build_computation_graph()
            all_node_sequences = {str(node_id): [] for node_id in id_to_node.keys()}
        
        # Generate sequences by iterating through complete cycles
        for iteration in range(num_complete_iterations):
            for complete_state in range(complete_states):
                # Map (iteration, complete_state) to global state index
                global_state = iteration * complete_states + complete_state
                
                # Set state with precomputed values (avoids redundant calculations)
                self.set_state(global_state, 
                              seed=seed,
                              sequential_pools=sequential_pools,
                              random_pools=random_pools,
                              complete_states=complete_states)
                seq = self.seq
                result.append(seq)
                
                # Collect design card metadata for this iteration
                if return_design_cards:
                    row = _collect_design_card_row(
                        global_state, seq, tracked_segments, mixed_segment_index
                    )
                    design_cards.append_row(row)
                
                # Collect node sequences for this iteration
                if return_computation_graph:
                    node_seqs = self._collect_node_sequences(node_to_id, id_to_node)
                    for node_id_str, node_seq in node_seqs.items():
                        all_node_sequences[node_id_str].append(node_seq)
        
        # Build return value
        if return_computation_graph or return_design_cards:
            ret = {"sequences": result}
            if return_computation_graph:
                ret["graph"] = graph
                ret["node_sequences"] = all_node_sequences
            if return_design_cards:
                ret["design_cards"] = design_cards
            return ret
        return result


# =============================================================================
# Design Card Helpers (Internal)
# =============================================================================

def _collect_design_card_row(
    sequence_id: int,
    sequence: str,
    tracked_segments: List[Dict],
    mixed_segment_index: 'MixedPoolSegmentIndex'
) -> Dict[str, Any]:
    """Collect metadata for a single sequence row.
    
    Args:
        sequence_id: Index of the sequence
        sequence: The generated sequence
        tracked_segments: List of tracked segments
        mixed_segment_index: Index for efficient MixedPool child lookup
        
    Returns:
        Dictionary with all metadata for this row
    """
    row = {
        'sequence_id': sequence_id,
        'sequence_length': len(sequence),
    }
    
    for seg in tracked_segments:
        prefix = seg.get('key_prefix')
        if not prefix:
            continue
        
        pool = seg['pool']
        abs_start = seg.get('abs_start')
        abs_end = seg.get('abs_end')
        
        # Check if this is a MixedPool child
        if seg.get('is_mixed_child'):
            mixed_parent = seg.get('mixed_parent')
            expected_child_index = seg.get('mixed_child_index')
            
            # Get actual selected child index
            actual_selected = mixed_segment_index.get_selected_child_index(mixed_parent)
            
            if actual_selected != expected_child_index:
                # Not selected - use precomputed suffixes to set None values
                # This avoids calling get_metadata() for unselected children
                suffixes = seg.get('metadata_key_suffixes', [])
                for suffix in suffixes:
                    row[f'{prefix}_{suffix}'] = None
                continue
        
        # Get metadata from pool
        metadata = pool.get_metadata(abs_start, abs_end)
        for key, value in metadata.items():
            row[f'{prefix}_{key}'] = value
    
    return row


class MixedPoolSegmentIndex:
    """Pre-indexed structure for efficient MixedPool child lookup.
    
    Groups segments by their MixedPool parent to allow O(1) lookup
    of which segments are active based on the MixedPool's selection.
    """
    
    def __init__(self, segments: List[Dict]):
        """Build index from segments.
        
        Args:
            segments: List of segments from _build_linear_structure()
        """
        # Map mixed_parent pool -> list of child segments grouped by child_index
        self._mixed_children: Dict[int, Dict[int, List[Dict]]] = {}
        
        for seg in segments:
            if seg.get('is_mixed_child'):
                parent = seg.get('mixed_parent')
                child_idx = seg.get('mixed_child_index')
                
                if parent is not None:
                    parent_id = id(parent)
                    if parent_id not in self._mixed_children:
                        self._mixed_children[parent_id] = {}
                    if child_idx not in self._mixed_children[parent_id]:
                        self._mixed_children[parent_id][child_idx] = []
                    self._mixed_children[parent_id][child_idx].append(seg)
    
    def get_selected_child_index(self, mixed_pool) -> Optional[int]:
        """Get the currently selected child index for a MixedPool.
        
        Args:
            mixed_pool: The MixedPool to query
            
        Returns:
            Index of currently selected child (0-based)
        """
        # MixedPool stores selection in internal state
        if hasattr(mixed_pool, '_get_selected_child_index'):
            return mixed_pool._get_selected_child_index()
        
        # Fallback: decompose state to find selection
        state = mixed_pool.get_state()
        num_children = len(mixed_pool.pools)
        return state % num_children
    
    def get_active_segments(self, mixed_pool) -> List[Dict]:
        """Get segments for the currently selected child.
        
        Args:
            mixed_pool: The MixedPool to query
            
        Returns:
            List of active segments for the selected child
        """
        parent_id = id(mixed_pool)
        if parent_id not in self._mixed_children:
            return []
        
        selected_idx = self.get_selected_child_index(mixed_pool)
        return self._mixed_children[parent_id].get(selected_idx, [])


def validate_unique_names(segments: List[Dict]) -> None:
    """Validate that different pool objects do not share the same name.
    
    Args:
        segments: List of segment dictionaries
        
    Raises:
        ValueError: If duplicate pool names are found from different objects
    """
    name_to_pool_id: Dict[str, int] = {}
    
    for seg in segments:
        pool = seg.get('pool')
        if pool and pool.name:
            name = pool.name
            pool_id = id(pool)
            
            if name in name_to_pool_id and name_to_pool_id[name] != pool_id:
                raise ValueError(
                    f"Duplicate pool name '{name}' used by different pool objects. "
                    f"Each named pool must have a unique name."
                )
            name_to_pool_id[name] = pool_id


# =============================================================================
# Visualization Helpers
# =============================================================================

def visualize_computation_graph(graph_dict, node_sequences=None, show_first_only=False, seq_display_length=50, indent=0, node_id=0, visited=None):
    """Visualize a computation graph in a tree format.
    
    Args:
        graph_dict: Dictionary with 'nodes' list from _build_computation_graph()
        node_sequences: Optional dict mapping node_id (as string) to sequences
        show_first_only: If True, display only the first sequence from each node's
            sequence list. If False (default), display the full list.
        seq_display_length: Maximum length to display for sequences before truncating
            with "..." (default: 50). Set to a larger value to see longer sequences.
        indent: Current indentation level (for recursion)
        node_id: Current node to print (for recursion)
        visited: Set of visited node IDs (for recursion)
        
    Example output:
        Node 0 (Pool, op: +)
          Node 1 (Pool, op: mutate)
            Node 3 (str: "ACGT...")
          Node 2 (str: ".BBBBB.")
    """
    if visited is None:
        visited = set()
    
    # Avoid infinite loops from circular references
    if node_id in visited:
        print("  " * indent + f"Node {node_id} (already shown)")
        return
    visited.add(node_id)
    
    # Find the node in the graph
    node = None
    for n in graph_dict["nodes"]:
        if n["node_id"] == node_id:
            node = n
            break
    
    if node is None:
        return
    
    # Build the display string
    prefix = "  " * indent
    
    # Start with node_id and type
    info_parts = [f"node_id: {node_id}"]
    
    if node["type"] == "Pool":
        info_parts.append(f"type: Pool")
        # Add operation for Pool nodes
        if node["op"]:
            info_parts.append(f"op: {node['op']}")
        else:
            info_parts.append(f"op: input")
    else:
        # For literal nodes, use the value_type (str, int, etc.)
        info_parts.append(f"type: {node['value_type']}")
    
    # Add parent_ids
    info_parts.append(f"parent_ids: {node['parent_ids']}")
    
    # Add mode for Pool nodes
    if node["type"] == "Pool":
        info_parts.append(f"mode: {node['mode']}")
    
    display = f"{prefix}{', '.join(info_parts)}"
    
    # Add sequence or value
    if node["type"] == "Pool":
        # Add sequence if available
        if node_sequences and str(node_id) in node_sequences:
            seq = node_sequences[str(node_id)]
            
            # If show_first_only is True and seq is a list, take only the first element
            if show_first_only and isinstance(seq, list) and len(seq) > 0:
                seq = seq[0]
            
            if isinstance(seq, str) and len(seq) > seq_display_length:
                seq_preview = seq[:seq_display_length-3] + "..."
            else:
                seq_preview = seq
            display += f" -> {seq_preview}"
    else:
        # Literal node - show value
        value = node["value"]
        if isinstance(value, str) and len(value) > seq_display_length:
            value = value[:seq_display_length-3] + "..."
        display += f" -> {repr(value)}"
    
    print(display)
    
    # Recursively print children
    for parent_id in node["parent_ids"]:
        visualize_computation_graph(graph_dict, node_sequences, show_first_only, seq_display_length, indent + 1, parent_id, visited)
