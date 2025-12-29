import random
from typing import List, Union, Dict, Any
from .pool import Pool


class MixedPool(Pool):
    """A class for mixing multiple Pool objects.
    
    In random mode, randomly selects which Pool to draw from based on weights.
    In sequential mode, iterates through each Pool sequentially in order.
    
    The pool has finite num_states equal to the sum of all child pool states,
    unless any child has infinite states (in which case MixedPool is infinite).
    """
    
    def __init__(self, 
                 pools: List[Pool], 
                 weights: List[float] = None,
                 max_num_states: int = None,
                 mode: str = 'random',
                 iteration_order: int | None = None,
                 name: str | None = None,
                 metadata: str = 'features'):
        """Initialize a MixedPool.
        
        Args:
            pools: List of Pool objects to mix
            weights: Optional list of relative probabilities for random selection.
                    If None, all pools have equal probability. Must have same
                    length as pools if provided.
            max_num_states: Maximum number of states before treating as infinite
            mode: Either 'random' or 'sequential' (default: 'random')
            iteration_order: Order for sequential iteration (default: auto-assigned based on creation order)
            
        Raises:
            ValueError: If pools is empty, weights length doesn't match pools,
                       pools have different seq_length values, or custom weights
                       are provided with mode='sequential'
        """
        if not pools:
            raise ValueError("pools list cannot be empty")
        
        # Validate that all pools have the same seq_length
        seq_lengths = [pool.seq_length for pool in pools]
        if len(set(seq_lengths)) > 1:
            raise ValueError(
                f"All pools in MixedPool must have the same seq_length. "
                f"Found seq_lengths: {set(seq_lengths)}"
            )
        
        # Validate that weights are not provided with sequential mode
        if mode == 'sequential' and weights is not None:
            raise ValueError(
                "Cannot specify custom weights with mode='sequential'. "
                "Sequential mode iterates through all child pool states deterministically, "
                "ignoring weights. Use mode='random' for weighted selection, or omit weights "
                "parameter for sequential mode."
            )
        
        # Validate no infinite children in sequential mode (fail fast)
        if mode == 'sequential':
            for pool in pools:
                if pool.num_states == float('inf'):
                    pool_name = pool.name or repr(pool)
                    raise ValueError(
                        f"MixedPool with mode='sequential' cannot contain pools with "
                        f"infinite states. Pool '{pool_name}' has infinite states. "
                        f"Use mode='random' instead."
                    )
            
        self.pools = pools
        
        # Set up weights (default to equal probability)
        if weights is None:
            self.weights = [1.0] * len(pools)
        else:
            if len(weights) != len(pools):
                raise ValueError(f"weights length ({len(weights)}) must match pools length ({len(pools)})")
            self.weights = list(weights)
        
        # Normalize weights to probabilities
        total_weight = sum(self.weights)
        if total_weight <= 0:
            raise ValueError("Sum of weights must be positive")
        self.probabilities = [w / total_weight for w in self.weights]
        
        super().__init__(op='mixed', max_num_states=max_num_states, mode=mode, iteration_order=iteration_order, name=name, metadata=metadata)
    
    def _calculate_num_internal_states(self) -> Union[int, float]:
        """Calculate total number of internal states as sum of all child pool states.
        
        MixedPool's internal states represent the choice of which pool to use,
        so it's the sum of all child pool states (not product).
        
        Returns float('inf') if any child pool has infinite states.
        """
        total = 0
        for pool in self.pools:
            if pool.num_states == float('inf'):
                return float('inf')
            total += pool.num_states
        return total
    
    def _calculate_seq_length(self) -> int:
        """Return the sequence length (all pools have the same length)."""
        # All pools have the same length (validated in __init__)
        return self.pools[0].seq_length
    
    def is_sequential_compatible(self) -> bool:
        """Check if this pool is compatible with sequential iteration.
        
        Returns False if any child pool has infinite states, otherwise
        uses parent class logic (checks if num_states <= max_num_states).
        """
        # If any child pool has infinite states, not sequential-compatible
        for pool in self.pools:
            if pool.num_states == float('inf'):
                return False
        
        # Otherwise use parent logic
        return super().is_sequential_compatible()
    
    def _compute_seq(self) -> str:
        """Compute sequence based on current mode.
        
        Sequential mode (finite pools only): deterministically iterates through
        all child pool states in order.
        
        Random mode: uses weighted random selection of pools, then random state
        selection within the chosen pool. Works for both finite and infinite pools.
        
        Randomness is controlled by the seed passed to generate_seqs(), which
        initializes internal_random_state via set_state().
        """
        # Check both mode and compatibility for sequential behavior
        if self.mode == 'sequential' and self.is_sequential_compatible():
            # Sequential mode: deterministic decomposition
            pool_index, pool_state = self._decompose_state(self.get_state())
            selected_pool = self.pools[pool_index]
            selected_pool.set_state(pool_state)
            return selected_pool.seq
        else:
            # Random mode: weighted selection using current state as seed
            # Use local RNG to avoid global state pollution
            rng = random.Random(self.get_state())
            
            # Select pool using weights
            selected_pool = rng.choices(self.pools, weights=self.probabilities)[0]
            
            # Sample state for selected pool
            if selected_pool.num_states == float('inf'):
                pool_state = rng.randint(0, 10**9)
            else:
                pool_state = rng.randint(0, int(selected_pool.num_states) - 1)
            
            selected_pool.set_state(pool_state)
            return selected_pool.seq
    
    def _decompose_state(self, state: int) -> tuple:
        """Decompose global state into (pool_index, state_within_pool).
        
        For sequential iteration, maps a global state index to which pool
        and which state within that pool.
        
        Args:
            state: Global state index
            
        Returns:
            Tuple of (pool_index, state_within_pool)
        """
        remaining = state
        for i, pool in enumerate(self.pools):
            if remaining < pool.num_states:
                return (i, remaining)
            remaining -= pool.num_states
        
        # If state exceeds total, wrap around
        total_states = sum(pool.num_states for pool in self.pools)
        wrapped_state = state % total_states
        return self._decompose_state(wrapped_state)
    
    def __repr__(self) -> str:
        weights_info = "" if all(w == self.weights[0] for w in self.weights) else f", weights={self.weights}"
        return f"MixedPool({len(self.pools)} pools{weights_info})"
    
    # =========================================================================
    # Design Cards Methods
    # =========================================================================
    
    def _get_selected_child_index(self) -> int:
        """Get the index of the currently selected child pool.
        
        In sequential mode, uses state decomposition.
        In random mode, uses the same RNG logic as _compute_seq.
        
        Returns:
            Index of the selected child pool (0-based)
        """
        if self.mode == 'sequential' and self.is_sequential_compatible():
            pool_index, _ = self._decompose_state(self.get_state())
            return pool_index
        else:
            # Random mode: use same RNG as _compute_seq for consistency
            rng = random.Random(self.get_state())
            # Find which pool was selected
            selected_pool = rng.choices(self.pools, weights=self.probabilities)[0]
            return self.pools.index(selected_pool)
    
    def get_metadata(self, abs_start: int, abs_end: int) -> Dict[str, Any]:
        """Return metadata for this MixedPool at the current state.
        
        Extends base Pool metadata with selection information.
        
        Metadata levels:
            - 'core': index, abs_start, abs_end only
            - 'features': core + selected, selected_name (default)
            - 'complete': features + value
        
        Args:
            abs_start: Absolute start position in the final sequence
            abs_end: Absolute end position in the final sequence
            
        Returns:
            Dictionary with metadata fields based on metadata level.
        """
        # Get base metadata (handles core fields and 'complete' level value)
        metadata = super().get_metadata(abs_start, abs_end)
        
        # Add MixedPool-specific fields for 'features' and 'complete' levels
        if self._metadata_level in ('features', 'complete'):
            selected_index = self._get_selected_child_index()
            selected_pool = self.pools[selected_index]
            
            metadata['selected'] = selected_index
            metadata['selected_name'] = selected_pool.name if selected_pool.name else None
        
        return metadata

