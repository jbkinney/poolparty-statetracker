from ..types import Union, Optional, Literal, Sequence, Real, ModeType, beartype
from ..operation import Operation
from ..pool import Pool
import numpy as np
import pandas as pd


@beartype
class MixOp(Operation):
    """Mix sequences from multiple pre-generated pools."""
    design_card_keys = ['pool_idx', 'pool_name', 'seq_idx']
  
    #########################################################
    # Constructor
    #########################################################
  
    def __init__(
        self,
        pools: Sequence[Pool],
        num_seqs_by_pool: Sequence[int],
        pool_names: Optional[Sequence[str]] = None,
        parent_pool_probs: Optional[Sequence[Real]] = None,
        mode: ModeType = 'random',
        name: Optional[str] = None,
        design_card_keys: Optional[Sequence[str]] = None,
    ) -> None:
        
        # Validate pools and num_seqs_by_pool
        if len(pools) == 0:
            raise ValueError("pools must be non-empty")
        if len(pools) != len(num_seqs_by_pool):
            raise ValueError(f"{len(pools)=} and {len(num_seqs_by_pool)=} don't match.")
        for i, n in enumerate(num_seqs_by_pool):
            if n <= 0:
                raise ValueError(f"num_seqs_by_pool[{i}] must be positive; got {n}")
        
        # Set pool_names
        if pool_names is not None:
            if len(pool_names) != len(pools):
                raise ValueError(f"{len(pools)=} and {len(pools)=} don't match.")
            self.pool_names = list(pool_names)
        else:
            self.pool_names = [f"pool_{i}" for i in range(len(pools))]
        
        # Generate sequences from each pool and build master DataFrame
        dfs = []
        pool_boundaries = [0]
        for i, (pool, num_seqs) in enumerate(zip(pools, num_seqs_by_pool)):
            # Generate library from this pool
            df = pool.generate_library(num_seqs=num_seqs)
            
            # Add source tracking columns
            df['pool_idx'] = i
            df['pool_name'] = self.pool_names[i]
            df['source_seq_index'] = np.arange(len(df))
            
            dfs.append(df)
            pool_boundaries.append(pool_boundaries[-1] + len(df))
        
        # Concatenate all DataFrames and reset index
        self._master_results_df = pd.concat(dfs, ignore_index=True)
        self._pool_boundaries = pool_boundaries
        self._num_pools = len(pools)
        
        # Update design card keys
        self.design_card_keys = list(self._master_results_df.columns)
        
        # Validate and normalize parent_pool_probs
        if parent_pool_probs is not None:
            if len(parent_pool_probs) != len(pools):
                raise ValueError(
                    f"parent_pool_probs must have same length as pools; "
                    f"got {len(parent_pool_probs)} and {len(pools)}"
                )
            arr = np.array(parent_pool_probs, dtype=float)
            if not np.all(np.isfinite(arr)):
                raise ValueError("parent_pool_probs has non-finite values")
            if np.any(arr < 0):
                raise ValueError("parent_pool_probs has negative values")
            if np.sum(arr) == 0:
                raise ValueError("parent_pool_probs sums to zero")
            self.parent_pool_probs = arr / np.sum(arr)
        else:
            self.parent_pool_probs = None
        
        # Compute seq_length: fixed if all same length, None if variable
        seq_lengths = set(len(s) for s in self._master_results_df['seq'])
        seq_length = seq_lengths.pop() if len(seq_lengths) == 1 else None
    
        # Initialize base class attributes
        super().__init__(
            parent_pools=[],
            num_states=len(self._master_results_df),
            mode=mode,
            seq_length=seq_length,
            name=name,
            design_card_keys=self.design_card_keys,
        )
    
    #########################################################
    # Results computation
    #########################################################
    
    def compute_results_row(
        self, 
        input_strings: Sequence[str], 
        sequential_state: int
    ) -> dict:
        """Return a dict with seq and design card data for one sequence."""
        total_seqs = len(self._master_results_df)
        
        if self.mode == 'sequential':
            index = sequential_state % total_seqs
        elif self.mode == 'random':
            if self.parent_pool_probs is not None:
                # First select a pool according to parent_pool_probs
                pool_idx = self.rng.choice(self._num_pools, p=self.parent_pool_probs)
                # Then uniformly select a sequence within that pool's range
                start = self._pool_boundaries[pool_idx]
                end = self._pool_boundaries[pool_idx + 1]
                index = self.rng.integers(start, end)
            else:
                # Uniform selection over all sequences
                index = self.rng.integers(0, total_seqs)
        else:
            raise ValueError(f"{self.mode=} is not 'sequential' or 'random'.")
        
        # Get row from master DataFrame
        row = self._master_results_df.iloc[index]
        return row.to_dict()


#########################################################
# Public factory function
#########################################################

@beartype
def mix(
    pools: Sequence[Pool],
    num_seqs_by_pool: Sequence[int],
    pool_names: Optional[Sequence[str]] = None,
    parent_pool_probs: Optional[Sequence[Real]] = None,
    mode: ModeType = 'random',
    name: str = 'mix',
    design_card_keys: Optional[Sequence[str]] = None,
) -> Pool:
    """Create a Pool by mixing sequences from multiple pre-generated pools.
    
    Args:
        pools: List of Pool objects to mix
        num_seqs_by_pool: Number of sequences to generate from each pool
        pool_names: Optional names for source pools (defaults to pool_0, pool_1, etc.)
        parent_pool_probs: Optional probabilities for selecting from each pool in random mode
        mode: 'random' or 'sequential'
        name: Name for this operation
        design_card_keys: Which design card keys to include
    
    Returns:
        A Pool that serves sequences from the mixed pools
    """
    return Pool(
        operation=MixOp(
            pools=pools,
            num_seqs_by_pool=num_seqs_by_pool,
            pool_names=pool_names,
            parent_pool_probs=parent_pool_probs,
            mode=mode,
            name=name,
            design_card_keys=design_card_keys,
        ),
    )
