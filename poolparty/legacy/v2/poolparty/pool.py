from .types import Operation_type, Pool_type, Optional, Union, Sequence, beartype
from .operation import Operation
import numpy as np
import pandas as pd
import random


#########################################################################
# OutputSelectorOp - Selects a specific output from a multi-output operation
#########################################################################

@beartype
class OutputSelectorOp(Operation):
    """Lightweight operation that selects a specific output from a multi-output source.
    
    This operation wraps a multi-output operation and extracts a specific output
    column (seq_0, seq_1, etc.) as its primary sequence output.
    
    The OutputSelectorOp includes the source operation in its DAG by storing a reference
    and specially handling run() to ensure proper traversal.
    """
    num_outputs = 1
    design_card_keys: Sequence[str] = ['seq']
    
    def __init__(
        self,
        source_op: Operation,
        output_index: int,
        name: Optional[str] = None,
    ):
        """Initialize OutputSelectorOp.
        
        Args:
            source_op: The multi-output operation to select from
            output_index: Which output to select (0, 1, 2, ...)
            name: Optional name for this operation
        """
        if output_index < 0 or output_index >= source_op.num_outputs:
            raise ValueError(
                f"output_index {output_index} out of range [0, {source_op.num_outputs})"
            )
        
        self._source_op = source_op
        self._output_index = output_index
        
        super().__init__(
            parent_pools=[],  # No direct DAG parents - handled via _source_op
            num_states=source_op.num_states,
            mode=source_op.mode,
            seq_length=None,  # Variable per segment
            name=name or f'{source_op.name}_out{output_index}',
        )
    
    def run(self, num_seqs: int) -> list[str]:
        """Run source operation and extract this output's column."""
        # Get input sequences from source operation's parent pools
        input_seq_lists = []
        for parent in self._source_op.parent_pools:
            parent_seqs = parent.operation.run(num_seqs)
            input_seq_lists.append(parent_seqs)
        
        # Get states for the source operation
        if self._source_op.mode == 'sequential':
            states = [int(s) for s in self._source_op.states]
        else:
            states = [0] * num_seqs
        
        # Compute results on the source operation
        self._source_op.compute_results(input_seq_lists, states)
        
        # Extract the selected output column
        col = f'seq_{self._output_index}'
        seqs = self._source_op._results_df[col].tolist()
        
        # Store in our own results
        self._results_df = pd.DataFrame({'seq': seqs})
        return seqs
    
    def compute_results_row(
        self, 
        input_strings: Sequence[str], 
        sequential_state: int,
    ) -> dict:
        """Not used - run() handles everything directly."""
        raise NotImplementedError("OutputSelectorOp uses run() directly")


#########################################################################
# MultiPool - Container for outputs from a multi-output operation
#########################################################################

@beartype
class MultiPool:
    """Container for outputs from a multi-output operation.
    
    Provides iteration, indexing, and unpacking of individual output Pools.
    Each output Pool wraps an OutputSelectorOp that extracts the specific
    output column from the source operation.
    
    Example:
        >>> multi = breakpoint_scan("ACGTACGT", num_breakpoints=2)
        >>> left, middle, right = multi  # Unpack
        >>> for segment in multi:  # Iterate
        ...     print(segment.seq)
        >>> print(multi[1].seq)  # Index access
    """
    
    def __init__(self, operation: Operation):
        """Initialize MultiPool.
        
        Args:
            operation: A multi-output operation (num_outputs > 1)
        
        Raises:
            ValueError: If operation.num_outputs <= 1
        """
        if operation.num_outputs <= 1:
            raise ValueError(
                f"MultiPool requires num_outputs > 1, got {operation.num_outputs}"
            )
        self._operation = operation
        
        # Create a Pool for each output
        self._pools: list[Pool] = []
        for i in range(operation.num_outputs):
            selector_op = OutputSelectorOp(operation, output_index=i)
            self._pools.append(Pool(selector_op))
    
    def __getitem__(self, i: int) -> 'Pool':
        """Get the Pool for output i."""
        if i < 0:
            i = len(self._pools) + i  # Support negative indexing
        if i < 0 or i >= len(self._pools):
            raise IndexError(f"Output index {i} out of range [0, {len(self._pools)})")
        return self._pools[i]
    
    def __iter__(self):
        """Iterate over output Pools."""
        return iter(self._pools)
    
    def __len__(self) -> int:
        """Number of outputs."""
        return len(self._pools)
    
    @property
    def operation(self) -> Operation:
        """The underlying multi-output operation."""
        return self._operation
    
    @property
    def num_outputs(self) -> int:
        """Number of output pools."""
        return self._operation.num_outputs
    
    def __repr__(self) -> str:
        return f"MultiPool(op={self._operation!r}, num_outputs={self.num_outputs})"


#########################################################################
# Pool - Main pool class
#########################################################################

@beartype
class Pool:
    """A pool of oligonucleotide sequences with eager evaluation."""
    
    #########################################################
    # Constructor 
    #########################################################
    
    def __init__(self, operation: Operation_type) -> None:
        self.operation = operation      # Set operation
        self.gather_op_info()           # Inspect DAG
        self.rng = random.Random()      # Create RNG
        self.set_sequential_op_states(0)               # Set state
    
    def gather_op_info(self) -> None:

        # Recursively gather operations from DAG; store in op_dict
        op_dict: dict[int, Operation_type] = {}
        def collect_ops(pool: Pool):
            op = pool.operation
            if op.id in op_dict and op_dict[op.id] is not op:
                raise ValueError(f"Duplicate operation {op!r} found in DAG. ")
            op_dict[op.id] = op
            
            # Handle OutputSelectorOp specially - include source op and its parents
            if isinstance(op, OutputSelectorOp):
                source_op = op._source_op
                if source_op.id not in op_dict:
                    op_dict[source_op.id] = source_op
                    for parent in source_op.parent_pools:
                        collect_ops(parent)
            
            for parent in pool.operation.parent_pools:
                collect_ops(parent)                
        collect_ops(self)
        
        # Sort, store, cache, and inspect operations
        self.all_ops = sorted(op_dict.values(), key=lambda op: op.id)
        self.sequential_ops = [op for op in self.all_ops if op.mode == 'sequential']
        self.random_ops = [op for op in self.all_ops if op.mode == 'random']
        self.fixed_ops = [op for op in self.all_ops if op.mode == 'fixed']
        self.num_sequential_states = int(np.prod([op.num_states for op in self.sequential_ops]))
        self.seq_length = self.operation.seq_length
    
    def set_state(self, state: int) -> None:
        """Set the state for all operations in the DAG."""
        self.state = state
        self.set_sequential_op_states(state=state)
        self.set_random_op_seeds(seed=state)
    
    def set_random_op_seeds(self, seed: int = 0) -> None:
        """Set the seeds for all random operations in the DAG."""
        for op in self.random_ops:
            op.initialize_rng(seed)
    
    def set_sequential_op_states(self, state: int) -> None:
        """Set the state for all sequential operations in the DAG."""
        if not state >= 0:
            raise ValueError(f"state must be nonnegative; got {state}")
        self.state = state
        for op in self.sequential_ops:
            op.state = state % op.num_states
            state //= op.num_states
            
    def set_all_sequential_op_states(self, init_state:int, num_seqs: int):
        if not init_state >= 0:
            raise ValueError(f"state must be nonnegative; got {init_state}")
        if not num_seqs >= 0:
            raise ValueError(f"state must be nonnegative; got {num_seqs}")
        states = np.arange(init_state, init_state+num_seqs, dtype=int)
        for op in self.sequential_ops:
            op.states = states % op.num_states
            states //= op.num_states
    
    #########################################################################
    # Results methods
    #########################################################################
    
    def _clear_all_results(self) -> None:
        """Clear results DataFrames on all operations in the DAG."""
        for op in self.all_ops:
            op.clear_results()
    
    def _collate_results(self, suppress_multiple_ops_in_keys=False) -> pd.DataFrame:
        """Collect _results_df from all ops in the DAG.
        
        Concatenates DataFrames from all operations, adding prefixes to columns.
        Prefixes: {ClassName}({id}):{column}
        e.g., from_seqs(0):seq_name, mutation_scan(3):positions
        
        Returns:
            pd.DataFrame: DataFrame with prefixed results from all operations
        """
        result_df = pd.DataFrame()
        for op in self.all_ops:
            if op._results_df is not None:
                df = op.get_results()  # filtered by active_design_card_keys
                
                col_rename_dict = {}
                for col in df.columns:
                    if suppress_multiple_ops_in_keys and len(col)>2 and col[:2]=='op':
                        prefix = ''
                    else:
                        prefix = f"op{op.id}({op.name})."
                    col_rename_dict[col] = f"{prefix}{col}"
                df = df.rename(columns=col_rename_dict)
                result_df = pd.concat([result_df, df], axis=1)
        return result_df
    
        
    #########################################################################
    # Sequence generation
    #########################################################################
    
    def generate_library(
        self,
        num_seqs: Optional[int] = None,
        num_complete_iterations: Optional[int] = None,
        init_state: Optional[int] = None,  
        seed: Optional[int] = None,
        advance: bool = True,
    ) -> pd.DataFrame:
        """Generate a library of sequences from this Pool.
        
        Args:
            num_seqs: Number of sequences to generate
            num_complete_iterations: Number of complete iterations through sequential operations
            init_state: Initial state to use for sequential operations
            seed: Optional seed for random generation
            advance: If True, advance the state after generation
        
        Returns:
            pd.DataFrame: DataFrame with 'seq' as first column, followed by design card columns.
        """
        
        # Inspect DAG
        self.gather_op_info()
        
        # Determine number of sequences to generate
        match num_seqs, num_complete_iterations:
            case (None, None):
                raise ValueError("Must specify either num_seqs or num_complete_iterations")
            case (_, None):
                pass
            case (None, _):
                num_seqs = num_complete_iterations * self.num_sequential_states
            case (_, _):
                raise ValueError("Cannot specify both num_seqs and num_complete_iterations")
        
        # Clear all results before generation
        self._clear_all_results()
        
        # Seed random operations if provided
        if seed is not None:
            self.set_random_op_seeds(seed)
        
        # Set states for sequential ops
        if init_state is None:
            init_state = self.state
        self.set_all_sequential_op_states(init_state, num_seqs)
        
        # Recursively compute all sequences in one DAG traversal
        seqs = self.operation.run(num_seqs)
        
        # Advance state
        if advance:
            self.state += num_seqs
        
        # Gather results from all operations
        card_df = self._collate_results(suppress_multiple_ops_in_keys=True)
        
        # Create result DataFrame with 'seq' as first column
        result_df = pd.DataFrame({'seq': seqs})
        result_df = pd.concat([result_df, card_df], axis=1)
        return result_df
    
    def generate_sequence(self, 
                          state: Optional[int] = None, 
                          seed: Optional[int] = None, 
                          advance: bool = True) -> pd.DataFrame:
        """Generates a single sequence; wrapper for Pool.generate_library()
        
        Returns:
            pd.DataFrame: Single-row DataFrame with 'seq' as first column, followed by design card columns.
        """
        return self.generate_library(num_seqs=1, init_state=state, seed=seed, advance=advance)
    
    @property
    def seq(self) -> str:
        df = self.generate_sequence(advance=False)
        return df['seq'].iloc[0]
    
    #########################################################################
    # Dunder methods
    #########################################################################
    
    def __str__(self) -> str:
        return self.seq
    
    def __repr__(self) -> str:
        return f"Pool(op={self.operation!r})"
    
    def __len__(self) -> int:
        return self.operation.seq_length
    
    def __add__(self, other: Union[Pool_type, str]) -> Pool_type:
        from .operations.concatenate_op import concatenate
        return concatenate([self, other], name='+')
    
    def __radd__(self, other: Union[Pool_type, str]) -> Pool_type:
        from .operations.concatenate_op import concatenate
        return concatenate([other, self], name='+')
    
    def __mul__(self, n: int) -> Pool_type:
        from .operations.repeat_op import repeat
        return repeat(parent=self, n=n, name='*')
    
    def __rmul__(self, n: int) -> Pool_type:
        return self.__mul__(n)
    
    def __getitem__(self, key: Union[int, slice]) -> Pool_type:
        from .operations.slice_op import subseq
        return subseq(parent=self, key=key, name='slice')


    