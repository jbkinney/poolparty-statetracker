"""Pool class for poolparty.

Pools are the nodes in the computation DAG. Each Pool references:
- An Operation that produces it
- An output_index indicating which output of the Operation it represents
"""

from dataclasses import dataclass

from .types import Union, beartype


@dataclass
class Pool:
    """A node in the computation DAG.
    
    Each Pool is the output of exactly one Operation. For multi-output
    operations, each output has its own Pool with a different output_index.
    
    Attributes:
        operation: The Operation that produces this pool
        output_index: Which output of the operation (0 for single-output ops)
        name: Optional name for this pool
    """
    operation: "Operation"
    output_index: int = 0
    name: str | None = None
    
    @property
    def parents(self) -> list["Pool"]:
        """Get parent pools from the operation."""
        return self.operation.parent_pools
    
    # --- Composition operators ---
    
    @beartype
    def __add__(self, other: Union["Pool", str]) -> "Pool":
        """Concatenate this pool with another pool or string."""
        from .operations.concatenate import concatenate
        return concatenate([self, other])
    
    @beartype
    def __radd__(self, other: str) -> "Pool":
        """Concatenate a string with this pool."""
        from .operations.concatenate import concatenate
        return concatenate([other, self])
    
    @beartype
    def __mul__(self, n: int) -> "Pool":
        """Repeat this pool n times."""
        from .operations.concatenate import concatenate
        return concatenate([self] * n)
    
    @beartype
    def __rmul__(self, n: int) -> "Pool":
        """Repeat this pool n times."""
        return self.__mul__(n)
    
    @beartype
    def __getitem__(self, key: Union[int, slice]) -> "Pool":
        """Slice this pool's sequences.
        
        Args:
            key: Integer index or slice object
        
        Returns:
            Pool with sliced sequences
        
        Example:
            >>> pool = from_seqs(['ACGTACGT'])
            >>> first_half = pool[0:4]  # 'ACGT'
            >>> last_char = pool[-1]  # 'T'
        """
        from .operations.slice_op import subseq
        return subseq(self, key)
    
    def __repr__(self) -> str:
        name_str = f", name={self.name!r}" if self.name else ""
        if self.operation.num_outputs > 1:
            return f"Pool(op={self.operation.name}, out={self.output_index}{name_str})"
        return f"Pool(op={self.operation.name}{name_str})"
