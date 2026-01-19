"""Operation - Abstract base class for state operations."""
from abc import ABC, abstractmethod


class Operation(ABC):
    """Base class for state operations."""
    
    @abstractmethod
    def compute_num_states(self, parent_num_values: tuple[int, ...]) -> int:
        """Compute num_values for this node given parent num_values."""
        pass
    
    @abstractmethod
    def decompose(self, value: int, parent_num_values: tuple[int, ...]) -> tuple[int, ...]:
        """Decompose this node's value into parent values."""
        pass
