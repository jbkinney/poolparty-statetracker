"""Operation - Abstract base class for counter operations."""
from abc import ABC, abstractmethod


class Operation(ABC):
    """Base class for counter operations."""
    
    @abstractmethod
    def compute_num_states(self, parent_num_states: tuple[int, ...]) -> int:
        """Compute num_states for this node given parent num_states."""
        pass
    
    @abstractmethod
    def decompose(self, state: int, parent_num_states: tuple[int, ...]) -> tuple[int, ...]:
        """Decompose this node's state into parent states."""
        pass
