"""Marker class for poolparty - represents a registered marker with its properties."""
from dataclasses import dataclass, field
from poolparty.types import Optional


@dataclass
class Marker:
    """
    Represents a registered marker in a poolparty Party.
    
    Markers identify regions of sequences for later modification. Each marker
    has a name and a seq_length that specifies the expected length of content
    within the marker tags.
    
    Attributes
    ----------
    name : str
        The marker name (used in XML tags like <name>...</name>).
    seq_length : Optional[int]
        The expected length of content within the marker:
        - None: Variable-length marker (content length not fixed)
        - 0: Zero-length marker (insertion point, <name/>)
        - >0: Fixed-length marker (content must be this length)
    _id : int
        Unique identifier assigned by the Party upon registration.
    
    Examples
    --------
    Markers are typically created through marker operations, not directly:
    
    >>> with pp.Party() as party:
    ...     # insert_marker registers a marker with the party
    ...     pool = pp.insert_marker(bg, 'orf', start=10, stop=100)
    ...     
    ...     # Retrieve the registered marker
    ...     marker = party.get_marker_by_name('orf')
    ...     print(marker.seq_length)  # 90
    """
    name: str
    seq_length: Optional[int]  # None for variable-length, 0 for zero-length, >0 for fixed
    _id: int = field(default=-1, repr=False)
    
    def __post_init__(self):
        """Validate marker attributes."""
        if not self.name:
            raise ValueError("Marker name cannot be empty")
        if not self.name.isidentifier():
            raise ValueError(
                f"Marker name '{self.name}' is not a valid identifier. "
                "Use only letters, numbers, and underscores, starting with a letter."
            )
        if self.seq_length is not None and self.seq_length < 0:
            raise ValueError(f"seq_length must be None or >= 0, got {self.seq_length}")
    
    @property
    def is_variable_length(self) -> bool:
        """True if this marker has variable length (seq_length is None)."""
        return self.seq_length is None
    
    @property
    def is_zero_length(self) -> bool:
        """True if this marker is a zero-length insertion point."""
        return self.seq_length == 0
    
    def __hash__(self):
        """Hash based on name (markers with same name should be the same)."""
        return hash(self.name)
    
    def __eq__(self, other):
        """Equality based on name."""
        if isinstance(other, Marker):
            return self.name == other.name
        return False
