"""Region class for poolparty - represents a registered region with its properties."""
from dataclasses import dataclass, field
from poolparty.types import Optional


@dataclass
class Region:
    """
    Represents a registered region in a poolparty Party.
    
    Regions identify sections of sequences for later modification. Each region
    has a name and a seq_length that specifies the expected length of content
    within the region tags.
    
    Attributes
    ----------
    name : str
        The region name (used in XML tags like <name>...</name>).
    seq_length : Optional[int]
        The expected length of content within the region:
        - None: Variable-length region (content length not fixed)
        - 0: Zero-length region (insertion point, <name/>)
        - >0: Fixed-length region (content must be this length)
    _id : int
        Unique identifier assigned by the Party upon registration.
    
    Examples
    --------
    Regions are typically created through region operations, not directly:
    
    >>> with pp.Party() as party:
    ...     # insert_tags registers a region with the party
    ...     pool = pp.insert_tags(bg, 'orf', start=10, stop=100)
    ...     
    ...     # Retrieve the registered region
    ...     region = party.get_region_by_name('orf')
    ...     print(region.seq_length)  # 90
    """
    name: str
    seq_length: Optional[int]  # None for variable-length, 0 for zero-length, >0 for fixed
    _id: int = field(default=-1, repr=False)
    
    def __post_init__(self):
        """Validate region attributes."""
        if not self.name:
            raise ValueError("Region name cannot be empty")
        if not self.name.isidentifier():
            raise ValueError(
                f"Region name '{self.name}' is not a valid identifier. "
                "Use only letters, numbers, and underscores, starting with a letter."
            )
        if self.seq_length is not None and self.seq_length < 0:
            raise ValueError(f"seq_length must be None or >= 0, got {self.seq_length}")
    
    @property
    def is_variable_length(self) -> bool:
        """True if this region has variable length (seq_length is None)."""
        return self.seq_length is None
    
    @property
    def is_zero_length(self) -> bool:
        """True if this region is a zero-length insertion point."""
        return self.seq_length == 0
    
    def __hash__(self):
        """Hash based on name (regions with same name should be the same)."""
        return hash(self.name)
    
    def __eq__(self, other):
        """Equality based on name."""
        if isinstance(other, Region):
            return self.name == other.name
        return False
