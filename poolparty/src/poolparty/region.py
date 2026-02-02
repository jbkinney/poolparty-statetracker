"""Region and OrfRegion classes for poolparty - represent registered regions with their properties."""

from dataclasses import dataclass, field

from poolparty.types import Optional

# Valid frame values for ORF regions
VALID_FRAMES = {-3, -2, -1, 1, 2, 3}


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class OrfRegion(Region):
    """
    Represents an ORF (Open Reading Frame) region with associated reading frame.

    Extends Region with frame information for ORF-aware operations like
    stylize_orf() and mutagenize_orf().

    Attributes
    ----------
    frame : int
        Reading frame and orientation. Valid values: +1, +2, +3, -1, -2, -3.
        Positive values indicate forward orientation (5'->3'),
        negative values indicate reverse orientation (3'->5').
        The absolute value indicates the frame offset (1-indexed).
    """

    frame: int = 1

    def __post_init__(self):
        """Validate OrfRegion attributes."""
        # Call parent validation
        super().__post_init__()
        # Validate frame
        if self.frame not in VALID_FRAMES:
            raise ValueError(f"frame must be one of {sorted(VALID_FRAMES)}, got {self.frame}")
