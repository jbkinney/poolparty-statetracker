"""Seq class for bundling sequence string, style, and name."""
from dataclasses import dataclass
from ..types import SeqStyle, Optional, Sequence, beartype


@beartype
@dataclass(frozen=True)
class Seq:
    """Immutable container bundling DNA sequence string, style, and name.
    
    This class provides atomic operations for slicing, joining, and manipulating
    sequences while maintaining their associated styling and naming information.
    """
    string: str                    # DNA sequence string
    style: SeqStyle                # Per-position styling
    name: Optional[str] = None     # Sequence name
    
    def __len__(self) -> int:
        """Return length of sequence string."""
        return len(self.string)
    
    def __repr__(self) -> str:
        """String representation."""
        name_str = f", name={self.name!r}" if self.name else ""
        return f"Seq(len={len(self.string)}, styles={len(self.style.style_list)}{name_str})"
    
    def __getitem__(self, key: slice) -> 'Seq':
        """Atomic slicing of string + style.
        
        Parameters
        ----------
        key : slice
            Slice to extract from sequence.
        
        Returns
        -------
        Seq
            New Seq with sliced string and style.
        
        Examples
        --------
        >>> seq = Seq.from_string('ACGTACGT')
        >>> seq[2:6]
        Seq(len=4, styles=0)
        """
        if not isinstance(key, slice):
            raise TypeError("Seq only supports slice indexing")
        return Seq(
            string=self.string[key],
            style=self.style[key],
            name=self.name,
        )
    
    @classmethod
    def join(cls, input_seqs: Sequence['Seq'], sep: str = '') -> 'Seq':
        """Join multiple Seq with optional separator.
        
        Handles string join, SeqStyle.join with automatic offset handling,
        and name merging (dot-separated).
        
        Parameters
        ----------
        input_seqs : Sequence[Seq]
            Sequence of Seq objects to join.
        sep : str, default=''
            Separator string to insert between sequences.
        
        Returns
        -------
        Seq
            New Seq with joined strings, styles, and names.
        
        Examples
        --------
        >>> a = Seq.from_string('ACG', name='a')
        >>> b = Seq.from_string('TGC', name='b')
        >>> Seq.join([a, b], sep='N')
        Seq(len=7, styles=0, name='a.b')
        """
        if not input_seqs:
            return cls.empty()
        
        strings = []
        styles = []
        names = []
        
        for i, s in enumerate(input_seqs):
            if i > 0 and sep:
                strings.append(sep)
                styles.append(SeqStyle.empty(len(sep)))
            strings.append(s.string)
            styles.append(s.style)
            if s.name:
                names.append(s.name)
        
        return cls(
            string=''.join(strings),
            style=SeqStyle.join(styles),
            name='.'.join(names) if names else None,
        )
    
    @classmethod
    def empty(cls) -> 'Seq':
        """Create empty Seq.
        
        Returns
        -------
        Seq
            Empty Seq with no string, style, or name.
        """
        return cls('', SeqStyle.empty(0), None)
    
    @classmethod
    def from_string(cls, string: str, style: SeqStyle | None = None, name: str | None = None) -> 'Seq':
        """Create Seq from string with optional style/name.
        
        Parameters
        ----------
        string : str
            DNA sequence string.
        style : SeqStyle | None, default=None
            Optional style. If None, creates empty style.
        name : str | None, default=None
            Optional sequence name.
        
        Returns
        -------
        Seq
            New Seq with given string, style, and name.
        """
        if style is None:
            style = SeqStyle.empty(len(string))
        return cls(string, style, name)
    
    def insert(self, pos: int, content: 'Seq') -> 'Seq':
        """Insert content at position.
        
        Parameters
        ----------
        pos : int
            Position to insert at (0-indexed).
        content : Seq
            Seq to insert.
        
        Returns
        -------
        Seq
            New Seq with content inserted.
        """
        return Seq.join([self[:pos], content, self[pos:]])
    
    def reversed(self, do_reverse: bool = True) -> 'Seq':
        """Reverse sequence and mirror style positions.
        
        Parameters
        ----------
        do_reverse : bool, default=True
            If False, returns self unchanged (convenient for conditional reversal).
        
        Returns
        -------
        Seq
            New Seq with reversed string and mirrored style positions.
        """
        if not do_reverse:
            return self
        
        from ..utils import dna_utils
        reversed_string = dna_utils.reverse_complement(self.string)
        reversed_style = self.style.reversed(do_reverse=True)
        
        return Seq(reversed_string, reversed_style, self.name)
    
    def with_name(self, name: str | None) -> 'Seq':
        """Return copy with updated name.
        
        Parameters
        ----------
        name : str | None
            New name for the sequence.
        
        Returns
        -------
        Seq
            New Seq with updated name.
        """
        return Seq(self.string, self.style, name)
    
    def with_style(self, style: SeqStyle) -> 'Seq':
        """Return copy with updated style.
        
        Parameters
        ----------
        style : SeqStyle
            New style for the sequence.
        
        Returns
        -------
        Seq
            New Seq with updated style.
        """
        return Seq(self.string, style, self.name)
    
    def add_style(self, style_spec: str, positions) -> 'Seq':
        """Return copy with additional style added.
        
        Parameters
        ----------
        style_spec : str
            Style specification (e.g., 'red', 'bold', '#ff0000').
        positions : np.ndarray
            Positions to apply style.
        
        Returns
        -------
        Seq
            New Seq with added style.
        """
        new_style = self.style.add_style(style_spec, positions)
        return Seq(self.string, new_style, self.name)
    
    @classmethod
    def combine_names(cls, seqs: Sequence['Seq']) -> str | None:
        """Combine names from multiple Seq objects (for parent name inheritance).
        
        Parameters
        ----------
        seqs : Sequence[Seq]
            Sequence of Seq objects whose names to combine.
        
        Returns
        -------
        str | None
            Dot-separated combined names, or None if no names present.
        """
        names = [s.name for s in seqs if s.name is not None]
        return '.'.join(names) if names else None
