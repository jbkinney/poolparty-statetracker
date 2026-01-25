"""Replace marker content with sequences from another Pool."""
from numbers import Real
from poolparty.types import Optional
import numpy as np

from .parsing import validate_single_marker
from ..operation import Operation
from .. import dna


def replace_marker_content(
    bg_pool,
    content_pool,
    marker_name: str,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
    # Internal parameters for insertion_scan composite naming
    _seq_name_prefix: Optional[str] = None,
    _seq_name_pos_prefix: Optional[str] = None,
    _seq_name_site_prefix: Optional[str] = None,
    _pos_state=None,  # State object for position naming
    _site_state=None,  # State object for site naming
    _num_sites: Optional[int] = None,
    _style_insertion: Optional[str] = None,
    _style_background: Optional[str] = None,
    _outer_region: Optional[str] = None,  # Outer region for style_background targeting
):
    """
    Replace a marker region with content from another Pool.

    The marker (including its tags and any content) is replaced with
    sequences from content_pool. If the marker has strand='-', the
    content is reverse-complemented before insertion.

    Parameters
    ----------
    bg_pool : Pool or str
        Background Pool or sequence string containing the marker.
    content_pool : Pool or str
        Pool or sequence string to insert at the marker position.
    marker_name : str
        Name of the marker to replace.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    _factory_name: Optional[str], default=None
        Sets default name of the resulting operation

    Returns
    -------
    Pool
        A Pool yielding bg_pool sequences with the marker replaced by
        content_pool sequences.

    Examples
    --------
    >>> with pp.Party():
    ...     # Replace marker with content from another pool
    ...     bg = pp.from_seq('ACGT<insert/>TTTT')
    ...     inserts = pp.from_seqs(['AAA', 'GGG'], mode='sequential')
    ...     result = pp.replace_marker_content(bg, inserts, 'insert')
    ...     # Result yields: 'ACGTAAATTTT', 'ACGTGGGTTTT'
    ...
    ...     # With strand='-', content is reverse-complemented
    ...     bg = pp.from_seq("ACGT<region strand='-'>XX</region>TTTT")
    ...     content = pp.from_seq('AAA')
    ...     result = pp.replace_marker_content(bg, content, 'region')
    ...     # Result: 'ACGTTTTTTTT' (TTT is reverse complement of AAA)
    """
    from ..fixed_ops.from_seq import from_seq
    from ..pool import Pool
    
    # Convert strings to pools if needed
    bg_pool = from_seq(bg_pool) if isinstance(bg_pool, str) else bg_pool
    content_pool = from_seq(content_pool) if isinstance(content_pool, str) else content_pool
    
    op = ReplaceMarkerContentOp(
        bg_pool=bg_pool,
        content_pool=content_pool,
        marker_name=marker_name,
        name=None,
        iter_order=iter_order,
        _factory_name=_factory_name,
        _seq_name_prefix=_seq_name_prefix,
        _seq_name_pos_prefix=_seq_name_pos_prefix,
        _seq_name_site_prefix=_seq_name_site_prefix,
        _pos_state=_pos_state,
        _site_state=_site_state,
        _num_sites=_num_sites,
        _style_insertion=_style_insertion,
        _style_background=_style_background,
        _outer_region=_outer_region,
    )
    result_pool = Pool(operation=op)
    
    # The marker is replaced, so remove it from the pool's marker set
    result_pool._untrack_marker(marker_name)
    
    return result_pool


class ReplaceMarkerContentOp(Operation):
    """Replace a marker region with content from another pool."""
    
    factory_name = "replace_marker_content"
    design_card_keys = []
    
    def __init__(
        self,
        bg_pool,
        content_pool,
        marker_name: str,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
        # Internal parameters for insertion_scan composite naming
        _seq_name_prefix: Optional[str] = None,
        _seq_name_pos_prefix: Optional[str] = None,
        _seq_name_site_prefix: Optional[str] = None,
        _pos_state=None,  # State object for position naming
        _site_state=None,  # State object for site naming
        _num_sites: Optional[int] = None,
        _style_insertion: Optional[str] = None,
        _style_background: Optional[str] = None,
        _outer_region: Optional[str] = None,  # Outer region for style_background targeting
    ) -> None:
        self.marker_name = marker_name
        
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name
        
        # Store naming parameters for insertion_scan composite naming
        self._seq_name_prefix = _seq_name_prefix
        self._seq_name_pos_prefix = _seq_name_pos_prefix
        self._seq_name_site_prefix = _seq_name_site_prefix
        self._pos_state = _pos_state
        self._site_state = _site_state
        self._num_sites = _num_sites
        self._insertion_naming = any([_seq_name_prefix, _seq_name_pos_prefix, _seq_name_site_prefix])
        self._style_insertion = _style_insertion
        self._style_background = _style_background
        self._outer_region = _outer_region
        
        # The operation itself has num_values=1 because it doesn't add its own states.
        # The total number of output states comes from the product of parent pool counters.
        # When content_pool has multiple states (e.g., from mutagenize), those states
        # are inherited via the parent counter product.
        super().__init__(
            parent_pools=[bg_pool, content_pool],
            num_values=1,  # Operation doesn't add states
            mode='fixed',  # Mode is determined by parent counters
            seq_length=None,  # Variable length
            name=name,
            iter_order=iter_order,
        )

    
    def compute(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
        parent_styles: list | None = None,
    ) -> dict:
        """Replace marker in bg_seq with content_seq."""
        from ..types import StyleList
        
        bg_seq = parent_seqs[0]
        content_seq = parent_seqs[1]
        
        # Find and validate the marker
        marker = validate_single_marker(bg_seq, self.marker_name)
        
        # If strand='-', reverse complement the content before insertion
        if marker.strand == '-':
            content_seq = dna.reverse_complement(content_seq)
        
        # Build result: prefix + content + suffix
        prefix = bg_seq[:marker.start]
        suffix = bg_seq[marker.end:]
        result_seq = prefix + content_seq + suffix
        
        # Adjust styles from bg_pool (first parent) with position shifts
        # Styles within the marker region are discarded, suffix styles are shifted
        output_styles: StyleList = []
        
        if parent_styles and len(parent_styles) > 0:
            bg_styles = parent_styles[0]
            marker_region_len = marker.end - marker.start
            new_content_len = len(content_seq)
            length_delta = new_content_len - marker_region_len
            
            for spec, positions in bg_styles:
                adjusted_positions = []
                for pos in positions:
                    if pos < marker.start:
                        # Before marker: unchanged
                        adjusted_positions.append(pos)
                    elif pos >= marker.end:
                        # After marker region: shift by length change
                        adjusted_positions.append(pos + length_delta)
                    # Positions inside the marker region are discarded
                if adjusted_positions:
                    output_styles.append((spec, np.array(adjusted_positions, dtype=np.int64)))
        
        # Handle content_pool styles (second parent) - inserted content retains its styling
        if parent_styles and len(parent_styles) > 1:
            content_styles = parent_styles[1]
            original_content_len = len(parent_seqs[1])
            
            for spec, positions in content_styles:
                adjusted_positions = []
                for pos in positions:
                    if marker.strand == '-':
                        # Flip position for reverse complement
                        pos = original_content_len - 1 - pos
                    # Shift by prefix length
                    new_pos = marker.start + pos
                    adjusted_positions.append(new_pos)
                if adjusted_positions:
                    output_styles.append((spec, np.array(adjusted_positions, dtype=np.int64)))
        
        # Apply style_insertion to all inserted content positions
        original_content_len = len(parent_seqs[1])
        ins_start = marker.start
        ins_end = ins_start + original_content_len
        
        if self._style_insertion is not None:
            ins_positions = np.arange(ins_start, ins_end, dtype=np.int64)
            output_styles.append((self._style_insertion, ins_positions))
        
        # Apply style_background to non-inserted positions within outer region only
        if self._style_background is not None:
            if self._outer_region is not None and isinstance(self._outer_region, str):
                # Find outer region bounds in result_seq and only style within those bounds
                outer_marker = validate_single_marker(result_seq, self._outer_region)
                region_start = outer_marker.content_start
                region_end = outer_marker.content_end
            else:
                # No outer region specified - style entire sequence (excluding inserted)
                region_start = 0
                region_end = len(result_seq)
            
            inserted_positions = set(range(ins_start, ins_end))
            bg_positions = np.array([p for p in range(region_start, region_end) 
                                     if p not in inserted_positions], dtype=np.int64)
            if len(bg_positions) > 0:
                output_styles.append((self._style_background, bg_positions))
        
        return {'seq': result_seq, 'style': output_styles}
    
    def compute_seq_names(
        self,
        parent_names: list[Optional[str]],
        card: dict,
    ) -> Optional[str]:
        """Compute output sequence names with optional insertion_scan composite naming."""
        if not self._insertion_naming:
            return super().compute_seq_names(parent_names, card)
        
        # Get position and site indices from the state objects
        pos_idx = self._pos_state.value if self._pos_state is not None and self._pos_state.value is not None else 0
        site_idx = self._site_state.value if self._site_state is not None and self._site_state.value is not None else 0
        
        # Build name parts in order: product index, position index, site index
        name_parts = []
        if self._seq_name_prefix:
            w = pos_idx * self._num_sites + site_idx
            name_parts.append(f'{self._seq_name_prefix}{w}')
        if self._seq_name_pos_prefix:
            name_parts.append(f'{self._seq_name_pos_prefix}{pos_idx}')
        if self._seq_name_site_prefix:
            name_parts.append(f'{self._seq_name_site_prefix}{site_idx}')
        
        return '.'.join(name_parts) if name_parts else None
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'bg_pool': self.parent_pools[0],
            'content_pool': self.parent_pools[1],
            'marker_name': self.marker_name,
            'name': None,
            'iter_order': self.iter_order,
            '_seq_name_prefix': self._seq_name_prefix,
            '_seq_name_pos_prefix': self._seq_name_pos_prefix,
            '_seq_name_site_prefix': self._seq_name_site_prefix,
            '_pos_state': self._pos_state,
            '_site_state': self._site_state,
            '_num_sites': self._num_sites,
            '_style_insertion': self._style_insertion,
            '_style_background': self._style_background,
            '_outer_region': self._outer_region,
        }
