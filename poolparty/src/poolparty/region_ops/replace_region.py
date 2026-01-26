"""Replace region content with sequences from another Pool."""
from numbers import Real
from poolparty.types import Optional, SeqStyle
import numpy as np

from ..utils.parsing_utils import validate_single_region
from ..operation import Operation
from ..utils import dna_utils


def replace_region(
    pool,
    content_pool,
    region_name: str,
    iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
    # Internal parameters for insertion_scan composite naming
    _seq_name_prefix: Optional[str] = None,
    _seq_name_pos_prefix: Optional[str] = None,
    _seq_name_site_prefix: Optional[str] = None,
    _pos_state=None,  # State object for position naming
    _site_state=None,  # State object for site naming
    _num_sites: Optional[int] = None,
    _style: Optional[str] = None,
):
    """
    Replace a region with content from another Pool.

    The region (including its tags and any content) is replaced with
    sequences from content_pool. If the region has strand='-', the
    content is reverse-complemented before insertion.

    Parameters
    ----------
    pool : Pool or str
        Background Pool or sequence string containing the region.
    content_pool : Pool or str
        Pool or sequence string to insert at the region position.
    region_name : str
        Name of the region to replace.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    _factory_name: Optional[str], default=None
        Sets default name of the resulting operation

    Returns
    -------
    Pool
        A Pool yielding pool sequences with the region replaced by
        content_pool sequences.

    Examples
    --------
    >>> with pp.Party():
    ...     # Replace region with content from another pool
    ...     bg = pp.from_seq('ACGT<insert/>TTTT')
    ...     inserts = pp.from_seqs(['AAA', 'GGG'], mode='sequential')
    ...     result = pp.replace_region(bg, inserts, 'insert')
    ...     # Result yields: 'ACGTAAATTTT', 'ACGTGGGTTTT'
    ...
    ...     # With strand='-', content is reverse-complemented
    ...     bg = pp.from_seq("ACGT<region strand='-'>XX</region>TTTT")
    ...     content = pp.from_seq('AAA')
    ...     result = pp.replace_region(bg, content, 'region')
    ...     # Result: 'ACGTTTTTTTT' (TTT is reverse complement of AAA)
    """
    from ..fixed_ops.from_seq import from_seq
    from ..pool import Pool
    
    # Convert strings to pools if needed
    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    content_pool = from_seq(content_pool) if isinstance(content_pool, str) else content_pool
    
    op = ReplaceRegionOp(
        parent_pool=pool_obj,
        content_pool=content_pool,
        region_name=region_name,
        name=None,
        iter_order=iter_order,
        _factory_name=_factory_name,
        _seq_name_prefix=_seq_name_prefix,
        _seq_name_pos_prefix=_seq_name_pos_prefix,
        _seq_name_site_prefix=_seq_name_site_prefix,
        _pos_state=_pos_state,
        _site_state=_site_state,
        _num_sites=_num_sites,
        _style=_style,
    )
    result_pool = Pool(operation=op)
    
    # The region is replaced, so remove it from the pool's region set
    result_pool._untrack_region(region_name)
    
    return result_pool


class ReplaceRegionOp(Operation):
    """Replace a region with content from another pool."""
    
    factory_name = "replace_region"
    design_card_keys = []
    
    def __init__(
        self,
        parent_pool,
        content_pool,
        region_name: str,
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
        _style: Optional[str] = None,
    ) -> None:
        self.region_name = region_name
        
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
        self._style = _style
        
        # The operation itself has num_values=1 because it doesn't add its own states.
        # The total number of output states comes from the product of parent pool counters.
        # When content_pool has multiple states (e.g., from mutagenize), those states
        # are inherited via the parent counter product.
        super().__init__(
            parent_pools=[parent_pool, content_pool],
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
        parent_styles: list[SeqStyle] | None = None,
    ) -> dict:
        """Replace region in bg_seq with content_seq."""
        bg_seq = parent_seqs[0]
        content_seq = parent_seqs[1]
        
        # Find and validate the region
        region = validate_single_region(bg_seq, self.region_name)
        
        # If strand='-', reverse complement the content before insertion
        if region.strand == '-':
            content_seq = dna_utils.reverse_complement(content_seq)
        
        # Build result: prefix + content + suffix
        prefix = bg_seq[:region.start]
        suffix = bg_seq[region.end:]
        result_seq = prefix + content_seq + suffix
        
        # Use SeqStyle for clean style assembly
        bg_style = SeqStyle.from_parent(parent_styles, 0, len(bg_seq))
        content_style = SeqStyle.from_parent(parent_styles, 1, len(parent_seqs[1]))
        
        output_style = SeqStyle.join([
            bg_style[:region.start],                           # Prefix
            content_style[:].reversed(region.strand == '-'),   # Content (maybe reversed)
            bg_style[region.end:],                             # Suffix
        ])
        
        # Apply style to all inserted content positions
        if self._style is not None:
            ins_start = region.start
            ins_end = ins_start + len(content_seq)
            ins_positions = np.arange(ins_start, ins_end, dtype=np.int64)
            output_style = output_style.add_style(self._style, ins_positions)
        
        return {'seq': result_seq, 'style': output_style}
    
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
            name_parts.append(f'{self._seq_name_prefix}_{w}')
        if self._seq_name_pos_prefix:
            name_parts.append(f'{self._seq_name_pos_prefix}_{pos_idx}')
        if self._seq_name_site_prefix:
            name_parts.append(f'{self._seq_name_site_prefix}_{site_idx}')
        
        return '.'.join(name_parts) if name_parts else None
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'parent_pool': self.parent_pools[0],
            'content_pool': self.parent_pools[1],
            'region_name': self.region_name,
            'name': None,
            'iter_order': self.iter_order,
            '_seq_name_prefix': self._seq_name_prefix,
            '_seq_name_pos_prefix': self._seq_name_pos_prefix,
            '_seq_name_site_prefix': self._seq_name_site_prefix,
            '_pos_state': self._pos_state,
            '_site_state': self._site_state,
            '_num_sites': self._num_sites,
            '_style': self._style,
        }
