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
    spacer_str: str = '',
    name: Optional[str] = None,
    op_name: Optional[str] = None,
    iter_order: Optional[Real] = None,
    op_iter_order: Optional[Real] = None,
    _factory_name: Optional[str] = None,
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
    name : Optional[str], default=None
        Name for the resulting Pool.
    op_name : Optional[str], default=None
        Name for the underlying Operation.
    iter_order : Optional[Real], default=None
        Iteration order priority for the resulting Pool.
    op_iter_order : Optional[Real], default=None
        Iteration order priority for the underlying Operation.
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
        spacer_str=spacer_str,
        name=op_name,
        iter_order=op_iter_order,
        _factory_name=_factory_name,
    )
    result_pool = Pool(operation=op, name=name, iter_order=iter_order)
    
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
        spacer_str: str = '',
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        self.marker_name = marker_name
        
        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name
        
        # The operation itself has num_states=1 because it doesn't add its own states.
        # The total number of output states comes from the product of parent pool counters.
        # When content_pool has multiple states (e.g., from mutagenize), those states
        # are inherited via the parent counter product.
        super().__init__(
            parent_pools=[bg_pool, content_pool],
            num_states=1,  # Operation doesn't add states
            mode='fixed',  # Mode is determined by parent counters
            seq_length=None,  # Variable length
            name=name,
            iter_order=iter_order,
            spacer_str=spacer_str,
        )
        
        # Store spacer_str for our manual handling in compute_seq_from_card
        # (we handle it manually since we don't use the base class's region-based handling)
        self._spacer_str = spacer_str

    
    def compute_design_card(
        self,
        parent_seqs: list[str],
        rng: Optional[np.random.Generator] = None,
    ) -> dict:
        """Return empty design card (no design decisions)."""
        return {}
    
    def compute_seq_from_card(
        self,
        parent_seqs: list[str],
        card: dict,
    ) -> dict:
        """Replace marker in bg_seq with content_seq."""
        bg_seq = parent_seqs[0]
        content_seq = parent_seqs[1]
        
        # Find and validate the marker
        marker = validate_single_marker(bg_seq, self.marker_name)
        
        # If strand='-', reverse complement the content before insertion
        if marker.strand == '-':
            content_seq = dna.reverse_complement(content_seq)
        
        # Apply spacer_str if specified
        if self._spacer_str:
            content_seq = self._spacer_str + content_seq + self._spacer_str
        
        # Build result: prefix + content + suffix
        prefix = bg_seq[:marker.start]
        suffix = bg_seq[marker.end:]
        result_seq = prefix + content_seq + suffix
        
        return {'seq_0': result_seq}
    
    def _get_copy_params(self) -> dict:
        """Return parameters needed to create a copy of this operation."""
        return {
            'bg_pool': self.parent_pools[0],
            'content_pool': self.parent_pools[1],
            'marker_name': self.marker_name,
            'spacer_str': self._spacer_str,
            'name': None,
            'iter_order': self.iter_order,
        }
