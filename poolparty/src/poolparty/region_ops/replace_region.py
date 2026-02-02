"""Replace region content with sequences from another Pool."""

from numbers import Real

import numpy as np

from poolparty.types import Optional, Seq

from ..operation import Operation
from ..utils.parsing_utils import validate_single_region_from_list


def replace_region(
    pool,
    content_pool,
    region_name: str,
    rc: bool = False,
    iter_order: Optional[Real] = None,
    prefix: Optional[str] = None,
    _factory_name: Optional[str] = None,
    _style: Optional[str] = None,
):
    """
    Replace a region with content from another Pool.

    The region (including its tags and any content) is replaced with
    sequences from content_pool.

    Parameters
    ----------
    pool : Pool or str
        Background Pool or sequence string containing the region.
    content_pool : Pool or str
        Pool or sequence string to insert at the region position.
    region_name : str
        Name of the region to replace.
    rc : bool, default=False
        If True, reverse-complement the content before insertion.
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
    ...     # With rc=True, content is reverse-complemented
    ...     bg = pp.from_seq("ACGT<region>XX</region>TTTT")
    ...     content = pp.from_seq('AAA')
    ...     result = pp.replace_region(bg, content, 'region', rc=True)
    ...     # Result: 'ACGTTTTTTTT' (TTT is reverse complement of AAA)
    """
    from ..fixed_ops.from_seq import from_seq
    from ..dna_pool import DnaPool

    # Convert strings to pools if needed
    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    content_pool = from_seq(content_pool) if isinstance(content_pool, str) else content_pool

    op = ReplaceRegionOp(
        parent_pool=pool_obj,
        content_pool=content_pool,
        region_name=region_name,
        rc=rc,
        name=None,
        iter_order=iter_order,
        prefix=prefix,
        _factory_name=_factory_name,
        _style=_style,
    )
    # Preserve the pool type from the first parent
    pool_class = type(pool_obj)
    result_pool = pool_class(operation=op)

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
        rc: bool = False,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        _factory_name: Optional[str] = None,
        _style: Optional[str] = None,
    ) -> None:
        self.region_name = region_name
        self.rc = rc

        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name

        self._style = _style

        # The operation itself has num_values=1 because it doesn't add its own states.
        # The total number of output states comes from the product of parent pool counters.
        # When content_pool has multiple states (e.g., from mutagenize), those states
        # are inherited via the parent counter product.
        super().__init__(
            parent_pools=[parent_pool, content_pool],
            num_states=1,  # Operation doesn't add states
            mode="fixed",  # Mode is determined by parent counters
            seq_length=None,  # Variable length
            name=name,
            iter_order=iter_order,
            prefix=prefix,
        )

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Replace region in bg_seq with content_seq."""
        bg_seq = parents[0]
        content_seq_obj = parents[1]

        # Find and validate the region (use pre-parsed regions if available)
        if bg_seq.regions:
            region = validate_single_region_from_list(
                bg_seq.regions, self.region_name, bg_seq.string
            )
        else:
            # Fall back to parsing if regions not available (e.g., from slicing)
            from ..utils.parsing_utils import validate_single_region

            region = validate_single_region(bg_seq.string, self.region_name)

        # If rc=True, reverse complement the content before insertion
        if self.rc:
            content_seq_obj = content_seq_obj.reversed()

        # Use Seq slicing for assembly (slices may have partial tags, but join works)
        prefix_seq = parents[0][: region.start]
        suffix_seq = parents[0][region.end :]

        # Join with content (from_string in join will parse the final result)
        output_seq = Seq.join([prefix_seq, content_seq_obj, suffix_seq])

        # Apply style to all inserted content positions
        if self._style is not None:
            ins_start = region.start
            ins_end = ins_start + len(content_seq_obj)
            ins_positions = np.arange(ins_start, ins_end, dtype=np.int64)
            output_seq = output_seq.add_style(self._style, ins_positions)

        return output_seq, {}
