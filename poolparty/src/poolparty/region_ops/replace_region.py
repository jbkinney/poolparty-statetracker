"""Replace region content with sequences from another Pool."""

from numbers import Real

import numpy as np

from poolparty.types import Optional, Seq

from ..operation import Operation
from ..utils.dna_seq import DnaSeq
from ..utils.parsing_utils import validate_single_region_from_list


def replace_region(
    pool,
    content_pool,
    region_name: str,
    rc: bool = False,
    sync: bool = False,
    keep_tags: bool = False,
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
    sync : bool, default=False
        If True, synchronize pool and content_pool so they iterate
        in lock-step (1:1 pairing) instead of a Cartesian product.
    keep_tags : bool, default=False
        If True, preserve the region's XML tags around the new content.
        The region remains tracked in the resulting pool.
    iter_order : Optional[Real], default=None
        Iteration order priority for the Operation.
    prefix : Optional[str], default=None
        Prefix for sequence names in the resulting Pool.

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
    ...     # With sync=True for 1:1 pairing
    ...     bg = pp.from_seqs(['ACGT<bc/>TTTT', 'CCCC<bc/>GGGG'], mode='sequential')
    ...     barcodes = pp.get_barcodes(num_barcodes=2, length=4, seed=42)
    ...     result = pp.replace_region(bg, barcodes, 'bc', sync=True)
    ...     # Each background gets a unique barcode (no Cartesian product)
    ...
    ...     # With keep_tags=True to preserve region tracking
    ...     result = pp.replace_region(bg, barcodes, 'bc', keep_tags=True)
    ...     # Region tags remain: 'ACGT<bc>XXXX</bc>TTTT'
    """
    from ..fixed_ops.from_seq import from_seq

    # Convert strings to pools if needed
    pool_obj = from_seq(pool) if isinstance(pool, str) else pool
    content_pool = from_seq(content_pool) if isinstance(content_pool, str) else content_pool

    if sync:
        from ..state_ops.sync import sync as _sync

        _sync([pool_obj, content_pool])

    op = ReplaceRegionOp(
        parent_pool=pool_obj,
        content_pool=content_pool,
        region_name=region_name,
        rc=rc,
        keep_tags=keep_tags,
        name=None,
        iter_order=iter_order,
        prefix=prefix,
        _factory_name=_factory_name,
        _style=_style,
    )
    # Preserve the pool type from the first parent
    pool_class = type(pool_obj)
    result_pool = pool_class(operation=op)

    if not keep_tags:
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
        keep_tags: bool = False,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        prefix: Optional[str] = None,
        _factory_name: Optional[str] = None,
        _style: Optional[str] = None,
    ) -> None:
        self.region_name = region_name
        self.rc = rc
        self.keep_tags = keep_tags

        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name

        self._style = _style

        # Compute output seq_length when all component lengths are known:
        # output = parent_bio_length - old_region_length + new_content_length
        from ..party import get_active_party

        seq_length = None
        parent_len = parent_pool.seq_length
        content_len = content_pool.seq_length
        party = get_active_party()
        region_len = None
        if party is not None and party.has_region(region_name):
            region_len = party.get_region(region_name).seq_length
        if parent_len is not None and region_len is not None and content_len is not None:
            seq_length = parent_len - region_len + content_len

        super().__init__(
            parent_pools=[parent_pool, content_pool],
            num_states=1,
            mode="fixed",
            seq_length=seq_length,
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

        if self.keep_tags:
            open_tag = f"<{self.region_name}>"
            close_tag = f"</{self.region_name}>"
            wrapped = f"{open_tag}{content_seq_obj.string}{close_tag}"
            if content_seq_obj.style is not None:
                from ..utils.style_utils import SeqStyle

                wrapped_style = SeqStyle.join([
                    SeqStyle.empty(len(open_tag)),
                    content_seq_obj.style,
                    SeqStyle.empty(len(close_tag)),
                ])
            else:
                wrapped_style = None
            content_seq_obj = DnaSeq.from_string(wrapped, wrapped_style)

        # Use Seq slicing for assembly (slices may have partial tags, but join works)
        prefix_seq = parents[0][: region.start]
        suffix_seq = parents[0][region.end :]

        # Join with content (from_string in join will parse the final result)
        output_seq = DnaSeq.join([prefix_seq, content_seq_obj, suffix_seq])

        # Apply style to all inserted content positions
        if self._style is not None:
            ins_start = region.start
            ins_end = ins_start + len(content_seq_obj)
            ins_positions = np.arange(ins_start, ins_end, dtype=np.int64)
            output_seq = output_seq.add_style(self._style, ins_positions)

        return output_seq, {}
