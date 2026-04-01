"""RegionScan operation - insert XML region tags at scanning positions."""

from numbers import Real

import numpy as np

from poolparty.types import CardsType, Integral, ModeType, Optional, PositionsType, RegionType, Seq, SeqStyle

from ..operation import Operation
from ..utils import build_scan_cache, validate_positions
from ..utils.dna_seq import DnaSeq
from ..utils.parsing_utils import build_region_tags, get_nontag_positions


def region_scan(
    pool,
    tag_name: str = "region",
    positions: PositionsType = None,
    region: RegionType = None,
    remove_tags: Optional[bool] = None,
    region_length: int = 0,
    prefix: Optional[str] = None,
    mode: ModeType = "random",
    num_states: Optional[Integral] = None,
    iter_order: Optional[Real] = None,
    cards: CardsType = None,
    _factory_name: Optional[str] = None,
):
    """
    Insert XML-style region tags at scanning positions in a sequence.

    Parameters
    ----------
    pool : Pool or str
        Input Pool or sequence string to insert tags into.
    tag_name : str, default='region'
        Name for the XML tag to insert.
    positions : PositionsType, default=None
        Valid insertion positions (0-based). If None, all positions are valid.
    region : RegionType, default=None
        Region to constrain the scan to. Can be region name (str) or [start, stop].
    remove_tags : Optional[bool], default=None
        If True and region is a region name, remove tags from output.
    region_length : Integral, default=0
        Length of sequence to encompass. 0 creates zero-length regions (<name/>),
        >0 creates region tags (<name>BASES</name>).
    mode : ModeType, default='random'
        Position selection mode: 'random' or 'sequential'.
    _factory_name : Optional[str], default=None
        Sets default name of the resulting operation

    Returns
    -------
    Pool
        A Pool yielding sequences with the region tags inserted at selected positions.
    """
    from ..fixed_ops.from_seq import from_seq
    from ..party import get_active_party

    pool = from_seq(pool) if isinstance(pool, str) else pool

    if mode not in ("random", "sequential"):
        raise ValueError(f"mode must be 'random' or 'sequential', got '{mode}'")

    if region_length < 0:
        raise ValueError(f"region_length must be >= 0, got {region_length}")

    party = get_active_party()

    op = RegionScanOp(
        parent_pool=pool,
        region_name=tag_name,
        positions=positions,
        region=region,
        remove_tags=remove_tags,
        region_length=int(region_length),
        prefix=prefix,
        mode=mode,
        num_states=num_states,
        name=None,
        iter_order=iter_order,
        cards=cards,
        _factory_name=_factory_name,
    )

    registered_region = party.register_region(tag_name, region_length)
    pool_class = type(pool)
    result_pool = pool_class(operation=op)
    result_pool.add_region(registered_region)

    return result_pool


class RegionScanOp(Operation):
    """Insert XML region tags at scanning positions."""

    factory_name = "region_scan"
    design_card_keys = [
        "position_index",
        "start",
        "end",
        "name",
        "region_seq",
    ]

    def __init__(
        self,
        parent_pool,
        region_name: str,
        positions: PositionsType = None,
        region: RegionType = None,
        remove_tags: Optional[bool] = None,
        spacer_str: str = "",
        region_length: int = 0,
        prefix: Optional[str] = None,
        mode: ModeType = "random",
        num_states: Optional[Integral] = None,
        name: Optional[str] = None,
        iter_order: Optional[Real] = None,
        cards: CardsType = None,
        _factory_name: Optional[str] = None,
    ) -> None:
        """Initialize RegionScanOp."""
        from ..party import get_active_party

        self.region_name = region_name
        self._positions = positions
        self._mode = mode
        self._region_length = region_length
        self._region = region  # Store early for logging
        self._valid_positions = None
        self._sequential_cache = None

        # Determine effective seq_length for cache building:
        # If region is a region name, use the region's registered length
        # If region is [start, stop], use stop - start
        # Otherwise, use the parent pool's seq_length
        if isinstance(region, str):
            party = get_active_party()
            try:
                constraint_region = party.get_region_by_name(region)
                self._seq_length = constraint_region.seq_length
            except (ValueError, KeyError):
                self._seq_length = parent_pool.seq_length
        elif region is not None:
            self._seq_length = int(region[1]) - int(region[0])
        else:
            self._seq_length = parent_pool.seq_length

        # Set factory name if provided
        if _factory_name is not None:
            self.factory_name = _factory_name

        natural_num_states = None
        if mode == "sequential":
            if self._seq_length is not None:
                natural_num_states = self._build_caches()
                if num_states is None:
                    num_states = natural_num_states
            else:
                raise ValueError(
                    "mode='sequential' requires known scan geometry. "
                    "Provide a parent with known seq_length, or a region "
                    "constraint with known length."
                )

        super().__init__(
            parent_pools=[parent_pool],
            num_states=num_states,
            mode=mode,
            seq_length=parent_pool.seq_length,
            name=name,
            iter_order=iter_order,
            prefix=prefix,
            region=region,
            remove_tags=remove_tags,
            _natural_num_states=natural_num_states,
            cards=cards,
        )

    def _build_caches(self) -> int:
        """Build caches for sequential enumeration based on seq_length."""
        return build_scan_cache(
            seq_length=self._seq_length,
            item_length=self._region_length,
            positions=self._positions,
            error_context="region tag insertion",
        )

    def _get_valid_region_positions(self, seq: str) -> tuple[list[int], list[int]]:
        """Get valid region tag insertion positions, excluding tag interiors.

        Returns tuple of (valid_nontag_indices, nontag_positions) where:
        - valid_nontag_indices: indices into nontag_positions that are valid start positions
        - nontag_positions: literal positions of all non-tag characters
        """
        # Get positions not inside existing tags
        nontag_positions = get_nontag_positions(seq)

        # For region tags, ensure there's room for region_length bases
        if self._region_length > 0:
            # Valid indices are those where we have room for region_length consecutive non-tag chars
            max_valid_idx = len(nontag_positions) - self._region_length
            if max_valid_idx < 0:
                all_valid_indices = []
            else:
                all_valid_indices = list(range(max_valid_idx + 1))
        else:
            # For zero-length regions, all positions are valid plus end (len of seq)
            all_valid_indices = list(range(len(nontag_positions) + 1))

        # Apply user position filter
        if self._positions is not None:
            indices = validate_positions(
                self._positions,
                max_position=len(all_valid_indices) - 1,
                min_position=0,
            )
            filtered_indices = [all_valid_indices[i] for i in indices]
            return filtered_indices, nontag_positions

        return all_valid_indices, nontag_positions

    def _compute_core(
        self,
        parents: list[Seq],
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[Seq, dict]:
        """Return Seq with region tags inserted and design card."""

        seq = parents[0].string

        valid_indices, nontag_positions = self._get_valid_region_positions(seq)
        if len(valid_indices) == 0:
            raise ValueError("No valid positions for region tag insertion")

        # Select position
        if self.mode == "random":
            if rng is None:
                raise RuntimeError(f"{self.mode.capitalize()} mode requires RNG")
            position_index = int(rng.integers(0, len(valid_indices)))
        else:
            # Use state 0 when inactive (state is None)
            state = self.state.value
            state = 0 if state is None else state
            position_index = state % len(valid_indices)

        # Build region tags - extract content using non-tag indices
        nontag_idx = valid_indices[position_index]
        if self._region_length > 0:
            # Extract content from non-tag characters only
            content = "".join(
                seq[nontag_positions[i]]
                for i in range(nontag_idx, nontag_idx + self._region_length)
            )
            region_tag = build_region_tags(self.region_name, content)
            start = nontag_idx
            stop = nontag_idx + self._region_length
            # Get raw sequence from literal start to end (including tags/gaps, excluding new region_tag)
            start_literal = nontag_positions[nontag_idx]
            end_nontag_idx = nontag_idx + self._region_length
            if end_nontag_idx < len(nontag_positions):
                end_literal = nontag_positions[end_nontag_idx]
            else:
                end_literal = nontag_positions[-1] + 1 if nontag_positions else len(seq)
            marked_seq = seq[start_literal:end_literal]
        else:
            region_tag = build_region_tags(self.region_name, "")
            start = nontag_idx
            stop = nontag_idx
            marked_seq = ""

        # Insert tags at position
        if self._region_length > 0:
            # Region tags: replace content with tags
            # Get literal start and end positions from non-tag indices
            start_literal = nontag_positions[nontag_idx]
            end_nontag_idx = nontag_idx + self._region_length
            # End position is the literal position of the first char AFTER the region
            if end_nontag_idx < len(nontag_positions):
                end_literal = nontag_positions[end_nontag_idx]
            else:
                # One past the last non-tag character (preserves trailing tags)
                end_literal = nontag_positions[-1] + 1 if nontag_positions else len(seq)
            result_seq = seq[:start_literal] + region_tag + seq[end_literal:]

            # Adjust parent styles for region tag insertion
            # Opening tag length is from start of region_tag to first '>' + 1
            opening_tag_end = region_tag.index(">") + 1
            opening_tag_len = opening_tag_end
            # Closing tag length is the rest
            closing_tag_len = len(region_tag) - opening_tag_len - len(content)
            total_tag_len = opening_tag_len + closing_tag_len
        else:
            # Zero-length region: insert at position
            if nontag_idx < len(nontag_positions):
                raw_position = nontag_positions[nontag_idx]
            else:
                raw_position = len(seq)  # Insert at end
            result_seq = seq[:raw_position] + region_tag + seq[raw_position:]

        # Adjust parent styles to account for tag insertion
        seq_len = len(seq)
        input_style = parents[0].style

        if input_style is None:
            # Styles suppressed
            output_style = None
        elif self._region_length > 0:
            # Region tags: split and reassemble with tag spacers
            output_style = SeqStyle.join(
                [
                    input_style[:start_literal],  # Before tag
                    SeqStyle.empty(opening_tag_len),  # Opening tag spacer
                    input_style[start_literal:end_literal],  # Inside region
                    SeqStyle.empty(closing_tag_len),  # Closing tag spacer
                    input_style[end_literal:],  # After tag
                ]
            )
        else:
            # Zero-length region: insert tag spacer at position
            output_style = SeqStyle.join(
                [
                    input_style[:raw_position],  # Before tag
                    SeqStyle.empty(len(region_tag)),  # Tag spacer
                    input_style[raw_position:],  # After tag
                ]
            )

        output_seq = DnaSeq(result_seq, output_style)

        if self._party.suppress_cards:
            return output_seq, {}

        return output_seq, {
            "position_index": position_index,
            "start": start,
            "end": stop,
            "name": self.region_name,
            "region_seq": region_tag,
        }
